//! A result as Arrow columns.
//!
//! This is the fast path out of the database and the reason the client
//! links the engine crates rather than the C ABI. A result comes back
//! from the executor as rows of engine values, and every one of them
//! that becomes a Python object costs an allocation, a type check and a
//! reference count. A result that becomes Arrow costs one buffer per
//! column and no Python objects at all, and pandas, polars and DuckDB
//! all read it without copying it again.
//!
//! What goes across is the Arrow C Data Interface, through the PyCapsule
//! protocol: `__arrow_c_stream__` hands out a capsule holding an
//! `ArrowArrayStream`, and every library that speaks Arrow knows how to
//! take one. There is no pyarrow dependency in the extension, and no
//! version of pyarrow it has to agree with, because the interface is a
//! C struct and not a Python API.
//!
//! A column has one type, which the values decide: the first one that
//! is not null settles it and every value after it has to fit. Integers
//! widen to floats where a column holds both, since that is the one
//! mixture a projection produces by accident and the one no reader is
//! surprised by. Everything else that does not fit is refused, naming
//! the column and the row, because a column that quietly became strings
//! is worse than one that would not build.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{
    ArrayRef, BooleanArray, Date32Array, DurationNanosecondArray, Float64Array, Int64Array,
    IntervalMonthDayNanoArray, ListArray, NullArray, StringArray, StructArray,
    Time64NanosecondArray, TimestampNanosecondArray, UInt64Array,
};
use arrow::buffer::{NullBuffer, OffsetBuffer};
use arrow::datatypes::{
    DataType, Field, FieldRef, Fields, IntervalMonthDayNano, IntervalUnit, Schema, TimeUnit,
};
use arrow::error::ArrowError;
use arrow::ffi_stream::FFI_ArrowArrayStream;
use arrow::record_batch::{RecordBatch, RecordBatchIterator};
use pyo3::prelude::*;
use zu_common::{DurationKind, Temporal};
use zudb::query::{QueryResult, Value};

use crate::value::Names;

/// How many rows go in one record batch.
///
/// A result is already in memory, so this is not about streaming a
/// table too big to hold: it is about the copy. Batching keeps the
/// Arrow buffers a reader has to allocate down to a working set that
/// fits in cache, and it is what every other Arrow producer does.
const BATCH: usize = 65_536;

/// What goes wrong here, with the GIL down and no way to raise yet.
///
/// The two Python classes are the two mistakes: a value of the wrong
/// type in a column is a `TypeError`, and a value of the right type
/// that will not fit is a `ValueError`. Arrow's own errors are neither,
/// and are internal until one of them turns out to be reachable.
pub enum Snag {
    Type(String),
    Value(String),
    Arrow(ArrowError),
}

impl Snag {
    /// The exception this is, once there is a GIL to raise it with.
    pub fn raise(self, _py: Python<'_>) -> PyErr {
        match self {
            Snag::Type(detail) => pyo3::exceptions::PyTypeError::new_err(detail),
            Snag::Value(detail) => pyo3::exceptions::PyValueError::new_err(detail),
            // Nothing here is meant to be reachable: the types are
            // decided before a buffer is filled, so an Arrow error is
            // this module getting it wrong rather than the caller.
            Snag::Arrow(err) => pyo3::exceptions::PyRuntimeError::new_err(format!(
                "arrow could not build the result: {err}"
            )),
        }
    }
}

impl From<ArrowError> for Snag {
    fn from(err: ArrowError) -> Snag {
        Snag::Arrow(err)
    }
}

/// The type of one column, as this module thinks about it.
///
/// Arrow's `DataType` is what it turns into, but not what it is
/// decided as: a node, a rel and a record all become structs, and
/// telling them apart afterwards by their field names would be reading
/// tea leaves. Deciding it once and carrying it is also what makes the
/// second pass, the one that fills the buffers, a match with no
/// re-inspection of the values in it.
#[derive(Clone, PartialEq)]
enum Kind {
    /// Nothing but nulls, which Arrow has a type for.
    Null,
    Bool,
    Int,
    Float,
    Str,
    Date,
    Time,
    LocalDatetime,
    /// A datetime with an offset, in minutes from UTC. The values are
    /// instants, so the offset is how the column prints and not what it
    /// holds; the first one in the column names the zone.
    ZonedDatetime(i16),
    YearMonth,
    DayTime,
    Node,
    Rel,
    Path,
    List(Box<Kind>),
    Record(Vec<(String, Kind)>),
}

impl Kind {
    fn name(&self) -> String {
        match self {
            Kind::Null => "nulls".into(),
            Kind::Bool => "booleans".into(),
            Kind::Int => "integers".into(),
            Kind::Float => "floats".into(),
            Kind::Str => "strings".into(),
            Kind::Date => "dates".into(),
            Kind::Time => "times".into(),
            Kind::LocalDatetime => "datetimes".into(),
            Kind::ZonedDatetime(_) => "zoned datetimes".into(),
            Kind::YearMonth => "year-month durations".into(),
            Kind::DayTime => "day-time durations".into(),
            Kind::Node => "nodes".into(),
            Kind::Rel => "rels".into(),
            Kind::Path => "paths".into(),
            Kind::List(of) => format!("lists of {}", of.name()),
            Kind::Record(_) => "records".into(),
        }
    }

    fn data_type(&self) -> DataType {
        match self {
            Kind::Null => DataType::Null,
            Kind::Bool => DataType::Boolean,
            Kind::Int => DataType::Int64,
            Kind::Float => DataType::Float64,
            Kind::Str => DataType::Utf8,
            Kind::Date => DataType::Date32,
            Kind::Time => DataType::Time64(TimeUnit::Nanosecond),
            Kind::LocalDatetime => DataType::Timestamp(TimeUnit::Nanosecond, None),
            Kind::ZonedDatetime(offset) => {
                DataType::Timestamp(TimeUnit::Nanosecond, Some(zone(*offset).into()))
            }
            // Arrow has a year-month interval, which is exactly what
            // this is, and pyarrow cannot build a Python array of one:
            // its type id has no class behind it, so reading such a
            // column raises `KeyError: 21`. Month-day-nano is the
            // interval every reader implements, and a year-month
            // duration is one with no days and no nanoseconds in it.
            Kind::YearMonth => DataType::Interval(IntervalUnit::MonthDayNano),
            Kind::DayTime => DataType::Duration(TimeUnit::Nanosecond),
            Kind::Node => DataType::Struct(node_fields()),
            Kind::Rel => DataType::Struct(rel_fields()),
            Kind::Path => DataType::Struct(path_fields()),
            Kind::List(of) => DataType::List(item(of.data_type())),
            Kind::Record(fields) => DataType::Struct(
                fields
                    .iter()
                    .map(|(name, kind)| Arc::new(Field::new(name, kind.data_type(), true)))
                    .collect(),
            ),
        }
    }
}

/// Every field is nullable, here and in the nested types, because a
/// null row of a struct column is a null in each of its children and
/// there is no other place to put it.
fn field(name: &str, data_type: DataType) -> FieldRef {
    Arc::new(Field::new(name, data_type, true))
}

fn item(data_type: DataType) -> FieldRef {
    field("item", data_type)
}

fn node_fields() -> Fields {
    Fields::from(vec![
        field("table", DataType::Utf8),
        field("offset", DataType::UInt64),
    ])
}

fn rel_fields() -> Fields {
    Fields::from(vec![
        field("table", DataType::Utf8),
        field("src", DataType::UInt64),
        field("dst", DataType::UInt64),
        field("ord", DataType::UInt64),
    ])
}

fn path_fields() -> Fields {
    Fields::from(vec![
        field(
            "nodes",
            DataType::List(item(DataType::Struct(node_fields()))),
        ),
        field("rels", DataType::List(item(DataType::Struct(rel_fields())))),
    ])
}

/// An offset in minutes as the name Arrow keeps a timezone under.
///
/// A fixed offset rather than a region, because a fixed offset is what
/// the value carries: the engine stores when a zoned datetime happened
/// and how far from UTC it was written, and no amount of arithmetic
/// recovers `Europe/Paris` from `+01:00`.
fn zone(offset: i16) -> String {
    let sign = if offset < 0 { '-' } else { '+' };
    let minutes = offset.unsigned_abs();
    format!("{sign}{:02}:{:02}", minutes / 60, minutes % 60)
}

/// The stream a result exports, batches and schema and all.
///
/// Built whole rather than lazily: the rows are already in memory, so a
/// reader that pulls one batch at a time would only be deferring a copy
/// it is going to ask for anyway, and building it here is what lets the
/// refusals happen while there is still a caller to raise them at.
pub fn stream(result: &QueryResult, names: &Names) -> Result<FFI_ArrowArrayStream, Snag> {
    let kinds = result
        .columns
        .iter()
        .enumerate()
        .map(|(at, name)| infer(name, result.rows.iter().map(|row| &row[at])))
        .collect::<Result<Vec<_>, Snag>>()?;
    let schema = Arc::new(Schema::new(
        result
            .columns
            .iter()
            .zip(&kinds)
            .map(|(name, kind)| field(name, kind.data_type()))
            .collect::<Fields>(),
    ));

    let mut batches = Vec::new();
    let mut at = 0;
    while at < result.rows.len() {
        let rows = &result.rows[at..(at + BATCH).min(result.rows.len())];
        let columns = kinds
            .iter()
            .enumerate()
            .map(|(ix, kind)| {
                let values: Vec<&Value> = rows.iter().map(|row| &row[ix]).collect();
                build(kind, &values, names)
            })
            .collect::<Result<Vec<_>, Snag>>()?;
        batches.push(RecordBatch::try_new(schema.clone(), columns)?);
        at += BATCH;
    }
    // A result with no rows is still a result: it has a schema, and a
    // reader that gets no batch at all cannot tell what the columns
    // were. One empty batch says both.
    if batches.is_empty() {
        let columns = kinds
            .iter()
            .map(|kind| build(kind, &[], names))
            .collect::<Result<Vec<_>, Snag>>()?;
        batches.push(RecordBatch::try_new(schema.clone(), columns)?);
    }

    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);
    Ok(FFI_ArrowArrayStream::new(Box::new(reader)))
}

/// The type of a column, from the values in it.
fn infer<'a>(name: &str, values: impl Iterator<Item = &'a Value>) -> Result<Kind, Snag> {
    let mut kind = Kind::Null;
    for (row, value) in values.enumerate() {
        let found = kind_of(name, row, value)?;
        let (held, arrived) = (kind.name(), found.name());
        kind = unify(kind, found).ok_or_else(|| {
            Snag::Type(format!(
                "column '{name}' mixes {held} and {arrived} at row {row}, and an Arrow column holds one type"
            ))
        })?;
    }
    Ok(kind)
}

/// The type of one value, on its own.
fn kind_of(name: &str, row: usize, value: &Value) -> Result<Kind, Snag> {
    Ok(match value {
        Value::Null => Kind::Null,
        Value::Bool(_) => Kind::Bool,
        Value::Int(_) => Kind::Int,
        Value::Float(_) => Kind::Float,
        Value::Str(_) => Kind::Str,
        Value::Node { .. } => Kind::Node,
        Value::Rel { .. } => Kind::Rel,
        Value::Path(_) => Kind::Path,
        Value::List(items) => {
            let mut of = Kind::Null;
            for item in items {
                let found = kind_of(name, row, item)?;
                let (held, arrived) = (of.name(), found.name());
                of = unify(of, found).ok_or_else(|| {
                    Snag::Type(format!(
                        "the list at row {row} of column '{name}' mixes {held} and {arrived}, and an Arrow list holds one type"
                    ))
                })?;
            }
            Kind::List(Box::new(of))
        }
        Value::Record(fields) => Kind::Record(
            fields
                .iter()
                .map(|(field, value)| Ok((field.clone(), kind_of(name, row, value)?)))
                .collect::<Result<Vec<_>, Snag>>()?,
        ),
        Value::Temporal(temporal) => match temporal {
            Temporal::Date(_) => Kind::Date,
            Temporal::LocalTime(_) => Kind::Time,
            Temporal::LocalDatetime(_) => Kind::LocalDatetime,
            Temporal::ZonedDatetime { offset, .. } => Kind::ZonedDatetime(*offset),
            Temporal::Duration(DurationKind::YearMonth, _) => Kind::YearMonth,
            Temporal::Duration(DurationKind::DayTime, _) => Kind::DayTime,
            // Arrow has a time and a timestamp and nothing in between:
            // there is no time-with-offset type to put this in, and
            // dropping the offset would move the value.
            Temporal::ZonedTime { .. } => {
                return Err(Snag::Type(format!(
                    "row {row} of column '{name}' is a time with an offset, which Arrow has no type for"
                )));
            }
        },
        // GV60 and GV61. A handle is a reference, and a column of
        // references is a column of nothing a frame can hold: the
        // graph is in the file and the binding table is behind the
        // handle. A caller who wants one in a frame reads the rows,
        // where it arrives as the string that names it, or projects
        // the columns of the table instead of the table.
        Value::Graph(_) | Value::BindingTable(_) => {
            return Err(Snag::Type(format!(
                "row {row} of column '{name}' is a reference to a graph or a binding table, which Arrow has no type for"
            )));
        }
        // Never in a result: the executor settles a chain into its
        // edges before the rows leave the pipeline.
        Value::Chain(_) => {
            return Err(Snag::Type(format!(
                "row {row} of column '{name}' is a path chain, which is internal to the executor"
            )));
        }
    })
}

/// The one type two types are both, or `None` when they are not.
fn unify(left: Kind, right: Kind) -> Option<Kind> {
    Some(match (left, right) {
        (Kind::Null, other) | (other, Kind::Null) => other,
        // The one widening: a projection that returns an integer for
        // one row and a float for another means a number, and every
        // reader of the column reads it as one.
        (Kind::Int, Kind::Float) | (Kind::Float, Kind::Int) => Kind::Float,
        // The first zoned value in the column names the zone. Later
        // rows may have been written elsewhere, and they are the same
        // instant either way, so this changes how a column prints and
        // never what it holds.
        (Kind::ZonedDatetime(offset), Kind::ZonedDatetime(_)) => Kind::ZonedDatetime(offset),
        (Kind::List(left), Kind::List(right)) => Kind::List(Box::new(unify(*left, *right)?)),
        (Kind::Record(left), Kind::Record(right)) => {
            if left.len() != right.len() {
                return None;
            }
            let mut fields = Vec::with_capacity(left.len());
            for ((name, left), (other, right)) in left.into_iter().zip(right) {
                if name != other {
                    return None;
                }
                fields.push((name, unify(left, right)?));
            }
            Kind::Record(fields)
        }
        (left, right) if left == right => left,
        _ => return None,
    })
}

/// One column's array, filled from the values in it.
fn build(kind: &Kind, values: &[&Value], names: &Names) -> Result<ArrayRef, Snag> {
    Ok(match kind {
        Kind::Null => Arc::new(NullArray::new(values.len())),
        Kind::Bool => Arc::new(
            values
                .iter()
                .map(|value| match value {
                    Value::Bool(b) => Some(*b),
                    _ => None,
                })
                .collect::<BooleanArray>(),
        ),
        Kind::Int => Arc::new(
            values
                .iter()
                .map(|value| match value {
                    Value::Int(n) => Some(*n),
                    _ => None,
                })
                .collect::<Int64Array>(),
        ),
        Kind::Float => Arc::new(
            values
                .iter()
                .map(|value| match value {
                    Value::Float(f) => Some(*f),
                    // Widened where the column holds both, which is
                    // the only place an integer reaches a float column.
                    Value::Int(n) => Some(*n as f64),
                    _ => None,
                })
                .collect::<Float64Array>(),
        ),
        Kind::Str => Arc::new(
            values
                .iter()
                .map(|value| match value {
                    Value::Str(s) => Some(s.as_str()),
                    _ => None,
                })
                .collect::<StringArray>(),
        ),
        Kind::Date => Arc::new(
            temporals(values)
                .map(|temporal| match temporal {
                    Some(Temporal::Date(days)) => Some(*days),
                    _ => None,
                })
                .collect::<Date32Array>(),
        ),
        Kind::Time => Arc::new(
            temporals(values)
                .map(|temporal| match temporal {
                    Some(Temporal::LocalTime(nanos)) => Some(*nanos),
                    _ => None,
                })
                .collect::<Time64NanosecondArray>(),
        ),
        Kind::LocalDatetime => Arc::new(
            temporals(values)
                .map(|temporal| match temporal {
                    Some(Temporal::LocalDatetime(nanos)) => Some(*nanos),
                    _ => None,
                })
                .collect::<TimestampNanosecondArray>(),
        ),
        Kind::ZonedDatetime(offset) => Arc::new(
            temporals(values)
                .map(|temporal| match temporal {
                    Some(Temporal::ZonedDatetime { nanos, .. }) => Some(*nanos),
                    _ => None,
                })
                .collect::<TimestampNanosecondArray>()
                .with_timezone(zone(*offset)),
        ),
        Kind::YearMonth => {
            let mut months = Vec::with_capacity(values.len());
            for (row, temporal) in temporals(values).enumerate() {
                months.push(match temporal {
                    Some(Temporal::Duration(DurationKind::YearMonth, count)) => {
                        // Arrow counts the months of an interval in 32
                        // bits and the engine counts them in 64, so the
                        // far end of the range has nowhere to go.
                        // Refusing it is the only honest answer;
                        // wrapping would move the value by centuries.
                        let count = i32::try_from(*count).map_err(|_| {
                            Snag::Value(format!(
                                "the duration at row {row} is {count} months, which is more than an Arrow interval holds"
                            ))
                        })?;
                        Some(IntervalMonthDayNano::new(count, 0, 0))
                    }
                    _ => None,
                });
            }
            Arc::new(months.into_iter().collect::<IntervalMonthDayNanoArray>())
        }
        Kind::DayTime => Arc::new(
            temporals(values)
                .map(|temporal| match temporal {
                    Some(Temporal::Duration(DurationKind::DayTime, nanos)) => Some(*nanos),
                    _ => None,
                })
                .collect::<DurationNanosecondArray>(),
        ),
        Kind::Node => nodes(values, names)?,
        Kind::Rel => rels(values, names)?,
        Kind::Path => paths(values, names)?,
        Kind::List(of) => {
            let mut offsets = Vec::with_capacity(values.len() + 1);
            let mut flat: Vec<&Value> = Vec::new();
            let mut valid = Vec::with_capacity(values.len());
            offsets.push(0i32);
            for value in values {
                if let Value::List(items) = value {
                    flat.extend(items.iter());
                    valid.push(true);
                } else {
                    valid.push(false);
                }
                offsets.push(flat.len() as i32);
            }
            Arc::new(ListArray::try_new(
                item(of.data_type()),
                OffsetBuffer::new(offsets.into()),
                build(of, &flat, names)?,
                Some(NullBuffer::from(valid)),
            )?)
        }
        Kind::Record(fields) => {
            let mut children: Vec<ArrayRef> = Vec::with_capacity(fields.len());
            for (at, (_, kind)) in fields.iter().enumerate() {
                let column: Vec<&Value> = values
                    .iter()
                    .map(|value| match value {
                        Value::Record(held) => &held[at].1,
                        _ => &Value::Null,
                    })
                    .collect();
                children.push(build(kind, &column, names)?);
            }
            Arc::new(StructArray::try_new(
                match kind.data_type() {
                    DataType::Struct(fields) => fields,
                    _ => unreachable!("a record is a struct"),
                },
                children,
                Some(present(values)),
            )?)
        }
    })
}

/// The temporal each value holds, or `None` for a value that is not one
/// and for a null.
fn temporals<'a>(values: &'a [&'a Value]) -> impl Iterator<Item = Option<&'a Temporal>> {
    values.iter().map(|value| match value {
        Value::Temporal(temporal) => Some(temporal),
        _ => None,
    })
}

/// Which rows of a struct column are there at all.
fn present(values: &[&Value]) -> NullBuffer {
    NullBuffer::from(
        values
            .iter()
            .map(|value| !matches!(value, Value::Null))
            .collect::<Vec<bool>>(),
    )
}

fn nodes(values: &[&Value], names: &Names) -> Result<ArrayRef, Snag> {
    let table = tables(
        values,
        |value| match value {
            Value::Node { table, .. } => Some(*table),
            _ => None,
        },
        |id| names.node_name(id),
    );
    let offset: UInt64Array = values
        .iter()
        .map(|value| match value {
            Value::Node { offset, .. } => Some(*offset),
            _ => None,
        })
        .collect();
    Ok(Arc::new(StructArray::try_new(
        node_fields(),
        vec![Arc::new(table), Arc::new(offset)],
        Some(present(values)),
    )?))
}

fn rels(values: &[&Value], names: &Names) -> Result<ArrayRef, Snag> {
    let table = tables(
        values,
        |value| match value {
            Value::Rel { table, .. } => Some(*table),
            _ => None,
        },
        |id| names.rel_name(id),
    );
    let end = |pick: fn(&Value) -> Option<u64>| -> UInt64Array {
        values.iter().map(|value| pick(value)).collect()
    };
    Ok(Arc::new(StructArray::try_new(
        rel_fields(),
        vec![
            Arc::new(table),
            Arc::new(end(|value| match value {
                Value::Rel { src, .. } => Some(*src),
                _ => None,
            })),
            Arc::new(end(|value| match value {
                Value::Rel { dst, .. } => Some(*dst),
                _ => None,
            })),
            Arc::new(end(|value| match value {
                Value::Rel { ord, .. } => Some(*ord),
                _ => None,
            })),
        ],
        Some(present(values)),
    )?))
}

/// The table name of every row, borrowed rather than copied.
///
/// The catalog owns the names and a column holds as many rows as the
/// result does, so the names go in by reference and the only string
/// built here is the stand-in for a table the catalog no longer has,
/// which is one per missing table rather than one per row.
fn tables<'a>(
    values: &[&Value],
    id_of: impl Fn(&Value) -> Option<u32>,
    name_of: impl Fn(u32) -> Option<&'a str>,
) -> StringArray {
    let mut gone: HashMap<u32, String> = HashMap::new();
    for value in values {
        if let Some(id) = id_of(value)
            && name_of(id).is_none()
        {
            gone.entry(id).or_insert_with(|| format!("#{id}"));
        }
    }
    values
        .iter()
        .map(|value| id_of(value).map(|id| name_of(id).unwrap_or_else(|| gone[&id].as_str())))
        .collect()
}

/// A path column, as the two lists a walk is.
///
/// A path is nodes and edges alternating, and Arrow has no type for a
/// list whose elements alternate between two structs. Two lists say the
/// same thing without a union in the middle of it: the nodes in the
/// order the walk visits them, the edges in the order it crosses them,
/// and one more node than edge.
fn paths(values: &[&Value], names: &Names) -> Result<ArrayRef, Snag> {
    let mut node_offsets = vec![0i32];
    let mut rel_offsets = vec![0i32];
    let mut walked_nodes: Vec<&Value> = Vec::new();
    let mut walked_rels: Vec<&Value> = Vec::new();
    for value in values {
        if let Value::Path(elements) = value {
            walked_nodes.extend(elements.iter().step_by(2));
            walked_rels.extend(elements.iter().skip(1).step_by(2));
        }
        node_offsets.push(walked_nodes.len() as i32);
        rel_offsets.push(walked_rels.len() as i32);
    }
    let nodes = ListArray::try_new(
        item(DataType::Struct(node_fields())),
        OffsetBuffer::new(node_offsets.into()),
        nodes(&walked_nodes, names)?,
        Some(present(values)),
    )?;
    let rels = ListArray::try_new(
        item(DataType::Struct(rel_fields())),
        OffsetBuffer::new(rel_offsets.into()),
        rels(&walked_rels, names)?,
        Some(present(values)),
    )?;
    Ok(Arc::new(StructArray::try_new(
        path_fields(),
        vec![Arc::new(nodes), Arc::new(rels)],
        Some(present(values)),
    )?))
}
