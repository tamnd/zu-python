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
//! The columns themselves are not built here. `zudb::query::column`
//! reads a result down its columns in the engine, in two passes over
//! the rows, and hands back one owned buffer per column in the layout
//! Arrow already uses: values end to end, a validity bitmap that is
//! absent when nothing is null, strings as bytes and offsets. This
//! module takes those buffers and puts an Arrow array around them,
//! which for integers, floats, booleans, strings, dates, times,
//! datetimes and durations is a move and not a copy. `docs/clients/duckdb.md`
//! in the engine tree is why: this file used to walk the whole result
//! once per column to infer a type and once per column per batch to
//! gather pointers, and that transpose was the twenty.
//!
//! What is left to build by hand is what no buffer covers: nodes, rels,
//! paths, lists and records, which arrive as borrowed values and become
//! structs and lists the way they always did. They are also the columns
//! nobody exports a million of.
//!
//! A column has one type, which the engine decides and this module only
//! translates. Two refusals stay here, because they are Arrow's facts
//! and not the engine's: a time with an offset has no Arrow type, and
//! neither has a handle to a graph or a binding table.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BooleanArray, Date32Array, DurationNanosecondArray, Float64Array, Int64Array,
    IntervalMonthDayNanoArray, LargeStringArray, ListArray, NullArray, StringArray, StructArray,
    Time64NanosecondArray, TimestampNanosecondArray, UInt64Array,
};
use arrow::buffer::{BooleanBuffer, Buffer, NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow::datatypes::{
    DataType, Field, FieldRef, Fields, IntervalMonthDayNano, IntervalUnit, Schema, SchemaRef,
    TimeUnit,
};
use arrow::error::ArrowError;
use arrow::ffi_stream::FFI_ArrowArrayStream;
use arrow::record_batch::{RecordBatch, RecordBatchOptions, RecordBatchReader};
use pyo3::prelude::*;
use zu_common::{DurationKind, Temporal};
use zudb::query::column::{ColumnData, ColumnType, Offsets, Validity};
use zudb::query::{QueryResult, Value};

use crate::value::Names;

/// How many rows go in one record batch.
///
/// A result is already in memory and the arrays are built whole, so a
/// batch is a view into them rather than a copy: the boundary exists
/// because readers expect one and because a working set that fits in
/// cache is faster to consume, not because anything is allocated at it.
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

/// A buffer that does not match the type the engine decided for it,
/// which is this module reading its own input wrong.
fn mismatch(name: &str, ty: &ColumnType) -> Snag {
    Snag::Arrow(ArrowError::SchemaError(format!(
        "column '{name}' came back as {} in a buffer that does not hold one",
        ty.name()
    )))
}

/// The refusal for a type Arrow has nowhere to put.
///
/// Two of them, and both are Arrow's facts rather than the engine's,
/// which is why they live in the client and not in `columnar()`.
fn unsupported(name: &str, ty: &ColumnType) -> Snag {
    match ty {
        // Arrow has a time and a timestamp and nothing in between:
        // there is no time-with-offset type to put this in, and
        // dropping the offset would move the value.
        ColumnType::ZonedTime { .. } => Snag::Type(format!(
            "column '{name}' holds a time with an offset, which Arrow has no type for"
        )),
        // GV60 and GV61. A handle is a reference, and a column of
        // references is a column of nothing a frame can hold: the graph
        // is in the file and the binding table is behind the handle. A
        // caller who wants one in a frame reads the rows, where it
        // arrives as the string that names it, or projects the columns
        // of the table instead of the table.
        ColumnType::Graph | ColumnType::BindingTable => Snag::Type(format!(
            "column '{name}' holds a reference to a graph or a binding table, which Arrow has no type for"
        )),
        _ => mismatch(name, ty),
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

/// The Arrow type a column type becomes, and the two places where the
/// answer is that it does not become one.
///
/// The column name rides along because a refusal without it sends
/// somebody to read a schema by hand, and because a nested refusal is
/// still about the column it is nested in.
fn data_type(name: &str, ty: &ColumnType) -> Result<DataType, Snag> {
    Ok(match ty {
        ColumnType::Null => DataType::Null,
        ColumnType::Bool => DataType::Boolean,
        ColumnType::Int => DataType::Int64,
        ColumnType::Float => DataType::Float64,
        ColumnType::Str => DataType::Utf8,
        ColumnType::Date => DataType::Date32,
        ColumnType::LocalTime => DataType::Time64(TimeUnit::Nanosecond),
        ColumnType::LocalDatetime => DataType::Timestamp(TimeUnit::Nanosecond, None),
        ColumnType::ZonedDatetime { offset } => {
            DataType::Timestamp(TimeUnit::Nanosecond, Some(zone(*offset).into()))
        }
        // Arrow has a year-month interval, which is exactly what this
        // is, and pyarrow cannot build a Python array of one: its type
        // id has no class behind it, so reading such a column raises
        // `KeyError: 21`. Month-day-nano is the interval every reader
        // implements, and a year-month duration is one with no days and
        // no nanoseconds in it.
        ColumnType::YearMonth => DataType::Interval(IntervalUnit::MonthDayNano),
        ColumnType::DayTime => DataType::Duration(TimeUnit::Nanosecond),
        ColumnType::Node => DataType::Struct(node_fields()),
        ColumnType::Rel => DataType::Struct(rel_fields()),
        ColumnType::Path => DataType::Struct(path_fields()),
        ColumnType::List(of) => DataType::List(item(data_type(name, of)?)),
        ColumnType::Record(fields) => DataType::Struct(
            fields
                .iter()
                .map(|(held, ty)| Ok(field(held, data_type(name, ty)?)))
                .collect::<Result<Fields, Snag>>()?,
        ),
        ColumnType::ZonedTime { .. } | ColumnType::Graph | ColumnType::BindingTable => {
            return Err(unsupported(name, ty));
        }
    })
}

/// The stream a result exports, batches and schema and all.
///
/// One array per column, built once out of the engine's buffers, and
/// batches that are slices of them. The arrays are built eagerly
/// because the refusals have to happen while there is still a caller to
/// raise them at; the batches are not, so `record_batches` no longer
/// builds a second copy of the table before it hands back a reader.
pub fn stream(result: &QueryResult, names: &Names) -> Result<FFI_ArrowArrayStream, Snag> {
    let columns = result
        .columnar()
        .map_err(|mixed| Snag::Type(mixed.to_string()))?;
    let rows = columns.rows;

    let mut fields = Vec::with_capacity(columns.len());
    let mut arrays = Vec::with_capacity(columns.len());
    for held in columns.columns {
        let array = column(
            held.name,
            &held.ty,
            held.data,
            held.validity,
            held.len,
            names,
        )?;
        fields.push(field(held.name, array.data_type().clone()));
        arrays.push(array);
    }

    let schema = Arc::new(Schema::new(Fields::from(fields)));
    Ok(FFI_ArrowArrayStream::new(Box::new(Slices {
        schema,
        arrays,
        rows,
        at: 0,
        given: 0,
    })))
}

/// The batches, cut out of the finished arrays as they are asked for.
///
/// A result with no rows still has a schema, and a reader that gets no
/// batch at all cannot tell what the columns were, so an empty result
/// gives one empty batch and then stops.
struct Slices {
    schema: SchemaRef,
    arrays: Vec<ArrayRef>,
    rows: usize,
    at: usize,
    given: usize,
}

impl Iterator for Slices {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Result<RecordBatch, ArrowError>> {
        if self.at >= self.rows && self.given > 0 {
            return None;
        }
        let take = BATCH.min(self.rows - self.at);
        let columns: Vec<ArrayRef> = self
            .arrays
            .iter()
            .map(|array| array.slice(self.at, take))
            .collect();
        self.at += take;
        self.given += 1;
        // The row count goes in by hand because a result with no
        // columns still has rows, and a batch of no columns cannot say
        // how many any other way.
        Some(RecordBatch::try_new_with_options(
            self.schema.clone(),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(take)),
        ))
    }
}

impl RecordBatchReader for Slices {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

/// The bitmap Arrow keeps beside a buffer, out of the one the engine
/// filled. Absent means every row has a value, in both layouts.
fn nulls(validity: Option<Validity>) -> Option<NullBuffer> {
    validity
        .map(|held| NullBuffer::new(BooleanBuffer::new(Buffer::from_vec(held.bits), 0, held.len)))
}

/// One whole column as an Arrow array.
///
/// Every flat arm here moves a `Vec` into an Arrow buffer and allocates
/// nothing: the engine filled it in the layout Arrow reads, and the
/// only work left is putting a type and a bitmap around it. The two
/// exceptions are year-month intervals, which are 96 bits in Arrow and
/// 64 in the engine, and the complex types, which have no buffer.
fn column(
    name: &str,
    ty: &ColumnType,
    data: ColumnData<'_>,
    validity: Option<Validity>,
    len: usize,
    names: &Names,
) -> Result<ArrayRef, Snag> {
    let valid = nulls(validity);
    Ok(match data {
        ColumnData::Null => Arc::new(NullArray::new(len)),
        ColumnData::Bool { bits } => Arc::new(BooleanArray::new(
            BooleanBuffer::new(Buffer::from_vec(bits), 0, len),
            valid,
        )),
        ColumnData::Int(values) => Arc::new(Int64Array::new(ScalarBuffer::from(values), valid)),
        ColumnData::Float(values) => Arc::new(Float64Array::new(ScalarBuffer::from(values), valid)),
        ColumnData::Str(held) => match held.offsets {
            Offsets::I32(offsets) => Arc::new(StringArray::try_new(
                OffsetBuffer::new(ScalarBuffer::from(offsets)),
                Buffer::from_vec(held.bytes),
                valid,
            )?),
            // Past two gigabytes of text in one column, which is where
            // a 32 bit offset stops addressing the bytes. Arrow's own
            // answer is the wider type and every reader has it.
            Offsets::I64(offsets) => Arc::new(LargeStringArray::try_new(
                OffsetBuffer::new(ScalarBuffer::from(offsets)),
                Buffer::from_vec(held.bytes),
                valid,
            )?),
        },
        ColumnData::Days(values) => Arc::new(Date32Array::new(ScalarBuffer::from(values), valid)),
        ColumnData::Nanos(values) => {
            let values = ScalarBuffer::from(values);
            match ty {
                ColumnType::LocalTime => Arc::new(Time64NanosecondArray::new(values, valid)),
                ColumnType::LocalDatetime => Arc::new(TimestampNanosecondArray::new(values, valid)),
                ColumnType::ZonedDatetime { offset } => Arc::new(
                    TimestampNanosecondArray::new(values, valid).with_timezone(zone(*offset)),
                ),
                ColumnType::DayTime => Arc::new(DurationNanosecondArray::new(values, valid)),
                // A time with an offset fills a nanosecond buffer like
                // any other time, and this is where it stops.
                _ => return Err(unsupported(name, ty)),
            }
        }
        ColumnData::Months(counts) => Arc::new(IntervalMonthDayNanoArray::new(
            ScalarBuffer::from(months(name, &counts)?),
            valid,
        )),
        // The types with no buffer: nodes, rels, paths, lists, records,
        // and the two handles, which reach here as values and are
        // refused there.
        ColumnData::Complex(values) => build(name, ty, &values, names)?,
    })
}

/// Month counts as the interval Arrow carries them in.
///
/// Arrow counts the months of an interval in 32 bits and the engine
/// counts them in 64, so the far end of the range has nowhere to go.
/// Refusing it is the only honest answer; wrapping would move the value
/// by centuries.
fn months(name: &str, counts: &[i64]) -> Result<Vec<IntervalMonthDayNano>, Snag> {
    let mut months = Vec::with_capacity(counts.len());
    for (row, count) in counts.iter().enumerate() {
        let count = i32::try_from(*count).map_err(|_| {
            Snag::Value(format!(
                "the duration at row {row} of column '{name}' is {count} months, which is more than an Arrow interval holds"
            ))
        })?;
        months.push(IntervalMonthDayNano::new(count, 0, 0));
    }
    Ok(months)
}

/// One column's array, walked out of the values in it.
///
/// This is the slow path and it is where the complex types live: the
/// top level reaches it only for nodes, rels, paths, lists and records,
/// and everything below the top level reaches it always, because a list
/// item and a record field are values wherever they sit.
fn build(name: &str, ty: &ColumnType, values: &[&Value], names: &Names) -> Result<ArrayRef, Snag> {
    Ok(match ty {
        ColumnType::Null => Arc::new(NullArray::new(values.len())),
        ColumnType::Bool => Arc::new(
            values
                .iter()
                .map(|value| match value {
                    Value::Bool(b) => Some(*b),
                    _ => None,
                })
                .collect::<BooleanArray>(),
        ),
        ColumnType::Int => Arc::new(
            values
                .iter()
                .map(|value| match value {
                    Value::Int(n) => Some(*n),
                    _ => None,
                })
                .collect::<Int64Array>(),
        ),
        ColumnType::Float => Arc::new(
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
        ColumnType::Str => Arc::new(
            values
                .iter()
                .map(|value| match value {
                    Value::Str(s) => Some(s.as_str()),
                    _ => None,
                })
                .collect::<StringArray>(),
        ),
        ColumnType::Date => Arc::new(
            temporals(values)
                .map(|temporal| match temporal {
                    Some(Temporal::Date(days)) => Some(*days),
                    _ => None,
                })
                .collect::<Date32Array>(),
        ),
        ColumnType::LocalTime => Arc::new(
            temporals(values)
                .map(|temporal| match temporal {
                    Some(Temporal::LocalTime(nanos)) => Some(*nanos),
                    _ => None,
                })
                .collect::<Time64NanosecondArray>(),
        ),
        ColumnType::LocalDatetime => Arc::new(
            temporals(values)
                .map(|temporal| match temporal {
                    Some(Temporal::LocalDatetime(nanos)) => Some(*nanos),
                    _ => None,
                })
                .collect::<TimestampNanosecondArray>(),
        ),
        ColumnType::ZonedDatetime { offset } => Arc::new(
            temporals(values)
                .map(|temporal| match temporal {
                    Some(Temporal::ZonedDatetime { nanos, .. }) => Some(*nanos),
                    _ => None,
                })
                .collect::<TimestampNanosecondArray>()
                .with_timezone(zone(*offset)),
        ),
        ColumnType::YearMonth => {
            let mut counts = Vec::with_capacity(values.len());
            let mut valid = Vec::with_capacity(values.len());
            for temporal in temporals(values) {
                match temporal {
                    Some(Temporal::Duration(DurationKind::YearMonth, count)) => {
                        counts.push(*count);
                        valid.push(true);
                    }
                    _ => {
                        counts.push(0);
                        valid.push(false);
                    }
                }
            }
            Arc::new(IntervalMonthDayNanoArray::new(
                ScalarBuffer::from(months(name, &counts)?),
                Some(NullBuffer::from(valid)),
            ))
        }
        ColumnType::DayTime => Arc::new(
            temporals(values)
                .map(|temporal| match temporal {
                    Some(Temporal::Duration(DurationKind::DayTime, nanos)) => Some(*nanos),
                    _ => None,
                })
                .collect::<DurationNanosecondArray>(),
        ),
        ColumnType::Node => nodes(values, names)?,
        ColumnType::Rel => rels(values, names)?,
        ColumnType::Path => paths(name, values, names)?,
        ColumnType::List(of) => {
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
                item(data_type(name, of)?),
                OffsetBuffer::new(offsets.into()),
                build(name, of, &flat, names)?,
                Some(NullBuffer::from(valid)),
            )?)
        }
        ColumnType::Record(fields) => {
            let mut children: Vec<ArrayRef> = Vec::with_capacity(fields.len());
            for (at, (_, ty)) in fields.iter().enumerate() {
                let column: Vec<&Value> = values
                    .iter()
                    .map(|value| match value {
                        Value::Record(held) => &held[at].1,
                        _ => &Value::Null,
                    })
                    .collect();
                children.push(build(name, ty, &column, names)?);
            }
            Arc::new(StructArray::try_new(
                match data_type(name, ty)? {
                    DataType::Struct(fields) => fields,
                    _ => return Err(mismatch(name, ty)),
                },
                children,
                Some(present(values)),
            )?)
        }
        ColumnType::ZonedTime { .. } | ColumnType::Graph | ColumnType::BindingTable => {
            return Err(unsupported(name, ty));
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
fn paths(name: &str, values: &[&Value], names: &Names) -> Result<ArrayRef, Snag> {
    let mut node_offsets = vec![0i32];
    let mut rel_offsets = vec![0i32];
    let mut walked_nodes: Vec<&Value> = Vec::new();
    let mut walked_rels: Vec<&Value> = Vec::new();
    for (row, value) in values.iter().enumerate() {
        match value {
            Value::Path(elements) => {
                walked_nodes.extend(elements.iter().step_by(2));
                walked_rels.extend(elements.iter().skip(1).step_by(2));
            }
            // Never in a result: the executor settles a chain into its
            // edges before the rows leave the pipeline.
            Value::Chain(_) => {
                return Err(Snag::Type(format!(
                    "row {row} of column '{name}' is a path chain, which is internal to the executor"
                )));
            }
            _ => {}
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
