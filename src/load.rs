//! Building a database out of columns and an edge list.
//!
//! A row at a time through `INSERT` is the wrong shape for loading
//! data and the wrong shape for making a graph: every row is parsed,
//! bound and committed, and a rel table cannot be made that way at
//! all, because the statement that would make one says which two
//! tables it joins only for the edge it is writing. This is the other
//! shape, and it is the one the C ABI's loader has: a table's columns
//! whole, an edge list whole, one file written once.
//!
//! What it writes is what a bulk load writes: a node table with a row
//! per element of every column, a rel table holding the edges between
//! those rows, and a primary-key index over the rows so a lookup by
//! key does not scan. Edges name rows by position, counting from zero,
//! because at load time a row has no other name.

use std::path::PathBuf;

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDate, PyDateTime, PyDelta, PyDict, PyTime};
use zu_common::DurationKind;
use zu_common::temporal::days_from_civil;
use zudb::zu1::file::Zu1File;
use zudb::zu1::graph::bulk_load_keyed;
use zudb::zu1::props::{PropValues, store_props};

use crate::error::to_py_err;
use crate::value::{Duration, clock_nanos};

/// One column, reduced to the vector the property store keeps it in.
///
/// Owned rather than borrowed from the caller's lists, because a
/// Python list holds objects and the store holds numbers: there is
/// nothing here to borrow. Which arm a column is comes from its first
/// value, and every value after it has to be that arm too.
enum Column {
    Int(Vec<u64>),
    Float(Vec<f64>),
    Bool(Vec<bool>),
    Str(Vec<Vec<u8>>),
    Date(Vec<i32>),
    LocalTime(Vec<i64>),
    LocalDatetime(Vec<i64>),
    Duration(DurationKind, Vec<i64>),
}

impl Column {
    /// How many values are in it, which is how many rows the table has
    /// if this is the first column and a refusal if it is not.
    fn len(&self) -> usize {
        match self {
            Column::Int(v) => v.len(),
            Column::Float(v) => v.len(),
            Column::Bool(v) => v.len(),
            Column::Str(v) => v.len(),
            Column::Date(v) => v.len(),
            Column::LocalTime(v) => v.len(),
            Column::LocalDatetime(v) => v.len(),
            Column::Duration(_, v) => v.len(),
        }
    }
}

/// Writes a new database at `path` and answers what went into it.
///
/// The path must not exist. A bulk load builds a database rather than
/// adding to one, so a path that already holds one is a caller who
/// meant a different path, and overwriting it would be the worst
/// possible reading of the call.
///
/// `columns` is a dictionary of column name to a list of values, all
/// of them the same length, which is the number of rows the node table
/// gets. `edges` is a sequence of pairs of row numbers. Either may be
/// left out: a graph with no properties is a graph, and so is one with
/// no edges.
#[pyfunction]
#[pyo3(signature = (path, *, nodes, rels = "rel", columns = None, edges = None, rows = None))]
pub fn load(
    py: Python<'_>,
    path: PathBuf,
    nodes: &str,
    rels: &str,
    columns: Option<&Bound<'_, PyDict>>,
    edges: Option<&Bound<'_, PyAny>>,
    rows: Option<u64>,
) -> PyResult<Py<PyDict>> {
    if nodes.is_empty() || rels.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "a table has a name, and a load names both the node table and the rel table",
        ));
    }
    let built = build(columns)?;
    let rows = match (rows, built.first()) {
        (Some(rows), Some((name, column))) if column.len() as u64 != rows => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "column '{name}' holds {} values against the {rows} rows this load asks for",
                column.len()
            )));
        }
        (Some(rows), _) => rows,
        (None, Some((_, column))) => column.len() as u64,
        (None, None) => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "a load with no columns has no rows to count, so it has to be told how many",
            ));
        }
    };
    let pairs = pairs(edges, rows)?;

    // Released for the write, which is the whole cost of a load: the
    // edges are sorted, the graph is built, and every column is
    // encoded and written to disk. Everything above this line was
    // reading Python objects, which needs the GIL and cannot be done
    // without it.
    let written = py.detach(|| -> zudb::Result<(usize, usize)> {
        let mut db = Zu1File::create(&path)?;
        let mut pairs = pairs;
        pairs.sort_unstable();
        pairs.dedup();
        bulk_load_keyed(&mut db, nodes, rels, rows, &pairs, None)?;
        if !built.is_empty() {
            // The store wants a slice of slices for a string column,
            // which a `Vec<Vec<u8>>` is not, so the row borrows are
            // built first and handed over after.
            let strings: Vec<Vec<&[u8]>> = built
                .iter()
                .map(|(_, column)| match column {
                    Column::Str(v) => v.iter().map(Vec::as_slice).collect(),
                    _ => Vec::new(),
                })
                .collect();
            let props: Vec<(&str, PropValues<'_>)> = built
                .iter()
                .zip(&strings)
                .map(|((name, column), strings)| {
                    let values = match column {
                        Column::Str(_) => PropValues::Str(strings),
                        Column::Int(v) => PropValues::Int(v),
                        Column::Float(v) => PropValues::Float(v),
                        Column::Bool(v) => PropValues::Bool(v),
                        Column::Date(v) => PropValues::Date(v),
                        Column::LocalTime(v) => PropValues::LocalTime(v),
                        Column::LocalDatetime(v) => PropValues::LocalDatetime(v),
                        Column::Duration(kind, v) => PropValues::Duration(*kind, v),
                    };
                    (name.as_str(), values)
                })
                .collect();
            store_props(&mut db, nodes, &props)?;
        }
        Ok((built.len(), pairs.len()))
    });
    let (written_columns, written_edges) = written.map_err(|err| to_py_err(py, err))?;

    let stats = PyDict::new(py);
    stats.set_item("nodes", rows)?;
    stats.set_item("rels", written_edges)?;
    stats.set_item("columns", written_columns)?;
    Ok(stats.unbind())
}

/// Every column, in the order the dictionary holds them, which is the
/// order they were written.
fn build(columns: Option<&Bound<'_, PyDict>>) -> PyResult<Vec<(String, Column)>> {
    let Some(columns) = columns else {
        return Ok(Vec::new());
    };
    let mut built: Vec<(String, Column)> = Vec::with_capacity(columns.len());
    for (name, values) in columns.iter() {
        let name = name.extract::<String>()?;
        if name.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "a column has a name",
            ));
        }
        let column = column(&name, &values)?;
        if let Some((first, had)) = built.first().map(|(name, column)| (name, column.len()))
            && column.len() != had
        {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "column '{name}' holds {} values and column '{first}' holds {had}, and a table is as wide as it is long",
                column.len()
            )));
        }
        built.push((name, column));
    }
    Ok(built)
}

/// One column, read out of a sequence of Python objects.
///
/// The first value settles what the column is and every value after it
/// has to be the same thing. There is no null: a column that holds one
/// cannot be loaded this way, so a null here could only be refused,
/// and refusing it where it is named is better than refusing it at the
/// end of a million rows.
fn column(name: &str, values: &Bound<'_, PyAny>) -> PyResult<Column> {
    let mut column: Option<Column> = None;
    for (row, value) in values.try_iter()?.enumerate() {
        let value = value?;
        let mismatch = |want: &str| {
            let got = value
                .get_type()
                .getattr("__name__")
                .and_then(|name| name.extract::<String>())
                .unwrap_or_else(|_| "unknown".to_string());
            pyo3::exceptions::PyTypeError::new_err(format!(
                "column '{name}' holds {want} and row {row} is of type '{got}'"
            ))
        };
        match column.as_mut() {
            None => column = Some(started(name, row, &value)?),
            // Bool before int, since in Python every bool is an int
            // and a column of `True` is not a column of ones.
            Some(Column::Bool(v)) => v.push(
                value
                    .cast::<PyBool>()
                    .map_err(|_| mismatch("booleans"))?
                    .is_true(),
            ),
            Some(Column::Int(v)) => {
                if value.cast::<PyBool>().is_ok() {
                    return Err(mismatch("integers"));
                }
                v.push(value.extract::<i64>().map_err(|_| mismatch("integers"))? as u64);
            }
            Some(Column::Float(v)) => {
                v.push(value.extract::<f64>().map_err(|_| mismatch("floats"))?)
            }
            Some(Column::Str(v)) => v.push(
                value
                    .extract::<String>()
                    .map_err(|_| mismatch("strings"))?
                    .into_bytes(),
            ),
            // Datetime before date, since a datetime is a date and
            // reading one as the other would throw the time away.
            Some(Column::LocalDatetime(v)) => {
                let dt = value
                    .cast::<PyDateTime>()
                    .map_err(|_| mismatch("datetimes"))?;
                v.push(datetime_nanos(dt)?);
            }
            Some(Column::Date(v)) => {
                if value.cast::<PyDateTime>().is_ok() {
                    return Err(mismatch("dates"));
                }
                let date = value.cast::<PyDate>().map_err(|_| mismatch("dates"))?;
                v.push(date_days(date)?);
            }
            Some(Column::LocalTime(v)) => {
                let time = value.cast::<PyTime>().map_err(|_| mismatch("times"))?;
                v.push(clock_nanos(time.as_any())?);
            }
            Some(Column::Duration(kind, v)) => v.push(duration_count(*kind, &value, mismatch)?),
        }
    }
    column.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "column '{name}' is empty, and an empty column says nothing about what it would hold"
        ))
    })
}

/// The column a first value starts, with that value already in it.
fn started(name: &str, row: usize, value: &Bound<'_, PyAny>) -> PyResult<Column> {
    if let Ok(b) = value.cast::<PyBool>() {
        return Ok(Column::Bool(vec![b.is_true()]));
    }
    if let Ok(s) = value.extract::<String>() {
        return Ok(Column::Str(vec![s.into_bytes()]));
    }
    if let Ok(n) = value.extract::<i64>() {
        return Ok(Column::Int(vec![n as u64]));
    }
    if let Ok(f) = value.extract::<f64>() {
        return Ok(Column::Float(vec![f]));
    }
    if let Ok(d) = value.extract::<Duration>() {
        return Ok(if d.months == 0 {
            Column::Duration(DurationKind::DayTime, vec![d.nanoseconds])
        } else {
            Column::Duration(DurationKind::YearMonth, vec![d.months])
        });
    }
    if let Ok(delta) = value.cast::<PyDelta>() {
        return Ok(Column::Duration(
            DurationKind::DayTime,
            vec![delta_nanos(delta)?],
        ));
    }
    if let Ok(dt) = value.cast::<PyDateTime>() {
        return Ok(Column::LocalDatetime(vec![datetime_nanos(dt)?]));
    }
    if let Ok(date) = value.cast::<PyDate>() {
        return Ok(Column::Date(vec![date_days(date)?]));
    }
    if let Ok(time) = value.cast::<PyTime>() {
        return Ok(Column::LocalTime(vec![clock_nanos(time.as_any())?]));
    }
    let got = value
        .get_type()
        .getattr("__name__")
        .and_then(|name| name.extract::<String>())
        .unwrap_or_else(|_| "unknown".to_string());
    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "column '{name}' starts at row {row} with a value of type '{got}', and a loaded column holds booleans, integers, floats, strings, dates, times, datetimes or durations"
    )))
}

/// A duration for a column that is already one of the two kinds, which
/// is the one place the two do not mix: a column of months has no room
/// for a count of nanoseconds and the other way about.
fn duration_count(
    kind: DurationKind,
    value: &Bound<'_, PyAny>,
    mismatch: impl Fn(&str) -> PyErr,
) -> PyResult<i64> {
    let want = match kind {
        DurationKind::YearMonth => "year-month durations",
        DurationKind::DayTime => "day-time durations",
    };
    if let Ok(d) = value.extract::<Duration>() {
        return match kind {
            DurationKind::YearMonth if d.months != 0 || d.nanoseconds == 0 => Ok(d.months),
            DurationKind::DayTime if d.months == 0 => Ok(d.nanoseconds),
            _ => Err(mismatch(want)),
        };
    }
    match (kind, value.cast::<PyDelta>()) {
        (DurationKind::DayTime, Ok(delta)) => delta_nanos(delta),
        _ => Err(mismatch(want)),
    }
}

fn date_days(date: &Bound<'_, PyDate>) -> PyResult<i32> {
    Ok(days_from_civil(
        date.getattr("year")?.extract()?,
        date.getattr("month")?.extract()?,
        date.getattr("day")?.extract()?,
    ))
}

fn datetime_nanos(dt: &Bound<'_, PyDateTime>) -> PyResult<i64> {
    const NANOS_PER_DAY: i64 = 86_400 * 1_000_000_000;
    let days = date_days(dt.as_any().cast::<PyDate>()?)?;
    Ok(i64::from(days) * NANOS_PER_DAY + clock_nanos(dt.as_any())?)
}

fn delta_nanos(delta: &Bound<'_, PyDelta>) -> PyResult<i64> {
    let days: i64 = delta.getattr("days")?.extract()?;
    let seconds: i64 = delta.getattr("seconds")?.extract()?;
    let micros: i64 = delta.getattr("microseconds")?.extract()?;
    Ok(days * 86_400 * 1_000_000_000 + seconds * 1_000_000_000 + micros * 1_000)
}

/// The edge list, as the pairs of row numbers it is.
///
/// An edge naming a row the table has not got is refused here rather
/// than written, because a graph builder handed one would either
/// invent the row or lose the edge and neither is what the caller
/// meant.
fn pairs(edges: Option<&Bound<'_, PyAny>>, rows: u64) -> PyResult<Vec<(u32, u32)>> {
    let Some(edges) = edges else {
        return Ok(Vec::new());
    };
    let mut pairs = Vec::new();
    for (at, edge) in edges.try_iter()?.enumerate() {
        let edge = edge?;
        let (from, to): (i64, i64) = edge.extract().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(format!(
                "edge {at} is not a pair of row numbers"
            ))
        })?;
        for end in [from, to] {
            if end < 0 || end as u64 >= rows {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "edge {at} joins row {end} of a table with {rows} rows in it"
                )));
            }
        }
        pairs.push((from as u32, to as u32));
    }
    Ok(pairs)
}
