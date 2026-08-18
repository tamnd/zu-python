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
use pyo3::types::PyDict;
use zudb::zu1::file::Zu1File;
use zudb::zu1::graph::bulk_load_keyed;
use zudb::zu1::props::{PropValues, store_props};

use crate::buffer::{Column, Mismatch, type_name};
use crate::error::to_py_err;

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
            // The store wants a slice of slices for a column of strings
            // or of bytes, which a vector of either is not, so the row
            // borrows are built first and handed over after.
            let runs: Vec<Vec<&[u8]>> = built
                .iter()
                .map(|(_, column)| match column {
                    Column::Str(v) => v.iter().map(String::as_bytes).collect(),
                    Column::Bytes(v) => v.iter().map(Vec::as_slice).collect(),
                    _ => Vec::new(),
                })
                .collect();
            let props: Vec<(&str, PropValues<'_>)> = built
                .iter()
                .zip(&runs)
                .map(|((name, column), runs)| {
                    let values = match column {
                        Column::Str(_) => PropValues::Str(runs),
                        Column::Bytes(_) => PropValues::Bytes(runs),
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
        // The store takes a column of bytes and every statement that
        // reads one back refuses it, so a load that wrote one would be
        // writing data the caller cannot get at again. Refused here
        // until the read side catches up, at which point this goes and
        // nothing else has to change.
        if matches!(column, Column::Bytes(_)) {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "column '{name}' holds byte strings, and no statement can read one back yet, so a load will not write a column of them"
            )));
        }
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
/// has to agree, which is [`Column`]'s rule and is the appender's rule
/// too. What this adds is the column's name, because a message about a
/// column is worth leading with the name of it.
fn column(name: &str, values: &Bound<'_, PyAny>) -> PyResult<Column> {
    let mut column: Option<Column> = None;
    for (row, value) in values.try_iter()?.enumerate() {
        let value = value?;
        match column.as_mut() {
            Some(column) => column.widening_push(&value).map_err(|why| match why {
                Mismatch::Wanted(holds) => pyo3::exceptions::PyTypeError::new_err(format!(
                    "column '{name}' holds {holds} and row {row} is of type '{}'",
                    type_name(&value)
                )),
                Mismatch::Python(err) => err,
            })?,
            None => {
                column = Some(Column::start(&value)?.ok_or_else(|| {
                    pyo3::exceptions::PyTypeError::new_err(format!(
                        "column '{name}' starts at row {row} with a value of type '{}', and a loaded column holds booleans, integers, floats, strings, dates, times, datetimes or durations",
                        type_name(&value)
                    ))
                })?);
            }
        }
    }
    column.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "column '{name}' is empty, and an empty column says nothing about what it would hold"
        ))
    })
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
