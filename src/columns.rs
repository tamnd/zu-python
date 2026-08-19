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
//! ## The translation is not written here
//!
//! It used to be, seven hundred lines of it, and the JavaScript client
//! had its own seven hundred that had to agree with them. `zu-arrow` in
//! the engine tree is the one answer about what a zu column becomes in
//! Arrow, and both clients export through it, so a year-month duration
//! is a month interval in both and a node names its table in both. The
//! engine is also where the buffers are: `zudb::query::column` reads a
//! result down its columns in two passes and hands back one owned
//! buffer per column in the layout Arrow already uses, and putting an
//! array around one of those is a move rather than a copy.
//!
//! What is left here is the three things a shared crate cannot know:
//! which Python exception each kind of refusal is, where the table
//! names live in this client, and how many rows a caller wanted in a
//! batch.

use pyo3::prelude::*;

use crate::value::Names;

/// What goes wrong here, with the GIL down and no way to raise yet.
///
/// The engine's error, under the name this client has always called it,
/// because what it means has not changed: something in a column could
/// not be said in Arrow.
pub use zu_arrow::Error as Snag;

/// Turning one into the exception it is.
///
/// An extension trait rather than a method, because [`Snag`] belongs to
/// the engine now and only this client knows that a value of the wrong
/// type is a `TypeError` here.
pub trait Raise {
    /// The exception this is, once there is a GIL to raise it with.
    fn raise(self, py: Python<'_>) -> PyErr;
}

impl Raise for Snag {
    fn raise(self, _py: Python<'_>) -> PyErr {
        match self {
            // The two Python classes are the two mistakes: a value of
            // the wrong type in a column is a `TypeError`, and a value
            // of the right type that will not fit is a `ValueError`.
            Snag::Type(detail) => pyo3::exceptions::PyTypeError::new_err(detail),
            Snag::Value(detail) => pyo3::exceptions::PyValueError::new_err(detail),
            // Nothing here is meant to be reachable: the types are
            // decided before a buffer is filled, so an Arrow error is
            // the translation getting it wrong rather than the caller.
            Snag::Arrow(err) => pyo3::exceptions::PyRuntimeError::new_err(format!(
                "arrow could not build the result: {err}"
            )),
        }
    }
}

/// The names this client already took off the catalog, offered to the
/// translation in the shape it asks for.
///
/// Borrowed rather than cloned, because a column of a hundred million
/// nodes is a hundred million lookups. The inherent `node` and `rel` on
/// [`Names`] answer with the `#id` fallback and stay where they are;
/// these two are the raw question, which is what the translation wants
/// so it can decide the fallback itself.
impl zu_arrow::Tables for Names {
    fn node(&self, id: u32) -> Option<&str> {
        self.node_name(id)
    }

    fn rel(&self, id: u32) -> Option<&str> {
        self.rel_name(id)
    }
}

/// How many rows go in one record batch, when a caller has no opinion.
pub const BATCH: usize = zu_arrow::BATCH;

/// The stream a result exports, batches and schema and all.
///
/// One array per column, built once out of the engine's buffers, and
/// batches that are slices of them. The arrays are built eagerly
/// because the refusals have to happen while there is still a caller to
/// raise them at; the batches are not, so a reader that stops early
/// stops paying.
pub fn stream(
    result: &zudb::query::QueryResult,
    names: &Names,
    rows: usize,
) -> Result<arrow::ffi_stream::FFI_ArrowArrayStream, Snag> {
    zu_arrow::stream(result, names, rows)
}
