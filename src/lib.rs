//! The zu extension module.
//!
//! What Python imports is the package `zudb`, and what the package
//! imports is this. Everything with an engine call behind it lives
//! here; everything that is only Python, the exception classes above
//! all, lives in the package, where it has a signature a reader can
//! see and a stub a checker can use.
//!
//! This links the engine crates directly rather than going through
//! `libzu`'s C ABI, which is ADR 0002: the hard part of a Python
//! binding is the runtime, not the FFI, and PyO3 is years of exactly
//! that work. What it owes in return is the ABI's semantics, and the
//! conformance corpus is what says whether it paid.

mod columns;
mod conn;
mod error;
mod load;
mod value;

use std::path::PathBuf;

use pyo3::prelude::*;

/// Opens the database at `path` and connects to it.
///
/// Creates one when the path holds nothing, which is what every Python
/// database module does and what a notebook expects. A read-only
/// connection never creates anything, so a mistyped path there is an
/// error rather than an empty database.
///
/// `memory_limit` is in bytes and `threads` is how many the executor
/// may use; both default to what the engine decides for the machine.
#[pyfunction]
#[pyo3(signature = (path, *, read_only = false, memory_limit = None, threads = None))]
fn connect(
    py: Python<'_>,
    path: PathBuf,
    read_only: bool,
    memory_limit: Option<usize>,
    threads: Option<usize>,
) -> PyResult<conn::Connection> {
    conn::Connection::open(py, path, read_only, memory_limit, threads)
}

#[pymodule]
fn _zudb(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__engine_version__", env!("CARGO_PKG_VERSION"))?;
    module.add("__abi_version__", zudb::C_ABI_VERSION)?;
    module.add_function(wrap_pyfunction!(connect, module)?)?;
    module.add_function(wrap_pyfunction!(load::load, module)?)?;
    module.add_class::<conn::Connection>()?;
    module.add_class::<conn::Result>()?;
    module.add_class::<value::Node>()?;
    module.add_class::<value::Rel>()?;
    module.add_class::<value::Path>()?;
    module.add_class::<value::Duration>()?;
    Ok(())
}
