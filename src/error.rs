//! Raising an engine condition as a Python exception.
//!
//! The classes are defined in `zudb/errors.py` rather than here. A
//! Python programmer reads exception classes, subclasses them and
//! catches them by name, and a class written in Python has a
//! docstring, a readable signature and a stub the type checkers agree
//! about. What this module owns is the mapping: which class a
//! condition raises, and the fields it arrives with.
//!
//! Every field the error model gives a binding is a field here, and
//! none of them is parsed back out of a message. That is the whole
//! point of the model: `except zudb.SyntaxError as e: e.line` is what
//! a caller writes, never a regular expression over `str(e)`.

use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyType};
use zudb::ZuError;

/// The `zudb.errors` module, imported once and kept.
///
/// Imported lazily rather than at module init, because this extension
/// is imported *by* `zudb/__init__.py` and reaching back into a
/// package that is still executing its first line would be a circular
/// import. By the time anything fails, the package is there.
static ERRORS: PyOnceLock<Py<PyModule>> = PyOnceLock::new();

fn errors(py: Python<'_>) -> PyResult<&Bound<'_, PyModule>> {
    ERRORS
        .get_or_try_init(py, || Ok(PyModule::import(py, "zudb.errors")?.unbind()))
        .map(|module| module.bind(py))
}

/// Turns an engine error into the Python exception it is.
///
/// A failure to build that exception is raised as itself, since a
/// broken import is a worse thing to hide than the condition that was
/// being reported.
pub fn to_py_err(py: Python<'_>, err: ZuError) -> PyErr {
    // A statement that was stopped is reported as whatever stopped it.
    // If a signal is what set the flag then Python has a
    // `KeyboardInterrupt` waiting to be raised, and raising anything
    // else would swallow the user's `Ctrl-C` and have them press it
    // again, harder.
    if matches!(err, ZuError::Interrupted)
        && let Err(signal) = py.check_signals()
    {
        return signal;
    }
    match build(py, &err) {
        Ok(raised) => raised,
        Err(broken) => broken,
    }
}

fn build(py: Python<'_>, err: &ZuError) -> PyResult<PyErr> {
    let class: Bound<'_, PyType> = errors(py)?.getattr(class_for(err))?.cast_into()?;
    let fields = PyDict::new(py);
    fields.set_item("retryable", err.retryable())?;
    if let Some(status) = err.gqlstatus() {
        fields.set_item("code", status.code())?;
        fields.set_item("condition", status.standard_text())?;
        fields.set_item("severity", severity(status.severity()))?;
        fields.set_item("doc_url", status.doc_url())?;
    }
    if let Some(at) = err.position() {
        fields.set_item("line", at.line)?;
        fields.set_item("column", at.column)?;
        fields.set_item("offset", at.offset)?;
    }
    if let Some(excerpt) = err.excerpt() {
        fields.set_item("excerpt", excerpt)?;
    }
    Ok(PyErr::from_value(
        class.call((err.to_string(),), Some(&fields))?,
    ))
}

/// The class a condition raises.
///
/// One exception type per GQLSTATUS class, which is what the two
/// characters that open a code are for. A caller catching
/// `zudb.DataError` catches every one of the forty-two conditions in
/// class 22 without listing them, and a condition zu adds to that
/// class later is caught by the same `except`.
fn class_for(err: &ZuError) -> &'static str {
    let Some(status) = err.gqlstatus() else {
        return match err {
            ZuError::Interrupted => "Interrupted",
            ZuError::InvalidArgument(_) => "ProgrammingError",
            ZuError::Conflict(_) => "TransactionError",
            // A file that is not there, or not readable, is the same
            // kind of thing as a database that cannot be reached, and
            // class 08 is what the standard calls that. Reporting a
            // missing path as an internal error would tell the caller
            // to file a bug about their own typo.
            ZuError::Io(_) => "ConnectionError",
            // And so is a file that is there and is not a database,
            // which is the same typo landing on a real file. Every
            // corruption a client can meet is a file it opened, so
            // this is the connection failing rather than the engine,
            // and the message says which file and what was wrong with
            // it.
            ZuError::Corrupt { .. } => "ConnectionError",
            _ => "InternalError",
        };
    };
    match status.class() {
        "08" => "ConnectionError",
        "22" => "DataError",
        "25" | "2D" | "40" => "TransactionError",
        "42" => "SyntaxError",
        _ => "Error",
    }
}

/// The severity letter's name, because `"exception"` is what a person
/// reading a traceback understands and `X` is what the standard's
/// table says.
fn severity(severity: zudb::Severity) -> &'static str {
    match severity {
        zudb::Severity::Success => "success",
        zudb::Severity::NoData => "no_data",
        zudb::Severity::Warning => "warning",
        zudb::Severity::Informational => "informational",
        zudb::Severity::Exception => "exception",
    }
}

/// A mistake the program made, as the class for one.
///
/// A `ValueError` would be wrong and a segfault would be worse:
/// `zudb.ProgrammingError` is the class for something the caller did
/// rather than for a condition the engine raised, and it is what a
/// call made on an object that is closed, or handed something it
/// cannot do anything with, raises.
pub fn programming(py: Python<'_>, message: &str) -> PyErr {
    match errors(py).and_then(|errors| {
        let class: Bound<'_, PyType> = errors.getattr("ProgrammingError")?.cast_into()?;
        Ok(PyErr::from_value(class.call1((message,))?))
    }) {
        Ok(raised) => raised,
        Err(broken) => broken,
    }
}

/// What a statement raises when it is run on a connection that is
/// closed.
pub fn closed(py: Python<'_>, what: &str) -> PyErr {
    programming(
        py,
        &format!("{what} is closed, so there is nothing left to run a statement on"),
    )
}
