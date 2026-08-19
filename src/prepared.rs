//! A statement compiled once and run many times.
//!
//! ```python
//! with conn.prepare("MATCH (p:person) WHERE p.name = $name RETURN p.id AS id") as find:
//!     for name in names:
//!         print(find.execute({"name": name}).fetchall())
//! ```
//!
//! What this buys is not what a driver's `prepare` buys, and it is
//! worth saying so here rather than letting a reader assume it. A
//! driver prepares to save a round trip to a server, and there is no
//! server and no round trip: the engine is in this process. It caches a
//! plan by the text of the statement, so the second `conn.execute` of
//! the same string is not compiled a second time either, and a loop
//! that prepares and a loop that repeats the same string run at the
//! same speed.
//!
//! Two things it does buy. The compile happens at the line that asked
//! for it, so a program that prepares its statements at startup finds a
//! statement that does not compile there, rather than on the first
//! request that needed it. And `params` comes back, which is what the
//! statement wants bound, so a layer binding from a record knows what
//! to look for without reading the text.
//!
//! The one thing that is a speedup is the case this is written beside:
//! a statement whose text is different every run, which is what a
//! program that formats its values into the string is writing. That one
//! pays the compile every time and no cache can help it, and binding
//! parameters is what fixes it, prepared or not.

use std::sync::atomic::{AtomicBool, Ordering};

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::conn::{Connection, Result, bind};
use crate::error::{programming, to_py_err};

/// A statement the engine has compiled and is holding for you.
///
/// Take one with `Connection.prepare`, run it with `execute`, and close
/// it, which gives the statement back to the connection. A `with` block
/// closes it at the end, and running one that has been closed is
/// refused rather than quietly recompiled.
#[pyclass(module = "zudb")]
pub struct Prepared {
    /// The connection it was compiled on, held rather than borrowed: a
    /// prepared statement whose connection was collected would be an id
    /// for a session that is gone.
    conn: Py<Connection>,
    /// The text it was compiled from, kept so that a caller holding one
    /// in a dictionary can see which it is.
    #[pyo3(get)]
    statement: String,
    params: Vec<String>,
    /// What the session pinned it under.
    id: u64,
    open: AtomicBool,
}

#[pymethods]
impl Prepared {
    /// The names this statement wants bound, in the order it uses them.
    #[getter]
    fn params(&self) -> Vec<String> {
        self.params.clone()
    }

    /// Whether this prepared statement has been closed.
    #[getter]
    fn closed(&self) -> bool {
        !self.open.load(Ordering::Acquire)
    }

    /// Runs it with these parameters and gives back its rows.
    ///
    /// The same answer `Connection.execute` gives, because it is the
    /// same statement: a `zudb.Result`, which is rows in memory and
    /// knows how to become an Arrow table or a DataFrame. A name the
    /// statement wants and the caller did not bind is an error from the
    /// engine at this call, not at the prepare, since a missing value
    /// is nothing to do with the statement.
    #[pyo3(signature = (params = None))]
    fn execute(
        &self,
        py: Python<'_>,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<crate::conn::Result> {
        self.alive(py)?;
        let params = bind(params)?;
        self.conn.borrow(py).query_prepared(py, self.id, params)
    }

    /// The same call, named for the way it reads in a notebook.
    #[pyo3(signature = (params = None))]
    fn sql(&self, py: Python<'_>, params: Option<&Bound<'_, PyDict>>) -> PyResult<Result> {
        self.execute(py, params)
    }

    /// Closes it and gives the statement back to the connection.
    ///
    /// Doing it twice does nothing, which is what a `with` block around
    /// a caller who closed it themselves needs. A prepared statement
    /// whose connection has already closed is closed too, since the
    /// session that was holding the id went with it, and closing that
    /// one is not an error either.
    fn close(&self, py: Python<'_>) {
        if self.open.swap(false, Ordering::AcqRel) {
            self.conn.borrow(py).release(py, self.id);
        }
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[pyo3(signature = (*_exception))]
    fn __exit__(&self, py: Python<'_>, _exception: &Bound<'_, PyTuple>) -> bool {
        self.close(py);
        // False, so an exception raised inside the block carries on out
        // of it.
        false
    }

    fn __repr__(&self) -> String {
        let closed = if self.closed() { ", closed" } else { "" };
        format!("<zudb.Prepared {:?}{closed}>", self.statement)
    }
}

impl Prepared {
    /// Compiles one, which is what makes one.
    pub fn compile(py: Python<'_>, conn: Py<Connection>, statement: &str) -> PyResult<Prepared> {
        let (id, params) = conn
            .borrow(py)
            .engine(py, |engine| engine.prepare(statement))?
            .map_err(|err| to_py_err(py, err))?;
        Ok(Prepared {
            conn,
            statement: statement.to_string(),
            params,
            id,
            open: AtomicBool::new(true),
        })
    }

    /// Refuses a run on one that has been closed.
    fn alive(&self, py: Python<'_>) -> PyResult<()> {
        if self.closed() {
            return Err(programming(
                py,
                "this prepared statement is closed, and a closed one has given its \
                 statement back to the connection",
            ));
        }
        Ok(())
    }
}
