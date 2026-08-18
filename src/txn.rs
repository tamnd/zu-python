//! Several statements as one unit of work.
//!
//! A statement written on its own already runs in a transaction of its
//! own, so this is not what makes a write atomic. What it holds is the
//! span: two statements are one unit, and the file keeps the state they
//! started from until one word or the other ends them.
//!
//! ```python
//! with conn.transaction():
//!     conn.execute("INSERT (a:account {uid: 1, balance: 100})")
//!     conn.execute("INSERT (b:account {uid: 2, balance: 0})")
//! ```
//!
//! The block commits when it ends and rolls back when it raises, which
//! is the whole reason this is a context manager rather than a pair of
//! methods: the failure that has to roll back is the one nobody wrote a
//! handler for.
//!
//! The three words underneath are `START TRANSACTION`, `COMMIT` and
//! `ROLLBACK`, and a caller who would rather write them can, on this
//! connection, today. What this adds is the rollback nobody remembers
//! to write, and a name for the state, so a program can ask whether it
//! is inside one rather than remembering.

use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::conn::Connection;
use crate::error::programming;

/// A transaction that has been started and not yet ended.
///
/// Take one with `Connection.transaction`. It starts when it is taken,
/// so a transaction that cannot start says so at the line that asked
/// rather than at the `with` below it, and the statements that run
/// inside it are the ones written on the connection it came from.
#[pyclass(module = "zudb")]
pub struct Transaction {
    /// The connection this runs on, held rather than borrowed, because
    /// a transaction whose connection was collected would be a span
    /// with nothing left to commit.
    conn: Py<Connection>,
    /// Whether it was started `READ ONLY`, which the engine refuses a
    /// write inside of at the statement that writes.
    #[pyo3(get)]
    read_only: bool,
    /// Whether a `COMMIT` or a `ROLLBACK` has already run here. A
    /// transaction ended inside its own block is ended, and the block
    /// closing over it does nothing, which is what lets a caller commit
    /// early and carry on.
    done: bool,
}

#[pymethods]
impl Transaction {
    /// Ends the transaction and keeps what it wrote.
    ///
    /// Doing it twice is refused rather than ignored. A second commit
    /// is a program that has lost track of where its transaction ends,
    /// and the statements between the two are in neither of them.
    fn commit(&mut self, py: Python<'_>) -> PyResult<()> {
        self.end(py, "COMMIT")
    }

    /// Ends the transaction and throws away what it wrote.
    fn rollback(&mut self, py: Python<'_>) -> PyResult<()> {
        self.end(py, "ROLLBACK")
    }

    /// Whether this transaction has already been committed or rolled
    /// back.
    #[getter]
    fn done(&self) -> bool {
        self.done
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    /// Commits at the end of the block, and rolls back when the block
    /// raised.
    ///
    /// A rollback that fails raises out of here in place of what was
    /// being raised, which reads badly and is still right: the first
    /// exception is the `__context__` of the second, and a program told
    /// only about the failed statement would go on believing the
    /// transaction unwound.
    #[pyo3(signature = (*exception))]
    fn __exit__(&mut self, py: Python<'_>, exception: &Bound<'_, PyTuple>) -> PyResult<bool> {
        // The first of the three is the exception's type, and it is
        // `None` when the block ended by falling off its own end. That
        // is the whole question here, so the other two go unread.
        let raised = exception.get_item(0).is_ok_and(|kind| !kind.is_none());
        if !self.done {
            self.end(py, if raised { "ROLLBACK" } else { "COMMIT" })?;
        }
        // False, so an exception raised inside the block carries on out
        // of it. A context manager that swallowed one would be a very
        // quiet way to lose an error, and a quieter way to lose the
        // rollback that answered it.
        Ok(false)
    }

    fn __repr__(&self) -> String {
        let read_only = if self.read_only { " read only" } else { "" };
        let done = if self.done { ", done" } else { "" };
        format!("<zudb.Transaction{read_only}{done}>")
    }
}

impl Transaction {
    /// Starts one, which is the statement that starts one.
    pub fn start(py: Python<'_>, conn: Py<Connection>, read_only: bool) -> PyResult<Transaction> {
        let statement = if read_only {
            "START TRANSACTION READ ONLY"
        } else {
            "START TRANSACTION"
        };
        conn.borrow(py).run(py, statement)?;
        Ok(Transaction {
            conn,
            read_only,
            done: false,
        })
    }

    fn end(&mut self, py: Python<'_>, statement: &str) -> PyResult<()> {
        if self.done {
            return Err(programming(
                py,
                "this transaction has already ended, and the statements after it \
                 belong to no transaction of yours",
            ));
        }
        self.conn.borrow(py).run(py, statement)?;
        // Written after the statement rather than before it, so a
        // commit the engine refused leaves a transaction a caller can
        // still roll back.
        self.done = true;
        Ok(())
    }
}
