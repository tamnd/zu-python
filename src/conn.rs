//! The connection and what a statement gives back.
//!
//! A connection is not thread-safe in the engine and every method
//! takes `&mut self` there, so the one held here sits behind a mutex.
//! That is not a way of making a connection concurrent: a statement
//! holds the mutex for as long as it runs, and two threads that want
//! to run statements at once want two connections. It is there so that
//! a program which shares one by accident waits rather than corrupts.

use std::ffi::CStr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyDict, PyList, PyTuple};
use zudb::query::{QueryResult, Value};
use zudb::{Config, Database, Interrupt};

use crate::appender::Appender;
use crate::columns;
use crate::error::{closed, programming, to_py_err};
use crate::interrupt;
use crate::txn::Transaction;
use crate::value::{Names, from_py, to_py};

/// What a capsule holding an Arrow stream is called. The name is part
/// of the protocol: a consumer checks it before it reads the pointer,
/// and a capsule named anything else is not one of these.
const STREAM: &CStr = c"arrow_array_stream";

/// One connection to one database.
///
/// Statements run on it in order, one at a time. It reads the database
/// as of when it was opened, which is why a program that wants to see
/// another writer's work takes a new connection rather than waiting on
/// this one.
#[pyclass(module = "zudb")]
pub struct Connection {
    /// `None` once closed, which is what makes a second `close()` do
    /// nothing and a statement after one an error rather than a crash.
    ///
    /// Reachable from the appender, which writes through the same
    /// connection and takes this same lock. One rule holds for every
    /// caller of it: take it with the GIL already released. A thread
    /// that waited for it holding the GIL would stop every other
    /// thread in the process for the length of somebody else's
    /// statement, and would deadlock against the thread inside that
    /// statement, which needs the GIL back to return.
    ///
    /// Counted rather than plain, because the thread a statement runs
    /// on takes a share of it: a job handed to that thread outlives the
    /// call that made it in the type system even though it never does
    /// in fact.
    pub(crate) inner: Arc<Mutex<Option<zudb::Connection>>>,
    /// The thread statements run on where a `Ctrl-C` can arrive,
    /// started at the first one that needs it and kept for the rest.
    runner: OnceLock<interrupt::Runner>,
    /// The word the running statement reads, held out here rather than
    /// reached through the lock. A stop that had to wait for the
    /// connection to be free could only ever arrive after the statement
    /// it was meant to stop.
    stop: Interrupt,
    /// Whether the connection is still open, kept beside the lock for
    /// the same reason: asking a connection whether it is closed, or
    /// asking it to stop, should not queue behind a ten second
    /// statement.
    alive: AtomicBool,
    #[pyo3(get)]
    path: PathBuf,
    #[pyo3(get)]
    read_only: bool,
}

#[pymethods]
impl Connection {
    /// Runs one statement and gives back its rows.
    ///
    /// The parameters are named, never positional, because zuQL names
    /// them: `$id` in the statement is `id` in the dictionary. A name
    /// the statement does not use is an error from the engine rather
    /// than a value quietly ignored.
    #[pyo3(signature = (statement, params = None))]
    fn execute(
        &self,
        py: Python<'_>,
        statement: &str,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Result> {
        let params = bind(params)?;
        // Owned rather than borrowed, both of them, because the thread
        // that runs the statement is not this one and the borrow would
        // have to say so. It is two allocations against a statement,
        // which is nothing beside the parse it is about to have.
        let statement = statement.to_string();
        // The GIL goes down for the whole statement, waiting for the
        // connection's own lock included. That is the point of a
        // compiled engine in a Python process: another thread runs
        // while this one is inside the executor. Where a `Ctrl-C` can
        // arrive the statement goes on the connection's own thread and
        // this one waits for it, which is the only way a press is felt
        // before the statement ends.
        let (result, names) =
            interrupt::watched(py, &self.runner, &self.inner, &self.stop, move |conn| {
                let borrowed: Vec<(&str, Value)> = params
                    .iter()
                    .map(|(name, value)| (name.as_str(), value.clone()))
                    .collect();
                let names = Names::of(conn.session_mut().catalog());
                conn.query_with(&statement, &borrowed)
                    .map(|result| (result, names))
            })
            .map_err(|stopped| stopped.raise(py))?
            .map_err(|err| to_py_err(py, err))?;
        Ok(Result {
            result,
            names,
            next: Mutex::new(0),
        })
    }

    /// The same call, named for the way it reads in a notebook:
    /// `conn.sql(...).to_pandas()`.
    #[pyo3(signature = (statement, params = None))]
    fn sql(
        &self,
        py: Python<'_>,
        statement: &str,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Result> {
        self.execute(py, statement, params)
    }

    /// Starts a transaction and hands it back for a `with` block.
    ///
    /// Several statements as one unit of work: the block commits when
    /// it ends and rolls back when it raises, which is the failure
    /// worth writing this for, since it is the one nobody wrote a
    /// handler for.
    ///
    /// It starts here rather than at the `with`, so a transaction that
    /// cannot start says so at the line that asked for one. A
    /// connection is inside one transaction at a time, and asking for a
    /// second while the first is running is refused by the engine
    /// rather than nested, because a transaction inside a transaction
    /// would have to invent an answer for what a rollback of the inner
    /// one undoes.
    ///
    /// `read_only=True` starts one that refuses to write, at the
    /// statement that writes rather than at the block that would have
    /// written.
    #[pyo3(signature = (*, read_only = false))]
    fn transaction(slf: Py<Self>, py: Python<'_>, read_only: bool) -> PyResult<Transaction> {
        Transaction::start(py, slf, read_only)
    }

    /// Whether an explicit transaction is running on this connection.
    ///
    /// The statements a program writes outside one are each a
    /// transaction of their own, so this is false on a connection that
    /// is writing perfectly well. It answers what is true between
    /// statements, which is the only moment a caller can ask in: asking
    /// while another thread is inside a statement waits for that
    /// statement, since the answer belongs to the session and the
    /// session is what the statement is holding.
    #[getter]
    fn in_transaction(&self, py: Python<'_>) -> PyResult<bool> {
        if !self.alive.load(Ordering::Acquire) {
            return Err(closed(py, "this connection"));
        }
        py.detach(|| {
            self.inner.lock().ok().and_then(|mut held| {
                held.as_mut()
                    .map(|conn| conn.session_mut().in_transaction())
            })
        })
        .ok_or_else(|| closed(py, "this connection"))
    }

    /// Opens an appender on `table`, for loading rows into a database
    /// that already exists.
    ///
    /// A statement writes one row at a time and commits each one, which
    /// is the wrong shape for a million rows. An appender buffers them
    /// in columns and writes each batch as one commit. The table has to
    /// be there already, since an appender adds rows to a table and
    /// does not make one.
    ///
    /// It is refused inside a transaction. An appender's batches are
    /// commits of their own and a rollback does not take them back, so
    /// an appender opened inside a `with conn.transaction()` would
    /// promise a span it is not in. Load first, then transact, or write
    /// the rows as statements.
    fn appender(slf: Py<Self>, py: Python<'_>, table: &str) -> PyResult<Appender> {
        if slf.borrow(py).in_transaction(py)? {
            return Err(programming(
                py,
                "an appender writes its own commits, which a rollback does not take \
                 back, so one cannot be opened inside a transaction",
            ));
        }
        Appender::open(py, slf, table)
    }

    /// Asks the statement running on this connection to stop.
    ///
    /// The one call meant to be made from another thread while the
    /// connection is busy, which is why it waits for nothing: the
    /// statement it stops is the one holding everything a call that
    /// waited would be waiting for. The statement raises
    /// `zudb.Interrupted` and the connection is exactly as it was, so
    /// the next statement on it starts warm. That is the difference
    /// between stopping a statement and closing a connection.
    ///
    /// With nothing running this does nothing. It does not arm a stop
    /// for the next statement, because a statement nobody has run yet
    /// is not one anybody has waited too long for.
    fn interrupt(&self, py: Python<'_>) -> PyResult<()> {
        if !self.alive.load(Ordering::Acquire) {
            return Err(closed(py, "this connection"));
        }
        self.stop.stop();
        Ok(())
    }

    /// How many rows the statement running on this connection has read
    /// out of storage, for showing a person that something is
    /// happening.
    ///
    /// Rows read rather than rows answered, because the statement
    /// somebody is waiting on is exactly the one that reads a hundred
    /// million rows to answer one. It starts at zero at each statement
    /// and holds its last value once one ends.
    #[getter]
    fn rows_read(&self) -> u64 {
        self.stop.rows()
    }

    /// Closes the connection and frees what it held.
    ///
    /// Doing it twice is not an error, because a `with` block that
    /// closed early would otherwise fail on the way out.
    fn close(&self, py: Python<'_>) {
        // Released here too, because closing waits for the statement
        // another thread is running and frees caches worth megabytes
        // once it has.
        py.detach(|| {
            if let Ok(mut held) = self.inner.lock() {
                drop(held.take());
            }
            // Written after the drop rather than before it, so that a
            // connection reports itself open until it really is not,
            // and a poisoned lock reports itself closed because
            // nothing can be run on one.
            self.alive.store(false, Ordering::Release);
        });
    }

    /// Whether this connection is still open.
    ///
    /// Answered from a word beside the lock rather than through it, so
    /// that asking does not queue behind whatever is running.
    #[getter]
    fn closed(&self) -> bool {
        !self.alive.load(Ordering::Acquire)
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[pyo3(signature = (*_exception))]
    fn __exit__(&self, py: Python<'_>, _exception: &Bound<'_, PyTuple>) -> bool {
        self.close(py);
        // False, so an exception raised inside the block carries on
        // out of it. A context manager that swallowed one would be a
        // very quiet way to lose an error.
        false
    }

    fn __repr__(&self) -> String {
        let state = if self.closed() { ", closed" } else { "" };
        format!("<zudb.Connection {}{state}>", self.path.display())
    }
}

impl Connection {
    /// Opens `path`, creating a database there when there is none.
    ///
    /// Creating is what every Python database module does and what a
    /// notebook expects, and it can only ever create where nothing
    /// was: a path that holds a database is opened, and a read-only
    /// connection never creates anything at all.
    pub fn open(
        py: Python<'_>,
        path: PathBuf,
        read_only: bool,
        memory_limit: Option<usize>,
        threads: Option<usize>,
    ) -> PyResult<Connection> {
        let mut config = Config::new().read_only(read_only);
        if let Some(bytes) = memory_limit {
            config = config.memory_limit(bytes);
        }
        if let Some(threads) = threads {
            config = config.threads(threads);
        }
        let missing = !path.exists();
        let opened = py.detach(|| {
            if missing && !read_only {
                Database::create_with(&path, config.clone())
            } else {
                Database::open_with(&path, config.clone())
            }
            .and_then(|db| db.connect())
        });
        let opened = opened.map_err(|err| to_py_err(py, err))?;
        Ok(Connection {
            // Taken here, once, because every later reader of it wants
            // it while the connection is busy and taking it then would
            // mean waiting for the statement it is there to stop.
            stop: opened.interrupt(),
            inner: Arc::new(Mutex::new(Some(opened))),
            runner: OnceLock::new(),
            alive: AtomicBool::new(true),
            path,
            read_only,
        })
    }

    /// Runs a statement that takes no parameters and gives back
    /// nothing, which is what the three transaction words are.
    ///
    /// It goes through `execute` rather than around it, so a `COMMIT`
    /// waits for the connection's lock, releases the GIL and feels a
    /// `Ctrl-C` exactly as every other statement does. The result is
    /// dropped, since the words return no columns.
    pub(crate) fn run(&self, py: Python<'_>, statement: &str) -> PyResult<()> {
        self.execute(py, statement, None).map(|_| ())
    }
}

/// The rows a statement gave back.
///
/// Held as the engine produced them and turned into Python objects
/// when they are read, so a result that is about to become an Arrow
/// table never pays for the objects it would have made.
#[pyclass(module = "zudb")]
pub struct Result {
    result: QueryResult,
    names: Names,
    /// Where `fetchone` has reached, which is the only cursor here.
    /// Iterating does not move it, because two callers reading one
    /// result should not be able to take each other's rows.
    next: Mutex<usize>,
}

#[pymethods]
impl Result {
    /// The column names, in the order the statement projected them.
    #[getter]
    fn columns(&self) -> Vec<String> {
        self.result.columns.clone()
    }

    /// Every row, as a list of tuples.
    fn fetchall<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let rows = PyList::empty(py);
        for row in &self.result.rows {
            rows.append(self.row(py, row)?)?;
        }
        Ok(rows)
    }

    /// The next row, or `None` when there are no more.
    fn fetchone<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyTuple>>> {
        let mut next = self.next.lock().map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err("this result was left locked by a panic")
        })?;
        let Some(row) = self.result.rows.get(*next) else {
            return Ok(None);
        };
        *next += 1;
        self.row(py, row).map(Some)
    }

    /// The warnings the statement raised, if it raised any. A notice
    /// is a condition that did not stop the statement, so it arrives
    /// beside the rows rather than instead of them.
    #[getter]
    fn notices<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let out = PyList::empty(py);
        for notice in &self.result.notices {
            let one = PyDict::new(py);
            one.set_item("code", notice.status.code())?;
            one.set_item("condition", notice.status.standard_text())?;
            one.set_item("detail", &notice.detail)?;
            out.append(one)?;
        }
        Ok(out)
    }

    /// The rows as an Arrow stream, for anything that speaks Arrow.
    ///
    /// This is the PyCapsule interface, which is how a producer hands
    /// Arrow data to a consumer without either of them importing the
    /// other: what comes back is a capsule holding an
    /// `ArrowArrayStream`, and `pyarrow.table(result)`,
    /// `polars.from_arrow(result)` and anything else that reads Arrow
    /// takes it from here. `requested_schema` is accepted and ignored,
    /// which the protocol allows: a result has the types it has, and
    /// casting them here would hide a conversion a caller can see.
    #[pyo3(signature = (requested_schema = None))]
    fn __arrow_c_stream__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let _ = requested_schema;
        // Released for the copy: filling Arrow buffers out of engine
        // values touches no Python object, and a result worth putting
        // in a DataFrame is big enough that another thread should get
        // to run while it happens.
        let stream = py
            .detach(|| columns::stream(&self.result, &self.names))
            .map_err(|snag| snag.raise(py))?;
        PyCapsule::new_with_value(py, stream, STREAM)
    }

    /// The rows as a `pyarrow.Table`.
    fn to_arrow<'py>(slf: PyRef<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        needed(py, "pyarrow", "arrow")?.call_method1("table", (slf,))
    }

    /// The rows as a `pandas.DataFrame`, with Arrow-backed dtypes.
    ///
    /// `ArrowDtype` rather than the NumPy dtypes pandas grew up with,
    /// because the data is already Arrow and converting it to NumPy
    /// would copy every column to lose null support on the way. It is
    /// also what pandas 3 wants.
    fn to_pandas<'py>(slf: PyRef<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let pandas = needed(py, "pandas", "pandas")?;
        let how = PyDict::new(py);
        how.set_item("types_mapper", pandas.getattr("ArrowDtype")?)?;
        Self::to_arrow(slf)?.call_method("to_pandas", (), Some(&how))
    }

    /// The rows as a `polars.DataFrame`.
    ///
    /// The constructor rather than `from_arrow`, because `from_arrow`
    /// on a stream is what polars 2 is going to hand back a `Series`
    /// for, and a result is a table however many columns it has.
    fn to_polars<'py>(slf: PyRef<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        needed(py, "polars", "polars")?.call_method1("DataFrame", (slf,))
    }

    /// The rows as a `pyarrow.RecordBatchReader`, a batch at a time.
    ///
    /// The same data as `to_arrow`, handed over in batches of sixty-five
    /// thousand rows instead of as one table, which is what a consumer
    /// that writes as it reads wants.
    fn record_batches<'py>(slf: PyRef<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        needed(py, "pyarrow", "arrow")?
            .getattr("RecordBatchReader")?
            .call_method1("from_stream", (slf,))
    }

    fn __len__(&self) -> usize {
        self.result.rows.len()
    }

    fn __iter__<'py>(slf: PyRef<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        slf.fetchall(py)?.call_method0("__iter__")
    }

    fn __repr__(&self) -> String {
        format!(
            "<zudb.Result {} columns, {} rows>",
            self.result.columns.len(),
            self.result.rows.len()
        )
    }
}

impl Result {
    fn row<'py>(&self, py: Python<'py>, row: &[Value]) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(
            py,
            row.iter()
                .map(|value| to_py(py, value, &self.names))
                .collect::<PyResult<Vec<_>>>()?,
        )
    }
}

/// A module the caller has to have installed for this call, imported.
///
/// A missing one is reported as the install that fixes it rather than
/// as `ModuleNotFoundError: No module named 'pyarrow'`, which is true
/// and says nothing about what to do. An import that fails for any
/// other reason is raised as it is, because a broken pandas is not a
/// missing one.
fn needed<'py>(py: Python<'py>, module: &str, extra: &str) -> PyResult<Bound<'py, PyModule>> {
    PyModule::import(py, module).map_err(|err| {
        if err.is_instance_of::<pyo3::exceptions::PyImportError>(py) {
            pyo3::exceptions::PyImportError::new_err(format!(
                "this needs {module}, which is not installed: pip install 'zudb[{extra}]'"
            ))
        } else {
            err
        }
    })
}

/// The parameter dictionary as the engine takes it.
fn bind(params: Option<&Bound<'_, PyDict>>) -> PyResult<Vec<(String, Value)>> {
    let Some(params) = params else {
        return Ok(Vec::new());
    };
    let mut out = Vec::with_capacity(params.len());
    for (name, value) in params.iter() {
        out.push((name.extract::<String>()?, from_py(&value)?));
    }
    Ok(out)
}
