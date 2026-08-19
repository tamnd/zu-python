//! The connection and what a statement gives back.
//!
//! A connection is not thread-safe in the engine and every method
//! takes `&mut self` there, so the one held here sits behind a mutex.
//! That is not a way of making a connection concurrent: a statement
//! holds the mutex for as long as it runs, and two threads that want
//! to run statements at once want two connections. It is there so that
//! a program which shares one by accident waits rather than corrupts.

use std::ffi::CStr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyDict, PyList, PyTuple};
use zudb::query::{QueryResult, Value};
use zudb::{Config, Database, Interrupt};

use crate::appender::Appender;
use crate::columns;
use crate::error::{closed, programming, to_py_err};
use crate::html;
use crate::interrupt;
use crate::plan;
use crate::prepared::Prepared;
use crate::register;
use crate::stream;
use crate::txn::Transaction;
use crate::value::{Names, from_py, to_py};

/// What a capsule holding an Arrow stream is called. The name is part
/// of the protocol: a consumer checks it before it reads the pointer,
/// and a capsule named anything else is not one of these.
const STREAM: &CStr = c"arrow_array_stream";

/// What a run is: the text of a statement, or the id the session pinned
/// a prepared one under.
///
/// One enum rather than two paths, so that a prepared statement gets
/// the interrupt handling, the GIL release and the catalog names that
/// every other statement gets, by being the same call.
pub(crate) enum Source {
    Text(String),
    Prepared(u64),
}

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
    /// The stream holding the connection, if one is. A stream ends when
    /// its reader says so, and a statement that queued behind a
    /// half-read one would be waiting for a loop that is waiting for
    /// it, so every other statement asks this and is told no rather
    /// than left to deadlock.
    pub(crate) feeding: Arc<stream::Feeding>,
    #[pyo3(get)]
    path: PathBuf,
    #[pyo3(get)]
    read_only: bool,
    /// Whether the database behind it is in memory, which is the one
    /// thing `path` cannot quite say: a file could be called
    /// `:memory:` on any filesystem that allows a colon.
    #[pyo3(get)]
    memory: bool,
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
        self.query(py, statement, bind(params)?)
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

    /// Runs one statement and hands back its rows a batch at a time, as
    /// the executor makes them.
    ///
    /// ```python
    /// with conn.stream("MATCH (p:person) RETURN p.name AS name") as rows:
    ///     for (name,) in rows:
    ///         print(name)
    /// ```
    ///
    /// What is in memory is a batch and not the answer, which is what
    /// this is for: a statement over ten million rows read by a program
    /// holding a thousand. A reader that stops early stops the scan,
    /// which is the other half of it, and `batch_rows` says how many
    /// rows a batch may hold when the rows are going somewhere with a
    /// size of its own. It is a ceiling rather than a size: the engine
    /// fills whole vectors and a batch holds as many of them as fit
    /// under the number, so a round figure comes back a little short
    /// unless a vector divides by it.
    ///
    /// The stream holds the connection until it ends, because a
    /// connection runs one statement at a time. Read it to the end,
    /// close it, or open it in a `with` block, which closes it however
    /// the block is left.
    #[pyo3(signature = (statement, params = None, *, batch_rows = None))]
    fn stream(
        &self,
        py: Python<'_>,
        statement: &str,
        params: Option<&Bound<'_, PyDict>>,
        batch_rows: Option<usize>,
    ) -> PyResult<stream::Stream> {
        stream::open(
            py,
            &self.inner,
            &self.stop,
            &self.feeding,
            self.alive.load(Ordering::Acquire),
            statement.to_string(),
            bind(params)?,
            batch_rows,
        )
    }

    /// Compiles a statement now and hands back something that runs it
    /// later, as often as you like, with different values bound each
    /// time.
    ///
    /// ```python
    /// with conn.prepare("MATCH (p:person) WHERE p.name = $name RETURN p.id AS id") as find:
    ///     rows = find.execute({"name": "ada"})
    /// ```
    ///
    /// It is not the speedup the word usually promises, and the class
    /// says why at length: there is no round trip to save here, and the
    /// engine already caches a plan by the text of the statement. What
    /// it buys is the compile happening at this line, at startup, and
    /// `params` coming back.
    fn prepare(slf: Py<Self>, py: Python<'_>, statement: &str) -> PyResult<Prepared> {
        Prepared::compile(py, slf, statement)
    }

    /// What the statement would do, without doing it.
    ///
    /// ```python
    /// print(conn.explain("MATCH (p:person) RETURN p.name AS name"))
    /// ```
    ///
    /// The plan comes back as the engine's own listing, which is what
    /// `print` gives, and as a tree of operators, which is what
    /// `plan.root` walks. It takes no parameters: a plan is chosen from
    /// the shape of the statement rather than from the values bound to
    /// it, and one asked for with values would suggest otherwise.
    fn explain(&self, py: Python<'_>, statement: &str) -> PyResult<plan::Plan> {
        let out = self
            .engine(py, |conn| conn.explain_plan(statement))?
            .map_err(|err| to_py_err(py, err))?;
        plan::planned(py, &out)
    }

    /// Runs the statement with the counters on and answers what its
    /// operators really did.
    ///
    /// ```python
    /// run = conn.profile("MATCH (p:person)-[:knows]->(q:person) RETURN q.name AS name")
    /// print(run)
    /// ```
    ///
    /// Beside every operator's rows is what the optimizer thought it
    /// would produce, so the operator that surprised it is the one to
    /// look at. A statement that writes is refused rather than
    /// profiled, because a measurement that also inserted two rows
    /// changed the thing it was measuring.
    #[pyo3(signature = (statement, params = None))]
    fn profile(
        &self,
        py: Python<'_>,
        statement: &str,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<plan::Profile> {
        let params = bind(params)?;
        let statement = statement.to_string();
        let out = interrupt::watched(py, &self.runner, &self.inner, &self.stop, move |conn| {
            let borrowed: Vec<(&str, Value)> = params
                .iter()
                .map(|(name, value)| (name.as_str(), value.clone()))
                .collect();
            conn.profile(&statement, &borrowed)
        })
        .map_err(|stopped| stopped.raise(py))?
        .map_err(|err| to_py_err(py, err))?;
        plan::profiled(py, &out)
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
    pub(crate) fn in_transaction(&self, py: Python<'_>) -> PyResult<bool> {
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

    /// Puts a DataFrame under a name a statement can match on.
    ///
    /// ```python
    /// conn.register("people", frame)
    /// conn.execute("MATCH (p:people) WHERE p.age > 40 RETURN p.name AS name")
    /// ```
    ///
    /// Anything that speaks Arrow goes in: a pandas or polars
    /// DataFrame, a pyarrow Table or RecordBatchReader, or a dictionary
    /// of lists. Nothing is copied. The engine is told where the
    /// columns are and reads them where they lie, so registering costs
    /// the same whether the frame has ten rows or ten million, and a
    /// frame is a view rather than a snapshot: write into the array
    /// behind it and the next statement answers what is there now. A
    /// dictionary of Python lists is the one exception, since a list
    /// holds objects rather than numbers and there is nothing in it to
    /// point at.
    ///
    /// The frame belongs to this connection and goes when it does.
    /// Nothing is written to the database and no other program sees it.
    /// Registering the same name again replaces what it stands for,
    /// columns and all. Registering over a table the database already
    /// holds is refused, since a statement naming it would mean the
    /// stored one. Gives back the number of rows the frame has.
    fn register(&self, py: Python<'_>, name: &str, data: &Bound<'_, PyAny>) -> PyResult<usize> {
        register::register(py, self, name, data)
    }

    /// Takes a registered frame's name away.
    ///
    /// The name stops standing for anything and the bytes go back to
    /// the caller, which is not necessarily this instant: a statement
    /// still reading the frame holds it until it ends.
    fn unregister(&self, py: Python<'_>, name: &str) -> PyResult<()> {
        register::unregister(py, self, name)
    }

    /// The names this connection has registered frames under, sorted.
    #[getter]
    fn registered(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        register::registered(py, self)
    }

    /// Another connection to the same database, made from this one.
    ///
    /// This is how a pool is written. `zudb.connect()` opens the file
    /// again and looks the database up by path; this forks off the one
    /// this connection already holds, which costs a schema load and no
    /// lookup, and works on a database in memory, where there is no
    /// path to open a second time.
    ///
    /// The two are connections in every sense rather than two names
    /// for one. Each has its own prepared statements, its own caches
    /// and its own transaction, so a thread that takes one from a pool
    /// is not in whatever transaction the last borrower left open. What
    /// they share is the write side: they queue behind each other to
    /// write and each sees what the other has committed, which is what
    /// two connections to one file have always done.
    ///
    /// `duplicate()` is the same call under the name that says what it
    /// does. `cursor()` is what every other embedded database calls
    /// it, and a caller who learned the word somewhere else should not
    /// have to learn another one here.
    fn cursor(&self, py: Python<'_>) -> PyResult<Connection> {
        let made = self
            .engine(py, |conn| conn.duplicate())?
            .map_err(|err| to_py_err(py, err))?;
        Ok(Connection {
            stop: made.interrupt(),
            inner: Arc::new(Mutex::new(Some(made))),
            runner: OnceLock::new(),
            alive: AtomicBool::new(true),
            feeding: Arc::new(stream::Feeding::new()),
            path: self.path.clone(),
            read_only: self.read_only,
            memory: self.memory,
        })
    }

    /// The same call as `cursor()`, under the name that says what it
    /// does rather than the name every embedded database uses for it.
    fn duplicate(&self, py: Python<'_>) -> PyResult<Connection> {
        self.cursor(py)
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
        // A stream is told first. Closing waits for the connection's
        // lock and a stream holds it for as long as its reader takes,
        // so a connection closed under a half-read one would wait for a
        // loop nobody is going to run again.
        self.feeding.hang_up();
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
    /// The name a database in memory is asked for by, and answers to.
    ///
    /// The spelling every embedded database has used for thirty years,
    /// which is the reason it is this and not something better: a
    /// caller who types it has already been taught what it means
    /// somewhere else.
    const MEMORY: &'static str = ":memory:";

    /// Opens `path`, creating a database there when there is none.
    ///
    /// Creating is what every Python database module does and what a
    /// notebook expects, and it can only ever create where nothing
    /// was: a path that holds a database is opened, and a read-only
    /// connection never creates anything at all.
    ///
    /// No path, or `":memory:"`, is a database in memory. It used to
    /// be a file called `:memory:` in the working directory, which was
    /// the worst of both worlds: the name said nothing was on disk and
    /// something was.
    pub fn open(
        py: Python<'_>,
        path: Option<PathBuf>,
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
        // The name is reported back as it was asked for rather than as
        // the engine spells it. The engine mints a unique one per
        // database so that two of them never share a writer, and that
        // counter is its business and not a caller's.
        let memory = path.as_deref().is_none_or(|p| p == Path::new(Self::MEMORY));
        let path = match memory {
            true => PathBuf::from(Self::MEMORY),
            false => path.expect("a path that is not the memory name"),
        };
        let missing = !memory && !path.exists();
        let opened = py.detach(|| {
            if memory {
                Database::memory_with(config.clone())
            } else if missing && !read_only {
                Database::create_with(&path, config.clone())
            } else {
                Database::open_with(&path, config.clone())
            }
            .and_then(|db| db.connect())
        });
        let opened = opened.map_err(|err| to_py_err(py, err))?;
        Ok(Connection {
            memory,
            // Taken here, once, because every later reader of it wants
            // it while the connection is busy and taking it then would
            // mean waiting for the statement it is there to stop.
            stop: opened.interrupt(),
            inner: Arc::new(Mutex::new(Some(opened))),
            runner: OnceLock::new(),
            alive: AtomicBool::new(true),
            feeding: Arc::new(stream::Feeding::new()),
            path,
            read_only,
        })
    }

    /// Runs one statement, with its parameters already engine values.
    ///
    /// The one path every statement takes, whether the parameters came
    /// from a caller's dictionary or were built here.
    pub(crate) fn query(
        &self,
        py: Python<'_>,
        statement: &str,
        params: Vec<(String, Value)>,
    ) -> PyResult<Result> {
        // Owned rather than borrowed, because the thread that runs the
        // statement is not this one and the borrow would have to say
        // so. It is one allocation against a statement, which is
        // nothing beside the parse it is about to have.
        self.run_source(py, Source::Text(statement.to_string()), params)
    }

    /// Runs a statement the session has already compiled, under the id
    /// it pinned it with.
    ///
    /// The same path as every other statement, so a prepared one
    /// releases the GIL, queues for the connection's lock and feels a
    /// `Ctrl-C` exactly as one written out does.
    pub(crate) fn query_prepared(
        &self,
        py: Python<'_>,
        id: u64,
        params: Vec<(String, Value)>,
    ) -> PyResult<Result> {
        self.run_source(py, Source::Prepared(id), params)
    }

    /// Gives a prepared statement's id back to the session.
    ///
    /// Every reason the connection could not be had is ignored, since
    /// they all mean the same thing here: a session that is gone is not
    /// holding the statement either.
    pub(crate) fn release(&self, py: Python<'_>, id: u64) {
        let _ = self.engine(py, |conn| -> std::result::Result<bool, ()> {
            Ok(conn.close_prepared(id))
        });
    }

    fn run_source(
        &self,
        py: Python<'_>,
        source: Source,
        params: Vec<(String, Value)>,
    ) -> PyResult<Result> {
        if self.feeding.busy() {
            return Err(programming(py, stream::STREAMING));
        }
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
                let out = match &source {
                    Source::Text(statement) => conn.query_with(statement, &borrowed),
                    Source::Prepared(id) => conn.execute_prepared(*id, &borrowed),
                };
                out.map(|result| (result, names))
            })
            .map_err(|stopped| stopped.raise(py))?
            .map_err(|err| to_py_err(py, err))?;
        Ok(Result {
            result,
            names,
            next: Mutex::new(0),
        })
    }

    /// Runs a statement that takes no parameters and gives back
    /// nothing, which is what the three transaction words are.
    ///
    /// It goes through the same path as every other statement, so a
    /// `COMMIT` waits for the connection's lock, releases the GIL and
    /// feels a `Ctrl-C` exactly as they do. The result is dropped,
    /// since the words return no columns.
    pub(crate) fn run(&self, py: Python<'_>, statement: &str) -> PyResult<()> {
        self.query(py, statement, Vec::new()).map(|_| ())
    }

    /// Work done on the engine connection itself, with the GIL down.
    ///
    /// For the things a statement cannot say: reading the catalog,
    /// opening the engine's own appender. The two failures are kept
    /// apart because they belong to different people. A connection that
    /// is closed or locked is this call's own answer and comes back as
    /// the outer error; anything the work itself decides is wrong comes
    /// back as the inner one, for the caller to turn into an exception
    /// now that there is an interpreter to build one with.
    pub(crate) fn engine<T, S, F>(
        &self,
        py: Python<'_>,
        work: F,
    ) -> PyResult<std::result::Result<T, S>>
    where
        T: Send,
        S: Send,
        F: FnOnce(&mut zudb::Connection) -> std::result::Result<T, S> + Send,
    {
        if !self.alive.load(Ordering::Acquire) {
            return Err(closed(py, "this connection"));
        }
        if self.feeding.busy() {
            return Err(programming(py, stream::STREAMING));
        }
        py.detach(|| {
            let mut held = self.inner.lock().map_err(|_| ())?;
            let conn = held.as_mut().ok_or(())?;
            Ok(work(conn))
        })
        .map_err(|()| closed(py, "this connection"))
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
    ///
    /// A batch is a view of the column it comes from rather than a copy
    /// of it, and it is cut when the reader asks for it, so a consumer
    /// that stops early stops paying.
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

    /// The rows as an HTML table, which is what a notebook shows.
    ///
    /// Jupyter asks for this before it falls back to `repr`, so a
    /// result in a cell is the rows and not a line about how many
    /// there are. Only the first hundred are drawn and the note
    /// underneath says how many there were, because a million rows of
    /// markup is a notebook file nobody can open.
    ///
    /// Reading it does not move the cursor `fetchone` uses, for the
    /// same reason iterating does not: a person who looked at a result
    /// has not taken any of its rows.
    fn _repr_html_(&self, py: Python<'_>) -> PyResult<String> {
        let rows = &self.result.rows;
        let columns = &self.result.columns;
        if columns.is_empty() {
            return Ok(html::wrap(
                "<span class=\"zu-note\">no columns, which is what a statement that \
                 writes gives back</span>",
            ));
        }
        let mut out = String::from("<table class=\"zu-table\"><thead><tr>");
        for name in columns {
            out.push_str(&format!("<th>{}</th>", html::escape(name)));
        }
        out.push_str("</tr></thead><tbody>");
        for row in rows.iter().take(html::SHOWN) {
            out.push_str("<tr>");
            for value in row {
                out.push_str(&html::cell(&to_py(py, value, &self.names)?)?);
            }
            out.push_str("</tr>");
        }
        out.push_str("</tbody></table>");
        out.push_str(&format!(
            "<div class=\"zu-note\">{}</div>",
            html::note(rows.len(), columns.len())
        ));
        Ok(html::wrap(&out))
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
pub(crate) fn bind(params: Option<&Bound<'_, PyDict>>) -> PyResult<Vec<(String, Value)>> {
    let Some(params) = params else {
        return Ok(Vec::new());
    };
    let mut out = Vec::with_capacity(params.len());
    for (name, value) in params.iter() {
        out.push((name.extract::<String>()?, from_py(&value)?));
    }
    Ok(out)
}
