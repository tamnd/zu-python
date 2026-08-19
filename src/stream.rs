//! A statement read a batch at a time, instead of all at once.
//!
//! A result that does not fit in memory is the reason this exists, and
//! a reader that will not read all of it is the reason it stops
//! properly. The engine's shape for both is a sink: it hands over a
//! batch of rows, the sink says whether it wants more, and a sink that
//! says no ends the scan at the boundary an interrupt is answered at.
//! That is a push and Python wants a pull, so the two are joined by a
//! queue of two batches and a thread of this statement's own.
//!
//! A thread of its own rather than the one the connection keeps for
//! interruptible statements, because a stream lasts as long as its
//! reader takes and that thread is the connection's: a statement parked
//! on it waiting for the body of a `for` loop to come round again would
//! be every later statement on the connection waiting too. The queue is
//! bounded because that is what backpressure is. A reader that stops
//! reading stops the scan two batches later rather than pulling a
//! database into memory behind it.
//!
//! The rows are copied out of the batch on the statement's thread,
//! where the GIL is not held, and turned into Python objects on the
//! reader's thread, where it is. So the copy is a batch and never the
//! answer, and the interpreter is free for the whole time the executor
//! is working.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use zudb::query::Value;
use zudb::{Batch, Flow, Interrupt, Streamed};

use crate::error::{closed, programming, to_py_err};
use crate::interrupt::on_main_thread;
use crate::value::{Names, to_py};

/// How many batches may sit between the statement and the reader.
///
/// Two, so that one is being turned into Python objects while the next
/// is being made, and no more, because everything past that is memory
/// spent to hide a reader slower than the scan. It is the whole of the
/// backpressure and it is deliberately small.
const QUEUE: usize = 2;

/// How long a reader on the main thread waits for a batch before it
/// asks Python whether a `Ctrl-C` arrived.
///
/// The same two milliseconds every other statement waits in, for the
/// same reason: the budget from the press to the exception is fifty
/// milliseconds and most of a press is this wait.
const TICK: Duration = Duration::from_millis(2);

/// How long `close` waits for the statement to notice that nobody is
/// reading before it stops it outright.
///
/// A stream that is handing rows over notices at its next batch, which
/// is immediately, and ends with a summary that says it was stopped. A
/// statement reading a million rows that answer nothing hands over no
/// batch to notice at, so after this it is interrupted instead, which
/// is the difference between a close that waits for a scan and one that
/// does not.
const GRACE: Duration = Duration::from_millis(50);

/// What every batch of one statement shares.
///
/// The column names and the table names belong to the statement rather
/// than to the batch, so they are made once and each batch carries a
/// share of them instead of a copy.
struct Head {
    columns: Vec<String>,
    names: Names,
}

/// What the thread running the statement sends back.
enum Chunk {
    /// Rows, copied out of the batch they were borrowed in.
    Rows {
        head: Arc<Head>,
        rows: Vec<Vec<Value>>,
    },
    /// The statement ended, and this is what it ended as. Always the
    /// last thing sent.
    Done(Box<Ending>),
}

/// How a statement finished, from the reader's side.
enum Ending {
    /// It ran to the end, or to where the reader stopped it.
    Ran(Streamed),
    /// It failed, and this is what a caller is told. Held as the
    /// exception rather than as the engine's error so that it can be
    /// raised more than once, since an iterator that is asked again
    /// after it failed should say the same thing again.
    Failed(PyErr),
    /// The connection was closed, or a panic left its lock poisoned,
    /// which for anything that would run on it is the same fact.
    Closed,
}

/// The queue between the statement and the reader, with both ends able
/// to walk away.
///
/// A bounded channel is what this is. The standard library's own would
/// do but for the wait: a reader on the main thread has to give up
/// every couple of milliseconds to ask Python whether a signal arrived,
/// and a receive with a deadline is not on the stable side of the
/// channel. So the queue is written out, with one condition variable
/// making room and arrival exact rather than polled.
struct Pipe {
    held: Mutex<Held>,
    /// One variable for both directions. There is one writer and, in
    /// any program worth calling correct, one reader, so waking both is
    /// waking at most one of each.
    moved: Condvar,
}

struct Held {
    queue: VecDeque<(Arc<Head>, Vec<Vec<Value>>)>,
    /// How the statement finished, taken by the reader that asks after
    /// the last batch.
    ending: Option<Ending>,
    /// Nobody is reading any more, so the statement stops.
    shut: bool,
    /// The statement is over and has said so, which is how a reader
    /// tells the end from a pause.
    over: bool,
}

impl Pipe {
    fn new() -> Pipe {
        Pipe {
            held: Mutex::new(Held {
                queue: VecDeque::with_capacity(QUEUE),
                ending: None,
                shut: false,
                over: false,
            }),
            moved: Condvar::new(),
        }
    }

    /// Hands a batch over, waiting for room. `false` means nobody is
    /// reading any more, which is what ends the scan.
    fn send(&self, head: Arc<Head>, rows: Vec<Vec<Value>>) -> bool {
        let Ok(mut held) = self.held.lock() else {
            return false;
        };
        while held.queue.len() >= QUEUE && !held.shut {
            let Ok(next) = self.moved.wait(held) else {
                return false;
            };
            held = next;
        }
        if held.shut {
            return false;
        }
        held.queue.push_back((head, rows));
        self.moved.notify_all();
        true
    }

    /// Says the statement is over. Kept beside the queue rather than
    /// put in it, so that a reader which hung up and threw the queued
    /// batches away still finds out how the statement ended.
    fn finish(&self, ending: Ending) {
        let Ok(mut held) = self.held.lock() else {
            return;
        };
        held.ending = Some(ending);
        held.over = true;
        self.moved.notify_all();
    }

    /// The next chunk, waiting up to `patience` for one, or forever
    /// when there is none. `None` is a wait that ran out.
    fn recv(&self, patience: Option<Duration>) -> Option<Chunk> {
        let held = self.held.lock().ok()?;
        let waiting = |held: &mut Held| held.queue.is_empty() && !held.over;
        let mut held = match patience {
            Some(patience) => {
                self.moved
                    .wait_timeout_while(held, patience, waiting)
                    .ok()?
                    .0
            }
            None => self.moved.wait_while(held, waiting).ok()?,
        };
        if let Some((head, rows)) = held.queue.pop_front() {
            self.moved.notify_all();
            return Some(Chunk::Rows { head, rows });
        }
        if held.over {
            // Taken rather than copied, and a second reader asking gets
            // the same answer a closed connection gets, because there
            // is no second reader in a program worth calling correct.
            return Some(Chunk::Done(Box::new(
                held.ending.take().unwrap_or(Ending::Closed),
            )));
        }
        None
    }

    /// Says that nobody is reading, and throws away what was waiting to
    /// be read. The statement finds out at its next batch, which is the
    /// same boundary an interrupt is answered at.
    fn hang_up(&self) {
        if let Ok(mut held) = self.held.lock() {
            held.shut = true;
            held.queue.clear();
            self.moved.notify_all();
        }
    }

    /// How the statement ended, for the reader that stopped it rather
    /// than read to it. `None` while it is still running.
    fn ending(&self) -> Option<Ending> {
        self.held.lock().ok()?.ending.take()
    }

    /// Waits for the statement to say it is over, up to `patience`.
    fn wait_over(&self, patience: Option<Duration>) -> bool {
        let Ok(held) = self.held.lock() else {
            return true;
        };
        let waiting = |held: &mut Held| !held.over;
        match patience {
            Some(patience) => self
                .moved
                .wait_timeout_while(held, patience, waiting)
                .map(|(held, _)| held.over)
                .unwrap_or(true),
            None => self
                .moved
                .wait_while(held, waiting)
                .map(|held| held.over)
                .unwrap_or(true),
        }
    }
}

/// What a connection keeps of the stream it is feeding.
///
/// A connection runs one statement at a time and a stream holds the
/// connection for as long as its reader takes, so this is the thing
/// every other statement asks before it queues: a statement that waited
/// for a half-read stream would wait for a loop that is waiting for it,
/// and a program deadlocked on itself is worse than a program that is
/// told no.
pub(crate) struct Feeding {
    now: Mutex<Option<Arc<Pipe>>>,
    /// Read without the lock, because refusing is the common answer and
    /// asking should cost a load.
    busy: AtomicBool,
}

impl Feeding {
    pub(crate) fn new() -> Feeding {
        Feeding {
            now: Mutex::new(None),
            busy: AtomicBool::new(false),
        }
    }

    /// Whether a stream is holding the connection now.
    pub(crate) fn busy(&self) -> bool {
        self.busy.load(Ordering::Acquire)
    }

    /// Tells the stream that is running, if one is, that nobody is
    /// reading. For a connection on its way out: the statement ends at
    /// its next batch and gives the lock back, which is what closing is
    /// waiting for.
    pub(crate) fn hang_up(&self) {
        if let Ok(now) = self.now.lock()
            && let Some(pipe) = now.as_ref()
        {
            pipe.hang_up();
        }
    }

    fn took(&self, pipe: &Arc<Pipe>) {
        if let Ok(mut now) = self.now.lock() {
            *now = Some(Arc::clone(pipe));
        }
        self.busy.store(true, Ordering::Release);
    }

    fn gave_back(&self) {
        if let Ok(mut now) = self.now.lock() {
            *now = None;
        }
        self.busy.store(false, Ordering::Release);
    }
}

/// What the reader has in hand between two calls.
#[derive(Default)]
struct State {
    /// Rows taken out of a batch and not yet handed to Python.
    pending: VecDeque<Vec<Value>>,
    /// What the last batch was made of, kept for the rows still in
    /// `pending` and for `columns` after the statement is over.
    head: Option<Arc<Head>>,
    /// `Some` once the statement has said it is over.
    ending: Option<Ending>,
}

/// The pieces of one stream that both sides hold.
struct Live {
    pipe: Arc<Pipe>,
    /// The connection's own, so that a `Ctrl-C` or a close reaches a
    /// statement that is producing nothing to notice at.
    stop: Interrupt,
    feeding: Arc<Feeding>,
    state: Mutex<State>,
    /// Set by the reader, so a second `close` costs nothing.
    hung_up: AtomicBool,
}

/// A statement's rows, read as the executor makes them.
///
/// Iterate it for rows and `batches()` for the lists they arrive in.
/// Either way what is in memory is a batch and not the answer, which is
/// the whole point: a statement over ten million rows is read by a
/// program holding a thousand.
///
/// It holds the connection until it ends, because a connection runs one
/// statement at a time. Read it to the end, close it, or open it in a
/// `with` block, which closes it however the block is left.
#[pyclass(module = "zudb")]
pub struct Stream {
    live: Arc<Live>,
    statement: String,
}

/// What a streamed statement did, known once it has ended.
///
/// The rows are gone by then, which is the point of streaming, so this
/// is what is worth keeping about a result nobody held: what it
/// projected, how much of it was read, whether the reader stopped it
/// early, and what the engine wanted to say along the way.
#[pyclass(module = "zudb", frozen)]
pub struct StreamSummary {
    /// The column names, in the order the statement projected them.
    #[pyo3(get)]
    columns: Vec<String>,
    /// How many rows were handed over, which is fewer than the
    /// statement would have returned when the reader stopped early.
    #[pyo3(get)]
    rows: u64,
    /// Whether the reader stopped it before it ran out of rows.
    #[pyo3(get)]
    stopped: bool,
    /// Whether the rows arrived as they were made, rather than the
    /// statement running whole and being handed over in batches
    /// afterwards. A statement that has to see every row before it can
    /// give one, which is `ORDER BY`, `DISTINCT` and the aggregates, is
    /// the second kind. The loop over it reads the same either way and
    /// what differs is what it cost.
    #[pyo3(get)]
    streamed: bool,
    notices: Vec<(String, String, String)>,
}

#[pymethods]
impl StreamSummary {
    /// The warnings the statement raised, in the shape a result reports
    /// them.
    #[getter]
    fn notices<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let out = PyList::empty(py);
        for (code, condition, detail) in &self.notices {
            let one = PyDict::new(py);
            one.set_item("code", code)?;
            one.set_item("condition", condition)?;
            one.set_item("detail", detail)?;
            out.append(one)?;
        }
        Ok(out)
    }

    fn __repr__(&self) -> String {
        format!(
            "<zudb.StreamSummary {} rows{}{}>",
            self.rows,
            if self.stopped { ", stopped" } else { "" },
            if self.streamed { "" } else { ", buffered" }
        )
    }
}

/// The same rows, in the batches they arrived in.
///
/// One object rather than a method that yields, so that closing the
/// stream is the one call it always was and a batch reader is the same
/// stream seen a list at a time.
#[pyclass(module = "zudb")]
pub struct StreamBatches {
    stream: Py<Stream>,
}

#[pymethods]
impl StreamBatches {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let stream = self.stream.borrow(py);
        let live = Arc::clone(&stream.live);
        drop(stream);
        loop {
            {
                let mut state = live.state.lock().map_err(|_| panicked())?;
                if !state.pending.is_empty() {
                    let head = state.head.clone().ok_or_else(panicked)?;
                    let rows: Vec<Vec<Value>> = state.pending.drain(..).collect();
                    drop(state);
                    let out = PyList::empty(py);
                    for row in &rows {
                        out.append(tuple(py, row, &head.names)?)?;
                    }
                    return Ok(out);
                }
            }
            if !pull(&live, py)? {
                return Err(pyo3::exceptions::PyStopIteration::new_err(()));
            }
        }
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        format!(
            "<zudb.StreamBatches of {}>",
            self.stream.borrow(py).__repr__()
        )
    }
}

#[pymethods]
impl Stream {
    /// The column names, in the order the statement projects them.
    ///
    /// Answering this reads the first batch and keeps it, because the
    /// names are the statement's and the statement does not say them
    /// until it has made a row. Nothing is lost by that: the batch is
    /// handed to the reader that asks next.
    #[getter]
    fn columns(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        loop {
            {
                let state = self.live.state.lock().map_err(|_| panicked())?;
                if let Some(head) = state.head.as_ref() {
                    return Ok(head.columns.clone());
                }
                if let Some(Ending::Ran(streamed)) = state.ending.as_ref() {
                    return Ok(streamed.columns.clone());
                }
            }
            if !pull(&self.live, py)? {
                return Ok(Vec::new());
            }
        }
    }

    /// What the statement did, once it has done it, and `None` while it
    /// is still running.
    ///
    /// A statement that failed has raised its failure at the row it
    /// happened on, so a stream that ended badly reports no summary
    /// rather than a summary of half a run.
    #[getter]
    fn summary(&self, py: Python<'_>) -> PyResult<Option<StreamSummary>> {
        let state = self.live.state.lock().map_err(|_| panicked())?;
        let _ = py;
        Ok(match state.ending.as_ref() {
            Some(Ending::Ran(streamed)) => Some(summarised(streamed)),
            _ => None,
        })
    }

    /// The rows in the batches they arrived in, as lists of tuples.
    ///
    /// For a reader that writes what it reads somewhere with a size of
    /// its own. A batch holds at most what `batch_rows` asked for, and
    /// the last one holds whatever was left.
    fn batches(slf: Py<Self>) -> StreamBatches {
        StreamBatches { stream: slf }
    }

    /// Stops the statement and gives the connection back.
    ///
    /// Doing it twice is not an error, and doing it to a stream that
    /// has already ended does nothing, so a `with` block that read to
    /// the end leaves the same way one that read a page does.
    fn close(&self, py: Python<'_>) {
        hang_up(&self.live, py);
    }

    /// Whether the statement is over, either because it ran out of rows
    /// or because this stream was closed.
    #[getter]
    fn closed(&self, py: Python<'_>) -> PyResult<bool> {
        let ended = self
            .live
            .state
            .lock()
            .map_err(|_| panicked())?
            .ending
            .is_some();
        let _ = py;
        Ok(ended || self.live.hung_up.load(Ordering::Acquire))
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        loop {
            {
                let mut state = self.live.state.lock().map_err(|_| panicked())?;
                if let Some(row) = state.pending.pop_front() {
                    let head = state.head.clone().ok_or_else(panicked)?;
                    drop(state);
                    return tuple(py, &row, &head.names);
                }
            }
            if !pull(&self.live, py)? {
                return Err(pyo3::exceptions::PyStopIteration::new_err(()));
            }
        }
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[pyo3(signature = (*_exception))]
    fn __exit__(&self, py: Python<'_>, _exception: &Bound<'_, PyTuple>) -> bool {
        self.close(py);
        false
    }

    fn __repr__(&self) -> String {
        let over = self
            .live
            .state
            .lock()
            .map(|state| state.ending.is_some())
            .unwrap_or(true);
        format!(
            "<zudb.Stream \"{}\"{}>",
            self.statement,
            if over { ", ended" } else { "" }
        )
    }
}

impl Drop for Stream {
    /// A stream nobody holds any more is a reader that walked away, and
    /// the statement behind it should not go on reading for one. The
    /// wait belongs to `close`, though, not here: a collector running
    /// this cannot afford to wait for a scan, so the word is said and
    /// the connection comes free a batch later.
    fn drop(&mut self) {
        if !self.live.hung_up.swap(true, Ordering::AcqRel) {
            self.live.pipe.hang_up();
            self.live.stop.stop();
        }
    }
}

/// Starts a statement on a thread of its own and hands back the stream
/// that reads it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn open(
    py: Python<'_>,
    inner: &Arc<Mutex<Option<zudb::Connection>>>,
    stop: &Interrupt,
    feeding: &Arc<Feeding>,
    alive: bool,
    statement: String,
    params: Vec<(String, Value)>,
    batch_rows: Option<usize>,
) -> PyResult<Stream> {
    if !alive {
        return Err(closed(py, "this connection"));
    }
    if feeding.busy() {
        return Err(programming(py, STREAMING));
    }
    if batch_rows == Some(0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "batch_rows is how many rows a batch may hold, so it starts at one",
        ));
    }
    let live = Arc::new(Live {
        pipe: Arc::new(Pipe::new()),
        stop: stop.clone(),
        feeding: Arc::clone(feeding),
        state: Mutex::new(State::default()),
        hung_up: AtomicBool::new(false),
    });
    feeding.took(&live.pipe);
    let held = Arc::clone(inner);
    let running = Arc::clone(&live);
    let text = statement.clone();
    let started = std::thread::Builder::new()
        .name("zudb stream".into())
        .spawn(move || run(&running, &held, &text, &params, batch_rows));
    if started.is_err() {
        feeding.gave_back();
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "a thread to stream a statement on could not be started",
        ));
    }
    Ok(Stream { live, statement })
}

/// What a connection a stream is still reading says to the next
/// statement somebody runs on it.
pub(crate) const STREAMING: &str = "a stream on this connection has not finished, and a connection runs one statement at a \
     time: read the stream to the end, close it, or open a second connection";

/// The statement, on its own thread, from the moment it takes the
/// connection to the moment it gives it back.
fn run(
    live: &Arc<Live>,
    inner: &Arc<Mutex<Option<zudb::Connection>>>,
    statement: &str,
    params: &[(String, Value)],
    batch_rows: Option<usize>,
) {
    let pipe = Arc::clone(&live.pipe);
    let ending = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Ok(mut held) = inner.lock() else {
            return Ending::Closed;
        };
        let Some(conn) = held.as_mut() else {
            return Ending::Closed;
        };
        // Cleared holding the lock, for the reason every other
        // statement clears it there: a stop that arrived while nothing
        // was running must not end the statement about to start.
        conn.interrupt().clear();
        let names = Names::of(conn.session_mut().catalog());
        let borrowed: Vec<(&str, Value)> = params
            .iter()
            .map(|(name, value)| (name.as_str(), value.clone()))
            .collect();
        let mut head: Option<Arc<Head>> = None;
        let mut sink = |batch: Batch<'_>| -> zudb::Result<Flow> {
            let head = head.get_or_insert_with(|| {
                Arc::new(Head {
                    columns: batch.columns().to_vec(),
                    names: names.clone(),
                })
            });
            // Copied out here, on this thread, with the GIL nowhere in
            // sight. The batch is borrowed for the length of this call
            // and the reader is going to hold it for the length of a
            // loop body, so there is nothing to keep instead.
            Ok(if pipe.send(Arc::clone(head), batch.rows().to_vec()) {
                Flow::More
            } else {
                Flow::Stop
            })
        };
        let out = match batch_rows {
            Some(rows) => conn.query_stream_batched(statement, &borrowed, rows, &mut sink),
            None => conn.query_stream(statement, &borrowed, &mut sink),
        };
        match out {
            Ok(streamed) => Ending::Ran(streamed),
            // Built here rather than on the reader's thread, which
            // means taking the GIL for as long as one exception takes
            // to make. It is the last thing this thread does and it is
            // what lets the same failure be raised twice.
            Err(err) => Python::attach(|py| Ending::Failed(to_py_err(py, err))),
        }
    }));
    // The lock is gone by here, so the connection is free before
    // anybody is told the statement is over.
    live.feeding.gave_back();
    let ending = ending.unwrap_or_else(|_| {
        // Raised again on the reader's thread rather than left to end
        // this one silently, because a panic inside the engine is
        // something the program that ran the statement should see.
        Ending::Failed(pyo3::exceptions::PyRuntimeError::new_err(
            "the statement behind this stream ended in a panic",
        ))
    });
    live.pipe.finish(ending);
}

/// Waits for the next chunk and puts it where the reader will find it.
///
/// `false` means the statement is over and there is nothing more
/// coming, which is the only answer a caller has to act on.
fn pull(live: &Arc<Live>, py: Python<'_>) -> PyResult<bool> {
    {
        let state = live.state.lock().map_err(|_| panicked())?;
        if let Some(ending) = state.ending.as_ref() {
            return match ending {
                Ending::Failed(err) => Err(err.clone_ref(py)),
                Ending::Closed => Err(closed(py, "this connection")),
                Ending::Ran(_) => Ok(false),
            };
        }
    }
    // Off the main thread a signal was never going to arrive, so the
    // wait is one wait. On it the wait is a couple of milliseconds at a
    // time with a question to Python between them, which is where a
    // press is felt.
    let chunk = if on_main_thread(py) {
        loop {
            if let Some(chunk) = py.detach(|| live.pipe.recv(Some(TICK))) {
                break chunk;
            }
            if let Err(signal) = py.check_signals() {
                // Stopped rather than left running, so the connection
                // is free by the time the exception reaches the caller
                // rather than busy with rows nobody wants.
                hang_up(live, py);
                return Err(signal);
            }
        }
    } else {
        py.detach(|| live.pipe.recv(None))
            .ok_or_else(|| closed(py, "this connection"))?
    };
    let mut state = live.state.lock().map_err(|_| panicked())?;
    match chunk {
        Chunk::Rows { head, rows } => {
            state.head = Some(head);
            state.pending.extend(rows);
            Ok(true)
        }
        Chunk::Done(ending) => {
            let out = match ending.as_ref() {
                Ending::Failed(err) => Err(err.clone_ref(py)),
                Ending::Closed => Err(closed(py, "this connection")),
                Ending::Ran(_) => Ok(false),
            };
            state.ending = Some(*ending);
            out
        }
    }
}

/// Says that nobody is reading and waits for the connection to come
/// back, stopping the statement outright if it does not.
fn hang_up(live: &Arc<Live>, py: Python<'_>) {
    if live.hung_up.swap(true, Ordering::AcqRel) {
        return;
    }
    live.pipe.hang_up();
    // Waited for rather than abandoned, so the connection is free by
    // the time this call returns and the next statement on it is not
    // told that a stream nobody holds is still reading.
    py.detach(|| {
        if !live.pipe.wait_over(Some(GRACE)) {
            live.stop.stop();
            live.pipe.wait_over(None);
        }
    });
    // Kept, so that a reader which stopped a stream can still ask what
    // it stopped: the summary of a run cut short is exactly the thing
    // that says how much of it was read.
    if let Some(ending) = live.pipe.ending()
        && let Ok(mut state) = live.state.lock()
        && state.ending.is_none()
    {
        state.ending = Some(ending);
    }
}

fn tuple<'py>(py: Python<'py>, row: &[Value], names: &Names) -> PyResult<Bound<'py, PyTuple>> {
    PyTuple::new(
        py,
        row.iter()
            .map(|value| to_py(py, value, names))
            .collect::<PyResult<Vec<_>>>()?,
    )
}

fn summarised(streamed: &Streamed) -> StreamSummary {
    StreamSummary {
        columns: streamed.columns.clone(),
        rows: streamed.rows,
        stopped: streamed.stopped,
        streamed: streamed.streamed,
        notices: streamed
            .notices
            .iter()
            .map(|notice| {
                (
                    notice.status.code().to_string(),
                    notice.status.standard_text().to_string(),
                    notice.detail.clone(),
                )
            })
            .collect(),
    }
}

fn panicked() -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(
        "this stream was left in an unknown state by a thread that panicked",
    )
}
