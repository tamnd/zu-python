//! Stopping a statement while it runs: a `Ctrl-C` from the person
//! waiting for it, and `interrupt()` from another thread.
//!
//! Python answers a signal by setting a flag in the handler and raising
//! the exception the next time the main thread runs bytecode, and a
//! thread inside the engine with the GIL released is not running
//! bytecode. That is why a press during a ten second statement is felt
//! ten seconds later in every binding that does nothing about it, and
//! it is why a statement that could be interrupted runs on a thread of
//! its own while the thread that asked for it waits, asking Python
//! every few milliseconds whether a press arrived.
//!
//! The thread is kept rather than made, and it is only made at all
//! where it buys something. A thread costs some thirty microseconds to
//! start and six to hand a job to once it is parked, and a small
//! statement costs less than either, so the connection keeps one and
//! wakes it. Python delivers a signal to the main thread and to no
//! other, so a statement anywhere else runs on the thread that asked
//! for it, exactly as it did before, and is stopped by `interrupt()` or
//! not at all.
//!
//! Stopping is the engine's [`Interrupt`], which the connection holds a
//! clone of from the moment it is opened. Holding it outside the lock
//! is the whole trick: an ask that had to wait for the connection to be
//! free could only ever arrive after the statement it was meant to
//! stop.

use std::cell::Cell;
use std::panic::AssertUnwindSafe;

use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use pyo3::prelude::*;
use zudb::Interrupt;

use crate::error::closed;

/// How long the waiting thread sleeps between two asks.
///
/// The budget is 50 ms from the press to the exception, and most of a
/// press is this sleep: the engine answers a stop at the boundary of a
/// chunk, which is a tenth of a millisecond of the budget, and the rest
/// of it is the wait between the press landing and the next ask. Two
/// puts the whole path at five milliseconds measured, forty five of it
/// spare for a machine under load, and a statement running for an hour
/// wakes this thread under two million times, which is a wakeup against
/// two milliseconds of work.
const TICK: Duration = Duration::from_millis(2);

/// How long the waiting thread looks for the answer before it sleeps
/// for it.
///
/// A sleep and the wake that ends it cost microseconds, which is more
/// than a small statement costs to run, so a thread that went straight
/// to sleep would add that to every quick statement in the process.
/// Two hundred microseconds is longer than a small statement and
/// nothing beside a statement anybody would reach for the keyboard
/// over, and it is spent on the thread with nothing else to do rather
/// than on the one running the query.
const SPIN: Duration = Duration::from_micros(200);

/// Why a statement gave back no answer at all.
pub enum Stopped {
    /// The connection was closed, or a panic left its lock poisoned,
    /// which for anything that would run on it is the same fact.
    Closed,
    /// A signal arrived while the statement ran. This is the exception
    /// Python had waiting, which is a `KeyboardInterrupt` unless the
    /// program installed a handler that raises something else.
    Signal(PyErr),
}

impl Stopped {
    pub fn raise(self, py: Python<'_>) -> PyErr {
        match self {
            Stopped::Closed => closed(py, "this connection"),
            Stopped::Signal(err) => err,
        }
    }
}

/// A statement on its way to the thread that runs it.
///
/// Type-erased, so that one thread serves statements answering
/// different things: what each job gives back it gives back through the
/// slot it was built around, and by then nobody needs to know what it
/// was.
type Job = Box<dyn FnOnce() + Send + 'static>;

/// The thread a connection keeps for the statements it wants to be able
/// to interrupt.
///
/// One thread per connection rather than one per statement, and started
/// at the first statement rather than at the connection, so a program
/// that never runs one on the main thread never pays for it.
pub struct Runner {
    post: Arc<Post>,
    /// Joined when the connection goes, so that nothing of this
    /// connection's is still running once nothing points at it. It can
    /// only ever be a statement that is already over: a statement being
    /// waited for is being waited for by somebody holding the
    /// connection.
    thread: Option<std::thread::JoinHandle<()>>,
}

/// Where the statements waiting for that thread sit.
#[derive(Default)]
struct Post {
    queue: Mutex<Queue>,
    knock: Condvar,
}

#[derive(Default)]
struct Queue {
    /// A queue rather than a slot, for the one caller who has two at
    /// once: a signal handler that runs a statement of its own while
    /// the statement it interrupted is still being waited for.
    jobs: std::collections::VecDeque<Job>,
    /// Set when the connection is dropped, which is what ends the
    /// thread.
    shut: bool,
}

impl Post {
    /// The next statement, or `None` once the connection is gone.
    ///
    /// Looked for before it is slept for. A statement handed to a
    /// thread that is still awake costs a lock, and one handed to a
    /// thread that has parked costs a wake, which on this machine is
    /// the difference between one microsecond and eighty. A program
    /// running statements in a loop hands them over inside the look
    /// every time, and one that has gone quiet pays the wake once.
    fn next(&self) -> Option<Job> {
        let looking = Instant::now();
        loop {
            if let Ok(mut queue) = self.queue.try_lock() {
                if let Some(job) = queue.jobs.pop_front() {
                    return Some(job);
                }
                if queue.shut {
                    return None;
                }
            }
            if looking.elapsed() >= SPIN {
                break;
            }
            std::hint::spin_loop();
        }
        let mut queue = self.queue.lock().unwrap_or_else(|held| held.into_inner());
        loop {
            if let Some(job) = queue.jobs.pop_front() {
                return Some(job);
            }
            if queue.shut {
                return None;
            }
            queue = self
                .knock
                .wait(queue)
                .unwrap_or_else(|held| held.into_inner());
        }
    }

    /// Hands a statement over, or says that the thread is gone.
    fn send(&self, job: Job) -> bool {
        let mut queue = self.queue.lock().unwrap_or_else(|held| held.into_inner());
        if queue.shut {
            return false;
        }
        queue.jobs.push_back(job);
        self.knock.notify_one();
        true
    }
}

impl Runner {
    fn start() -> Runner {
        let post = Arc::new(Post::default());
        let taking = Arc::clone(&post);
        let thread = std::thread::Builder::new()
            .name("zudb statement".into())
            .spawn(move || {
                while let Some(job) = taking.next() {
                    job();
                }
            })
            // A process that cannot start a thread cannot run a
            // statement on one, and there is nothing to say about that
            // which the operating system's own message does not.
            .expect("a thread to run statements on");
        Runner {
            post,
            thread: Some(thread),
        }
    }
}

impl Drop for Runner {
    fn drop(&mut self) {
        self.post
            .queue
            .lock()
            .unwrap_or_else(|held| held.into_inner())
            .shut = true;
        self.post.knock.notify_all();
        if let Some(thread) = self.thread.take() {
            // A panic inside a statement was caught where it happened
            // and raised again on the thread that asked for it, so
            // there is nothing here for one to have left behind.
            let _ = thread.join();
        }
    }
}

/// What a statement leaves for the thread waiting on it.
struct Answer<T> {
    /// `None` until the statement is over. A panic inside the engine
    /// arrives here as the payload it unwound with, to be raised again
    /// on the thread that asked for the statement, which is where a
    /// Python caller can see it.
    slot: Mutex<Option<std::thread::Result<T>>>,
    over: Condvar,
}

impl<T> Answer<T> {
    fn new() -> Answer<T> {
        Answer {
            slot: Mutex::new(None),
            over: Condvar::new(),
        }
    }

    fn done(&self, out: std::thread::Result<T>) {
        *self.slot.lock().unwrap_or_else(|held| held.into_inner()) = Some(out);
        self.over.notify_all();
    }

    /// The answer if it is there, without waiting for it.
    fn peek(&self) -> Option<std::thread::Result<T>> {
        self.slot.try_lock().ok().and_then(|mut slot| slot.take())
    }

    /// The answer, waiting up to `patience` for it, or forever when
    /// there is none.
    fn wait(&self, patience: Option<Duration>) -> Option<std::thread::Result<T>> {
        let slot = self.slot.lock().unwrap_or_else(|held| held.into_inner());
        let waiting = |slot: &mut Option<std::thread::Result<T>>| slot.is_none();
        let mut slot = match patience {
            Some(patience) => {
                self.over
                    .wait_timeout_while(slot, patience, waiting)
                    .unwrap_or_else(|held| held.into_inner())
                    .0
            }
            None => self
                .over
                .wait_while(slot, waiting)
                .unwrap_or_else(|held| held.into_inner()),
        };
        slot.take()
    }
}

/// Runs `work` on the connection, with a `Ctrl-C` able to stop it.
///
/// `stop` is the connection's own handle, taken when it was opened
/// rather than fetched from inside the lock, because the point of it is
/// to be reachable while the lock is held by the statement it stops.
///
/// A press that arrives while the statement is still running raises,
/// and whatever the statement was about to answer is dropped: the
/// person pressed the key, and the answer to that is the exception
/// rather than the rows they interrupted. A press with nothing running
/// is Python's to deliver as it always was.
pub fn watched<T: Send + 'static>(
    py: Python<'_>,
    runner: &OnceLock<Runner>,
    inner: &Arc<Mutex<Option<zudb::Connection>>>,
    stop: &Interrupt,
    work: impl FnOnce(&mut zudb::Connection) -> T + Send + 'static,
) -> Result<T, Stopped> {
    let held = Arc::clone(inner);
    let run = move || -> Result<T, Stopped> {
        let mut held = held.lock().map_err(|_| Stopped::Closed)?;
        let conn = held.as_mut().ok_or(Stopped::Closed)?;
        // Cleared here, holding the lock, rather than before the wait.
        // An ask that arrived while nothing was running must not end
        // the statement about to start, and an ask meant for the
        // statement another thread is running must not be cleared by
        // this one queuing up behind it.
        conn.interrupt().clear();
        Ok(work(conn))
    };
    if !on_main_thread(py) {
        // The GIL goes down for the whole statement, waiting for the
        // connection's own lock included, which is what lets another
        // thread run while this one is inside the executor. Waiting
        // for the lock with the GIL held would be worse than slow: the
        // thread inside the statement has to take the GIL back to
        // return, and it could not.
        return py.detach(run);
    }
    let answer: Arc<Answer<Result<T, Stopped>>> = Arc::new(Answer::new());
    let done = Arc::clone(&answer);
    let job: Job = Box::new(move || {
        // The panic is caught rather than left to end the thread this
        // connection keeps: an engine that panicked on one statement
        // is still a connection somebody is going to call `close` on,
        // and a thread that died holding the answer would leave the
        // caller waiting for it forever.
        done.done(std::panic::catch_unwind(AssertUnwindSafe(run)));
    });
    if !runner.get_or_init(Runner::start).post.send(job) {
        // The thread is gone, which happens when the connection it
        // belongs to is dropped, which is a connection nothing can run
        // on anyway.
        return Err(Stopped::Closed);
    }
    let out = py.detach(|| {
        // Most statements are over in less time than a sleep and a wake
        // cost, so the first wait is a look.
        let looking = Instant::now();
        loop {
            if let Some(out) = answer.peek() {
                return Some(out);
            }
            if looking.elapsed() >= SPIN {
                return None;
            }
            std::hint::spin_loop();
        }
    });
    let out = match out {
        Some(out) => out,
        None => loop {
            if let Some(out) = py.detach(|| answer.wait(Some(TICK))) {
                break out;
            }
            // Where the press is felt. Python runs the handler here, on
            // the thread it is meant to run on, and hands back what it
            // raised.
            if let Err(signal) = py.check_signals() {
                stop.stop();
                // Waited for rather than abandoned, so that the next
                // statement finds the connection free rather than
                // locked by one nobody is listening to.
                py.detach(|| answer.wait(None));
                return Err(Stopped::Signal(signal));
            }
        },
    };
    // Raised again here, on the thread that asked for the statement,
    // where PyO3 turns it into an exception a caller can see rather
    // than into a process that stops.
    out.unwrap_or_else(|panicked| std::panic::resume_unwind(panicked))
}

thread_local! {
    /// Whether this thread is the one a signal can reach, asked once
    /// per thread and kept, because the answer cannot change while the
    /// thread lives.
    static IS_MAIN: Cell<Option<bool>> = const { Cell::new(None) };
}

/// Whether a `Ctrl-C` can arrive on this thread at all.
///
/// `threading` rather than the interpreter's own thread state, because
/// the limited ABI these wheels are built against does not expose the
/// second one, and asking Python is a couple of attribute lookups on a
/// module that is imported long before anything here runs.
fn on_main_thread(py: Python<'_>) -> bool {
    IS_MAIN.with(|known| {
        if let Some(known) = known.get() {
            return known;
        }
        let asked = idents(py).is_ok_and(|(this, main)| this == main);
        known.set(Some(asked));
        asked
    })
}

fn idents(py: Python<'_>) -> PyResult<(u64, u64)> {
    let threading = py.import("threading")?;
    let this: u64 = threading.call_method0("get_ident")?.extract()?;
    let main: u64 = threading
        .call_method0("main_thread")?
        .getattr("ident")?
        .extract()?;
    Ok((this, main))
}
