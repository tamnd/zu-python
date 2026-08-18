//! Appending rows to a table that already exists.
//!
//! `INSERT` is the wrong shape for loading data. Every row is parsed,
//! bound, planned and committed, and the commit is the expensive part,
//! so a million rows is a million commits and the load is dominated by
//! durability work nobody asked for. `load` is the right shape for a
//! database that does not exist yet, and this is the right shape for
//! one that does: rows go into per-column buffers in memory, and a
//! flush turns the whole buffer into one commit.
//!
//! ```python
//! with conn.appender("person") as rows:
//!     for uid, name in enumerate(names):
//!         rows.append_row([uid, name])
//! ```
//!
//! A row is every column of the table, in the order the table declares
//! them, and a column is a position rather than a name: naming the
//! columns per row would cost a lookup per value on the one path where
//! per-value cost is the whole story, and a loader knows its own column
//! order.
//!
//! The engine's appender borrows the connection for as long as it
//! lives, which is a promise a Python object cannot make, so this one
//! buffers here and opens an engine appender for the length of a flush.
//! That is a catalog read per flush, against a commit and a fold that
//! cost time proportional to the table, so it is not where a load
//! spends its time. What it buys is an appender that can be held in a
//! variable, passed to a function and closed by a `with` block, which
//! is what a Python caller expects of it.

use std::sync::{Mutex, MutexGuard};

use pyo3::exceptions::PyResourceWarning;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use zudb::zu1::catalog::Catalog;
use zudb::{Field, ZuError};

use crate::buffer::{Column, Mismatch, type_name};
use crate::conn::Connection;
use crate::error::{programming, to_py_err};

/// Rows on their way into a table, buffered until they are flushed.
///
/// Take one with `Connection.appender`, append rows to it, and close
/// it. What is buffered is columnar and typed from the table's own
/// columns, read when the appender opened, so a value that does not
/// belong in a column is refused by the call that appended it rather
/// than at the flush that would have carried it, and the message names
/// the column it did not fit.
#[pyclass(module = "zudb")]
pub struct Appender {
    /// The connection this writes through, held rather than borrowed:
    /// an appender whose connection was collected would be a buffer
    /// with nowhere to go.
    conn: Py<Connection>,
    #[pyo3(get)]
    table: String,
    state: Mutex<State>,
}

/// One column of the table, and what has been buffered for it.
struct Buffer {
    name: String,
    values: Column,
    /// The node table whose rows this column names, for the two columns
    /// of a rel table and for nothing else: a row of one is an offset
    /// into the table the edge runs from and an offset into the table
    /// it runs to. A negative offset is no row of anything and is
    /// refused where it was appended; whether the row is there at all
    /// is a question only the flush can answer, since the table may be
    /// being appended to at the same time.
    ends: Option<u32>,
}

/// What is buffered, and how much of it has gone in.
struct State {
    /// One buffer per column of the table, in the order the table
    /// declares them, built when the appender opened. A flush empties
    /// these and keeps them, since the next batch is the same shape as
    /// the last.
    cols: Vec<Buffer>,
    /// Rows buffered and not yet written, which is every column's
    /// length and is kept beside them so that a table with no columns
    /// can still answer for itself.
    buffered: u64,
    /// Rows this appender has committed, across every flush.
    committed: u64,
    open: bool,
}

/// What can go wrong with the GIL down, where there is no way to build
/// a Python exception yet.
enum Snag {
    Closed,
    Locked,
    Finished,
    Engine(ZuError),
    /// A row the engine's appender would not take, named by where it
    /// sits in the batch, because that is the row the caller can go and
    /// look at.
    Row(u64, ZuError),
    /// An edge to a row that is not there, caught before the write
    /// rather than after it.
    Offset {
        row: u64,
        offset: i64,
        table: String,
        rows: u64,
    },
}

impl Snag {
    fn raise(self, py: Python<'_>) -> PyErr {
        match self {
            Snag::Closed => programming(
                py,
                "the connection this appender writes through is closed, so its \
                 buffered rows have nowhere to go",
            ),
            Snag::Locked => programming(
                py,
                "this appender was left locked by a panic, and what it holds \
                 cannot be trusted to be a rectangle any more",
            ),
            Snag::Finished => programming(
                py,
                "this appender is closed, and a closed appender has already \
                 written everything it was given",
            ),
            Snag::Engine(err) => to_py_err(py, err),
            // The engine reports the value and the column; which row of
            // the batch it was is the part only this side knows, and it
            // is the part that says where to look.
            Snag::Row(at, err) => {
                let raised = to_py_err(py, err);
                programming(py, &format!("row {at} of this batch: {}", raised.value(py)))
            }
            Snag::Offset {
                row,
                offset,
                table,
                rows,
            } => pyo3::exceptions::PyValueError::new_err(format!(
                "row {row} of this batch joins row {offset} of '{table}', which has {rows} \
                 rows in it, so the rows an edge joins have to be written before the edge is"
            )),
        }
    }
}

#[pymethods]
impl Appender {
    /// Appends one row, which is a sequence of one value per column of
    /// the table, in the order the table declares them.
    ///
    /// The values go into memory and nothing else happens, so this is a
    /// conversion and a push per column. A row of the wrong width, or
    /// with a value that does not fit the column, is refused with
    /// nothing of it kept, so the appender is still usable once the
    /// caller has fixed the row.
    fn append_row(&self, py: Python<'_>, row: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut state = self.writable(py)?;
        state.append(&self.table, row)
    }

    /// Appends every row of an iterable of rows.
    ///
    /// The same thing in a loop, and worth a call of its own because it
    /// is one lock and one attribute lookup for the batch rather than
    /// one per row. A row that is refused stops the call where it was
    /// refused and the rows before it stay buffered: nothing here is a
    /// transaction until the flush, and throwing away work the caller
    /// can keep would not make it one.
    fn append_rows(&self, py: Python<'_>, rows: &Bound<'_, PyAny>) -> PyResult<usize> {
        let mut state = self.writable(py)?;
        let mut taken = 0;
        for row in rows.try_iter()? {
            state.append(&self.table, &row?)?;
            taken += 1;
        }
        Ok(taken)
    }

    /// Writes every buffered row and makes it readable, and answers how
    /// many rows this appender has committed in all.
    ///
    /// One commit, whatever the buffer holds, with the GIL released for
    /// it: the values are sealed into the file, one frame naming them
    /// is synced to the log, and the fold that follows puts them where
    /// every query looks. On return the buffer is empty and the rows
    /// are there. A flush with nothing buffered touches no file, so a
    /// loader can flush on a timer without writing empty commits.
    ///
    /// A flush that fails keeps its rows, so that what did not go in is
    /// still there to be looked at and tried again.
    fn flush(&self, py: Python<'_>) -> PyResult<u64> {
        let held = self.conn.bind(py).borrow();
        let conn: &Connection = &held;
        // Everything from here down is Rust over buffers that are
        // already typed, so the GIL is not needed for any of it, and a
        // flush is where a load spends its time. The buffer's own lock
        // is taken down here too, and not above: a lock waited for with
        // the GIL held is a lock that stops every other thread in the
        // process for as long as the wait.
        py.detach(|| -> Result<u64, Snag> {
            let mut state = self.state.lock().map_err(|_| Snag::Locked)?;
            if !state.open {
                return Err(Snag::Finished);
            }
            state.write(conn, &self.table)
        })
        .map_err(|snag| snag.raise(py))
    }

    /// Rows buffered and not yet written.
    #[getter]
    fn buffered(&self, py: Python<'_>) -> PyResult<u64> {
        Ok(self.locked(py)?.buffered)
    }

    /// Rows this appender has committed, across every flush.
    #[getter]
    fn committed(&self, py: Python<'_>) -> PyResult<u64> {
        Ok(self.locked(py)?.committed)
    }

    /// Throws away what is buffered and answers how many rows that was.
    ///
    /// The way out of a load that went wrong halfway. A caller who has
    /// noticed that the rows are wrong wants them gone, and closing
    /// would write them. Rows an earlier flush committed are committed,
    /// and this does not reach them.
    fn discard(&self, py: Python<'_>) -> PyResult<u64> {
        let mut state = self.writable(py)?;
        let dropped = state.buffered;
        state.empty();
        Ok(dropped)
    }

    /// Flushes what is left and answers how many rows this appender
    /// committed in all.
    ///
    /// Closing twice is not an error and writes nothing the second
    /// time, because a `with` block that closed early would otherwise
    /// fail on the way out.
    fn close(&self, py: Python<'_>) -> PyResult<u64> {
        let held = self.conn.bind(py).borrow();
        let conn: &Connection = &held;
        py.detach(|| -> Result<u64, Snag> {
            let mut state = self.state.lock().map_err(|_| Snag::Locked)?;
            if !state.open {
                return Ok(state.committed);
            }
            let committed = state.write(conn, &self.table)?;
            state.open = false;
            Ok(committed)
        })
        .map_err(|snag| snag.raise(py))
    }

    /// Whether this appender has been closed.
    #[getter]
    fn closed(&self, py: Python<'_>) -> PyResult<bool> {
        Ok(!self.locked(py)?.open)
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Closes on the way out, which flushes, and does it whether the
    /// block ended well or badly.
    ///
    /// A block that raised is the case worth thinking about, and the
    /// rows still go in. That is the answer the Rust appender gives by
    /// flushing when it is dropped, for the same reason: a loader that
    /// stopped partway is better served by its rows arriving than by
    /// them vanishing, and a caller who wants the other answer writes
    /// `discard()` and gets exactly it. An error the flush raises
    /// carries the block's own exception as its context, so neither of
    /// the two is lost.
    #[pyo3(signature = (*_exception))]
    fn __exit__(&self, py: Python<'_>, _exception: &Bound<'_, PyTuple>) -> PyResult<bool> {
        self.close(py)?;
        // False, so an exception raised inside the block carries on out
        // of it.
        Ok(false)
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let state = self.locked(py)?;
        let closed = if state.open { "" } else { ", closed" };
        Ok(format!(
            "<zudb.Appender {}, {} buffered, {} committed{closed}>",
            self.table, state.buffered, state.committed
        ))
    }
}

/// An appender the collector took while it still held rows says so.
///
/// The rows are gone by then, and going quietly is the failure mode
/// worth a warning: the caller wrote a loop that appended a million
/// rows, never closed the appender, and got a database with nothing in
/// it and no complaint about it. Flushing here instead is what the
/// Rust appender does when it is dropped, and it is not available to
/// this one: a collector runs whenever it likes, including while
/// another thread is inside a statement on the same connection, and a
/// commit from there would be a write nobody asked for at a moment
/// nobody chose.
///
/// `ResourceWarning` is the class Python already uses for a file that
/// was never closed, which is the same mistake with the same cure, and
/// it is silent by default and loud under `-W error` and under pytest.
impl Drop for Appender {
    fn drop(&mut self) {
        let Ok(state) = self.state.lock() else { return };
        if !state.open || state.buffered == 0 {
            return;
        }
        let Ok(message) = std::ffi::CString::new(format!(
            "appender on '{}' was collected with {} row{} buffered and never closed, \
             so they were discarded: close it, or hold it in a `with` block",
            self.table,
            state.buffered,
            if state.buffered == 1 { "" } else { "s" },
        )) else {
            return;
        };
        Python::attach(|py| {
            // A collector runs wherever it likes, including in the
            // middle of raising something else, and warning while an
            // exception is set is an error in itself. The one being
            // raised is put aside and put back, so the warning is an
            // aside rather than a thing that replaces the failure the
            // caller is about to see.
            let raising = PyErr::take(py);
            // A warning turned into an error by the caller's filters
            // has nowhere to go from a destructor, so it is written to
            // stderr the way Python writes any exception raised in one.
            if let Err(raised) = PyErr::warn(py, &py.get_type::<PyResourceWarning>(), &message, 1) {
                raised.write_unraisable(py, None);
            }
            if let Some(raising) = raising {
                raising.restore(py);
            }
        });
    }
}

impl Appender {
    /// Opens an appender on `table`, which is a node table or a rel
    /// table of the graph this connection reads.
    ///
    /// An engine appender is opened here and dropped, purely to find
    /// out whether it can be opened at all: a table nothing declares, a
    /// column that holds a null, a table a keyed rel table is built
    /// over, and a connection that is read-only are all refused here
    /// rather than at the first flush. A caller about to buffer a
    /// million rows wants to hear about them now.
    pub fn open(py: Python<'_>, conn: Py<Connection>, table: &str) -> PyResult<Appender> {
        let cols = {
            let held = conn.bind(py).borrow();
            let held: &Connection = &held;
            py.detach(|| -> Result<Vec<Buffer>, Snag> {
                let mut engine = held.inner.lock().map_err(|_| Snag::Closed)?;
                let engine = engine.as_mut().ok_or(Snag::Closed)?;
                engine.appender(table).map(drop).map_err(Snag::Engine)?;
                shape(engine, table)
            })
            .map_err(|snag| snag.raise(py))?
        };
        Ok(Appender {
            conn,
            table: table.to_string(),
            state: Mutex::new(State {
                cols,
                buffered: 0,
                committed: 0,
                open: true,
            }),
        })
    }

    /// The buffers, for a call that only reads them.
    ///
    /// The lock is taken with the GIL held, which a flush never does.
    /// Waiting for it here waits for at most one conversion of one row,
    /// and a flush that is under way has already let go of it by the
    /// time it needs the GIL back.
    fn locked(&self, py: Python<'_>) -> PyResult<MutexGuard<'_, State>> {
        self.state.lock().map_err(|_| Snag::Locked.raise(py))
    }

    /// The buffers, for a call that adds to them, which a closed
    /// appender has no business doing.
    ///
    /// The connection is checked as well as the appender, because a row
    /// appended through a closed connection has nowhere to go and the
    /// buffer is the only thing that would take it. Left to the flush,
    /// the same call would be refused or not depending on whether the
    /// batch happened to fill, which is a rule nobody can hold in their
    /// head. It is refused here instead, at the call that made the
    /// mistake, whatever the buffer is holding.
    fn writable(&self, py: Python<'_>) -> PyResult<MutexGuard<'_, State>> {
        let state = self.locked(py)?;
        if !state.open {
            return Err(Snag::Finished.raise(py));
        }
        if self
            .conn
            .bind(py)
            .borrow()
            .inner
            .lock()
            .is_ok_and(|c| c.is_none())
        {
            return Err(Snag::Closed.raise(py));
        }
        Ok(state)
    }
}

impl State {
    /// One row into the buffers, or nothing at all.
    fn append(&mut self, table: &str, row: &Bound<'_, PyAny>) -> PyResult<()> {
        let width = self.cols.len();
        let mut at = 0;
        for value in row.try_iter()? {
            let value = match value {
                Ok(value) => value,
                Err(err) => return self.refuse(at, err),
            };
            if at == width {
                return self.refuse(
                    at,
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "this row carries more than the {width} values '{table}' takes: {}",
                        self.names()
                    )),
                );
            }
            if let Err(err) = self.cols[at].take(&value, table, at) {
                return self.refuse(at, err);
            }
            at += 1;
        }
        if at != width {
            return self.refuse(
                at,
                pyo3::exceptions::PyValueError::new_err(format!(
                    "this row carries {at} values and '{table}' takes {width}: {}",
                    self.names()
                )),
            );
        }
        self.buffered += 1;
        Ok(())
    }

    /// The columns of the table, named, for a message about a row that
    /// is the wrong shape. A caller who miscounted wants to see what
    /// the count was supposed to be made of.
    fn names(&self) -> String {
        self.cols
            .iter()
            .map(|column| column.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Takes back the values a refused row managed to write, so that a
    /// refused row is a row that never happened rather than half of one
    /// nobody can find. A ragged buffer would be refused by the ingest
    /// at the flush, a long way from the row that caused it.
    fn refuse(&mut self, written: usize, err: PyErr) -> PyResult<()> {
        for column in self.cols.iter_mut().take(written) {
            column.values.pop();
        }
        Err(err)
    }

    /// The write itself, with the GIL already down.
    fn write(&mut self, conn: &Connection, table: &str) -> Result<u64, Snag> {
        if self.buffered == 0 {
            return Ok(self.committed);
        }
        let rows = self.buffered;
        {
            let mut engine = conn.inner.lock().map_err(|_| Snag::Closed)?;
            let engine = engine.as_mut().ok_or(Snag::Closed)?;
            self.reachable(engine, rows)?;
            let mut appender = engine.appender(table).map_err(Snag::Engine)?;
            // One vector, refilled per row rather than allocated per
            // row, which over a million rows is one allocation rather
            // than a million. The fields borrow the buffers, which is
            // what keeps a string column to one copy on the way in and
            // one on the way out rather than three.
            let mut row: Vec<Field<'_>> = Vec::with_capacity(self.cols.len());
            for at in 0..rows as usize {
                row.clear();
                row.extend(self.cols.iter().map(|column| column.values.field(at)));
                appender
                    .append_row(&row[..])
                    .map_err(|err| Snag::Row(at as u64, err))?;
            }
            appender.close().map_err(Snag::Engine)?;
        }
        self.empty();
        self.committed += rows;
        Ok(self.committed)
    }

    /// Every edge joins two rows that are there, checked against the
    /// row counts as they stand at the flush.
    ///
    /// This is the flush's own check and not the engine's, because the
    /// engine's comes too late: an edge to a row that is not there is
    /// refused when the write is folded into the graph, which is after
    /// the write is durable, and the frame it leaves behind is refused
    /// again by every writer that opens the database afterwards. Caught
    /// here, the batch is refused and the file is untouched.
    ///
    /// The counts are read at the flush and not when the appender
    /// opened, because the rows a later edge names may be written by an
    /// earlier flush of another appender on the same connection, and an
    /// edge to a row that arrived in the meantime is a good edge.
    fn reachable(&self, engine: &mut zudb::Connection, rows: u64) -> Result<(), Snag> {
        if self.cols.iter().all(|column| column.ends.is_none()) {
            return Ok(());
        }
        let catalog = catalog(engine)?;
        for column in &self.cols {
            let Some(end) = column.ends else { continue };
            let Some(node) = catalog.node_by_id(end) else {
                continue;
            };
            for at in 0..rows as usize {
                let Field::Int(offset) = column.values.field(at) else {
                    continue;
                };
                if offset as u64 >= node.node_count {
                    return Err(Snag::Offset {
                        row: at as u64,
                        offset,
                        table: node.name.clone(),
                        rows: node.node_count,
                    });
                }
            }
        }
        Ok(())
    }

    fn empty(&mut self) {
        self.cols
            .iter_mut()
            .for_each(|column| column.values.clear());
        self.buffered = 0;
    }
}

impl Buffer {
    /// One value into this column, or the reason it does not go there.
    ///
    /// The column's own name is in the message, and its position too,
    /// because a row is written by position and read by name and a
    /// caller who has them the wrong way round needs both to see it.
    fn take(&mut self, value: &Bound<'_, PyAny>, table: &str, at: usize) -> PyResult<()> {
        if self.ends.is_some()
            && let Ok(offset) = value.extract::<i64>()
            && offset < 0
        {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "value {at} of this row is {offset}, and column '{}' of '{table}' holds row \
                 offsets, which count from zero",
                self.name
            )));
        }
        self.values.push(value).map_err(|why| match why {
            Mismatch::Wanted(holds) => pyo3::exceptions::PyTypeError::new_err(format!(
                "value {at} of this row is of type '{}' and column '{}' of '{table}' holds {holds}",
                type_name(value),
                self.name
            )),
            Mismatch::Python(err) => err,
        })
    }
}

/// The catalog as the file has it, which is not always the one the
/// session is holding.
///
/// A session reloads its catalog when a statement runs, and an appender
/// is not a statement: a flush commits its rows and folds them in
/// without the session hearing about it, so the row counts the session
/// remembers are the counts from the last statement. Loading it costs a
/// read of a block chain, once per appender opened and once per flush of
/// a rel table, which is nothing beside the commit either of them is
/// about to do.
fn catalog(engine: &mut zudb::Connection) -> Result<Catalog, Snag> {
    let file = engine.session_mut().file_mut().map_err(Snag::Engine)?;
    Catalog::load(file).map_err(Snag::Engine)
}

/// The columns of the table an appender was opened on, in the order it
/// declares them.
///
/// A node table's columns are the ones the property store holds, with
/// the types it holds them as, which is what the engine's appender
/// checks a row against. A rel table has no property columns: a row of
/// one is the two ends of an edge, as offsets into the tables it runs
/// between, so those are the two columns and they are named for what
/// they are.
///
/// Read here rather than left to the flush so that a value that does
/// not belong in a column is refused by the call that appended it. A
/// row is refused a million rows before the flush that would have
/// carried it, and the message names the column rather than guessing at
/// it from the values that came before.
fn shape(engine: &mut zudb::Connection, table: &str) -> Result<Vec<Buffer>, Snag> {
    let catalog = catalog(engine)?;
    if let Some(rel) = catalog.rel_by_name(table) {
        let ends = [rel.from, rel.to];
        let named = |id: u32, fallback: &str| {
            catalog
                .node_tables()
                .iter()
                .find(|node| node.id == id)
                .map_or_else(|| fallback.to_string(), |node| node.name.clone())
        };
        // Named for the tables the edge runs between, since that is
        // what a row of a rel table is and there is nothing else to
        // call the two columns.
        return Ok(vec![
            Buffer {
                name: format!("from {}", named(ends[0], "the source table")),
                values: Column::Int(Vec::new()),
                ends: Some(ends[0]),
            },
            Buffer {
                name: format!("to {}", named(ends[1], "the destination table")),
                values: Column::Int(Vec::new()),
                ends: Some(ends[1]),
            },
        ]);
    }
    let id = catalog
        .node_by_name(table)
        .map(|node| node.id)
        .ok_or_else(|| {
            Snag::Engine(ZuError::InvalidArgument(format!(
                "no node table or rel table '{table}'"
            )))
        })?;
    let file = engine.session_mut().file_mut().map_err(Snag::Engine)?;
    let directory = zudb::zu1::props::load_props(file, id)
        .map_err(Snag::Engine)?
        .ok_or_else(|| {
            Snag::Engine(ZuError::InvalidArgument(format!(
                "'{table}' stores no properties, so it has no columns to append to"
            )))
        })?;
    directory
        .columns
        .iter()
        .map(|column| {
            Ok(Buffer {
                name: column.name.clone(),
                values: Column::for_type(&column.ty).ok_or_else(|| {
                    Snag::Engine(ZuError::InvalidArgument(format!(
                        "column '{}' of '{table}' holds {}, which this engine cannot yet \
                         append to",
                        column.name, column.ty
                    )))
                })?,
                ends: None,
            })
        })
        .collect()
}
