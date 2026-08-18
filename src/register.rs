//! Frames, registered under a name a statement can match on.
//!
//! ```python
//! conn.register("people", frame)
//! conn.execute("MATCH (p:people) WHERE p.age > 40 RETURN p.name AS name")
//! ```
//!
//! What this is not is a scan of the frame where it sits. DuckDB's
//! `register` is zero copy because its executor can call back out to
//! the Python object holding the data; this engine has no such callback
//! yet, so a registered frame is copied in and becomes a table like any
//! other. The call is the one it would be either way, and the day the
//! engine grows a scan it becomes the cheap thing under the same name.
//!
//! The copy is not the slow kind. A frame arrives as Arrow buffers and
//! goes in through the same appender a bulk load uses, which is one
//! commit for the batch and no Python object per cell, so what it costs
//! is a memcpy per column and the write.
//!
//! Registering owns the name. A name that is already a table of the
//! database is refused rather than overwritten, since a frame knows
//! nothing about the rows that were there, and a name this connection
//! registered before is emptied and rewritten, which is what rerunning
//! a cell means by it.
//!
//! What a name cannot do is change shape. A table's columns are the
//! ones its first row declared and no statement of this engine alters
//! or drops one, so a frame with different columns under a used name is
//! refused rather than half written, and `unregister` empties the table
//! rather than removing it. Both of those are the engine showing
//! through, and both are better said than worked around quietly.

use pyo3::prelude::*;
use zu_common::{DurationKind, Temporal};
use zudb::query::Value;
use zudb::zu1::catalog::Catalog;
use zudb::{Field, ZuError};

use crate::conn::Connection;
use crate::error::{programming, to_py_err};
use crate::frame::{self, Frame};

/// What can go wrong with the GIL down, where an exception cannot be
/// built yet.
enum Snag {
    Engine(ZuError),
    /// A row the engine's appender would not take, named by where it
    /// sits in the frame, since that is the row a caller can go and
    /// look at.
    Row(u64, ZuError),
    /// A column the table declares and the frame has not got, which is
    /// a table that was written to while it was being registered.
    Missing(String),
}

impl Snag {
    fn raise(self, py: Python<'_>, name: &str) -> PyErr {
        match self {
            Snag::Engine(err) => to_py_err(py, err),
            Snag::Row(row, err) => {
                let raised = to_py_err(py, err);
                programming(
                    py,
                    &format!("row {row} of this frame: {}", raised.value(py)),
                )
            }
            Snag::Missing(column) => programming(
                py,
                &format!(
                    "'{name}' declares a column '{column}' that this frame has not got, which is \
                     a table that was written to while it was being registered"
                ),
            ),
        }
    }
}

/// Copies a frame in as a node table called `name`.
pub fn register(
    py: Python<'_>,
    conn: Py<Connection>,
    name: &str,
    data: &Bound<'_, PyAny>,
) -> PyResult<usize> {
    identifier(py, name, "a registered frame")?;
    let frame = frame::read(py, data)?;
    if frame.columns.is_empty() {
        return Err(programming(
            py,
            "this frame has no columns, and a table whose rows hold nothing is not a table",
        ));
    }
    if frame.rows == 0 {
        return Err(programming(
            py,
            "this frame has no rows, and the columns of a table are declared by the first row \
             written into it",
        ));
    }
    for (column, _) in &frame.columns {
        identifier(py, column, "a column of a registered frame")?;
    }

    let held = conn.bind(py).borrow();
    let held: &Connection = &held;
    // Refused inside a transaction for the reason an appender is: the
    // rows after the first go in through the appender, whose batches are
    // commits of their own, so half the frame would survive a rollback
    // and half of it would not.
    if held.in_transaction(py)? {
        return Err(programming(
            py,
            "registering a frame writes its own commits, which a rollback does not take back, so \
             it cannot be done inside a transaction",
        ));
    }
    let mine = held.made_here(name);
    if !mine && has_table(py, held, name)? {
        return Err(programming(
            py,
            &format!(
                "'{name}' is already a table of this database, and registering over one would \
                 write across rows this frame knows nothing about"
            ),
        ));
    }
    // A name registered again is the same name holding something else,
    // which is what a caller who reruns a cell means by it. The rows go
    // first, so the frame that replaces them is the whole table rather
    // than the half of it that is new.
    //
    // The columns cannot change, though, and that is worth saying at the
    // call rather than at the row the engine refuses: a table's columns
    // are the ones the first row declared, no statement of this engine
    // alters or drops one, and the empty table left behind by an
    // `unregister` still has them.
    if mine {
        same_columns(py, held, name, &frame)?;
        held.run(py, &empty_out(name))?;
    }

    // The first row through a statement, because a table nothing
    // declares is declared by the row that is written into it and no
    // statement of this engine declares one on its own. Every row after
    // it goes through the appender, which is the fast way and needs the
    // table to be there.
    held.run(py, &first(py, name, &frame)?)?;
    if frame.rows > 1 {
        rest(py, held, name, &frame)?;
    }
    held.remember(name);
    Ok(frame.rows)
}

/// Empties the table a registered frame was copied into and forgets the
/// name.
pub fn unregister(py: Python<'_>, conn: &Connection, name: &str) -> PyResult<()> {
    if !conn.registered_here(name) {
        return Err(programming(
            py,
            &format!(
                "nothing is registered here as '{name}', and a name this connection did not \
                 register is a table of the database rather than a frame of the caller's"
            ),
        ));
    }
    conn.run(py, &empty_out(name))?;
    conn.forget(name);
    Ok(())
}

/// The statement that takes the rows of a registered frame back out.
///
/// `DETACH`, because a registered table's rows may have been joined to
/// since, and a delete that refused would leave the name registered
/// over rows nobody can replace.
fn empty_out(name: &str) -> String {
    format!("MATCH (frame:{name}) DETACH DELETE frame")
}

/// The `INSERT` that writes the first row and, in writing it, declares
/// the table.
///
/// The values go in as literals rather than as parameters, because a
/// column is declared by what is written into it and a parameter is
/// worked out rather than written: the engine refuses `{uid: $uid}` on
/// a table that does not exist yet, and it is right to, since the
/// statement alone does not say what the column would hold.
fn first(py: Python<'_>, name: &str, frame: &Frame) -> PyResult<String> {
    let mut fields = Vec::with_capacity(frame.columns.len());
    for (column, values) in &frame.columns {
        let value = values
            .value(0)
            .and_then(|value| literal(&value))
            .ok_or_else(|| {
                programming(
                    py,
                    &format!(
                        "column '{column}' holds values no statement can write down, and the \
                         columns of a table are declared by the first row written into it"
                    ),
                )
            })?;
        fields.push(format!("{column}: {value}"));
    }
    Ok(format!("INSERT (frame:{name} {{{}}})", fields.join(", ")))
}

/// One value as a statement writes it, or `None` when no statement
/// writes one at all.
///
/// The two that come back `None` are byte strings, which have no
/// literal, and year-month durations, whose only spelling is the one a
/// day-time duration already has.
fn literal(value: &Value) -> Option<String> {
    Some(match value {
        Value::Int(n) => n.to_string(),
        // Debug rather than Display, because Display writes `1` for a
        // float that holds one and the column would come out an
        // integer. Non-finite is refused for want of a literal: no
        // spelling of infinity parses, and a column cannot be declared
        // by a value that cannot be written.
        Value::Float(f) if f.is_finite() => format!("{f:?}"),
        Value::Bool(b) => b.to_string(),
        Value::Str(s) => quoted(s),
        Value::Temporal(t) => {
            let kind = match t {
                Temporal::Date(_) => "DATE",
                Temporal::LocalTime(_) => "LOCAL TIME",
                Temporal::LocalDatetime(_) => "LOCAL DATETIME",
                Temporal::Duration(DurationKind::DayTime, _) => "DURATION",
                _ => return None,
            };
            format!("{kind} {}", quoted(&t.to_string()))
        }
        _ => return None,
    })
}

/// A string as a statement carries it, with the five characters that
/// end a literal or start an escape written as escapes.
fn quoted(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('\'');
    for ch in text.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\'' => out.push_str("\\'"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(ch),
        }
    }
    out.push('\'');
    out
}

/// Every row after the first, through the engine's appender.
///
/// The columns are taken in the order the table declares them rather
/// than the order the frame holds them, because the appender writes a
/// row as a list and the two orders are only the same by luck.
fn rest(py: Python<'_>, conn: &Connection, name: &str, frame: &Frame) -> PyResult<()> {
    conn.engine(py, |engine| {
        let order = column_order(engine, name)?;
        // The frame's columns in the table's order, worked out once
        // rather than looked up per value, since the whole point of the
        // appender is that a value costs nothing but a copy.
        let mut taken: Vec<usize> = Vec::with_capacity(order.len());
        for column in &order {
            let at = frame
                .columns
                .iter()
                .position(|(held, _)| held == column)
                .ok_or_else(|| Snag::Missing(column.clone()))?;
            taken.push(at);
        }
        let mut appender = engine.appender(name).map_err(Snag::Engine)?;
        let mut row: Vec<Field<'_>> = Vec::with_capacity(taken.len());
        for at in 1..frame.rows {
            row.clear();
            row.extend(
                taken
                    .iter()
                    .map(|&column| frame.columns[column].1.field(at)),
            );
            appender
                .append_row(&row[..])
                .map_err(|err| Snag::Row(at as u64, err))?;
        }
        appender.close().map(|_| ()).map_err(Snag::Engine)
    })?
    .map_err(|snag| snag.raise(py, name))
}

/// That a frame going in under a name this connection already used has
/// the columns that name was declared with.
///
/// Names rather than types, because the engine refuses a value that does
/// not fit a column at the row that carries it and says so well, where a
/// column that is simply not there reads as a mistake about the data
/// rather than about the name.
fn same_columns(py: Python<'_>, conn: &Connection, name: &str, frame: &Frame) -> PyResult<()> {
    let mut declared = conn
        .engine(py, |engine| column_order(engine, name))?
        .map_err(|snag| snag.raise(py, name))?;
    let mut holds: Vec<String> = frame
        .columns
        .iter()
        .map(|(column, _)| column.clone())
        .collect();
    declared.sort();
    holds.sort();
    if declared != holds {
        return Err(programming(
            py,
            &format!(
                "'{name}' was registered with the columns {}, and this frame has {}, which no \
                 statement of this engine can turn one into the other: register it under another \
                 name",
                declared.join(", "),
                holds.join(", ")
            ),
        ));
    }
    Ok(())
}

/// Whether the database already holds a table of this name.
///
/// Node tables and rel tables both, because a name is a label in a
/// statement either way and registering over either of them would be
/// the same mistake.
fn has_table(py: Python<'_>, conn: &Connection, name: &str) -> PyResult<bool> {
    conn.engine(py, |engine| {
        let catalog = catalog(engine)?;
        Ok(catalog.node_by_name(name).is_some() || catalog.rel_by_name(name).is_some())
    })?
    .map_err(|snag: Snag| snag.raise(py, name))
}

/// The columns of a node table, in the order it declares them, which is
/// the order the appender writes a row in.
fn column_order(engine: &mut zudb::Connection, table: &str) -> Result<Vec<String>, Snag> {
    let id = catalog(engine)?
        .node_by_name(table)
        .map(|node| node.id)
        .ok_or_else(|| {
            Snag::Engine(ZuError::InvalidArgument(format!(
                "no node table '{table}', which is a table that went away while it was being \
                 registered"
            )))
        })?;
    let file = engine.session_mut().file_mut().map_err(Snag::Engine)?;
    let directory = zudb::zu1::props::load_props(file, id)
        .map_err(Snag::Engine)?
        .ok_or_else(|| {
            Snag::Engine(ZuError::InvalidArgument(format!(
                "'{table}' stores no properties, so it has no columns to write into"
            )))
        })?;
    Ok(directory
        .columns
        .iter()
        .map(|column| column.name.clone())
        .collect())
}

/// The catalog as the file has it, which is not always the one the
/// session is holding: an appender's commit folds rows in without the
/// session hearing about it.
fn catalog(engine: &mut zudb::Connection) -> Result<Catalog, Snag> {
    let file = engine.session_mut().file_mut().map_err(Snag::Engine)?;
    Catalog::load(file).map_err(Snag::Engine)
}

/// Refuses a name a statement could not carry.
///
/// A table name goes into a statement as itself, so a name that is not
/// an identifier is a name that would either fail to parse or parse as
/// something else, and the second of those is the one worth refusing
/// for.
fn identifier(py: Python<'_>, name: &str, what: &str) -> PyResult<()> {
    let mut chars = name.chars();
    let starts = chars
        .next()
        .is_some_and(|first| first.is_ascii_alphabetic() || first == '_');
    if !starts || !chars.all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return Err(programming(
            py,
            &format!(
                "'{name}' is not a name a statement can carry, and {what} is named by a letter or \
                 an underscore followed by letters, digits or underscores"
            ),
        ));
    }
    Ok(())
}
