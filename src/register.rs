//! Frames, registered under a name a statement can match on.
//!
//! ```python
//! conn.register("people", frame)
//! conn.execute("MATCH (p:people) WHERE p.age > 40 RETURN p.name AS name")
//! ```
//!
//! This is the replacement scan, and it copies nothing. What the engine
//! is told is where the caller's columns are, how wide their values are
//! and what they mean; a statement that matches the name builds vectors
//! that point straight at those buffers, so a frame of ten million rows
//! is registered in the time it takes to describe its columns and read
//! at the speed of the memory it already sits in.
//!
//! Because it is not a copy, a registered frame is a view and not a
//! snapshot: write into the array behind it and the next statement
//! answers what is there now. The one exception is a dictionary of
//! Python lists, which has to be read into buffers of this client's own
//! because a list holds objects rather than numbers.
//!
//! A frame belongs to the connection it was registered on and goes when
//! that connection does. It is not written to the database, no other
//! program opening the same file sees it, and nothing writes to it: a
//! statement that tries to insert into or delete from one is refused
//! with the reason. `unregister` takes the name away entirely, and the
//! bytes go back to the caller when the last statement reading them has
//! finished with them.

use pyo3::prelude::*;
use zudb::ZuError;
use zudb::zu1::catalog::Catalog;

use crate::conn::Connection;
use crate::error::{programming, to_py_err};
use crate::frame;

/// Registers a frame as a table called `name`.
pub fn register(
    py: Python<'_>,
    conn: &Connection,
    name: &str,
    data: &Bound<'_, PyAny>,
) -> PyResult<usize> {
    identifier(py, name, "a registered frame")?;
    let described = frame::read(py, data)?;
    if described.columns.is_empty() {
        return Err(programming(
            py,
            "this frame has no columns, and a table whose rows hold nothing is not a table",
        ));
    }
    for column in &described.columns {
        identifier(py, &column.name, "a column of a registered frame")?;
    }
    // Refused here rather than at the statement that would have hit it.
    // The engine keeps frames in an id space of their own and would
    // take this name happily; what it could not do is bind it, because
    // a label in a statement is one thing and the stored table would
    // win. Better said at the call that made the clash.
    if has_table(py, conn, name)? {
        return Err(programming(
            py,
            &format!(
                "'{name}' is already a table of this database, and registering over one would \
                 hide rows this frame knows nothing about"
            ),
        ));
    }

    let rows = described.rows as usize;
    // The description carries raw pointers, so it travels into the
    // detached call inside the type that says who keeps them alive. The
    // walk `Frame::new` does over the unsigned, scaled and string
    // columns happens in there, with the GIL down.
    conn.engine(py, move |engine| described.register(engine, name))?
        .map_err(|err| to_py_err(py, err))?;
    Ok(rows)
}

/// Takes a registered frame's name away and gives the bytes back.
///
/// The bytes go when the last statement reading them lets go, which is
/// usually now and is never before: a frame a running statement is
/// still scanning is held until it ends.
pub fn unregister(py: Python<'_>, conn: &Connection, name: &str) -> PyResult<()> {
    let dropped = conn
        .engine(py, |engine| engine.unregister(name))?
        .map_err(|err| to_py_err(py, err))?;
    if !dropped {
        return Err(programming(
            py,
            &format!(
                "nothing is registered here as '{name}', and a name this connection did not \
                 register is a table of the database rather than a frame of the caller's"
            ),
        ));
    }
    Ok(())
}

/// The names frames are registered under on this connection, sorted.
pub fn registered(py: Python<'_>, conn: &Connection) -> PyResult<Vec<String>> {
    conn.engine(py, |engine| Ok::<_, ZuError>(engine.registered()))?
        .map_err(|err| to_py_err(py, err))
}

/// Whether the database already holds a table of this name.
///
/// Node tables and rel tables both, because a name is a label in a
/// statement either way and registering over either of them would be
/// the same mistake.
fn has_table(py: Python<'_>, conn: &Connection, name: &str) -> PyResult<bool> {
    conn.engine(py, |engine| {
        let file = engine.session_mut().file_mut()?;
        let catalog = Catalog::load(file)?;
        Ok::<_, ZuError>(
            catalog.node_by_name(name).is_some() || catalog.rel_by_name(name).is_some(),
        )
    })?
    .map_err(|err| to_py_err(py, err))
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
