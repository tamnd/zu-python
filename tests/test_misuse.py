"""Deliberately wrong programs, and what each of them is told.

DX2 asks for a misuse suite in both clients: no crash, no leak, and a
clear error for every program that is wrong on purpose. Clear is the
hard word of the three, so it is spelled out here as three things a
message has to do. It names the thing the caller named, being the file
they opened, the column they appended to, the parameter they passed.
It says what was expected instead wherever there is something to say.
And it is the engine's own sentence rather than a syscall's, because
"failed to fill whole buffer" is a true statement about a read that
tells nobody which file was not a database.

Clear also means the right class, since a Python caller catches classes
and reads messages. A mistake the program made is a `ProgrammingError`,
a value Python has and zu does not is a `TypeError`, a value of the
right type and the wrong shape is a `ValueError`, and a file that is
not a database is a `ConnectionError`, the same as a file that is not
there: both are a path that does not lead to a database, and telling a
caller who mistyped one to file a bug would be the wrong answer twice.

No crash is the suite running at all. No leak is checked from outside
the call that would cause one, twice: every case is followed by a read
on the connection it was aimed at, and the failing connects are
repeated five hundred times, which is past the descriptor limit a
process starts with.

The last two tests are the half of a misuse suite that is usually
missing. The programs that look wrong and are not, each of which is a
decision somebody would otherwise reverse by accident, and the one
mistake that cannot be reported where it happens: an appender the
collector took while it still held rows, which warns instead.
"""

from __future__ import annotations

import gc
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest
import zudb

READ = "MATCH (p:person) RETURN p.uid AS uid"


@dataclass(frozen=True)
class Misuse:
    """One deliberately wrong program.

    `run` gets a connection to a database with three people in it and a
    directory to make a mess in. Returning normally fails the test:
    every program in the table is wrong.
    """

    what: str
    run: Callable[[zudb.Connection, Path], object]
    raises: type[BaseException]
    says: tuple[str, ...]


def junk(where: Path, name: str, contents: bytes) -> Path:
    """A file that is not a database, written where a caller would have
    a file that is not a database."""
    path = where / name
    path.write_bytes(contents)
    return path


def read_only(where: Path) -> zudb.Connection:
    """A database of its own, written, closed and opened read-only.

    Written and closed first, because a read-only open reads the base
    file as the last fold left it and a database still being written
    has a fold outstanding.
    """
    path = where / "reader.zu1"
    if not path.exists():
        with zudb.connect(path) as conn:
            conn.execute("INSERT (p:person {uid: 1, name: 'ada'})")
    return zudb.connect(path, read_only=True)


MISUSES: tuple[Misuse, ...] = (
    Misuse(
        "connects to a file too small to be a database",
        lambda conn, tmp: zudb.connect(junk(tmp, "small.zu1", b"not a database at all")),
        zudb.ConnectionError,
        ("small.zu1", "21 bytes", "too short to be a zu1 database"),
    ),
    Misuse(
        "connects to a file the right size and the wrong kind",
        lambda conn, tmp: zudb.connect(junk(tmp, "big.zu1", b"x" * 40960)),
        zudb.ConnectionError,
        ("big.zu1", "not a zu1 file"),
    ),
    Misuse(
        "connects read-only to a database that is not there",
        # Read-only, because `connect` on a path with nothing at it
        # makes a database and that is the documented answer. Asking
        # for read-only is asking for one that already exists.
        lambda conn, tmp: zudb.connect(tmp / "nowhere.zu1", read_only=True),
        zudb.ConnectionError,
        ("nowhere.zu1",),
    ),
    Misuse(
        "writes through a connection it opened read-only",
        lambda conn, tmp: read_only(tmp).execute("INSERT (p:person {uid: 4, name: 'zoe'})"),
        zudb.ProgrammingError,
        ("reader.zu1", "read-only"),
    ),
    Misuse(
        "opens an appender on a connection it opened read-only",
        lambda conn, tmp: read_only(tmp).appender("person"),
        zudb.ProgrammingError,
        ("appender writes", "read-only"),
    ),
    Misuse(
        "runs text that will not parse",
        lambda conn, tmp: conn.execute("MATCH (p:person) RETRUN p.uid"),
        zudb.SyntaxError,
        ("42001",),
    ),
    Misuse(
        "leaves out a parameter the statement reads",
        lambda conn, tmp: conn.execute("MATCH (p:person) WHERE p.uid = $uid RETURN p.uid AS uid"),
        zudb.SyntaxError,
        ("42002", "missing parameter $uid"),
    ),
    Misuse(
        "passes a parameter of a type zu has no value for",
        lambda conn, tmp: conn.execute("RETURN $x AS x", {"x": object()}),
        TypeError,
        ("a parameter cannot be a object", "zu holds"),
    ),
    Misuse(
        "passes the parameters as a list",
        lambda conn, tmp: conn.execute("RETURN $x AS x", [1, 2]),
        TypeError,
        ("dict",),
    ),
    Misuse(
        "passes a statement that is not a string",
        lambda conn, tmp: conn.execute(42),
        TypeError,
        ("str",),
    ),
    Misuse(
        "opens an appender on a table that is not there",
        lambda conn, tmp: conn.appender("nope"),
        zudb.ProgrammingError,
        ("no node table or rel table 'nope'",),
    ),
    Misuse(
        "appends a row shorter than the table",
        lambda conn, tmp: conn.appender("person").append_row([4]),
        ValueError,
        ("this row carries 1 values and 'person' takes 3", "uid, name, score"),
    ),
    Misuse(
        "appends a row with a value the column does not take",
        lambda conn, tmp: conn.appender("person").append_row(["four", "zoe", 1.0]),
        TypeError,
        ("value 0 of this row", "column 'uid' of 'person' holds integers"),
    ),
    Misuse(
        "appends a row that is not a row",
        lambda conn, tmp: conn.appender("person").append_row(4),
        TypeError,
        ("not iterable",),
    ),
    Misuse(
        "appends to an appender it closed",
        lambda conn, tmp: closed_appender(conn).append_row([4, "zoe", 1.0]),
        zudb.ProgrammingError,
        ("this appender is closed",),
    ),
    Misuse(
        "runs a statement on a connection it closed",
        lambda conn, tmp: closed_connection(tmp).execute(READ),
        zudb.ProgrammingError,
        ("this connection is closed",),
    ),
    Misuse(
        "flushes an appender whose connection is closed",
        lambda conn, tmp: orphaned_appender(tmp).flush(),
        zudb.ProgrammingError,
        ("the connection this appender writes through is closed",),
    ),
    Misuse(
        "loads columns of different lengths",
        lambda conn, tmp: zudb.load(
            tmp / "ragged.zu1",
            nodes="person",
            columns={"uid": [1, 2], "name": ["ada"]},
        ),
        ValueError,
        ("column 'name' holds 1 values and column 'uid' holds 2",),
    ),
    Misuse(
        "loads a column that holds two types",
        lambda conn, tmp: zudb.load(
            tmp / "mixed.zu1",
            nodes="person",
            columns={"uid": [1, "two"]},
        ),
        TypeError,
        ("column 'uid' holds integers and row 1 is of type 'str'",),
    ),
    Misuse(
        "loads an edge to a row that is not there",
        lambda conn, tmp: zudb.load(
            tmp / "dangling.zu1",
            nodes="person",
            rels="knows",
            columns={"uid": [1, 2]},
            edges=[(0, 9)],
        ),
        ValueError,
        ("edge 0 joins row 9 of a table with 2 rows in it",),
    ),
    Misuse(
        "loads an edge that is not a pair of rows",
        lambda conn, tmp: zudb.load(
            tmp / "single.zu1",
            nodes="person",
            rels="knows",
            columns={"uid": [1, 2]},
            edges=[(0,)],
        ),
        TypeError,
        ("edge 0 is not a pair of row numbers",),
    ),
    Misuse(
        "loads over a database that is already there",
        lambda conn, tmp: zudb.load(conn.path, nodes="person", columns={"uid": [1]}),
        zudb.ConnectionError,
        ("social.zu1",),
    ),
    Misuse(
        "loads nothing at all",
        lambda conn, tmp: zudb.load(tmp / "nothing.zu1", nodes="person"),
        ValueError,
        ("a load with no columns has no rows to count",),
    ),
)


def closed_appender(conn: zudb.Connection) -> zudb.Appender:
    """An appender that has been closed, which is the state a `with`
    block leaves one in."""
    rows = conn.appender("person")
    rows.close()
    return rows


def closed_connection(where: Path) -> zudb.Connection:
    """A connection of its own, closed. Its own, because the case that
    uses it would otherwise close the connection every other case is
    checked against afterwards."""
    conn = zudb.connect(where / "closed.zu1")
    conn.close()
    return conn


def orphaned_appender(where: Path) -> zudb.Appender:
    """An appender holding a row, whose connection was closed under it."""
    conn = zudb.connect(where / "orphan.zu1")
    conn.execute("INSERT (p:person {uid: 1, name: 'ada'})")
    rows = conn.appender("person")
    rows.append_row([2, "grace"])
    conn.close()
    return rows


@pytest.mark.parametrize("case", [pytest.param(case, id=case.what) for case in MISUSES])
def test_a_wrong_program_is_told_what_is_wrong(
    case: Misuse, social: zudb.Connection, tmp_path: Path
) -> None:
    with pytest.raises(case.raises) as raised:
        case.run(social, tmp_path)
    message = str(raised.value)
    for phrase in case.says:
        assert phrase in message, f"the message is missing '{phrase}': {message}"
    # The engine's sentence, not the read that noticed.
    assert "failed to fill whole buffer" not in message
    # And the connection it was aimed at is still a connection: the
    # failure took nothing with it.
    assert len(social.execute(READ).fetchall()) == 3


def test_every_condition_is_catchable_as_the_class_a_caller_writes() -> None:
    """The classes the table above uses, as a caller sees them.

    A caller writes `except zudb.Error` around a database call and
    `except (TypeError, ValueError)` around the values they built, so
    which of the two a mistake lands in is part of the contract rather
    than an implementation detail.
    """
    for case in MISUSES:
        if issubclass(case.raises, zudb.Error):
            continue
        assert case.raises in (TypeError, ValueError), case.what
        assert not issubclass(case.raises, zudb.Error), case.what


def test_five_hundred_failed_connections_leave_nothing_open(tmp_path: Path) -> None:
    """A failure that keeps the file open is a failure a program can
    only make a few hundred times, and the few hundredth is where it is
    found, in production, as a limit nobody thought was near."""
    missing = tmp_path / "nowhere.zu1"
    small = junk(tmp_path, "small.zu1", b"not a database at all")

    for _ in range(500):
        with pytest.raises(zudb.ConnectionError):
            zudb.connect(missing, read_only=True)
        with pytest.raises(zudb.ConnectionError):
            zudb.connect(small)

    # A descriptor per failure would have run out long ago, and a
    # database made now is readable and writable.
    with zudb.connect(tmp_path / "after.zu1") as conn:
        conn.execute("INSERT (p:person {uid: 1, name: 'ada'})")
        assert conn.execute(READ).fetchall() == [(1,)]

    # And nothing is holding a connection open behind the collector's
    # back, which is the other way a client leaks: the objects are gone
    # and the files they held went with them.
    gc.collect()
    alive = [obj for obj in gc.get_objects() if isinstance(obj, zudb.Connection)]
    assert alive == []


def test_a_statement_that_failed_wrote_nothing_and_left_the_connection_alone(
    social: zudb.Connection,
) -> None:
    """A statement that fails partway is where "no crash" is not
    enough: the connection has to be where it was, and so does the
    data."""
    with pytest.raises(zudb.DataError):
        social.execute("INSERT (p:person {uid: 1 / 0, name: 'zoe', score: 1.0})")
    assert len(social.execute(READ).fetchall()) == 3

    # The same for an appender: a value the column does not take is
    # refused by the call that appended it, and the rows already
    # buffered are still there to flush.
    with social.appender("person") as rows:
        rows.append_row([40, "zoe", 1.0])
        with pytest.raises(TypeError):
            rows.append_row(["fifty", "yvonne", 2.0])
    assert len(social.execute(READ).fetchall()) == 4


def test_the_programs_that_look_like_misuse_and_are_not(
    social: zudb.Connection, tmp_path: Path
) -> None:
    """Each of these is a decision, and a decision nobody wrote down is
    a decision somebody reverses by accident."""
    # A parameter the statement does not read is not an error. A caller
    # that passes one dict to several statements is doing something
    # reasonable, and refusing it would make the dict the union of what
    # every statement wants.
    assert len(social.execute(READ, {"unread": 1}).fetchall()) == 3

    # A label nothing carries matches nothing. A pattern with no answer
    # is the ordinary answer to a question about a graph, and the other
    # reading gives a query that fails on the day the last row of a
    # label is deleted.
    assert social.execute("MATCH (p:nobody) RETURN p.uid AS uid").fetchall() == []

    # A value of the wrong type against a column is a comparison that
    # is false rather than a failure, which is what GQL says about
    # comparing across types.
    assert (
        social.execute(
            f"{READ.rsplit(' RETURN', 1)[0]} WHERE p.uid = $uid RETURN p.uid AS uid", {"uid": "ada"}
        ).fetchall()
        == []
    )

    # A result is rows that were already read, so reading it twice
    # gives the same rows twice rather than an empty second answer,
    # which is the mistake every cursor API teaches people to expect.
    result = social.execute(READ)
    assert result.fetchall() == result.fetchall()
    assert len(list(result)) == 3

    # Closing twice is not an error. The second close has nothing to
    # do, and a caller cleaning up in a loop is allowed to ask.
    conn = zudb.connect(tmp_path / "twice.zu1")
    conn.close()
    conn.close()

    # A stop with nothing running is dropped rather than kept. This
    # client clears the flag before every statement, because the thread
    # that asked cannot know whether the statement it meant to stop had
    # already finished, and a `Ctrl-C` pressed a moment too late would
    # otherwise end the next thing the notebook runs.
    social.interrupt()
    assert len(social.execute(READ).fetchall()) == 3


def test_an_appender_the_collector_took_with_rows_in_it_says_so(
    social: zudb.Connection,
) -> None:
    """The one mistake that cannot be reported where it happens.

    Rows buffered in an appender nobody closed are discarded when the
    collector takes it, and going quietly is the failure mode worth a
    warning: a loop that appended a million rows, no `close`, a
    database with nothing in it and no complaint about it. Flushing
    from a destructor is the other answer and is not available, since a
    collector runs whenever it likes, including while another thread is
    inside a statement on the same connection.
    """
    rows = social.appender("person")
    rows.append_row([40, "zoe", 1.0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        del rows
        gc.collect()
    assert len(caught) == 1, [str(warning.message) for warning in caught]
    assert caught[0].category is ResourceWarning
    message = str(caught[0].message)
    assert "appender on 'person'" in message
    assert "1 row buffered" in message
    assert "`with` block" in message
    # The rows went nowhere, which is what the warning is about.
    assert len(social.execute(READ).fetchall()) == 3

    # An appender with nothing in it is not a mistake and says nothing:
    # a warning nobody can act on is a warning people learn to filter.
    empty = social.appender("person")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        del empty
        gc.collect()
    assert caught == []
