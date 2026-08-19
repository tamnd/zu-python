"""zu through the interface every Python database has.

    import zudb.dbapi

    conn = zudb.dbapi.connect("social.zu1")
    cur = conn.cursor()
    cur.execute("MATCH (p:person) WHERE p.score > ? RETURN p.name AS name", (40,))
    print(cur.fetchall())
    conn.commit()

PEP 249 is what a Python program expects a database to look like:
`connect`, a connection with `commit` and `rollback`, a cursor with
`execute` and `fetchone`, and a fixed list of exception classes. It is
worth having even where the native client is nicer, because the code
that reads it is not always code anyone can change: a dashboard, a test
harness, a notebook helper someone wrote against sqlite3 five years ago.

This is a layer and not a second client. A statement goes to the same
`execute` underneath, the rows are the same rows, and the exception a
statement raises is the exception it would have raised, only carrying a
PEP 249 class as well. Nothing here reimplements anything.

Two things about it are worth reading before writing against it.

Parameters are `?`, which PEP 249 calls `qmark`, and they are rewritten
into the engine's own `$name` before the statement is run. GQL cannot
use the `named` style: `:name` is how a node pattern names its label,
so `(p:person)` and `WHERE p.uid = :uid` cannot be told apart without
parsing the statement. `?` is a character GQL has no meaning for at
all, which makes it unambiguous. Passing a dict instead of a sequence
hands the statement over untouched, so `$name` still works for anyone
writing zu statements rather than generating them.

Transactions are implicit, which PEP 249 requires and the native client
does not do: a transaction opens before the first statement after each
`commit` or `rollback`, and closing a connection rolls back what was
not committed. `connect(..., autocommit=True)` turns that off and gives
back the native behaviour, where every statement stands alone.
"""

from __future__ import annotations

import contextlib
import datetime
import os
import time
from collections import deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any, NoReturn

import zudb
from zudb.types import Value

__all__ = [
    "apilevel",
    "threadsafety",
    "paramstyle",
    "connect",
    "Connection",
    "Cursor",
    "Warning",
    "Error",
    "InterfaceError",
    "DatabaseError",
    "DataError",
    "OperationalError",
    "IntegrityError",
    "InternalError",
    "ProgrammingError",
    "NotSupportedError",
    "ConnectionError",
    "TransactionError",
    "Interrupted",
    "SyntaxError",
    "STRING",
    "BINARY",
    "NUMBER",
    "DATETIME",
    "ROWID",
    "Date",
    "Time",
    "Timestamp",
    "DateFromTicks",
    "TimeFromTicks",
    "TimestampFromTicks",
    "Binary",
]

#: The version of the interface this module implements. There has never
#: been another one.
apilevel = "2.0"

#: 2 means threads may share the module and the connections made from
#: it, but not the cursors. A connection is one lock and statements on
#: it queue behind it, which is what makes sharing safe; a cursor holds
#: a position in a result and two threads taking rows from one would
#: each get half of them.
threadsafety = 2

#: `?`, positional. See the note at the top for why not `named`.
paramstyle = "qmark"

#: Rows read ahead of the caller to work out what type each column
#: holds, since a statement's result does not declare one. A hundred is
#: enough to type a column that starts with a null and few enough that
#: a result of ten million rows is not walked to answer a question
#: about its shape.
AHEAD = 100


class Warning(Exception):  # noqa: A001 - PEP 249 names it, builtin or not
    """PEP 249 requires the name. Nothing raises it.

    A condition that does not stop a statement arrives on the result as
    a notice rather than through Python's warning machinery, so
    `cur.result.notices` is where they are and there is nothing here to
    catch.
    """


#: PEP 249's root class, which is zu's own root class.
#:
#: The two hierarchies meet rather than sitting side by side: everything
#: the engine raises is already a `zudb.Error`, so making that the
#: `Error` PEP 249 asks for means one `except` catches the same set
#: whichever spelling the caller reached for.
Error = zudb.Error


class InterfaceError(zudb.ProgrammingError, Error):
    """A mistake in the use of this layer rather than in the database.

    A cursor used after it was closed, a connection used after it was
    closed. Nothing reached the engine, which is why it is also a
    `zudb.ProgrammingError`: that is the class the native client raises
    for the same mistakes.
    """


class DatabaseError(Error):
    """Everything the database itself reports."""


class DataError(DatabaseError, zudb.DataError):
    """Class 22: a value was wrong. Division by zero, a bad cast, a number that did not fit."""


class OperationalError(DatabaseError):
    """Something outside the statement went wrong.

    The database could not be opened, the transaction could not go on,
    the statement was interrupted. The three classes below are the ones
    that actually arrive, each of them an `OperationalError` and the
    native class it would have been.
    """


class IntegrityError(DatabaseError):
    """PEP 249 requires the name. Nothing raises it yet.

    It is what a constraint violation would be, and zu has no
    constraints to violate: no primary key, no foreign key, no check.
    When it has them, they land here.
    """


class InternalError(DatabaseError, zudb.InternalError):
    """A failure the engine could not describe as a condition.

    A corrupt file, an assumption that did not hold. Worth reporting at
    https://github.com/tamnd/zu/issues.
    """


class ProgrammingError(DatabaseError, zudb.ProgrammingError):
    """The caller made a mistake the engine or the client caught.

    A parameter of a type zu has no place for, a fetch before an
    execute, a statement given a different number of parameters than it
    has placeholders.
    """


class NotSupportedError(DatabaseError):
    """PEP 249 requires the name. Nothing raises it.

    The optional methods this layer does not implement, `callproc` and
    `nextset`, are left out entirely rather than defined and refused,
    which is what PEP 249 asks for: a caller can ask whether a method
    is there and cannot ask whether it works.
    """


class ConnectionError(OperationalError, zudb.ConnectionError):  # noqa: A001 - zu's name for class 08
    """Class 08: the database could not be reached or could not be read."""


class TransactionError(OperationalError, zudb.TransactionError):
    """Classes 25, 2D and 40: the transaction is what went wrong.

    Check `retryable` before running it again. A rollback with nothing
    done can be retried; a statement whose completion is unknown cannot.
    """


class Interrupted(OperationalError, zudb.Interrupted):
    """The statement was asked to stop and did.

    An `OperationalError` because that is where a caller written
    against PEP 249 looks for a statement that did not finish for a
    reason outside itself.
    """


class SyntaxError(ProgrammingError, zudb.SyntaxError):  # noqa: A001 - the standard's name for class 42
    """Class 42: the statement could not be parsed, or named something that is not there."""


#: Which PEP 249 class each native one arrives as. A class missing from
#: here is raised untouched, which is still a `dbapi.Error`, because
#: losing the class the engine chose would be worse than not having a
#: PEP 249 name for it.
_CLASSES: dict[type[zudb.Error], type[zudb.Error]] = {
    zudb.ConnectionError: ConnectionError,
    zudb.DataError: DataError,
    zudb.TransactionError: TransactionError,
    zudb.SyntaxError: SyntaxError,
    zudb.ProgrammingError: ProgrammingError,
    zudb.InternalError: InternalError,
    zudb.Interrupted: Interrupted,
}


def _reraise(failure: zudb.Error) -> NoReturn:
    """Raises the same failure again as its PEP 249 class.

    Every field goes across, so nothing is lost by passing through
    here: the code, the position, the excerpt and the documentation
    link are the ones the engine wrote. The original traceback is kept
    and the chain is not, because an exception that is an instance of
    the class it came from does not need to be printed twice.
    """
    kind = _CLASSES.get(type(failure))
    if kind is None:
        raise failure
    raise kind(
        str(failure),
        code=failure.code,
        condition=failure.condition,
        severity=failure.severity,
        line=failure.line,
        column=failure.column,
        offset=failure.offset,
        excerpt=failure.excerpt,
        doc_url=failure.doc_url,
        retryable=failure.retryable,
    ).with_traceback(failure.__traceback__) from None


@contextlib.contextmanager
def _translating() -> Iterator[None]:
    """Around every call into the native client."""
    try:
        yield
    except zudb.Error as failure:
        _reraise(failure)


class _Type:
    """One of PEP 249's type objects, which is a set of Python types.

    A result does not declare what its columns hold, so the type code
    in `Cursor.description` is the Python type of the values in it,
    read from the first rows. These objects compare equal to the codes
    they cover, so `cur.description[0][1] == dbapi.STRING` answers what
    it looks like it answers and `is str` works too.
    """

    __slots__ = ("_kinds", "_name")

    def __init__(self, name: str, kinds: tuple[type, ...]) -> None:
        self._name = name
        self._kinds = frozenset(kinds)

    def __eq__(self, other: object) -> Any:
        if isinstance(other, _Type):
            return other._name == self._name
        if isinstance(other, type):
            return other in self._kinds
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._name)

    def __repr__(self) -> str:
        return f"<zudb.dbapi.{self._name}>"


STRING = _Type("STRING", (str,))
BINARY = _Type("BINARY", (bytes, bytearray, memoryview))
NUMBER = _Type("NUMBER", (int, float))
DATETIME = _Type("DATETIME", (datetime.date, datetime.time, datetime.datetime))
#: The values that identify a row, which in a graph are the ones that
#: carry a table and an offset in it.
ROWID = _Type("ROWID", (zudb.Node, zudb.Rel))

# PEP 249's constructors. Python's own types are the ones zu takes and
# gives back, so these are the types themselves rather than wrappers
# around them, and a program that builds a date either way builds the
# same date.
Date = datetime.date
Time = datetime.time
Timestamp = datetime.datetime
Binary = bytes


def DateFromTicks(ticks: float) -> datetime.date:  # noqa: N802 - PEP 249 names it
    """The date at a Unix timestamp, in local time, as PEP 249 defines it."""
    return Date(*time.localtime(ticks)[:3])


def TimeFromTicks(ticks: float) -> datetime.time:  # noqa: N802 - PEP 249 names it
    """The time of day at a Unix timestamp, in local time."""
    return Time(*time.localtime(ticks)[3:6])


def TimestampFromTicks(ticks: float) -> datetime.datetime:  # noqa: N802 - PEP 249 names it
    """The moment at a Unix timestamp, in local time."""
    return Timestamp(*time.localtime(ticks)[:6])


def _closing(statement: str, opened: int, *, escapes: bool, doubled: bool) -> int:
    """Where the thing quoted at `opened` ends, one past its closer.

    Three kinds of quoting and three rules, which are the lexer's:
    `'a\\'b'` escapes with a backslash, `@'a''b'` has no escapes and
    doubles the quote instead, and a backtick-quoted name ends at the
    next backtick and has neither. An unterminated one runs to the end
    of the text, because the statement is about to be handed to the
    engine and the engine says where the string started.
    """
    quote = statement[opened]
    at = opened + 1
    while at < len(statement):
        letter = statement[at]
        if escapes and letter == "\\":
            at += 2
            continue
        if letter == quote:
            if doubled and statement[at + 1 : at + 2] == quote:
                at += 2
                continue
            return at + 1
        at += 1
    return len(statement)


def _placeholders(statement: str) -> list[int]:
    """Where the `?` markers are, and only those.

    A `?` inside a string, a quoted name or a comment is text somebody
    wrote, not a parameter, so this walks the statement the way the
    lexer does rather than counting characters. It is the whole of the
    parsing this layer does: everything else about the statement is the
    engine's business.
    """
    found: list[int] = []
    at = 0
    while at < len(statement):
        letter = statement[at]
        if letter == "?":
            found.append(at)
            at += 1
        elif letter in "'\"":
            at = _closing(statement, at, escapes=True, doubled=False)
        elif letter == "`":
            at = _closing(statement, at, escapes=False, doubled=False)
        elif letter == "@" and statement[at + 1 : at + 2] in ("'", '"'):
            at = _closing(statement, at + 1, escapes=False, doubled=True)
        elif statement[at : at + 2] == "//":
            end = statement.find("\n", at)
            at = len(statement) if end < 0 else end + 1
        elif statement[at : at + 2] == "/*":
            end = statement.find("*/", at + 2)
            at = len(statement) if end < 0 else end + 2
        else:
            at += 1
    return found


def _bound(
    statement: str, parameters: Sequence[Value] | Mapping[str, Value] | None
) -> tuple[str, dict[str, Value] | None]:
    """The statement the engine will run, and the parameters for it.

    A mapping goes through untouched, which is how a caller writing zu
    statements keeps `$name`. A sequence is bound to the `?` markers,
    each one becoming a name the engine can find.
    """
    if parameters is None:
        return statement, None
    if isinstance(parameters, Mapping):
        return statement, dict(parameters)
    if isinstance(parameters, (str, bytes)):
        raise ProgrammingError(
            f"parameters must be a sequence or a mapping, not {type(parameters).__name__}"
        )
    values = list(parameters)
    marks = _placeholders(statement)
    if len(marks) != len(values):
        raise ProgrammingError(
            f"the statement has {len(marks)} placeholders and {len(values)} parameters were given"
        )
    if not marks:
        return statement, None
    # A name the statement cannot already be using. Almost always `_1`
    # on the first look, and a caller who really has written `$_1` gets
    # `$__1` instead of a collision nobody would ever find.
    prefix = "_"
    while f"${prefix}" in statement:
        prefix = "_" + prefix
    out: list[str] = []
    last = 0
    for place, at in enumerate(marks, start=1):
        out.append(statement[last:at])
        out.append(f"${prefix}{place}")
        last = at + 1
    out.append(statement[last:])
    return "".join(out), {f"{prefix}{place}": value for place, value in enumerate(values, start=1)}


class Cursor:
    """A statement, its rows, and where the caller has read up to.

    Cursors are cheap and are not shared between threads. Making one
    per statement is the usual shape and costs nothing here: the
    connection underneath is what holds the engine, and a cursor is a
    position in a result that already exists.
    """

    #: PEP 249's optional extension: the exception classes reachable
    #: from the cursor, for code holding one and nothing else.
    Warning = Warning
    Error = Error
    InterfaceError = InterfaceError
    DatabaseError = DatabaseError
    DataError = DataError
    OperationalError = OperationalError
    IntegrityError = IntegrityError
    InternalError = InternalError
    ProgrammingError = ProgrammingError
    NotSupportedError = NotSupportedError

    def __init__(self, connection: Connection) -> None:
        self._connection = connection
        self._closed = False
        self._result: zudb.Result | None = None
        self._ahead: deque[tuple[Value, ...]] = deque()
        self._description: tuple[tuple[Any, ...], ...] | None = None
        self._rowcount = -1
        #: Rows `fetchmany` takes when it is not told how many. One,
        #: which PEP 249 asks for and nobody should leave alone: rows
        #: are already in memory here, so a bigger number costs
        #: nothing and saves calls.
        self.arraysize = 1

    @property
    def connection(self) -> Connection:
        """The connection this cursor was made from."""
        return self._connection

    @property
    def closed(self) -> bool:
        """Whether this cursor has been closed."""
        return self._closed or self._connection.closed

    @property
    def description(self) -> tuple[tuple[Any, ...], ...] | None:
        """One seven-item tuple per column, or `None` after a statement
        that returned no columns.

        `(name, type_code, display_size, internal_size, precision,
        scale, null_ok)`, of which zu answers the first two. The type
        code is the Python type of the values in that column, read from
        the first rows of the result, and `None` for a column that is
        null in all of them. A result carries no declared types to
        report instead: what a column holds is what the statement put
        in it.
        """
        return self._description

    @property
    def rowcount(self) -> int:
        """Rows the last statement produced, or -1 when there is no answer.

        Rows produced, never rows affected: a statement that writes
        gives back no columns and the engine does not count what it
        touched, so a write leaves this at -1 rather than at a number
        somebody would believe.
        """
        return self._rowcount

    @property
    def result(self) -> zudb.Result | None:
        """The native result the last statement gave back.

        The way out of this layer and into the rest of the client:
        `cur.result.to_arrow()` and `cur.result.notices` are there
        without opening a second connection. Reading rows from it moves
        this cursor's own position, since they are the same rows.
        """
        return self._result

    def execute(
        self, operation: str, parameters: Sequence[Value] | Mapping[str, Value] | None = None
    ) -> Cursor:
        """Runs one statement and keeps its rows to be fetched."""
        self._usable()
        statement, values = _bound(operation, parameters)
        self._connection._begin()
        with _translating():
            result = self._connection._conn.execute(statement, values)
        self._result = result
        self._ahead.clear()
        if not result.columns:
            self._description = None
            self._rowcount = -1
        else:
            self._description = self._typed(result)
            self._rowcount = len(result)
        return self

    def executemany(
        self, operation: str, seq_of_parameters: Iterable[Sequence[Value] | Mapping[str, Value]]
    ) -> Cursor:
        """Runs one statement once for each set of parameters.

        For writing rows, which is what PEP 249 has it for. Rows any of
        them produce are not kept, so `description` and `rowcount` are
        empty afterwards rather than describing whichever one happened
        to be last.
        """
        self._usable()
        self._connection._begin()
        for parameters in seq_of_parameters:
            statement, values = _bound(operation, parameters)
            with _translating():
                self._connection._conn.execute(statement, values)
        self._result = None
        self._ahead.clear()
        self._description = None
        self._rowcount = -1
        return self

    def fetchone(self) -> tuple[Value, ...] | None:
        """The next row, or `None` when there are no more."""
        result = self._rows()
        if self._ahead:
            return self._ahead.popleft()
        with _translating():
            return result.fetchone()

    def fetchmany(self, size: int | None = None) -> list[tuple[Value, ...]]:
        """The next `size` rows, or as many as are left."""
        self._rows()
        wanted = self.arraysize if size is None else size
        got: list[tuple[Value, ...]] = []
        while len(got) < wanted:
            row = self.fetchone()
            if row is None:
                break
            got.append(row)
        return got

    def fetchall(self) -> list[tuple[Value, ...]]:
        """Every row that has not been fetched yet."""
        result = self._rows()
        taken = list(self._ahead)
        self._ahead.clear()
        with _translating():
            while (row := result.fetchone()) is not None:
                taken.append(row)
        return taken

    def setinputsizes(self, sizes: Iterable[object]) -> None:
        """Nothing. PEP 249 requires the method, not an effect.

        It is there for drivers that have to reserve a buffer per
        parameter before the statement runs. Parameters here are Python
        objects handed straight across.
        """

    def setoutputsizes(self, size: int, column: int | None = None) -> None:
        """Nothing, for the same reason as `setinputsizes`."""

    def close(self) -> None:
        """Closes the cursor and lets go of its rows.

        The connection is left alone. Closing twice is allowed, since
        the second call has nothing to do and refusing it would only
        make callers write a flag.
        """
        self._closed = True
        self._result = None
        self._ahead.clear()

    def _usable(self) -> None:
        """That there is something to run a statement on."""
        if self._closed:
            raise InterfaceError("this cursor is closed")
        if self._connection.closed:
            raise InterfaceError("the connection this cursor was made from is closed")

    def _rows(self) -> zudb.Result:
        """The result to fetch from, if the last statement left one."""
        self._usable()
        if self._result is None or self._description is None:
            raise ProgrammingError(
                "there are no rows to fetch: the last statement on this cursor returned no columns"
            )
        return self._result

    def _typed(self, result: zudb.Result) -> tuple[tuple[Any, ...], ...]:
        """`description` for a result, and the rows read to work it out.

        Reading stops as soon as every column has been seen holding
        something, so the usual cost is one row. The rows read are kept
        and handed to the caller first, so typing a result takes none
        of it away.
        """
        names = result.columns
        kinds: list[type | None] = [None] * len(names)
        missing = len(names)
        with _translating():
            while missing and len(self._ahead) < AHEAD:
                row = result.fetchone()
                if row is None:
                    break
                self._ahead.append(row)
                for at, value in enumerate(row):
                    if kinds[at] is None and value is not None:
                        kinds[at] = type(value)
                        missing -= 1
        return tuple(
            (name, kind, None, None, None, None, None)
            for name, kind in zip(names, kinds, strict=True)
        )

    def __iter__(self) -> Iterator[tuple[Value, ...]]:
        """Rows one at a time, which PEP 249 lists as an extension."""
        while (row := self.fetchone()) is not None:
            yield row

    def __enter__(self) -> Cursor:
        return self

    def __exit__(self, *_exception: object) -> bool:
        self.close()
        return False

    def __repr__(self) -> str:
        if self.closed:
            return "<zudb.dbapi.Cursor closed>"
        if self._description is None:
            return "<zudb.dbapi.Cursor no rows>"
        return f"<zudb.dbapi.Cursor {len(self._description)} columns, {self._rowcount} rows>"


class Connection:
    """One connection, with the transaction PEP 249 says is running.

    The native connection is underneath and reachable as `zu`, so
    nothing this layer does not have is out of reach: appenders,
    registered frames and `interrupt()` are all still there, on the
    same connection and inside the same transaction.
    """

    #: The exception classes, reachable from the connection. PEP 249
    #: lists this as an extension and it is the only way for code that
    #: was handed a connection to catch what it raises without knowing
    #: which module made it.
    Warning = Warning
    Error = Error
    InterfaceError = InterfaceError
    DatabaseError = DatabaseError
    DataError = DataError
    OperationalError = OperationalError
    IntegrityError = IntegrityError
    InternalError = InternalError
    ProgrammingError = ProgrammingError
    NotSupportedError = NotSupportedError

    def __init__(self, connection: zudb.Connection, *, autocommit: bool = False) -> None:
        self._conn = connection
        self._autocommit = autocommit
        self._work: zudb.Transaction | None = None

    @property
    def zu(self) -> zudb.Connection:
        """The native connection this one wraps."""
        return self._conn

    @property
    def autocommit(self) -> bool:
        """Whether each statement stands alone rather than joining a transaction."""
        return self._autocommit

    @property
    def closed(self) -> bool:
        """Whether this connection is still open."""
        return self._conn.closed

    def cursor(self) -> Cursor:
        """A new cursor on this connection."""
        if self.closed:
            raise InterfaceError("this connection is closed")
        return Cursor(self)

    def commit(self) -> None:
        """Keeps what the transaction wrote and ends it.

        A no-op when nothing has been written since the last one, which
        is what lets a caller commit in a loop without asking whether
        there is anything to commit.
        """
        if self.closed:
            raise InterfaceError("this connection is closed")
        work, self._work = self._work, None
        if work is not None:
            with _translating():
                work.commit()

    def rollback(self) -> None:
        """Throws away what the transaction wrote and ends it."""
        if self.closed:
            raise InterfaceError("this connection is closed")
        work, self._work = self._work, None
        if work is not None:
            with _translating():
                work.rollback()

    def close(self) -> None:
        """Rolls back what was not committed and closes the connection.

        PEP 249 is explicit that an uncommitted transaction is lost
        here rather than kept, which is the one place the two clients
        differ on what a program meant: a statement written outside a
        transaction on the native client is committed when it finishes.
        """
        if self.closed:
            return
        try:
            self.rollback()
        finally:
            self._conn.close()

    def _begin(self) -> None:
        """Starts the transaction the next statement will run in.

        Lazily, because a connection nobody has run anything on is not
        holding a transaction open, and read-only when the connection
        is, since that is the only kind it could start.
        """
        if self._autocommit or self._work is not None:
            return
        with _translating():
            self._work = self._conn.transaction(read_only=self._conn.read_only)

    def __enter__(self) -> Connection:
        return self

    def __exit__(self, kind: object, *_rest: object) -> bool:
        """Commits a block that finished and rolls back one that raised.

        The connection stays open, which is what sqlite3 and psycopg
        both do: the block is the unit of work, not the connection. A
        block that closed the connection itself is left alone, since
        closing already decided what happens to the transaction and
        raising here would bury whatever the block was doing.
        """
        if self.closed:
            return False
        if kind is None:
            self.commit()
        else:
            self.rollback()
        return False

    def __repr__(self) -> str:
        if self.closed:
            return "<zudb.dbapi.Connection closed>"
        return f"<zudb.dbapi.Connection {str(self._conn.path)!r}>"


def connect(
    path: str | os.PathLike[str] | None = None,
    *,
    read_only: bool = False,
    memory_limit: int | None = None,
    threads: int | None = None,
    autocommit: bool = False,
) -> Connection:
    """Opens the database at `path` and connects to it.

    The same arguments `zudb.connect` takes, no path or `":memory:"`
    for a database in memory included, and one more: with `autocommit`
    every statement stands alone the way it does on the native client,
    instead of joining a transaction that runs until `commit` or
    `rollback`.
    """
    with _translating():
        conn = zudb.connect(path, read_only=read_only, memory_limit=memory_limit, threads=threads)
    return Connection(conn, autocommit=autocommit)
