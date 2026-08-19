"""zu on an event loop: the same calls, awaited.

    import zudb.aio

    async with zudb.aio.connect("social.zu1") as conn:
        rows = await conn.execute("MATCH (p:person) RETURN p.name AS name")
        for (name,) in rows:
            print(name)

A statement runs inside Rust with the GIL down, and a call like that
comes back when it comes back. Run one straight from a coroutine and
every other task on the loop waits for it, including the ones answering
requests that have nothing to do with the database. So each connection
here keeps a thread of its own, and every call that could wait is handed
to that thread and awaited.

A thread per connection rather than a pool shared by all of them,
because a connection is one lock and statements on it queue anyway. A
pool would add no parallelism the engine can use and would let two
statements on one connection run in an order neither caller wrote. Two
connections do run at the same time, on their own threads, since the
engine puts the GIL down for the work.

What comes back is a `zudb.Result`, which is rows already in memory, so
nothing about reading them waits and nothing about reading them is
awaited: `for row in rows` and `rows.to_arrow()` are the calls they were.
Only the ways in are awaited, and only the ones that can wait: `path`,
`read_only`, `closed` and `rows_read` are answered from beside the lock
rather than through it, so they stay properties here too.

Cancelling the task that awaits a statement interrupts the statement.
The engine is asked to stop, and the coroutine does not return until it
has, so the connection is idle again by the time the `CancelledError`
reaches the caller rather than busy with a statement nobody is waiting
for. A statement still queued when the cancellation arrives is dropped
without running.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import os
import pathlib
import threading
from collections.abc import Callable, Coroutine, Generator, Iterable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Generic, TypeVar

from . import _zudb
from ._zudb import (
    Appender,
    Connection,
    Plan,
    Prepared,
    Profile,
    Result,
    Stream,
    StreamBatches,
    StreamSummary,
    Transaction,
)
from .types import Value

__all__ = [
    "connect",
    "AsyncConnection",
    "AsyncTransaction",
    "AsyncAppender",
    "AsyncPrepared",
    "AsyncStream",
    "AsyncStreamBatches",
]

T = TypeVar("T")

#: Handed to `next` so that the end of a stream arrives as a value on the
#: connection's thread. A `StopIteration` raised there would cross a
#: future on its way back, and a `StopIteration` crossing a future is the
#: one exception asyncio cannot let through.
_NOTHING = object()


def connect(
    path: str | os.PathLike[str] | None = None,
    *,
    read_only: bool = False,
    memory_limit: int | None = None,
    threads: int | None = None,
) -> _Opening[AsyncConnection]:
    """Opens the database at `path` and connects to it.

    The arguments are `zudb.connect`'s, and so is the behaviour: a path
    holding nothing becomes a new database unless the connection is
    read-only, and no path at all, or `":memory:"`, is a database in
    memory that makes no file. Opening reads the file, so it happens on
    the connection's own thread like everything else.

    Await it for a connection to close yourself, or open it with `async
    with` for one that closes at the end of the block:

        conn = await zudb.aio.connect("social.zu1")
        async with zudb.aio.connect("social.zu1") as conn:
            ...
    """
    return _Opening(
        functools.partial(
            _open,
            path,
            read_only=read_only,
            memory_limit=memory_limit,
            threads=threads,
        )
    )


async def _open(
    path: str | os.PathLike[str] | None,
    *,
    read_only: bool,
    memory_limit: int | None,
    threads: int | None,
) -> AsyncConnection:
    """Starts the thread, opens the database on it, and joins the two."""
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="zudb aio")
    opening = pool.submit(
        functools.partial(
            _zudb.connect,
            path,
            read_only=read_only,
            memory_limit=memory_limit,
            threads=threads,
        )
    )
    try:
        conn = await asyncio.wrap_future(opening)
    except BaseException:
        # An open that was cancelled is still an open file once the
        # thread reaches it, so what arrives is closed rather than left
        # for the collector to find.
        opening.add_done_callback(_discard)
        pool.shutdown(wait=False)
        raise
    return AsyncConnection(conn, pool)


def _discard(opened: Future[Connection]) -> None:
    """Closes a connection nobody is waiting for any more."""
    if not opened.cancelled() and opened.exception() is None:
        opened.result().close()


class _Opening(Generic[T]):
    """Something to `await`, or to open with `async with`.

    Opening a connection, starting a transaction and taking an appender
    all reach the engine, so all three are coroutines, and a coroutine
    is not a context manager. This is the both of them: awaiting it
    opens the thing, and `async with` opens it and then enters it, so
    the two spellings a caller might reach for are the same call.
    """

    __slots__ = ("_open", "_held")

    def __init__(self, open_: Callable[[], Coroutine[Any, Any, T]]) -> None:
        self._open = open_
        self._held: T | None = None

    def __await__(self) -> Generator[Any, None, T]:
        return self._open().__await__()

    async def __aenter__(self) -> T:
        self._held = await self._open()
        return await self._held.__aenter__()  # type: ignore[attr-defined,no-any-return]

    async def __aexit__(self, *exception: Any) -> bool:
        return await self._held.__aexit__(*exception)  # type: ignore[attr-defined,no-any-return]


class AsyncConnection:
    """One connection to one database, and the thread its work runs on.

    Take one with `zudb.aio.connect`. Every method that could wait is a
    coroutine, and they queue in the order they were awaited, because
    the thread underneath runs one at a time.
    """

    __slots__ = ("_conn", "_pool", "_guard", "_running", "_shut")

    def __init__(self, conn: Connection, pool: ThreadPoolExecutor) -> None:
        self._conn = conn
        self._pool = pool
        # Held across the two lines that say which job is on the thread,
        # so that a cancellation either interrupts its own statement or
        # interrupts nothing. Without it a job that finished a moment
        # ago could have its stop land on the one after it.
        self._guard = threading.Lock()
        self._running: object | None = None
        self._shut = False

    @property
    def path(self) -> pathlib.Path:
        """The file this connection was opened on."""
        return self._conn.path

    @property
    def memory(self) -> bool:
        """Whether the database behind it is in memory."""
        return self._conn.memory

    @property
    def read_only(self) -> bool:
        """Whether it was opened read-only."""
        return self._conn.read_only

    @property
    def closed(self) -> bool:
        """Whether this connection is still open."""
        return self._conn.closed

    @property
    def rows_read(self) -> int:
        """How many rows the statement running on this connection has
        read out of storage.

        A property and not a coroutine on purpose: this is what a
        progress bar reads while a statement runs, and a call that had
        to queue behind that statement could only answer once it was
        over.
        """
        return self._conn.rows_read

    def interrupt(self) -> None:
        """Asks the statement running on this connection to stop.

        Not awaited, for the same reason: an ask that queued behind the
        statement it means to stop would arrive after it. Cancelling the
        task that awaits a statement does this for you.
        """
        self._conn.interrupt()

    async def execute(self, statement: str, params: Mapping[str, Value] | None = None) -> Result:
        """Runs one statement and gives back its rows."""
        return await self._call(functools.partial(self._conn.execute, statement, params))

    async def sql(self, statement: str, params: Mapping[str, Value] | None = None) -> Result:
        """The same call, named for the way it reads in a notebook."""
        return await self._call(functools.partial(self._conn.sql, statement, params))

    def prepare(self, statement: str) -> _Opening[AsyncPrepared]:
        """Compiles a statement now and hands back something that runs
        it later, as often as you like.

            statement = "MATCH (p:person) WHERE p.name = $name RETURN p.id AS id"
            async with conn.prepare(statement) as find:
                rows = await find.execute({"name": "ada"})

        Compiling reaches the engine, so it is awaited like everything
        else here, and what it buys is what the sync client's docstring
        says it buys: the compile at this line rather than at the first
        request, and the names the statement wants.
        """
        return _Opening(functools.partial(self._prepare, statement))

    async def _prepare(self, statement: str) -> AsyncPrepared:
        compiled = await self._call(functools.partial(self._conn.prepare, statement))
        return AsyncPrepared(self, compiled)

    async def explain(self, statement: str) -> Plan:
        """What the statement would do, without doing it."""
        return await self._call(functools.partial(self._conn.explain, statement))

    async def profile(self, statement: str, params: Mapping[str, Value] | None = None) -> Profile:
        """Runs the statement with the counters on and answers what its
        operators really did.
        """
        return await self._call(functools.partial(self._conn.profile, statement, params))

    def stream(
        self,
        statement: str,
        params: Mapping[str, Value] | None = None,
        *,
        batch_rows: int | None = None,
    ) -> _Opening[AsyncStream]:
        """Runs one statement and hands back its rows as the engine
        makes them.

            async with conn.stream("MATCH (p:person) RETURN p.name AS name") as rows:
                async for (name,) in rows:
                    await write(name)

        This is the call for a result too big to want in memory and for
        one whose first rows are worth having before the last are made.
        The statement runs on a thread of its own, not this connection's,
        and every wait for a batch is handed off the loop like every
        other wait here, so a task reading a stream leaves the loop free
        between batches.

        A stream holds the connection until it ends, so a statement run
        on the same connection while one is open is refused rather than
        queued behind a loop that may never finish. Read it to the end,
        close it, or open it with `async with`.
        """
        return _Opening(functools.partial(self._stream, statement, params, batch_rows))

    async def _stream(
        self,
        statement: str,
        params: Mapping[str, Value] | None,
        batch_rows: int | None,
    ) -> AsyncStream:
        opened = await self._call(
            functools.partial(self._conn.stream, statement, params, batch_rows=batch_rows)
        )
        return AsyncStream(self, opened)

    def transaction(self, *, read_only: bool = False) -> _Opening[AsyncTransaction]:
        """Starts a transaction and hands it back for an `async with`
        block.

            async with conn.transaction():
                await conn.execute("INSERT (a:account {uid: 1, balance: 100})")
                await conn.execute("INSERT (b:account {uid: 2, balance: 0})")

        The block commits when it ends and rolls back when it raises,
        and a task cancelled inside one leaves through the rollback like
        any other exception.
        """
        return _Opening(functools.partial(self._transaction, read_only))

    async def _transaction(self, read_only: bool) -> AsyncTransaction:
        started = await self._call(functools.partial(self._conn.transaction, read_only=read_only))
        return AsyncTransaction(self, started)

    async def in_transaction(self) -> bool:
        """Whether an explicit transaction is running on this
        connection.

        A method where the sync client has a property, because the
        answer is inside the lock and so has to be awaited, and a
        property that gave back a coroutine would read as `if
        conn.in_transaction:`, which is true whatever the answer.
        """
        return await self._call(lambda: self._conn.in_transaction)

    def appender(self, table: str) -> _Opening[AsyncAppender]:
        """Opens an appender on `table`, for loading rows into a
        database that already exists.
        """
        return _Opening(functools.partial(self._appender, table))

    async def _appender(self, table: str) -> AsyncAppender:
        opened = await self._call(functools.partial(self._conn.appender, table))
        return AsyncAppender(self, opened)

    async def register(self, name: str, data: Any) -> int:
        """Puts a DataFrame under a name a statement can match on."""
        return await self._call(functools.partial(self._conn.register, name, data))

    async def unregister(self, name: str) -> None:
        """Takes a registered frame's name away."""
        await self._call(functools.partial(self._conn.unregister, name))

    async def registered(self) -> list[str]:
        """The names this connection has registered frames under,
        sorted.

        A method where the sync client has a property, for the reason
        `in_transaction` is one: the names live in the engine, behind
        the lock, so asking has to be awaited.
        """
        return await self._call(lambda: self._conn.registered)

    def cursor(self) -> _Opening[AsyncConnection]:
        """Another connection to the same database, made from this one.

            async with conn.cursor() as other:
                rows = await other.execute("MATCH (p:person) RETURN p.name AS name")

        This is how a pool is written, and on an event loop it is the
        only way to have two statements in flight at once, since one
        connection runs one statement at a time by design. It forks off
        the database this connection already holds rather than opening
        the file again, so it works on a database in memory, where
        there is nothing to open a second time.

        The new connection gets a thread of its own, which is the whole
        point: two of them run at once. It gets its own prepared
        statements, its own caches and its own transaction, and shares
        only the write side, so the two queue behind each other to
        write and each sees what the other has committed.

        Closing this one does not close that one. Forking reaches the
        engine, so it is awaited like everything else here.
        """
        return _Opening(self._cursor)

    def duplicate(self) -> _Opening[AsyncConnection]:
        """The same call as `cursor()`, under the name that says what
        it does.
        """
        return _Opening(self._cursor)

    async def _cursor(self) -> AsyncConnection:
        made = await self._call(self._conn.cursor)
        # A thread of its own, started here rather than by the engine
        # call, so that a fork that failed leaves no thread behind.
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="zudb aio")
        return AsyncConnection(made, pool)

    async def close(self) -> None:
        """Closes the connection, frees what it held, and ends the
        thread it ran on.

        Closing waits for whatever is still running, which is why it is
        awaited. Doing it twice is not an error.
        """
        if self._shut:
            return
        await self._call(self._conn.close)
        self._shut = True
        # False, because the thread has just finished the close and has
        # nothing else to do, and waiting for a thread from inside a
        # coroutine is the one thing this module exists to avoid.
        self._pool.shutdown(wait=False)

    async def __aenter__(self) -> AsyncConnection:
        return self

    async def __aexit__(self, *_exception: Any) -> bool:
        await self.close()
        # False, so an exception raised inside the block carries on out
        # of it.
        return False

    def __repr__(self) -> str:
        state = ", closed" if self.closed else ""
        return f"<zudb.aio.AsyncConnection {self.path}{state}>"

    async def _call(self, work: Callable[[], T]) -> T:
        """Runs `work` on this connection's thread and waits for it.

        Everything here that can wait goes through this, so the order
        statements run in is the order they were awaited in, and the
        loop is free the whole time any of them is running.
        """
        if self._shut:
            # The thread is gone and there is nothing to run on, but
            # every call into a closed connection is refused before it
            # reaches the engine, so running it here raises exactly what
            # the thread would have raised, without one.
            return work()
        token = object()
        queued = self._pool.submit(self._job, token, work)
        try:
            return await asyncio.shield(asyncio.wrap_future(queued))
        except asyncio.CancelledError:
            if not queued.cancel():
                self._stop(token)
                # Waited for rather than abandoned, so that the next
                # statement finds the connection free rather than locked
                # by one nobody is listening to. Whatever it ends with,
                # including the rows it managed to finish, is dropped:
                # the caller asked to stop, and the answer to that is
                # the cancellation.
                with contextlib.suppress(BaseException):
                    await asyncio.shield(asyncio.wrap_future(queued))
            raise

    def _job(self, token: object, work: Callable[[], T]) -> T:
        """The work, with a note of whose it is while it runs."""
        with self._guard:
            self._running = token
        try:
            return work()
        finally:
            with self._guard:
                self._running = None

    def _stop(self, token: object) -> None:
        """Interrupts the statement `token` names, if it is the one
        running.
        """
        with self._guard:
            if self._running is token:
                # Refused on a connection that was closed underneath
                # us, which is not something a cancellation should
                # raise about: the statement is over either way.
                with contextlib.suppress(Exception):
                    self._conn.interrupt()


class AsyncTransaction:
    """A transaction that has been started and not yet ended.

    The transaction underneath is `zudb.Transaction` and the rules are
    its rules: it starts when it is taken, ending it twice is refused,
    and the statements inside it are the ones written on the connection
    it came from.
    """

    __slots__ = ("_conn", "_txn")

    def __init__(self, conn: AsyncConnection, txn: Transaction) -> None:
        self._conn = conn
        self._txn = txn

    @property
    def read_only(self) -> bool:
        """Whether it was started `READ ONLY`."""
        return self._txn.read_only

    @property
    def done(self) -> bool:
        """Whether this transaction has already been committed or
        rolled back.
        """
        return self._txn.done

    async def commit(self) -> None:
        """Ends the transaction and keeps what it wrote."""
        await self._conn._call(self._txn.commit)

    async def rollback(self) -> None:
        """Ends the transaction and throws away what it wrote."""
        await self._conn._call(self._txn.rollback)

    async def __aenter__(self) -> AsyncTransaction:
        return self

    async def __aexit__(self, *exception: Any) -> bool:
        """Commits at the end of the block, and rolls back when the
        block raised.

        A block cancelled partway is a block that raised, so it unwinds:
        the rollback is the reason to write the block this way, and a
        task that stopped halfway through two writes is exactly the case
        it is there for.
        """
        raised = bool(exception) and exception[0] is not None
        if not self.done:
            await (self.rollback() if raised else self.commit())
        return False

    def __repr__(self) -> str:
        read_only = " read only" if self.read_only else ""
        done = ", done" if self.done else ""
        return f"<zudb.aio.AsyncTransaction{read_only}{done}>"


class AsyncAppender:
    """Rows on their way into a table, buffered until they are flushed.

    Every call here is awaited, the ones that only touch the buffer
    included. The buffer's lock is held for the whole of a flush, and a
    flush is inside the engine with the GIL down: an append from the
    loop's own thread would wait on that lock while holding the GIL,
    which is the one wait the flush cannot get out from under. Handing
    them all to the connection's thread keeps them in one queue where
    they cannot meet.
    """

    __slots__ = ("_conn", "_appender")

    def __init__(self, conn: AsyncConnection, appender: Appender) -> None:
        self._conn = conn
        self._appender = appender

    @property
    def table(self) -> str:
        """The table this appender writes to."""
        return self._appender.table

    async def buffered(self) -> int:
        """Rows buffered and not yet written."""
        return await self._conn._call(lambda: self._appender.buffered)

    async def committed(self) -> int:
        """Rows this appender has committed, across every flush."""
        return await self._conn._call(lambda: self._appender.committed)

    async def closed(self) -> bool:
        """Whether this appender has been closed."""
        return await self._conn._call(lambda: self._appender.closed)

    async def append_row(self, row: Iterable[Value]) -> None:
        """Appends one row, which is a sequence of one value per column
        of the table.

        A row is a conversion and a push per column, so a loader with
        rows to hand should reach for `append_rows` and pay for one
        crossing rather than one per row.
        """
        await self._conn._call(functools.partial(self._appender.append_row, row))

    async def append_rows(self, rows: Iterable[Iterable[Value]]) -> int:
        """Appends every row of an iterable of rows.

        The iterable is read on the connection's thread, so a generator
        passed here runs there: keep it to arranging values, and do the
        awaiting that produced them before the call.
        """
        return await self._conn._call(functools.partial(self._appender.append_rows, rows))

    async def flush(self) -> int:
        """Writes every buffered row and makes it readable."""
        return await self._conn._call(self._appender.flush)

    async def discard(self) -> int:
        """Throws away what is buffered and answers how many rows that
        was.
        """
        return await self._conn._call(self._appender.discard)

    async def close(self) -> int:
        """Flushes what is left and answers how many rows this appender
        committed in all.
        """
        return await self._conn._call(self._appender.close)

    async def __aenter__(self) -> AsyncAppender:
        return self

    async def __aexit__(self, *_exception: Any) -> bool:
        """Closes on the way out, which flushes, and does it whether the
        block ended well or badly, like the appender underneath.
        """
        await self.close()
        return False

    def __repr__(self) -> str:
        return f"<zudb.aio.AsyncAppender {self.table}>"


class AsyncPrepared:
    """A statement the engine has compiled and is holding for you.

    The prepared statement underneath is `zudb.Prepared` and the rules
    are its rules: running one that has been closed is refused, closing
    twice does nothing, and a run that binds no value the statement
    wants fails at the run rather than at the compile.
    """

    __slots__ = ("_conn", "_prepared")

    def __init__(self, conn: AsyncConnection, prepared: Prepared) -> None:
        self._conn = conn
        self._prepared = prepared

    @property
    def statement(self) -> str:
        """The text it was compiled from."""
        return self._prepared.statement

    @property
    def params(self) -> list[str]:
        """The names this statement wants bound, in the order it uses
        them.

        A property rather than a coroutine, unlike most of what is here:
        the names were read at the compile and are held beside the id,
        so asking for them reaches nothing that could wait.
        """
        return self._prepared.params

    @property
    def closed(self) -> bool:
        """Whether this prepared statement has been closed."""
        return self._prepared.closed

    async def execute(self, params: Mapping[str, Value] | None = None) -> Result:
        """Runs it with these parameters and gives back its rows."""
        return await self._conn._call(functools.partial(self._prepared.execute, params))

    async def sql(self, params: Mapping[str, Value] | None = None) -> Result:
        """The same call, named for the way it reads in a notebook."""
        return await self._conn._call(functools.partial(self._prepared.sql, params))

    async def close(self) -> None:
        """Closes it and gives the statement back to the connection."""
        await self._conn._call(self._prepared.close)

    async def __aenter__(self) -> AsyncPrepared:
        return self

    async def __aexit__(self, *_exception: Any) -> bool:
        """Closes on the way out, whether the block ended well or badly.

        There is nothing to undo and nothing to write, so unlike a
        transaction's exit this one has only the one thing it could
        mean.
        """
        await self.close()
        return False

    def __repr__(self) -> str:
        closed = ", closed" if self.closed else ""
        return f"<zudb.aio.AsyncPrepared {self.statement!r}{closed}>"


class AsyncStream:
    """Rows arriving one batch at a time, awaited a row at a time.

    Take one with `AsyncConnection.stream`. The stream underneath is
    `zudb.Stream` and the rules are its rules: it holds the connection
    until it ends, closing it twice does nothing, and the summary is
    there once the statement is over whether it ran out of rows or was
    stopped.

    Waiting for the next batch is handed to the connection's thread, so
    the loop is free while the engine is working. The rows of a batch
    already in hand are turned into tuples on the loop's thread, which is
    the same work `Result` does when it is read, and there is nothing to
    wait for in it.
    """

    __slots__ = ("_conn", "_stream")

    def __init__(self, conn: AsyncConnection, stream: Stream) -> None:
        self._conn = conn
        self._stream = stream

    async def columns(self) -> list[str]:
        """The column names, in the order the statement projects them.

        A method where the sync client has a property, because answering
        reads the first batch and reading a batch can wait.
        """
        return await self._conn._call(lambda: self._stream.columns)

    @property
    def summary(self) -> StreamSummary | None:
        """What the statement did, once it has done it, and `None` while
        it is still running.

        A property and not a coroutine: the answer is beside the queue
        rather than behind the engine, so asking reaches nothing that
        could wait.
        """
        return self._stream.summary

    @property
    def closed(self) -> bool:
        """Whether the statement is over, by running out of rows or by
        being closed.
        """
        return self._stream.closed

    def batches(self) -> AsyncStreamBatches:
        """The rows in the batches they arrived in, as lists of tuples.

        For a writer with a size of its own: one await per batch rather
        than one per row, and one call into whatever is being written to.
        """
        return AsyncStreamBatches(self._conn, self._stream.batches())

    async def close(self) -> None:
        """Stops the statement and gives the connection back.

        Awaited because it waits for the statement to stop, so the
        connection is free by the time the call returns. Doing it twice
        is not an error.
        """
        await self._conn._call(self._stream.close)

    def __aiter__(self) -> AsyncStream:
        return self

    async def __anext__(self) -> tuple[Value, ...]:
        row = await self._conn._call(functools.partial(next, self._stream, _NOTHING))
        if row is _NOTHING:
            raise StopAsyncIteration
        return row  # type: ignore[return-value]

    async def __aenter__(self) -> AsyncStream:
        return self

    async def __aexit__(self, *_exception: Any) -> bool:
        """Closes on the way out, whether the block ended well or badly,
        so the connection comes back either way.
        """
        await self.close()
        return False

    def __repr__(self) -> str:
        return f"<zudb.aio.AsyncStream of {self._stream!r}>"


class AsyncStreamBatches:
    """The same rows, in the batches they arrived in."""

    __slots__ = ("_conn", "_batches")

    def __init__(self, conn: AsyncConnection, batches: StreamBatches) -> None:
        self._conn = conn
        self._batches = batches

    def __aiter__(self) -> AsyncStreamBatches:
        return self

    async def __anext__(self) -> list[tuple[Value, ...]]:
        batch = await self._conn._call(functools.partial(next, self._batches, _NOTHING))
        if batch is _NOTHING:
            raise StopAsyncIteration
        return batch  # type: ignore[return-value]

    def __repr__(self) -> str:
        return f"<zudb.aio.AsyncStreamBatches of {self._batches!r}>"
