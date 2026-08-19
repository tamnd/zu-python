"""The same calls, awaited.

There are three claims in `zudb.aio` and everything here is one of
them. The loop keeps running while a statement does, which is the whole
reason the module exists. Statements on one connection arrive in the
order they were awaited, because the thread underneath runs one at a
time. Cancelling the task that awaits a statement stops the statement
and leaves the connection free rather than busy with work nobody is
waiting for.

The rest is the sync client's behaviour, checked once through the async
spelling to show it survived the crossing.
"""

from __future__ import annotations

import asyncio
import functools
import time
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any, TypeVar

import pytest
import zudb
import zudb.aio

T = TypeVar("T")

# Every pair of people, filtered, which is a statement that runs for
# long enough to watch rather than one that is over before the loop is
# scheduled.
WORK = "MATCH (a:person), (b:person) WHERE a.uid < b.uid RETURN count(a) AS n"

# Three thousand people is about a second of that work on the machine
# this was written on, and six thousand about three seconds. The first
# is for the tests that wait for the statement to end and the second
# for the ones that stop it partway, which need it still running a
# fifth of a second in on a machine some multiple faster than this one.
WATCHED = 3_000
LONG = 6_000

# Two connections are timed against each other rather than against a
# budget, so what matters is that the statement is long enough to time
# and short enough to run three times.
TIMED = 1_500


def pairs(people: int) -> int:
    return people * (people - 1) // 2


def run(test: Callable[..., Coroutine[Any, Any, None]]) -> Callable[..., None]:
    """Runs a coroutine test on a loop of its own.

    pytest does not await, and this suite has no plugin that teaches it
    to, so each test is a coroutine and this is the call that runs one.
    A loop per test rather than one shared, because a loop that outlived
    a failure would carry whatever that failure left running into the
    test after it. `functools.wraps` keeps the signature, which is what
    pytest reads the fixtures out of.
    """

    @functools.wraps(test)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        asyncio.run(test(*args, **kwargs))

    return wrapper


async def crowded(path: Path, people: int) -> zudb.aio.AsyncConnection:
    """A connection to a database with `people` people in it.

    Loaded rather than inserted because a row at a time is a commit at
    a time, and this is scaffolding rather than the thing under test.
    The load is the sync call, since there is nothing to overlap it
    with and the connection comes after it either way.
    """
    zudb.load(
        path,
        nodes="person",
        rels="knows",
        columns={"uid": list(range(people))},
        edges=[(0, 1)],
    )
    return await zudb.aio.connect(path)


@run
async def test_a_statement_runs_and_gives_back_its_rows(tmp_path: Path) -> None:
    async with zudb.aio.connect(tmp_path / "one.zu1") as conn:
        await conn.execute("INSERT (p:person {uid: 10, name: 'ada'})")
        rows = await conn.execute("MATCH (p:person) RETURN p.name AS name")
        assert rows.fetchall() == [("ada",)]


@run
async def test_what_comes_back_is_read_without_awaiting(tmp_path: Path) -> None:
    """Rows are already in memory, so nothing about reading them waits."""
    async with zudb.aio.connect(tmp_path / "read.zu1") as conn:
        await conn.execute("INSERT (p:person {uid: 10, name: 'ada'})")
        rows = await conn.execute("MATCH (p:person) RETURN p.uid AS uid, p.name AS name")
        assert rows.columns == ["uid", "name"]
        assert len(rows) == 1
        assert [row for row in rows] == [(10, "ada")]


@run
async def test_the_loop_runs_while_a_statement_does(tmp_path: Path) -> None:
    """The claim the module is for.

    A statement is inside Rust with the GIL down, and while it is there
    a task that only wants the loop keeps getting it. The count is
    asserted low enough to be about whether the loop moved at all
    rather than about how fast this machine is.
    """
    async with await crowded(tmp_path / "ticking.zu1", WATCHED) as conn:
        ticks = 0
        statement = asyncio.ensure_future(conn.execute(WORK))
        while not statement.done():
            await asyncio.sleep(0.001)
            ticks += 1
        assert (await statement).fetchone() == (pairs(WATCHED),)
        assert ticks > 20, f"the loop only got round {ticks} times"


@run
@pytest.mark.timing
async def test_two_connections_run_at_the_same_time(tmp_path: Path) -> None:
    """A thread each, and the engine puts the GIL down for the work, so
    two statements together cost about what one costs rather than two.
    """
    first = await crowded(tmp_path / "first.zu1", TIMED)
    second = await crowded(tmp_path / "second.zu1", TIMED)
    async with first, second:
        # Once each before the clock starts, because the first
        # statement on a connection reads the file in and the number
        # wanted here is about two threads and not about a cold cache.
        await asyncio.gather(first.execute(WORK), second.execute(WORK))

        started = time.perf_counter()
        await first.execute(WORK)
        alone = time.perf_counter() - started

        started = time.perf_counter()
        await asyncio.gather(first.execute(WORK), second.execute(WORK))
        together = time.perf_counter() - started
    assert together < alone * 1.8, f"{together:.3f}s together against {alone:.3f}s alone"


@run
async def test_statements_arrive_in_the_order_they_were_awaited(tmp_path: Path) -> None:
    async with zudb.aio.connect(tmp_path / "ordered.zu1") as conn:
        await conn.execute("INSERT (p:person {uid: 0, name: 'seed'})")
        await asyncio.gather(
            *(
                conn.execute("INSERT (p:person {uid: $uid, name: 'p'})", {"uid": uid})
                for uid in range(1, 6)
            )
        )
        rows = await conn.execute("MATCH (p:person) RETURN p.uid AS uid")
        assert [uid for (uid,) in rows] == [0, 1, 2, 3, 4, 5]


@run
async def test_cancelling_the_task_stops_the_statement(tmp_path: Path) -> None:
    async with await crowded(tmp_path / "cancelled.zu1", LONG) as conn:
        statement = asyncio.ensure_future(conn.execute(WORK))
        # Long enough that it is inside the executor rather than still
        # being parsed when the cancellation arrives.
        await asyncio.sleep(0.2)
        statement.cancel()
        with pytest.raises(asyncio.CancelledError):
            await statement


@run
@pytest.mark.timing
async def test_a_cancelled_statement_leaves_the_connection_free(tmp_path: Path) -> None:
    """The reason the cancellation waits for the statement it stopped.

    A connection still running work nobody is listening to would make
    the next statement queue behind all of it, so the one after the
    cancellation is timed rather than merely run.
    """
    async with await crowded(tmp_path / "free.zu1", LONG) as conn:
        statement = asyncio.ensure_future(conn.execute(WORK))
        await asyncio.sleep(0.2)
        statement.cancel()
        with pytest.raises(asyncio.CancelledError):
            await statement

        started = time.perf_counter()
        rows = await conn.execute("MATCH (p:person) RETURN count(p) AS n")
        took = time.perf_counter() - started
        assert rows.fetchone() == (LONG,)
        assert took < 1.0, f"the next statement waited {took:.3f}s"


@run
async def test_a_statement_still_queued_is_dropped_without_running(tmp_path: Path) -> None:
    async with await crowded(tmp_path / "queued.zu1", LONG) as conn:
        running = asyncio.ensure_future(conn.execute(WORK))
        waiting = asyncio.ensure_future(conn.execute("INSERT (p:person {uid: 999999})"))
        await asyncio.sleep(0.2)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        rows = await conn.execute("MATCH (p:person) WHERE p.uid = 999999 RETURN p.uid AS uid")
        assert rows.fetchall() == []


@run
async def test_interrupt_is_not_awaited_and_stops_what_is_running(tmp_path: Path) -> None:
    async with await crowded(tmp_path / "interrupted.zu1", LONG) as conn:
        statement = asyncio.ensure_future(conn.execute(WORK))
        await asyncio.sleep(0.2)
        conn.interrupt()
        with pytest.raises(zudb.Interrupted):
            await statement


@run
async def test_a_transaction_block_commits_what_it_wrote(tmp_path: Path) -> None:
    async with zudb.aio.connect(tmp_path / "committed.zu1") as conn:
        await conn.execute("INSERT (p:person {uid: 10, name: 'ada'})")
        async with conn.transaction():
            await conn.execute("INSERT (p:person {uid: 20, name: 'grace'})")
            await conn.execute("INSERT (p:person {uid: 30, name: 'kay'})")
        rows = await conn.execute("MATCH (p:person) RETURN p.uid AS uid")
        assert [uid for (uid,) in rows] == [10, 20, 30]


@run
async def test_a_transaction_block_that_raises_rolls_back(tmp_path: Path) -> None:
    async with zudb.aio.connect(tmp_path / "rolled.zu1") as conn:
        await conn.execute("INSERT (p:person {uid: 10, name: 'ada'})")
        with pytest.raises(RuntimeError, match="halfway"):
            async with conn.transaction():
                await conn.execute("INSERT (p:person {uid: 20, name: 'grace'})")
                raise RuntimeError("halfway through")
        rows = await conn.execute("MATCH (p:person) RETURN p.uid AS uid")
        assert [uid for (uid,) in rows] == [10]


@run
async def test_a_transaction_cancelled_partway_rolls_back(tmp_path: Path) -> None:
    """The case the block is written this way for.

    A task stopped between two writes is a task that raised, so the
    block unwinds through the rollback and neither write is kept.
    """
    conn = await zudb.aio.connect(tmp_path / "half.zu1")
    await conn.execute("INSERT (p:person {uid: 10, name: 'ada'})")
    reached = asyncio.Event()

    async def both() -> None:
        async with conn.transaction():
            await conn.execute("INSERT (p:person {uid: 20, name: 'grace'})")
            reached.set()
            await asyncio.sleep(30)
            await conn.execute("INSERT (p:person {uid: 30, name: 'kay'})")

    task = asyncio.ensure_future(both())
    await reached.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    rows = await conn.execute("MATCH (p:person) RETURN p.uid AS uid")
    assert [uid for (uid,) in rows] == [10]
    await conn.close()


@run
async def test_in_transaction_is_awaited(tmp_path: Path) -> None:
    async with zudb.aio.connect(tmp_path / "asking.zu1") as conn:
        assert await conn.in_transaction() is False
        async with conn.transaction() as txn:
            assert await conn.in_transaction() is True
            assert txn.read_only is False
            assert txn.done is False
        assert await conn.in_transaction() is False


@run
async def test_a_transaction_ended_by_hand_is_done(tmp_path: Path) -> None:
    async with zudb.aio.connect(tmp_path / "byhand.zu1") as conn:
        txn = await conn.transaction()
        await conn.execute("INSERT (p:person {uid: 10, name: 'ada'})")
        await txn.rollback()
        assert txn.done is True
        rows = await conn.execute("MATCH (p:person) RETURN p.uid AS uid")
        assert rows.fetchall() == []


@run
async def test_an_appender_loads_rows(tmp_path: Path) -> None:
    async with zudb.aio.connect(tmp_path / "appended.zu1") as conn:
        await conn.execute("INSERT (p:person {uid: 0, name: 'seed'})")
        async with conn.appender("person") as appender:
            assert appender.table == "person"
            await appender.append_row([1, "ada"])
            await appender.append_rows([[2, "grace"], [3, "kay"]])
            assert await appender.buffered() == 3
            assert await appender.flush() == 3
            assert await appender.committed() == 3
        rows = await conn.execute("MATCH (p:person) RETURN p.name AS name")
        assert [name for (name,) in rows] == ["seed", "ada", "grace", "kay"]


@run
async def test_a_frame_registers_and_a_statement_matches_it(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    async with zudb.aio.connect(tmp_path / "framed.zu1") as conn:
        frame = pa.table({"uid": [1, 2, 3], "score": [10.0, 20.0, 30.0]})
        assert await conn.register("people", frame) == 3
        assert await conn.registered() == ["people"]
        rows = await conn.execute("MATCH (p:people) WHERE p.uid > 1 RETURN sum(p.score) AS total")
        assert rows.fetchone() == (50.0,)
        await conn.unregister("people")
        assert await conn.registered() == []


@run
async def test_the_ways_that_cannot_wait_are_properties(tmp_path: Path) -> None:
    """`rows_read` is what a progress bar reads while a statement runs,
    so it is answered from beside the lock rather than through it.
    """
    path = tmp_path / "beside.zu1"
    async with await crowded(path, LONG) as conn:
        assert conn.path == path
        assert conn.read_only is False
        assert conn.closed is False
        statement = asyncio.ensure_future(conn.execute(WORK))
        await asyncio.sleep(0.2)
        assert conn.rows_read > 0
        statement.cancel()
        with pytest.raises(asyncio.CancelledError):
            await statement


@run
async def test_a_connection_closes_at_the_end_of_the_block(tmp_path: Path) -> None:
    async with zudb.aio.connect(tmp_path / "block.zu1") as conn:
        assert conn.closed is False
    assert conn.closed is True


@run
async def test_awaiting_the_open_gives_a_connection_to_close_yourself(tmp_path: Path) -> None:
    conn = await zudb.aio.connect(tmp_path / "byhand.zu1")
    assert conn.closed is False
    await conn.close()
    assert conn.closed is True
    # Twice is not an error.
    await conn.close()
    assert conn.closed is True


@run
async def test_a_call_on_a_closed_connection_is_refused(tmp_path: Path) -> None:
    """The thread is gone, so the refusal has to come from here, and it
    is the refusal the engine would have given.
    """
    conn = await zudb.aio.connect(tmp_path / "gone.zu1")
    await conn.close()
    with pytest.raises(zudb.ProgrammingError, match="closed"):
        await conn.execute("MATCH (p:person) RETURN p.uid AS uid")


@run
async def test_a_read_only_connection_says_so(tmp_path: Path) -> None:
    path = tmp_path / "readonly.zu1"
    async with zudb.aio.connect(path) as writer:
        await writer.execute("INSERT (p:person {uid: 10, name: 'ada'})")
    async with zudb.aio.connect(path, read_only=True) as reader:
        assert reader.read_only is True
        rows = await reader.execute("MATCH (p:person) RETURN p.name AS name")
        assert rows.fetchall() == [("ada",)]


@run
async def test_opening_a_database_that_is_not_one_raises(tmp_path: Path) -> None:
    """The open is on the thread too, so its failure arrives awaited."""
    path = tmp_path / "junk.zu1"
    path.write_bytes(b"not a database")
    with pytest.raises(zudb.Error):
        await zudb.aio.connect(path)


@run
async def test_a_statement_that_is_wrong_raises_what_it_would_have(tmp_path: Path) -> None:
    async with zudb.aio.connect(tmp_path / "wrong.zu1") as conn:
        with pytest.raises(zudb.SyntaxError, match="expected"):
            await conn.execute("MATCH (p:person RETURN p")


@run
async def test_repr_names_the_file_and_says_when_it_is_closed(tmp_path: Path) -> None:
    conn = await zudb.aio.connect(tmp_path / "shown.zu1")
    assert "shown.zu1" in repr(conn)
    assert "closed" not in repr(conn)
    await conn.close()
    assert "closed" in repr(conn)


@run
async def test_sql_is_execute_under_another_name(tmp_path: Path) -> None:
    async with zudb.aio.connect(tmp_path / "notebook.zu1") as conn:
        await conn.sql("INSERT (p:person {uid: 10, name: 'ada'})")
        rows = await conn.sql("MATCH (p:person) RETURN p.name AS name")
        assert rows.fetchall() == [("ada",)]
