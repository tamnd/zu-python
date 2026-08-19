"""Frames, registered under a name a statement can match on.

The point of the call is that a DataFrame a program already has becomes
something a statement can read, and that reading it costs nothing: the
engine is told where the columns are and reads them where they lie. So
most of these register one and then match it, and the ones that matter
most prove the two halves of that claim, which are that the bytes are
never copied and that they are handed back when the name goes.

pandas and polars are asked for by the tests that need them and skipped
when they are not installed, because the wheel depends on neither.
"""

from __future__ import annotations

import datetime
import gc
import struct
import time
from pathlib import Path

import pytest
import zudb

pa = pytest.importorskip("pyarrow")


def names(conn: zudb.Connection, table: str) -> list[str]:
    """The `name` column of a registered frame, in the order it went in."""
    return [name for (name,) in conn.execute(f"MATCH (f:{table}) RETURN f.name AS name")]


def test_a_dictionary_of_lists_is_a_frame(empty: zudb.Connection) -> None:
    """The way in for a caller with no frame library installed."""
    assert empty.register("people", {"uid": [1, 2, 3], "name": ["ada", "grace", "lynn"]}) == 3
    assert names(empty, "people") == ["ada", "grace", "lynn"]


def test_a_statement_reads_a_registered_frame_like_any_other_table(
    empty: zudb.Connection,
) -> None:
    empty.register(
        "people",
        {"uid": [1, 2, 3], "name": ["ada", "grace", "lynn"], "age": [36, 45, 52]},
    )
    rows = empty.execute("MATCH (p:people) WHERE p.age > 40 RETURN p.name AS name").fetchall()
    assert rows == [("grace",), ("lynn",)]


def test_a_pandas_frame_goes_in(empty: zudb.Connection) -> None:
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"uid": [1, 2], "name": ["ada", "grace"]})
    assert empty.register("people", frame) == 2
    assert names(empty, "people") == ["ada", "grace"]


def test_a_polars_frame_goes_in(empty: zudb.Connection) -> None:
    """polars holds strings as views, which is a third Arrow layout and
    the same characters."""
    pl = pytest.importorskip("polars")
    frame = pl.DataFrame({"uid": [1, 2], "name": ["ada", "grace"]})
    assert empty.register("people", frame) == 2
    assert names(empty, "people") == ["ada", "grace"]


def test_a_pyarrow_table_goes_in(empty: zudb.Connection) -> None:
    assert empty.register("people", pa.table({"uid": [1, 2], "name": ["ada", "grace"]})) == 2
    assert names(empty, "people") == ["ada", "grace"]


def test_a_stream_of_several_batches_is_one_frame(empty: zudb.Connection) -> None:
    """A column of a table is one run of bytes, so this is the one shape
    that costs a memcpy per column on the way in."""
    schema = pa.schema([("uid", pa.int64()), ("name", pa.string())])
    batches = [
        pa.record_batch([pa.array([1, 2]), pa.array(["ada", "grace"])], schema=schema),
        pa.record_batch([pa.array([3]), pa.array(["lynn"])], schema=schema),
    ]
    reader = pa.RecordBatchReader.from_batches(schema, batches)
    assert empty.register("people", reader) == 3
    assert names(empty, "people") == ["ada", "grace", "lynn"]


def test_a_sliced_column_is_read_from_the_row_it_starts_at(empty: zudb.Connection) -> None:
    """A slice is an array with a row offset, which is the one thing a
    bare pointer cannot say, so it is copied down to itself first."""
    frame = pa.table({"uid": pa.array([1, 2, 3, 4]), "name": pa.array(["a", "b", "c", "d"])})
    assert empty.register("people", frame.slice(1, 2)) == 2
    assert names(empty, "people") == ["b", "c"]


def test_every_kind_of_column_a_row_can_hold_arrives_as_itself(empty: zudb.Connection) -> None:
    frame = pa.table(
        {
            "yes": pa.array([True, False]),
            "small": pa.array([1, 2], pa.int8()),
            "wide": pa.array([3, 4], pa.uint32()),
            "narrow": pa.array([1.5, 2.5], pa.float32()),
            "word": pa.array(["a", "b"]),
            "day": pa.array([datetime.date(2024, 1, 1), datetime.date(2024, 2, 1)]),
            "clock": pa.array([datetime.time(1, 2, 3), datetime.time(4, 5, 6)], pa.time64("us")),
            "moment": pa.array(
                [datetime.datetime(2024, 1, 1, 1, 2, 3), datetime.datetime(2024, 2, 1)]
            ),
            "span": pa.array([datetime.timedelta(seconds=90)] * 2, pa.duration("us")),
        }
    )
    assert empty.register("kinds", frame) == 2
    row = empty.execute(
        "MATCH (k:kinds) RETURN k.yes AS yes, k.small AS small, k.wide AS wide, "
        "k.narrow AS narrow, k.word AS word, k.day AS day, k.clock AS clock, "
        "k.moment AS moment, k.span AS span"
    ).fetchone()
    assert row == (
        True,
        1,
        3,
        1.5,
        "a",
        datetime.date(2024, 1, 1),
        datetime.time(1, 2, 3),
        datetime.datetime(2024, 1, 1, 1, 2, 3),
        zudb.Duration(nanoseconds=90_000_000_000),
    )


def held_column(values: list[int]) -> tuple[bytearray, object]:
    """An Arrow column over memory this test keeps and can write into.

    `from_buffers` is the way to hand pyarrow bytes that are already
    laid out, so what comes back points at the bytearray rather than at
    a copy of it. There is no validity buffer, which is what `None`
    first says, because a column of a row of this engine holds a value
    everywhere.
    """
    held = bytearray(struct.pack(f"<{len(values)}q", *values))
    return held, pa.Array.from_buffers(pa.int64(), len(values), [None, pa.py_buffer(held)])


def test_a_frame_is_read_where_it_lies_and_not_copied(empty: zudb.Connection) -> None:
    """The whole point of the call, proved the only way it can be.

    The column is written into between two statements and the second one
    answers the new number, which no copy taken at registration could
    do.
    """
    held, column = held_column([10, 20, 30])
    empty.register("numbers", pa.table({"n": column}))
    assert empty.execute("MATCH (x:numbers) RETURN sum(x.n) AS total").fetchone() == (60,)
    struct.pack_into("<q", held, 0, 1000)
    assert empty.execute("MATCH (x:numbers) RETURN sum(x.n) AS total").fetchone() == (1050,)


def test_the_bytes_go_back_when_the_frame_is_unregistered(empty: zudb.Connection) -> None:
    """A bytearray refuses to resize while anything is holding a buffer
    of it, so whether it will is exactly the question of whether the
    engine has let go."""
    held, column = held_column([1, 2, 3])
    empty.register("numbers", pa.table({"n": column}))
    del column
    with pytest.raises(BufferError):
        held.append(0)
    empty.unregister("numbers")
    gc.collect()
    held.append(0)


def test_a_dictionary_of_lists_is_copied_because_a_list_is_not_a_column(
    empty: zudb.Connection,
) -> None:
    """The one way in that does copy, and the reason it has to."""
    lists = {"uid": [1], "name": ["ada"]}
    empty.register("people", lists)
    lists["name"][0] = "grace"
    assert names(empty, "people") == ["ada"]


def test_a_frame_belongs_to_the_connection_that_registered_it(
    empty: zudb.Connection, tmp_path: Path
) -> None:
    """Nothing is written to the database, so another program opening the
    same file has never heard of it."""
    empty.register("people", {"uid": [1], "name": ["ada"]})
    with zudb.connect(tmp_path / "empty.zu1") as other:
        assert other.registered == []
        assert names(other, "people") == []


def test_registered_says_what_is_registered_here(empty: zudb.Connection) -> None:
    assert empty.registered == []
    empty.register("second", {"a": [1]})
    empty.register("first", {"a": [1]})
    assert empty.registered == ["first", "second"]


def test_registering_a_name_again_replaces_what_it_stands_for(empty: zudb.Connection) -> None:
    """Which is what rerunning a cell means by it."""
    empty.register("people", {"uid": [1, 2], "name": ["ada", "grace"]})
    assert empty.register("people", {"uid": [3], "name": ["lynn"]}) == 1
    assert names(empty, "people") == ["lynn"]


def test_a_name_registered_again_may_hold_a_different_shape(empty: zudb.Connection) -> None:
    """A frame is not a table, so nothing about the first registration
    survives the second."""
    empty.register("people", {"uid": [1], "name": ["ada"]})
    empty.register("people", {"name": ["grace"], "age": [45]})
    row = empty.execute("MATCH (p:people) RETURN p.name AS name, p.age AS age").fetchone()
    assert row == ("grace", 45)


def test_unregister_takes_the_name_away(empty: zudb.Connection) -> None:
    empty.register("people", {"uid": [1, 2], "name": ["ada", "grace"]})
    empty.unregister("people")
    assert empty.registered == []
    assert names(empty, "people") == []


def test_a_name_that_was_unregistered_can_be_registered_again(empty: zudb.Connection) -> None:
    empty.register("people", {"uid": [1], "name": ["ada"]})
    empty.unregister("people")
    assert empty.register("people", {"uid": [2], "name": ["grace"]}) == 1
    assert names(empty, "people") == ["grace"]


def test_unregistering_twice_is_refused(empty: zudb.Connection) -> None:
    empty.register("people", {"a": [1]})
    empty.unregister("people")
    with pytest.raises(zudb.ProgrammingError, match="nothing is registered here"):
        empty.unregister("people")


def test_unregistering_a_table_nobody_registered_is_refused(social: zudb.Connection) -> None:
    with pytest.raises(zudb.ProgrammingError, match="nothing is registered here"):
        social.unregister("person")


def test_registering_over_a_table_of_the_database_is_refused(social: zudb.Connection) -> None:
    """A statement naming it would mean the stored one."""
    with pytest.raises(zudb.ProgrammingError, match="already a table of this database"):
        social.register("person", {"uid": [1]})


def test_nothing_writes_to_a_registered_frame(empty: zudb.Connection) -> None:
    """It is the caller's memory, read where it lies, and a statement
    that wrote into it would be writing into the DataFrame."""
    empty.register("people", {"uid": [1], "name": ["ada"]})
    with pytest.raises(zudb.TransactionError, match="never written"):
        empty.execute("INSERT (p:people {uid: 2, name: 'grace'})")
    with pytest.raises(zudb.TransactionError, match="never written"):
        empty.execute("MATCH (p:people) DETACH DELETE p")


def test_a_null_anywhere_is_refused_by_column_and_row(empty: zudb.Connection) -> None:
    """A property that is null is one no row of this engine can hold."""
    frame = pa.table({"uid": pa.array([1, 2, 3]), "name": pa.array(["ada", None, "lynn"])})
    with pytest.raises(ValueError, match="column 'name' has no value at row 1"):
        empty.register("people", frame)


def test_a_frame_with_no_rows_registers_and_matches_nothing(empty: zudb.Connection) -> None:
    """A frame knows what its columns are without being told by a row, so
    a filter that came back empty is still a table to match on."""
    assert empty.register("people", pa.table({"uid": pa.array([], pa.int64())})) == 0
    assert empty.execute("MATCH (p:people) RETURN count(*) AS n").fetchone() == (0,)


def test_a_frame_with_no_columns_is_refused(empty: zudb.Connection) -> None:
    with pytest.raises(zudb.ProgrammingError, match="no columns"):
        empty.register("people", {})


def test_a_name_a_statement_could_not_carry_is_refused(empty: zudb.Connection) -> None:
    with pytest.raises(zudb.ProgrammingError, match="not a name a statement can carry"):
        empty.register("two words", {"a": [1]})
    with pytest.raises(zudb.ProgrammingError, match="a column of a registered frame"):
        empty.register("people", {"two words": [1]})


def test_a_zoned_timestamp_is_refused_with_what_to_do_about_it(empty: zudb.Connection) -> None:
    frame = pa.table({"when": pa.array([1_700_000_000_000_000], pa.timestamp("us", "UTC"))})
    with pytest.raises(TypeError, match="nowhere to keep"):
        empty.register("moments", frame)


def test_a_column_of_bytes_is_refused(empty: zudb.Connection) -> None:
    """Naming one would be naming data no statement reads back."""
    with pytest.raises(TypeError, match="column of bytes"):
        empty.register("blobs", pa.table({"raw": pa.array([b"x"])}))


def test_an_integer_too_large_for_a_column_is_refused_by_row(empty: zudb.Connection) -> None:
    """Checked once, at registration, so that reading a frame cannot
    fail: the engine's lane is signed and this value is not in it."""
    frame = pa.table({"big": pa.array([1, 2**63], pa.uint64())})
    with pytest.raises(zudb.ProgrammingError, match="at row 1"):
        empty.register("numbers", frame)


def test_something_that_is_not_a_frame_at_all_is_refused_with_the_list(
    empty: zudb.Connection,
) -> None:
    with pytest.raises(TypeError, match="__arrow_c_stream__"):
        empty.register("people", [1, 2, 3])


def test_registering_inside_a_transaction_is_refused(empty: zudb.Connection) -> None:
    """A frame is registered on the session, which is the thing a
    transaction is running on, and a rollback has nothing to say about
    memory the caller owns."""
    with empty.transaction():
        with pytest.raises(zudb.TransactionError, match="not inside a transaction"):
            empty.register("people", {"a": [1, 2]})


def test_a_closed_connection_registers_nothing(empty: zudb.Connection) -> None:
    empty.close()
    with pytest.raises(zudb.ProgrammingError, match="closed"):
        empty.register("people", {"a": [1]})


@pytest.mark.timing
def test_registering_costs_the_same_whatever_the_frame_holds(empty: zudb.Connection) -> None:
    """Nothing is copied, so nothing about the call is per row.

    Five million rows against ten, and the budget is a millisecond
    against the 30 microseconds either of them takes on a laptop. It is
    here to catch a way in that started walking the rows rather than to
    hold a number, which is why it is loose by a factor of thirty.
    """
    frames = {rows: pa.table({"n": pa.array(range(rows))}) for rows in (10, 5_000_000)}
    best = {}
    for rows, frame in frames.items():
        best[rows] = float("inf")
        for _ in range(5):
            started = time.perf_counter()
            empty.register("numbers", frame)
            best[rows] = min(best[rows], time.perf_counter() - started)
    assert best[5_000_000] < 1e-3, f"registering 5m rows took {best[5_000_000] * 1e3:.1f} ms"
