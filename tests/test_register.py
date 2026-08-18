"""Frames, registered under a name a statement can match on.

The point of the call is that a DataFrame a program already has becomes
something a statement can read, so most of these register one and then
match it. The rest are the refusals, which are where the engine shows
through: a table's columns are declared by its first row and nothing
alters or drops one, so a name that has been used holds the shape it was
used with.

pandas and polars are asked for by the tests that need them and skipped
when they are not installed, because the wheel depends on neither.
"""

from __future__ import annotations

import datetime
import time

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
    """Read a batch at a time, so a frame bigger than memory is not two
    frames."""
    schema = pa.schema([("uid", pa.int64()), ("name", pa.string())])
    batches = [
        pa.record_batch([pa.array([1, 2]), pa.array(["ada", "grace"])], schema=schema),
        pa.record_batch([pa.array([3]), pa.array(["lynn"])], schema=schema),
    ]
    reader = pa.RecordBatchReader.from_batches(schema, batches)
    assert empty.register("people", reader) == 3
    assert names(empty, "people") == ["ada", "grace", "lynn"]


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


def test_a_frame_is_a_snapshot_and_not_a_view(empty: zudb.Connection) -> None:
    """The one thing about this being a copy that a caller has to know."""
    held = {"uid": [1], "name": ["ada"]}
    empty.register("people", held)
    held["name"][0] = "grace"
    assert names(empty, "people") == ["ada"]


def test_registered_says_what_is_registered_here(empty: zudb.Connection) -> None:
    assert empty.registered == []
    empty.register("second", {"a": [1]})
    empty.register("first", {"a": [1]})
    assert empty.registered == ["first", "second"]


def test_registering_a_name_again_replaces_the_rows(empty: zudb.Connection) -> None:
    """Which is what rerunning a cell means by it."""
    empty.register("people", {"uid": [1, 2], "name": ["ada", "grace"]})
    assert empty.register("people", {"uid": [3], "name": ["lynn"]}) == 1
    assert names(empty, "people") == ["lynn"]


def test_unregister_takes_the_rows_back_out(empty: zudb.Connection) -> None:
    empty.register("people", {"uid": [1, 2], "name": ["ada", "grace"]})
    empty.unregister("people")
    assert empty.registered == []
    assert names(empty, "people") == []


def test_a_name_that_was_unregistered_can_be_registered_again(empty: zudb.Connection) -> None:
    """The empty table it left behind is still this connection's."""
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
    """A frame knows nothing about the rows that were already there."""
    with pytest.raises(zudb.ProgrammingError, match="already a table of this database"):
        social.register("person", {"uid": [1]})


def test_a_name_that_has_been_used_keeps_the_columns_it_was_used_with(
    empty: zudb.Connection,
) -> None:
    """No statement of this engine alters a table, so this is refused at
    the call rather than halfway through the rows."""
    empty.register("people", {"uid": [1], "name": ["ada"]})
    with pytest.raises(zudb.ProgrammingError, match="register it under another name"):
        empty.register("people", {"uid": [1], "name": ["ada"], "age": [36]})
    assert names(empty, "people") == ["ada"]


def test_a_null_anywhere_is_refused_by_column_and_row(empty: zudb.Connection) -> None:
    """A property that is null is one no row of this engine can hold."""
    frame = pa.table({"uid": pa.array([1, 2, 3]), "name": pa.array(["ada", None, "lynn"])})
    with pytest.raises(ValueError, match="column 'name' has no value at row 1"):
        empty.register("people", frame)


def test_a_frame_with_no_rows_is_refused(empty: zudb.Connection) -> None:
    """The columns of a table are declared by the first row written into
    it, and there is none."""
    with pytest.raises(zudb.ProgrammingError, match="no rows"):
        empty.register("people", pa.table({"uid": pa.array([], pa.int64())}))


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
    """Writing one would be writing data no statement reads back."""
    with pytest.raises(TypeError, match="column of bytes"):
        empty.register("blobs", pa.table({"raw": pa.array([b"x"])}))


def test_an_integer_too_large_for_a_column_is_refused_by_row(empty: zudb.Connection) -> None:
    frame = pa.table({"big": pa.array([1, 2**63], pa.uint64())})
    with pytest.raises(ValueError, match="at row 1"):
        empty.register("numbers", frame)


def test_something_that_is_not_a_frame_at_all_is_refused_with_the_list(
    empty: zudb.Connection,
) -> None:
    with pytest.raises(TypeError, match="__arrow_c_stream__"):
        empty.register("people", [1, 2, 3])


def test_registering_inside_a_transaction_is_refused(empty: zudb.Connection) -> None:
    """Its rows go in through the appender, whose batches are commits of
    their own that no rollback reaches."""
    with empty.transaction():
        with pytest.raises(zudb.ProgrammingError, match="own commits"):
            empty.register("people", {"a": [1, 2]})


def test_a_closed_connection_registers_nothing(empty: zudb.Connection) -> None:
    empty.close()
    with pytest.raises(zudb.ProgrammingError, match="closed"):
        empty.register("people", {"a": [1]})


def test_reading_the_frame_costs_a_memcpy_and_not_a_python_object_per_cell(
    empty: zudb.Connection,
) -> None:
    """The read is the part this client owns, so it is the part measured.

    A frame whose column name no statement could carry is read in full
    and then refused, which times the way in without the write behind
    it. The budget is loose by a factor of fifty against the 1 ms this
    takes on a laptop, because it is here to catch a way in that started
    making a Python object per cell rather than to hold a number.
    """
    rows = 200_000
    frame = pa.table({"two words": pa.array(range(rows))})
    best = float("inf")
    for _ in range(3):
        started = time.perf_counter()
        with pytest.raises(zudb.ProgrammingError):
            empty.register("numbers", frame)
        best = min(best, time.perf_counter() - started)
    assert best < 50e-3, f"reading {rows} rows took {best * 1e3:.0f} ms"
