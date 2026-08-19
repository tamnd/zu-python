"""A result as Arrow columns, and as the frames built on them.

The interface is the PyCapsule one, so most of these go through
pyarrow: it is the reference consumer, and what it reads out of the
capsule is what pandas, polars and everything else read too. The tests
that need pandas or polars say so and skip when they are not installed,
because the wheel depends on neither.
"""

from __future__ import annotations

import datetime
import threading
from pathlib import Path

import pytest
import zudb

pa = pytest.importorskip("pyarrow")


def test_a_result_is_a_table(loaded: zudb.Connection) -> None:
    table = loaded.execute("MATCH (p:person) RETURN p.uid AS uid, p.name AS name").to_arrow()
    assert table.num_rows == 3
    assert table.column_names == ["uid", "name"]
    assert table.to_pylist() == [
        {"uid": 10, "name": "ada"},
        {"uid": 20, "name": "grace"},
        {"uid": 30, "name": "kay"},
    ]


def test_the_capsule_is_the_interface(loaded: zudb.Connection) -> None:
    result = loaded.execute("MATCH (p:person) RETURN p.uid AS uid")
    capsule = result.__arrow_c_stream__()
    assert type(capsule).__name__ == "PyCapsule"
    # Consumed through the protocol rather than through `to_arrow`,
    # which is what every library that is not pyarrow does.
    assert pa.table(result).num_rows == 3


def test_a_requested_schema_is_accepted_and_the_result_is_what_it_is(
    loaded: zudb.Connection,
) -> None:
    # The protocol lets a consumer ask for a schema and lets a producer
    # ignore it, so asking for an int32 column is not an error and does
    # not get one either.
    result = loaded.execute("MATCH (p:person) RETURN p.uid AS uid")
    wanted = pa.schema([pa.field("uid", pa.int32())])
    capsule = result.__arrow_c_stream__(wanted.__arrow_c_schema__())
    reader = pa.RecordBatchReader._import_from_c_capsule(capsule)
    assert reader.schema.field("uid").type == pa.int64()
    assert reader.read_all().num_rows == 3


@pytest.mark.parametrize(
    "statement,params,arrow_type,answer",
    [
        ("RETURN 1 AS v", {}, pa.int64(), 1),
        ("RETURN 1.5 AS v", {}, pa.float64(), 1.5),
        ("RETURN 'ada' AS v", {}, pa.string(), "ada"),
        ("RETURN true AS v", {}, pa.bool_(), True),
        ("RETURN null AS v", {}, pa.null(), None),
        ("RETURN [1, 2] AS v", {}, pa.list_(pa.field("item", pa.int64())), [1, 2]),
        (
            "RETURN $v AS v",
            {"v": datetime.date(2024, 1, 2)},
            pa.date32(),
            datetime.date(2024, 1, 2),
        ),
        (
            "RETURN $v AS v",
            {"v": datetime.time(1, 2, 3)},
            pa.time64("ns"),
            datetime.time(1, 2, 3),
        ),
        (
            "RETURN $v AS v",
            {"v": datetime.datetime(2024, 1, 2, 3, 4, 5)},
            pa.timestamp("ns"),
            datetime.datetime(2024, 1, 2, 3, 4, 5),
        ),
        (
            "RETURN $v AS v",
            {"v": datetime.timedelta(hours=1)},
            pa.duration("ns"),
            datetime.timedelta(hours=1),
        ),
    ],
)
def test_a_value_becomes_the_arrow_type_it_is(
    empty: zudb.Connection,
    statement: str,
    params: dict,
    arrow_type: object,
    answer: object,
) -> None:
    table = empty.execute(statement, params).to_arrow()
    assert table.schema.field("v").type == arrow_type
    got = table.column("v")[0].as_py()
    if isinstance(answer, datetime.datetime):
        # pyarrow gives back a pandas Timestamp when pandas is there
        # and a datetime when it is not, and both compare equal to the
        # datetime that went in.
        assert got == answer
    else:
        assert got == answer


def test_a_zoned_datetime_carries_the_offset_it_was_written_with(empty: zudb.Connection) -> None:
    zone = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    written = datetime.datetime(2024, 1, 2, 3, 4, 5, tzinfo=zone)
    table = empty.execute("RETURN $v AS v", {"v": written}).to_arrow()
    assert table.schema.field("v").type == pa.timestamp("ns", tz="+05:30")
    assert table.column("v")[0].as_py() == written


def test_a_year_month_duration_is_an_interval_of_months(empty: zudb.Connection) -> None:
    # Arrow has a year-month interval and pyarrow cannot build a Python
    # array of one, so what goes across is the month-day-nano interval
    # every reader implements, with the days and nanoseconds zero.
    table = empty.execute("RETURN $v AS v", {"v": zudb.Duration(months=14)}).to_arrow()
    assert table.schema.field("v").type == pa.month_day_nano_interval()
    assert table.column("v")[0].as_py() == pa.MonthDayNano([14, 0, 0])


def test_a_time_with_an_offset_has_no_arrow_type(empty: zudb.Connection) -> None:
    zone = datetime.timezone(datetime.timedelta(hours=2))
    with pytest.raises(TypeError, match="time with an offset, which Arrow has no type for"):
        empty.execute("RETURN $v AS v", {"v": datetime.time(1, 2, 3, tzinfo=zone)}).to_arrow()


def test_a_node_is_a_struct_of_table_and_offset(loaded: zudb.Connection) -> None:
    table = loaded.execute("MATCH (p:person) RETURN p AS node").to_arrow()
    assert table.schema.field("node").type == pa.struct(
        [pa.field("table", pa.string()), pa.field("offset", pa.uint64())]
    )
    assert table.column("node")[0].as_py() == {"table": "person", "offset": 0}


def test_a_rel_is_a_struct_of_its_ends(loaded: zudb.Connection) -> None:
    table = loaded.execute("MATCH ()-[r:knows]->() RETURN r AS edge").to_arrow()
    assert table.column("edge").to_pylist() == [
        {"table": "knows", "src": 0, "dst": 1, "ord": 0},
        {"table": "knows", "src": 1, "dst": 2, "ord": 1},
    ]


def test_a_path_is_its_nodes_and_its_rels(loaded: zudb.Connection) -> None:
    table = loaded.execute(
        "MATCH q = (a:person)-[:knows]->(b:person) WHERE a.uid = 10 RETURN q AS walk"
    ).to_arrow()
    assert table.column("walk").to_pylist() == [
        {
            "nodes": [{"table": "person", "offset": 0}, {"table": "person", "offset": 1}],
            "rels": [{"table": "knows", "src": 0, "dst": 1, "ord": 0}],
        }
    ]


def test_a_record_is_a_struct_of_its_fields(empty: zudb.Connection) -> None:
    table = empty.execute("RETURN {a: 1, b: 'x'} AS rec").to_arrow()
    assert table.schema.field("rec").type == pa.struct(
        [pa.field("a", pa.int64()), pa.field("b", pa.string())]
    )
    assert table.column("rec")[0].as_py() == {"a": 1, "b": "x"}


@pytest.mark.parametrize(
    "statement,answer",
    [
        ("UNWIND [1, null, 3] AS v RETURN v", [1, None, 3]),
        ("UNWIND ['a', null] AS v RETURN v", ["a", None]),
        ("UNWIND [[1, 2], null, []] AS v RETURN v", [[1, 2], None, []]),
        ("UNWIND [{a: 1}, null] AS v RETURN v", [{"a": 1}, None]),
    ],
)
def test_a_null_is_a_null_in_the_column_it_is_in(
    empty: zudb.Connection, statement: str, answer: list
) -> None:
    assert empty.execute(statement).to_arrow().column("v").to_pylist() == answer


def test_a_column_of_integers_and_floats_is_a_column_of_floats(empty: zudb.Connection) -> None:
    table = empty.execute("UNWIND [1, 2.5] AS v RETURN v").to_arrow()
    assert table.schema.field("v").type == pa.float64()
    assert table.column("v").to_pylist() == [1.0, 2.5]


def test_a_column_of_two_types_is_refused(empty: zudb.Connection) -> None:
    with pytest.raises(TypeError, match="mixes integers and strings at row 1"):
        empty.execute("UNWIND [1, 'x'] AS v RETURN v").to_arrow()


def test_a_list_of_two_types_is_refused(empty: zudb.Connection) -> None:
    with pytest.raises(TypeError, match="mixes integers and strings"):
        empty.execute("RETURN [1, 'x'] AS v").to_arrow()


def test_a_result_with_no_rows_still_has_its_columns(loaded: zudb.Connection) -> None:
    table = loaded.execute("MATCH (p:person) WHERE p.uid = 99 RETURN p.uid AS uid").to_arrow()
    assert table.num_rows == 0
    assert table.column_names == ["uid"]
    # Nothing said what the column holds, so it holds nothing, which
    # Arrow has a type for.
    assert table.schema.field("uid").type == pa.null()


def test_the_batches_are_the_same_rows(tmp_path: Path) -> None:
    rows = 70_000
    zudb.load(tmp_path / "big.zu1", nodes="n", rels="r", columns={"uid": list(range(rows))})
    with zudb.connect(tmp_path / "big.zu1", read_only=True) as conn:
        reader = conn.execute("MATCH (x:n) RETURN x.uid AS uid").record_batches()
        batches = list(reader)
    # More rows than fit in one batch, so this is two of them, and they
    # add up to the result.
    assert len(batches) == 2
    assert sum(batch.num_rows for batch in batches) == rows


def test_a_result_reads_as_arrow_and_as_objects_and_says_the_same_thing(
    loaded: zudb.Connection,
) -> None:
    statement = "MATCH (p:person) RETURN p.uid AS uid, p.name AS name"
    objects = loaded.execute(statement).fetchall()
    columns = loaded.execute(statement).to_arrow().to_pylist()
    assert [tuple(row.values()) for row in columns] == objects


def test_a_dataframe_comes_back_with_arrow_dtypes(loaded: zudb.Connection) -> None:
    pytest.importorskip("pandas")
    frame = loaded.execute("MATCH (p:person) RETURN p.uid AS uid, p.name AS name").to_pandas()
    assert list(frame.columns) == ["uid", "name"]
    assert [str(dtype) for dtype in frame.dtypes] == ["int64[pyarrow]", "string[pyarrow]"]
    assert frame["name"].tolist() == ["ada", "grace", "kay"]


def test_polars_reads_the_same_rows(loaded: zudb.Connection) -> None:
    polars = pytest.importorskip("polars")
    frame = loaded.execute("MATCH (p:person) RETURN p.uid AS uid, p.name AS name").to_polars()
    assert isinstance(frame, polars.DataFrame)
    assert frame.columns == ["uid", "name"]
    assert frame["name"].to_list() == ["ada", "grace", "kay"]


@pytest.mark.timing
def test_python_keeps_running_while_a_result_becomes_arrow(tmp_path: Path) -> None:
    rows = 400_000
    zudb.load(
        tmp_path / "big.zu1",
        nodes="person",
        rels="knows",
        columns={"uid": list(range(rows)), "name": [f"p{uid}" for uid in range(rows)]},
    )
    with zudb.connect(tmp_path / "big.zu1", read_only=True) as conn:
        result = conn.execute("MATCH (x:person) RETURN x.uid AS uid, x.name AS name, x AS node")
        # Imported before the thread starts, so what the loop measures
        # is the copy and not pyarrow's first import.
        result.to_arrow()
        tables: list[object] = []
        ticks = 0
        done = threading.Event()

        def run() -> None:
            tables.append(result.to_arrow())
            done.set()

        worker = threading.Thread(target=run)
        worker.start()
        while not done.is_set():
            ticks += 1
        worker.join(timeout=120)
    assert not worker.is_alive()
    assert tables[0].num_rows == rows
    # Half a million turns on this machine, and none at all if the
    # GIL were held for the copy.
    assert ticks > 10_000, f"the main thread only got {ticks} turns"
