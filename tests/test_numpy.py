"""A result as numpy arrays.

The third way out of a result, and the one with the fewest dependencies
under it: numpy and nothing else. What these check is the mapping, which
is where a columnar export goes wrong quietly, and the two claims worth
making about it, which are that the buffer is not copied on the way and
that a null does not become a number.
"""

from __future__ import annotations

import datetime

import pytest
import zudb

np = pytest.importorskip("numpy")


def test_every_column_comes_back_under_its_own_name(social: zudb.Connection) -> None:
    result = social.execute("MATCH (p:person) RETURN p.uid AS uid, p.name AS name ORDER BY p.uid")
    arrays = result.fetchnumpy()
    assert list(arrays) == ["uid", "name"], "the order the statement projected them in"
    assert arrays["uid"].tolist() == [10, 20, 30]
    assert arrays["name"].tolist() == ["ada", "grace", "kay"]


@pytest.mark.parametrize(
    "statement,params,dtype,answer",
    [
        ("RETURN 1 AS v", {}, "int64", 1),
        ("RETURN 1.5 AS v", {}, "float64", 1.5),
        ("RETURN true AS v", {}, "bool", True),
        ("RETURN 'ada' AS v", {}, "object", "ada"),
        ("RETURN $v AS v", {"v": datetime.date(2020, 1, 2)}, "datetime64[D]", "2020-01-02"),
        (
            "RETURN $v AS v",
            {"v": datetime.datetime(2020, 1, 2, 3, 4, 5)},
            "datetime64[ns]",
            "2020-01-02T03:04:05.000000000",
        ),
        ("RETURN $v AS v", {"v": datetime.time(1, 2, 3)}, "timedelta64[ns]", 3723000000000),
        (
            "RETURN $v AS v",
            {"v": datetime.timedelta(hours=1, microseconds=5)},
            "timedelta64[ns]",
            3600000005000,
        ),
        ("RETURN DURATION 'P1Y2M' AS v", {}, "timedelta64[M]", 14),
        ("RETURN null AS v", {}, "object", None),
    ],
)
def test_a_column_becomes_the_numpy_type_that_holds_it(
    empty: zudb.Connection, statement: str, params: dict, dtype: str, answer: object
) -> None:
    array = empty.execute(statement, params).fetchnumpy()["v"]
    assert array.dtype == np.dtype(dtype)
    assert array[0] == np.array([answer], dtype=dtype)[0]


def test_a_time_of_day_is_nanoseconds_since_midnight(empty: zudb.Connection) -> None:
    """numpy has no clock reading, and this is the reading and not a stand-in."""
    array = empty.execute("RETURN $v AS v", {"v": datetime.time(1, 2, 3, 500000)}).fetchnumpy()["v"]
    assert array[0] == np.timedelta64(3723500000000, "ns")
    assert array[0].astype("timedelta64[s]") == np.timedelta64(3723, "s")


def test_a_datetime_with_an_offset_comes_back_as_the_instant_in_utc(
    empty: zudb.Connection,
) -> None:
    """numpy has nowhere to keep the offset, and the instant is what the buffer holds."""
    zone = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    when = datetime.datetime(2020, 1, 2, 3, 4, 5, tzinfo=zone)
    array = empty.execute("RETURN $v AS v", {"v": when}).fetchnumpy()["v"]
    assert array[0] == np.datetime64("2020-01-01T21:34:05.000000000")


def test_a_time_with_an_offset_is_refused_rather_than_moved(empty: zudb.Connection) -> None:
    zone = datetime.timezone(datetime.timedelta(hours=2))
    with pytest.raises(TypeError, match="time with an offset"):
        empty.execute("RETURN $v AS v", {"v": datetime.time(1, 2, 3, tzinfo=zone)}).fetchnumpy()


@pytest.mark.parametrize(
    "statement,dtype,answer",
    [
        ("UNWIND [1, null, 3] AS v RETURN v", "int64", [1, 3]),
        ("UNWIND [1.5, null, 3.5] AS v RETURN v", "float64", [1.5, 3.5]),
        ("UNWIND [true, null, false] AS v RETURN v", "bool", [True, False]),
    ],
)
def test_a_column_with_a_null_in_it_is_masked(
    empty: zudb.Connection, statement: str, dtype: str, answer: list
) -> None:
    """numpy has no missing integer, so the mask is where the null goes."""
    array = empty.execute(statement).fetchnumpy()["v"]
    assert isinstance(array, np.ma.MaskedArray)
    assert array.dtype == np.dtype(dtype)
    assert array.mask.tolist() == [False, True, False]
    assert array.compressed().tolist() == answer


def test_a_column_with_nothing_missing_is_a_plain_array(empty: zudb.Connection) -> None:
    """A mask nothing is masked by is a second buffer nobody asked for."""
    array = empty.execute("UNWIND [1, 2, 3] AS v RETURN v").fetchnumpy()["v"]
    assert not isinstance(array, np.ma.MaskedArray)
    assert array.tolist() == [1, 2, 3]


@pytest.mark.parametrize(
    "statement,answer",
    [
        ("UNWIND ['a', null, 'ccc'] AS v RETURN v", ["a", None, "ccc"]),
        ("UNWIND [[1, 2], null] AS v RETURN v", [[1, 2], None]),
        ("UNWIND [{a: 1}, null] AS v RETURN v", [{"a": 1}, None]),
    ],
)
def test_an_object_column_carries_the_null_in_the_cell(
    empty: zudb.Connection, statement: str, answer: list
) -> None:
    """An object array has somewhere to put `None`, so a mask would say it twice."""
    array = empty.execute(statement).fetchnumpy()["v"]
    assert array.dtype == np.dtype("object")
    assert not isinstance(array, np.ma.MaskedArray)
    assert array.tolist() == answer


def test_nodes_and_rels_come_back_as_the_objects_the_rows_hold(loaded: zudb.Connection) -> None:
    statement = "MATCH (p:person)-[k:knows]->(q:person) RETURN p AS p, k AS k"
    arrays = loaded.execute(statement).fetchnumpy()
    assert arrays["p"].dtype == np.dtype("object")
    assert all(isinstance(node, zudb.Node) for node in arrays["p"])
    assert all(isinstance(rel, zudb.Rel) for rel in arrays["k"])


def test_the_buffer_is_moved_into_numpy_and_not_copied(empty: zudb.Connection) -> None:
    """The claim the whole path is for: the array is the engine's buffer.

    `owndata` false with a base object is numpy's own way of saying the
    memory came from somewhere else and is being kept alive by whoever
    it came from, which here is the extension holding the `Vec`.
    """
    array = empty.execute("UNWIND [1, 2, 3] AS v RETURN v").fetchnumpy()["v"]
    assert not array.flags.owndata
    assert array.base is not None
    # And it is the caller's to write in: the result kept no second
    # reference to it, so there is nothing for a write to corrupt.
    assert array.flags.writeable
    array[0] = 99
    assert array.tolist() == [99, 2, 3]


def test_a_result_with_no_rows_gives_empty_arrays_of_the_right_type(
    empty: zudb.Connection,
) -> None:
    arrays = empty.execute("UNWIND [] AS v RETURN v").fetchnumpy()
    assert list(arrays) == ["v"]
    assert len(arrays["v"]) == 0


def test_a_statement_that_writes_gives_no_columns(empty: zudb.Connection) -> None:
    assert empty.execute("INSERT (p:person {uid: 1, name: 'ada'})").fetchnumpy() == {}


def test_two_columns_of_the_same_name_are_refused(empty: zudb.Connection) -> None:
    """A dict holds one of each name, and dropping the other one quietly is worse."""
    with pytest.raises(ValueError, match="two columns called 'v'"):
        empty.execute("RETURN 1 AS v, 2 AS v").fetchnumpy()


def test_reading_the_columns_does_not_move_the_cursor(social: zudb.Connection) -> None:
    """Like every other way of reading a result whole."""
    result = social.execute("MATCH (p:person) RETURN p.uid AS uid ORDER BY p.uid")
    assert len(result.fetchnumpy()["uid"]) == 3
    assert result.fetchone() == (10,)
