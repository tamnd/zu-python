"""What an engine value is once it is a Python object."""

from __future__ import annotations

import datetime

import pytest
import zudb


def test_a_node_carries_its_table_and_offset(social: zudb.Connection) -> None:
    rows = social.execute("MATCH (p:person) RETURN p ORDER BY p.uid")
    nodes = [node for (node,) in rows]
    assert [node.table for node in nodes] == ["person"] * 3
    assert [node.offset for node in nodes] == [0, 1, 2]
    assert repr(nodes[0]) == "Node(person, 0)"


def test_two_reads_of_one_node_are_equal_and_hash_alike(social: zudb.Connection) -> None:
    one = social.execute("MATCH (p:person) WHERE p.uid = 10 RETURN p").fetchone()[0]
    two = social.execute("MATCH (p:person) WHERE p.uid = 10 RETURN p").fetchone()[0]
    assert one == two
    assert len({one, two}) == 1


@pytest.mark.parametrize(
    "statement,answer",
    [
        ("RETURN 1 AS v", 1),
        ("RETURN 1.5 AS v", 1.5),
        ("RETURN 'ada' AS v", "ada"),
        ("RETURN true AS v", True),
        ("RETURN false AS v", False),
        ("RETURN null AS v", None),
        ("RETURN [1, 2, 3] AS v", [1, 2, 3]),
        ("RETURN [[1], [2, 3]] AS v", [[1], [2, 3]]),
        ("RETURN [] AS v", []),
    ],
)
def test_a_literal_reads_back_as_the_python_object_it_is(
    empty: zudb.Connection, statement: str, answer: object
) -> None:
    got = empty.execute(statement).fetchone()[0]
    assert got == answer
    assert type(got) is type(answer)


def test_a_duration_counts_months_or_nanoseconds_and_never_both() -> None:
    with pytest.raises(ValueError, match="never both"):
        zudb.Duration(months=1, nanoseconds=1)


def test_a_duration_says_which_kind_it_is() -> None:
    assert zudb.Duration(months=14).kind == "year_month"
    assert zudb.Duration(nanoseconds=1).kind == "day_time"
    assert zudb.Duration().kind == "day_time"


def test_a_day_time_duration_converts_to_a_timedelta() -> None:
    hour = zudb.Duration(nanoseconds=3_600_000_000_000)
    assert hour.to_timedelta() == datetime.timedelta(hours=1)


def test_a_conversion_that_rounds_rounds_towards_zero() -> None:
    assert zudb.Duration(nanoseconds=1_500).to_timedelta() == datetime.timedelta(microseconds=1)
    assert zudb.Duration(nanoseconds=-1_500).to_timedelta() == datetime.timedelta(microseconds=-1)


def test_a_year_month_duration_has_no_timedelta() -> None:
    with pytest.raises(ValueError, match="no timedelta"):
        zudb.Duration(months=3).to_timedelta()


def test_a_duration_repr_names_the_count_it_carries() -> None:
    assert repr(zudb.Duration(months=3)) == "Duration(months=3)"
    assert repr(zudb.Duration(nanoseconds=3)) == "Duration(nanoseconds=3)"
