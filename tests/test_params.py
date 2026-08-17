"""Parameters, which are named and typed and never string formatting."""

from __future__ import annotations

import datetime

import pytest
import zudb


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        False,
        0,
        -7,
        1 << 40,
        1.5,
        "ada",
        "",
        [1, 2, 3],
        [1, "a", None],
        [],
        datetime.date(2020, 1, 2),
        datetime.time(1, 2, 3),
        datetime.time(1, 2, 3, 400_000),
        datetime.datetime(2020, 1, 2, 3, 4, 5),
        datetime.datetime(2020, 1, 2, 3, 4, 5, 6),
        datetime.datetime(2020, 1, 2, 3, 4, 5, tzinfo=datetime.UTC),
        zudb.Duration(months=14),
        zudb.Duration(nanoseconds=90_061_000_000_000),
    ],
)
def test_a_parameter_comes_back_as_what_went_in(empty: zudb.Connection, value: object) -> None:
    assert empty.execute("RETURN $p AS p", {"p": value}).fetchone() == (value,)


def test_a_tuple_is_a_list_on_the_way_in(empty: zudb.Connection) -> None:
    assert empty.execute("RETURN $p AS p", {"p": (1, 2)}).fetchone() == ([1, 2],)


def test_a_timedelta_becomes_a_day_time_duration(empty: zudb.Connection) -> None:
    delta = datetime.timedelta(days=2, seconds=3, microseconds=4)
    got = empty.execute("RETURN $p AS p", {"p": delta}).fetchone()[0]
    assert got == zudb.Duration(nanoseconds=172_803_000_004_000)
    assert got.to_timedelta() == delta


def test_several_parameters_are_told_apart_by_name(empty: zudb.Connection) -> None:
    rows = empty.execute("RETURN $a AS a, $b AS b", {"a": 1, "b": "two"})
    assert rows.fetchall() == [(1, "two")]


def test_a_parameter_is_a_value_and_never_a_fragment_of_the_statement(
    social: zudb.Connection,
) -> None:
    hostile = "ada' RETURN 1 AS pwned MATCH (p:person) WHERE p.name = 'x"
    rows = social.execute("MATCH (p:person) WHERE p.name = $n RETURN p.name AS n", {"n": hostile})
    assert rows.fetchall() == []


def test_a_parameter_filters_the_same_way_a_literal_does(social: zudb.Connection) -> None:
    by_name = social.execute(
        "MATCH (p:person) WHERE p.name = $n RETURN p.uid AS uid", {"n": "grace"}
    )
    assert by_name.fetchall() == [(20,)]


def test_no_parameters_at_all_is_fine(social: zudb.Connection) -> None:
    assert social.execute("MATCH (p:person) RETURN count(p) AS n").fetchall() == [(3,)]


def test_a_type_zu_does_not_hold_is_refused_by_name(empty: zudb.Connection) -> None:
    with pytest.raises(TypeError, match="cannot be a object"):
        empty.execute("RETURN $p AS p", {"p": object()})


def test_a_bool_is_a_bool_and_not_the_integer_one(empty: zudb.Connection) -> None:
    got = empty.execute("RETURN $p AS p", {"p": True}).fetchone()[0]
    assert got is True
    assert type(got) is bool
