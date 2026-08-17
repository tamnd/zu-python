"""Running a statement and reading the rows back."""

from __future__ import annotations

import pytest
import zudb
from conftest import PEOPLE


def test_columns_come_back_in_the_order_they_were_projected(social: zudb.Connection) -> None:
    rows = social.execute("MATCH (p:person) RETURN p.name AS name, p.uid AS uid")
    assert rows.columns == ["name", "uid"]


def test_fetchall_gives_every_row_as_a_tuple(social: zudb.Connection) -> None:
    rows = social.execute("MATCH (p:person) RETURN p.uid AS uid, p.name AS name ORDER BY uid")
    assert rows.fetchall() == [(uid, name) for uid, name, _ in PEOPLE]


def test_fetchone_walks_the_rows_and_then_answers_none(social: zudb.Connection) -> None:
    rows = social.execute("MATCH (p:person) RETURN p.uid AS uid ORDER BY uid")
    assert rows.fetchone() == (10,)
    assert rows.fetchone() == (20,)
    assert rows.fetchone() == (30,)
    assert rows.fetchone() is None
    assert rows.fetchone() is None


def test_iterating_does_not_move_the_cursor(social: zudb.Connection) -> None:
    rows = social.execute("MATCH (p:person) RETURN p.uid AS uid ORDER BY uid")
    assert [uid for (uid,) in rows] == [10, 20, 30]
    assert rows.fetchone() == (10,)
    assert [uid for (uid,) in rows] == [10, 20, 30]


def test_a_result_knows_how_many_rows_it_holds(social: zudb.Connection) -> None:
    rows = social.execute("MATCH (p:person) RETURN p.uid AS uid")
    assert len(rows) == 3
    assert len(social.execute("MATCH (p:person) WHERE p.uid > 100 RETURN p.uid AS uid")) == 0


def test_repr_says_the_shape(social: zudb.Connection) -> None:
    rows = social.execute("MATCH (p:person) RETURN p.uid AS uid, p.name AS name")
    assert repr(rows) == "<zudb.Result 2 columns, 3 rows>"


def test_sql_is_the_same_call_under_the_name_a_notebook_uses(social: zudb.Connection) -> None:
    assert social.sql("MATCH (p:person) RETURN count(p) AS n").fetchall() == [(3,)]


def test_aggregation_and_ordering_run(social: zudb.Connection) -> None:
    rows = social.execute("MATCH (p:person) RETURN p.name AS name ORDER BY p.score DESC LIMIT 2")
    assert rows.fetchall() == [("grace",), ("ada",)]


def test_a_statement_that_writes_answers_no_columns(empty: zudb.Connection) -> None:
    rows = empty.execute("INSERT (p:person {uid: 1, name: 'ada'})")
    assert rows.columns == []
    assert len(rows) == 0


def test_a_statement_with_nothing_to_say_has_no_notices(social: zudb.Connection) -> None:
    assert social.execute("MATCH (p:person) RETURN p.uid AS uid").notices == []


def test_two_statements_on_one_connection_run_in_order(empty: zudb.Connection) -> None:
    empty.execute("INSERT (p:person {uid: 1, name: 'ada'})")
    empty.execute("INSERT (p:person {uid: 2, name: 'kay'})")
    assert empty.execute("MATCH (p:person) RETURN count(p) AS n").fetchall() == [(2,)]


@pytest.mark.parametrize(
    "statement,answer",
    [
        ("RETURN 1 + 1 AS n", 2),
        ("RETURN 'a' + 'b' AS s", "ab"),
        ("UNWIND [1, 2, 3] AS n RETURN sum(n) AS total", 6),
    ],
)
def test_expressions_answer_without_touching_the_graph(
    empty: zudb.Connection, statement: str, answer: object
) -> None:
    assert empty.execute(statement).fetchall() == [(answer,)]
