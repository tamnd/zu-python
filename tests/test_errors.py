"""Conditions as exceptions, with the fields the error model promises."""

from __future__ import annotations

import pytest
import zudb


def test_a_syntax_error_carries_its_code_and_its_place(empty: zudb.Connection) -> None:
    with pytest.raises(zudb.SyntaxError) as raised:
        empty.execute("MATCH (p:person) RETRUN p")
    err = raised.value
    assert err.code.startswith("42")
    assert err.condition
    assert err.severity == "exception"
    assert err.line == 1
    assert err.column > 1
    assert err.offset > 0
    assert err.retryable is False


def test_the_doc_url_points_at_the_page_for_the_code(empty: zudb.Connection) -> None:
    with pytest.raises(zudb.SyntaxError) as raised:
        empty.execute("MATCH (p:person) RETRUN p")
    err = raised.value
    assert err.doc_url == f"https://zu.dev/docs/errors/{err.code.lower()}"


def test_an_excerpt_and_a_caret_point_at_the_token(empty: zudb.Connection) -> None:
    statement = "MATCH (p:person) RETRUN p"
    with pytest.raises(zudb.SyntaxError) as raised:
        empty.execute(statement)
    err = raised.value
    assert err.excerpt == statement
    assert statement[err.offset :].startswith("RETRUN")
    quoted, pointer = err.caret().splitlines()
    assert quoted == statement
    assert pointer.index("^") == err.column - 1


def test_a_name_nothing_bound_is_a_reference_error(empty: zudb.Connection) -> None:
    with pytest.raises(zudb.Error) as raised:
        empty.execute("RETURN nobody AS x")
    assert raised.value.code == "42002"


def test_a_pattern_the_graph_cannot_satisfy_matches_nothing(empty: zudb.Connection) -> None:
    """A label nothing declares is not an error, it is a pattern with no
    answer, which is the reading a client that composes a query out of
    optional parts depends on."""
    result = empty.execute("MATCH (p:person)-[:follows]->(q:person) RETURN p")
    assert result.columns == ["p"]
    assert result.fetchall() == []


def test_every_condition_is_catchable_as_the_base_class(empty: zudb.Connection) -> None:
    with pytest.raises(zudb.Error):
        empty.execute("this is not a statement")


def test_a_condition_is_an_exception_and_reads_like_one(empty: zudb.Connection) -> None:
    with pytest.raises(zudb.Error) as raised:
        empty.execute("MATCH (p:person) RETURN")
    assert isinstance(raised.value, Exception)
    assert str(raised.value)


def test_a_wrong_type_in_a_value_is_a_data_error(social: zudb.Connection) -> None:
    with pytest.raises(zudb.DataError) as raised:
        social.execute("INSERT (p:person {uid: 'ten', name: 'zoe', score: 1.0})")
    assert raised.value.code.startswith("22")


def test_a_mistake_the_program_made_is_a_programming_error(empty: zudb.Connection) -> None:
    empty.close()
    with pytest.raises(zudb.ProgrammingError):
        empty.execute("RETURN 1 AS one")


def test_a_missing_database_is_a_connection_error(tmp_path) -> None:
    with pytest.raises(zudb.ConnectionError):
        zudb.connect(tmp_path / "not-here.zu1", read_only=True)


def test_the_error_classes_are_zu_classes_and_not_the_builtins() -> None:
    assert zudb.SyntaxError is not SyntaxError
    assert issubclass(zudb.SyntaxError, zudb.Error)
    assert issubclass(zudb.DataError, zudb.Error)
    assert issubclass(zudb.ConnectionError, zudb.Error)
    assert issubclass(zudb.TransactionError, zudb.Error)
    assert issubclass(zudb.ProgrammingError, zudb.Error)
    assert issubclass(zudb.InternalError, zudb.Error)
    assert issubclass(zudb.Interrupted, zudb.Error)


def test_a_condition_that_never_happened_leaves_the_fields_empty() -> None:
    err = zudb.Error("nothing in particular")
    assert err.code is None
    assert err.line is None
    assert err.caret() is None
    assert str(err) == "nothing in particular"
