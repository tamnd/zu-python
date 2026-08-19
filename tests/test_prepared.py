"""A statement compiled once and run many times.

The rows a prepared statement gives back are the rows `execute` gives
back, and they are asserted here mostly to say that the two ways of
asking are the same way. What the rest of it is about is the lifetime:
what a prepared statement is before it is closed, what it says after,
what happens to one whose connection went first, and whether the names
it reports are the names the statement wants.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import zudb

BY_NAME = "MATCH (p:person) WHERE p.name = $name RETURN p.uid AS uid"


def test_it_reports_its_text_and_the_names_it_wants(social: zudb.Connection) -> None:
    with social.prepare(BY_NAME) as find:
        assert isinstance(find, zudb.Prepared)
        assert find.statement == BY_NAME
        assert find.params == ["name"]
        assert find.closed is False


def test_a_statement_that_takes_no_parameters_reports_none(social: zudb.Connection) -> None:
    with social.prepare("MATCH (p:person) RETURN p.name AS name") as everyone:
        assert everyone.params == []


def test_it_runs_as_often_as_it_is_asked_with_different_bindings(social: zudb.Connection) -> None:
    with social.prepare(BY_NAME) as find:
        assert find.execute({"name": "ada"}).fetchall() == [(10,)]
        assert find.execute({"name": "grace"}).fetchall() == [(20,)]
        assert find.execute({"name": "nobody"}).fetchall() == []
        assert find.execute({"name": "ada"}).fetchall() == [(10,)]


def test_what_comes_back_is_the_result_a_statement_gives(social: zudb.Connection) -> None:
    with social.prepare(BY_NAME) as find:
        rows = find.execute({"name": "ada"})
        assert isinstance(rows, zudb.Result)
        assert rows.columns == ["uid"]
        assert rows.notices == []
        assert len(rows) == 1


def test_sql_is_the_same_call_under_the_notebook_name(social: zudb.Connection) -> None:
    with social.prepare(BY_NAME) as find:
        assert find.sql({"name": "grace"}).fetchall() == [(20,)]


def test_a_prepared_write_writes(social: zudb.Connection) -> None:
    with social.prepare("INSERT (p:person {uid: $uid, name: $name, score: $score})") as add:
        add.execute({"uid": 40, "name": "hedy", "score": 51.0})
        add.execute({"uid": 50, "name": "edith", "score": 33.5})

    rows = social.execute("MATCH (p:person) RETURN count(*) AS n")
    assert rows.fetchall() == [(5,)]


def test_a_statement_that_does_not_compile_fails_at_the_prepare(social: zudb.Connection) -> None:
    with pytest.raises(zudb.SyntaxError):
        social.prepare("MATCH (")


def test_a_name_the_caller_did_not_bind_fails_at_the_run(social: zudb.Connection) -> None:
    with social.prepare(BY_NAME) as find:
        with pytest.raises(zudb.SyntaxError, match=r"\$name"):
            find.execute()
        # And the statement is still there to be run properly, since a
        # missing binding is nothing to do with the statement.
        assert find.execute({"name": "ada"}).fetchall() == [(10,)]


def test_closing_it_twice_does_nothing(social: zudb.Connection) -> None:
    find = social.prepare(BY_NAME)
    find.close()
    assert find.closed is True
    find.close()
    assert find.closed is True


def test_a_closed_prepared_statement_refuses_to_run(social: zudb.Connection) -> None:
    find = social.prepare(BY_NAME)
    find.close()

    with pytest.raises(zudb.ProgrammingError, match="closed"):
        find.execute({"name": "ada"})
    with pytest.raises(zudb.ProgrammingError, match="closed"):
        find.sql({"name": "ada"})


def test_the_block_closes_it_at_the_end(social: zudb.Connection) -> None:
    with social.prepare(BY_NAME) as find:
        assert find.closed is False
    assert find.closed is True


def test_an_exception_inside_the_block_closes_it_and_carries_on(social: zudb.Connection) -> None:
    find = social.prepare(BY_NAME)
    with pytest.raises(ZeroDivisionError), find:
        raise ZeroDivisionError
    assert find.closed is True


def test_one_whose_connection_closed_says_the_connection_is_closed(tmp_path: Path) -> None:
    conn = zudb.connect(tmp_path / "gone.zu1")
    conn.execute("INSERT (p:person {uid: 1, name: 'ada'})")
    find = conn.prepare(BY_NAME)
    conn.close()

    with pytest.raises(zudb.ProgrammingError, match="closed"):
        find.execute({"name": "ada"})
    # Closing it is still fine, and still does nothing: the session that
    # was holding the id went when the connection did.
    find.close()
    assert find.closed is True


def test_a_read_only_connection_prepares_and_runs_a_read(loaded: zudb.Connection) -> None:
    with loaded.prepare("MATCH (p:person) RETURN p.name AS name") as everyone:
        assert len(everyone.execute()) == 3


def test_a_connection_prepares_as_many_statements_as_it_likes(social: zudb.Connection) -> None:
    first = social.prepare(BY_NAME)
    second = social.prepare("MATCH (p:person) RETURN count(*) AS n")
    third = social.prepare("MATCH (p:person) RETURN p.name AS name")

    assert first.execute({"name": "grace"}).fetchall() == [(20,)]
    assert second.execute().fetchall() == [(3,)]
    assert len(third.execute()) == 3

    for statement in (first, second, third):
        statement.close()


def test_preparing_on_a_closed_connection_is_refused(tmp_path: Path) -> None:
    conn = zudb.connect(tmp_path / "shut.zu1")
    conn.close()

    with pytest.raises(zudb.ProgrammingError, match="closed"):
        conn.prepare("MATCH (p:person) RETURN p.name AS name")


def test_a_statement_that_is_not_a_string_is_refused(social: zudb.Connection) -> None:
    with pytest.raises(TypeError):
        social.prepare(42)


def test_it_says_what_it_is_and_whether_it_is_closed(social: zudb.Connection) -> None:
    find = social.prepare(BY_NAME)
    assert repr(find) == f'<zudb.Prepared "{BY_NAME}">'
    find.close()
    assert repr(find) == f'<zudb.Prepared "{BY_NAME}", closed>'


def test_the_class_is_the_one_the_package_exports(social: zudb.Connection) -> None:
    with social.prepare(BY_NAME) as find:
        assert type(find) is zudb.Prepared
        assert type(find).__module__ == "zudb"
