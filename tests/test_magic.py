"""`%gql` and `%%gql`, against a real shell.

IPython is not a dependency of the wheel and these skip without it,
which is the same rule pyarrow and pandas get. What it is not is a
mock: the magics are registered with `%load_ext` and run through
`run_line_magic` and `run_cell_magic`, so what is checked is what a
notebook does rather than what a method returns when called directly.

Most of these are about which connection a cell runs on, because that
is the whole of the magic's own behaviour. The statement is handed
straight to `execute` and everything else in the suite says what that
does.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import zudb

pytest.importorskip("IPython")

from IPython.core.error import UsageError  # noqa: E402
from IPython.core.interactiveshell import InteractiveShell  # noqa: E402
from zudb import magic  # noqa: E402


@pytest.fixture
def shell() -> Any:
    """A shell with the extension loaded, thrown away afterwards.

    `InteractiveShell` is a singleton, so the instance is cleared at
    the end and the next test gets a namespace of its own rather than
    the connections this one left in it.
    """
    shell = InteractiveShell.instance()
    shell.run_line_magic("load_ext", "zudb.magic")
    yield shell
    InteractiveShell.clear_instance()


def gql(shell: Any, cell: str, line: str = "") -> Any:
    return shell.run_cell_magic("gql", line, cell)


def test_the_line_magic_opens_a_connection_the_cell_magic_uses(shell: Any, tmp_path: Path) -> None:
    conn = shell.run_line_magic("gql", str(tmp_path / "opened.zu1"))
    assert isinstance(conn, zudb.Connection)
    gql(shell, "INSERT (p:person {uid: 10, name: 'ada'})")
    result = gql(shell, "MATCH (p:person) RETURN p.name AS name")
    assert result.fetchall() == [("ada",)]


def test_the_result_is_the_value_of_the_cell(shell: Any, tmp_path: Path) -> None:
    """Which is what makes the notebook draw it as a table."""
    shell.run_line_magic("gql", str(tmp_path / "shown.zu1"))
    gql(shell, "INSERT (p:person {uid: 10, name: 'ada'})")
    result = gql(shell, "MATCH (p:person) RETURN p.name AS name")
    assert isinstance(result, zudb.Result)
    assert "<table" in result._repr_html_()


def test_the_line_magic_with_nothing_says_which_connection_is_open(
    shell: Any, tmp_path: Path
) -> None:
    opened = shell.run_line_magic("gql", str(tmp_path / "asked.zu1"))
    assert shell.run_line_magic("gql", "") is opened


def test_the_line_magic_with_nothing_and_no_connection_says_how_to_open_one(
    shell: Any,
) -> None:
    with pytest.raises(UsageError, match="%gql <path>"):
        shell.run_line_magic("gql", "")


def test_a_read_only_connection_is_opened_when_asked_for(shell: Any, tmp_path: Path) -> None:
    path = tmp_path / "readonly.zu1"
    zudb.load(path, nodes="person", columns={"uid": [1, 2, 3]})
    conn = shell.run_line_magic("gql", f"--read-only {path}")
    assert conn.read_only is True
    assert len(gql(shell, "MATCH (p:person) RETURN p.uid AS uid")) == 3


def test_opening_a_second_database_closes_the_first(shell: Any, tmp_path: Path) -> None:
    """Nothing else here has a name for it, so leaving it open would
    leave a file held by nobody."""
    first = shell.run_line_magic("gql", str(tmp_path / "first.zu1"))
    second = shell.run_line_magic("gql", str(tmp_path / "second.zu1"))
    assert first.closed is True
    assert second.closed is False


def test_close_shuts_the_connection_the_magic_opened(shell: Any, tmp_path: Path) -> None:
    conn = shell.run_line_magic("gql", str(tmp_path / "closing.zu1"))
    assert shell.run_line_magic("gql", "--close") is None
    assert conn.closed is True
    with pytest.raises(UsageError, match="no connection is open"):
        shell.run_line_magic("gql", "")


def test_the_cell_magic_finds_the_one_connection_the_notebook_made(
    shell: Any, tmp_path: Path
) -> None:
    """A notebook that already called `zudb.connect` needs no `%gql`."""
    shell.user_ns["conn"] = zudb.connect(tmp_path / "theirs.zu1")
    gql(shell, "INSERT (p:person {uid: 10, name: 'ada'})")
    assert gql(shell, "MATCH (p:person) RETURN p.name AS name").fetchall() == [("ada",)]


def test_two_connections_in_the_notebook_means_the_cell_has_to_say_which(
    shell: Any, tmp_path: Path
) -> None:
    shell.user_ns["reader"] = zudb.connect(tmp_path / "one.zu1")
    shell.user_ns["writer"] = zudb.connect(tmp_path / "two.zu1")
    with pytest.raises(UsageError, match="reader, writer"):
        gql(shell, "MATCH (p:person) RETURN p.uid AS uid")
    shell.user_ns["writer"].execute("INSERT (p:person {uid: 10, name: 'ada'})")
    result = gql(shell, "MATCH (p:person) RETURN p.name AS name", "--conn writer")
    assert result.fetchall() == [("ada",)]


def test_a_name_that_is_not_a_connection_is_refused(shell: Any, tmp_path: Path) -> None:
    shell.run_line_magic("gql", str(tmp_path / "named.zu1"))
    shell.user_ns["other"] = "not a connection"
    with pytest.raises(UsageError, match="not a zudb.Connection"):
        gql(shell, "MATCH (p:person) RETURN p.uid AS uid", "--conn other")


def test_no_connection_anywhere_says_both_ways_to_get_one(shell: Any) -> None:
    with pytest.raises(UsageError, match="zudb.connect"):
        gql(shell, "MATCH (p:person) RETURN p.uid AS uid")


def test_parameters_come_out_of_a_variable(shell: Any, tmp_path: Path) -> None:
    shell.run_line_magic("gql", str(tmp_path / "params.zu1"))
    gql(shell, "INSERT (p:person {uid: 10, name: 'ada'})")
    gql(shell, "INSERT (p:person {uid: 20, name: 'grace'})")
    shell.user_ns["args"] = {"uid": 20}
    result = gql(
        shell, "MATCH (p:person) WHERE p.uid = $uid RETURN p.name AS name", "--params args"
    )
    assert result.fetchall() == [("grace",)]


def test_a_params_variable_that_is_not_a_dict_is_refused(shell: Any, tmp_path: Path) -> None:
    shell.run_line_magic("gql", str(tmp_path / "wrong.zu1"))
    shell.user_ns["args"] = [1, 2, 3]
    with pytest.raises(UsageError, match="not a dict"):
        gql(shell, "MATCH (p:person) RETURN p.uid AS uid", "--params args")


def test_out_puts_the_result_in_a_variable_and_shows_nothing(shell: Any, tmp_path: Path) -> None:
    shell.run_line_magic("gql", str(tmp_path / "out.zu1"))
    gql(shell, "INSERT (p:person {uid: 10, name: 'ada'})")
    assert gql(shell, "MATCH (p:person) RETURN p.name AS name", "--out rows") is None
    assert shell.user_ns["rows"].fetchall() == [("ada",)]


def test_a_trailing_semicolon_is_taken_off(shell: Any, tmp_path: Path) -> None:
    """A person types it out of habit and a statement does not take one."""
    shell.run_line_magic("gql", str(tmp_path / "semi.zu1"))
    gql(shell, "INSERT (p:person {uid: 10, name: 'ada'});")
    assert gql(shell, "MATCH (p:person) RETURN p.name AS name ;\n").fetchall() == [("ada",)]


def test_an_empty_cell_says_so(shell: Any, tmp_path: Path) -> None:
    shell.run_line_magic("gql", str(tmp_path / "blank.zu1"))
    with pytest.raises(UsageError, match="no statement"):
        gql(shell, "\n  \n")


def test_an_option_neither_magic_takes_is_refused(shell: Any, tmp_path: Path) -> None:
    with pytest.raises(UsageError, match="--verbose"):
        shell.run_line_magic("gql", "--verbose social.zu1")
    shell.run_line_magic("gql", str(tmp_path / "options.zu1"))
    with pytest.raises(UsageError, match="--verbose"):
        gql(shell, "MATCH (p:person) RETURN p.uid AS uid", "--verbose")


def test_an_option_with_no_variable_after_it_is_refused(shell: Any, tmp_path: Path) -> None:
    shell.run_line_magic("gql", str(tmp_path / "dangling.zu1"))
    with pytest.raises(UsageError, match="names a variable"):
        gql(shell, "MATCH (p:person) RETURN p.uid AS uid", "--out")


def test_two_paths_at_once_is_refused(shell: Any, tmp_path: Path) -> None:
    with pytest.raises(UsageError, match="more than one"):
        shell.run_line_magic("gql", f"{tmp_path / 'a.zu1'} {tmp_path / 'b.zu1'}")


def test_a_statement_that_is_wrong_raises_what_it_would_have(shell: Any, tmp_path: Path) -> None:
    """The magic is a way in and not a layer: the engine's exception
    arrives as the engine's exception."""
    shell.run_line_magic("gql", str(tmp_path / "wrong.zu1"))
    with pytest.raises(zudb.SyntaxError):
        gql(shell, "MATCH (p:person RETURN p")


def test_a_path_with_a_space_in_it_is_quoted_and_opens(shell: Any, tmp_path: Path) -> None:
    """Which is the reason the line is lexed rather than split on
    whitespace, and the only reason."""
    room = tmp_path / "two words"
    room.mkdir()
    path = room / "spaced.zu1"
    conn = shell.run_line_magic("gql", f'"{path}"')
    assert conn.closed is False
    assert conn.execute("RETURN 1 AS n").fetchone() == (1,)


@pytest.mark.skipif(os.name != "nt", reason="a backslash is only a separator on Windows")
def test_a_windows_path_keeps_its_separators() -> None:
    r"""The failure this is here for: `shlex.split` treats a backslash
    as an escape, so `C:\data\social.zu1` came out of it as
    `C:datasocial.zu1` and the magic then said the system could not
    find a file the person could see in front of them."""
    assert magic.words(r"--read-only C:\data\social.zu1") == [
        "--read-only",
        r"C:\data\social.zu1",
    ]


@pytest.mark.skipif(os.name == "nt", reason="a backslash is an escape everywhere else")
def test_a_backslash_is_still_an_escape_where_a_shell_says_it_is() -> None:
    """Nothing is taken away from the platforms that were right: a
    person on a Unix quotes a space with a backslash and expects that
    to go on working."""
    assert magic.words(r"/tmp/two\ words/spaced.zu1") == ["/tmp/two words/spaced.zu1"]


def test_a_hash_in_a_name_is_a_name_and_not_a_comment() -> None:
    """A line magic is not a script, and `#` is a legal character in a
    file name on every platform this runs on."""
    assert magic.words("a#b.zu1 --read-only") == ["a#b.zu1", "--read-only"]
