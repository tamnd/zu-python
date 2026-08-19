"""PEP 249, and the two places zu is not sqlite3.

Most of this is the specification read back: the module globals, the
seven-item description, the exception hierarchy, what a fetch before an
execute is told. It is worth writing down because PEP 249 is a contract
somebody else's code holds us to, and code that holds a driver to a
contract is exactly the code nobody runs until it breaks.

The two parts that are ours are the `?` markers and the implicit
transaction. `?` has to survive strings, quoted names and comments,
because a question mark inside a string is text somebody wrote. The
transaction has to open by itself and close on `commit`, `rollback` and
`close`, which is what a program written against any other driver
already assumes.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest
import zudb
import zudb.dbapi as dbapi

PEOPLE = [(10, "ada", 36.5), (20, "grace", 45.0), (30, "kay", 22.25)]


@pytest.fixture
def conn(tmp_path: Path) -> dbapi.Connection:
    """The three people, through the layer that is being tested."""
    connection = dbapi.connect(tmp_path / "social.zu1")
    cur = connection.cursor()
    first, rest = PEOPLE[0], PEOPLE[1:]
    cur.execute(f"INSERT (p:person {{uid: {first[0]}, name: '{first[1]}', score: {first[2]}}})")
    for uid, name, score in rest:
        cur.execute("INSERT (p:person {uid: ?, name: ?, score: ?})", (uid, name, score))
    connection.commit()
    yield connection
    connection.close()


def test_the_module_says_what_it_is() -> None:
    """The three globals every driver has to carry."""
    assert dbapi.apilevel == "2.0"
    assert dbapi.threadsafety == 2
    assert dbapi.paramstyle == "qmark"


def test_a_statement_runs_and_its_rows_come_back(conn: dbapi.Connection) -> None:
    cur = conn.cursor()
    cur.execute("MATCH (p:person) RETURN p.uid AS uid, p.name AS name")
    assert cur.fetchall() == [(10, "ada"), (20, "grace"), (30, "kay")]


def test_rows_come_back_one_and_a_few_at_a_time(conn: dbapi.Connection) -> None:
    cur = conn.cursor()
    cur.execute("MATCH (p:person) RETURN p.name AS name")
    assert cur.fetchone() == ("ada",)
    assert cur.fetchmany(2) == [("grace",), ("kay",)]
    assert cur.fetchmany(2) == []
    assert cur.fetchone() is None


def test_fetchmany_takes_arraysize_when_it_is_not_told(conn: dbapi.Connection) -> None:
    cur = conn.cursor()
    cur.execute("MATCH (p:person) RETURN p.name AS name")
    assert cur.arraysize == 1, "what PEP 249 says the default is"
    assert cur.fetchmany() == [("ada",)]
    cur.arraysize = 5
    assert cur.fetchmany() == [("grace",), ("kay",)]


def test_a_block_reaches_across_the_rows_read_to_type_the_columns(
    conn: dbapi.Connection,
) -> None:
    """One call, and the join in the middle of it does not show."""
    cur = conn.cursor()
    cur.execute("MATCH (p:person) RETURN p.name AS name")
    assert cur.fetchmany(3) == [("ada",), ("grace",), ("kay",)]


def test_a_block_bigger_than_what_is_left_gives_what_is_left(conn: dbapi.Connection) -> None:
    cur = conn.cursor()
    cur.execute("MATCH (p:person) RETURN p.name AS name")
    assert cur.fetchmany(1000) == [("ada",), ("grace",), ("kay",)]
    assert cur.fetchmany(1000) == []


def test_a_block_of_no_rows_takes_none(conn: dbapi.Connection) -> None:
    """A page size read from configuration can be zero, and it means zero."""
    cur = conn.cursor()
    cur.execute("MATCH (p:person) RETURN p.name AS name")
    assert cur.fetchmany(0) == []
    assert cur.fetchone() == ("ada",)


def test_a_block_of_fewer_than_no_rows_is_refused(conn: dbapi.Connection) -> None:
    """An empty list would read as the end of the rows and end a loop early."""
    cur = conn.cursor()
    cur.execute("MATCH (p:person) RETURN p.name AS name")
    with pytest.raises(dbapi.ProgrammingError, match="-1 is not one"):
        cur.fetchmany(-1)
    assert cur.fetchall() == [("ada",), ("grace",), ("kay",)]


def test_a_cursor_iterates(conn: dbapi.Connection) -> None:
    """An extension PEP 249 names, and the way anyone actually reads rows."""
    cur = conn.cursor()
    cur.execute("MATCH (p:person) RETURN p.name AS name")
    assert [name for (name,) in cur] == ["ada", "grace", "kay"]


def test_description_names_the_columns_and_their_types(conn: dbapi.Connection) -> None:
    cur = conn.cursor()
    cur.execute("MATCH (p:person) RETURN p.uid AS uid, p.name AS name, p.score AS score")
    assert [column[0] for column in cur.description] == ["uid", "name", "score"]
    assert [len(column) for column in cur.description] == [7, 7, 7]
    assert [column[1] for column in cur.description] == [int, str, float]
    assert cur.description[0][1] == dbapi.NUMBER
    assert cur.description[1][1] == dbapi.STRING
    assert cur.description[2][1] == dbapi.NUMBER


def test_a_column_that_is_null_the_whole_way_down_has_no_type(conn: dbapi.Connection) -> None:
    """There is nothing to read a type off, and a guess would be a lie."""
    cur = conn.cursor()
    cur.execute("MATCH (p:person) RETURN p.name AS name, null AS nothing")
    assert [column[1] for column in cur.description] == [str, None]


def test_working_out_the_types_takes_none_of_the_rows(conn: dbapi.Connection) -> None:
    """The rows read to type the columns are handed back first."""
    cur = conn.cursor()
    cur.execute("MATCH (p:person) RETURN p.name AS name")
    assert cur.description[0][1] is str
    assert cur.fetchall() == [("ada",), ("grace",), ("kay",)]


def test_rowcount_is_the_rows_a_statement_produced(conn: dbapi.Connection) -> None:
    cur = conn.cursor()
    cur.execute("MATCH (p:person) RETURN p.name AS name")
    assert cur.rowcount == 3
    cur.execute("MATCH (p:person) WHERE p.uid > 99 RETURN p.name AS name")
    assert cur.rowcount == 0


def test_a_statement_that_writes_has_no_description_and_no_count(conn: dbapi.Connection) -> None:
    """PEP 249 asks for -1 when there is no answer, and there is none:
    a write gives back no columns and the engine does not count what it
    touched."""
    cur = conn.cursor()
    cur.execute("INSERT (p:person {uid: 40, name: 'hopper', score: 1.0})")
    assert cur.description is None
    assert cur.rowcount == -1


def test_fetching_after_a_statement_that_wrote_says_why_it_cannot(conn: dbapi.Connection) -> None:
    cur = conn.cursor()
    cur.execute("INSERT (p:person {uid: 40, name: 'hopper', score: 1.0})")
    with pytest.raises(dbapi.ProgrammingError, match="no rows to fetch"):
        cur.fetchone()


def test_fetching_before_anything_ran_says_the_same(conn: dbapi.Connection) -> None:
    with pytest.raises(dbapi.ProgrammingError, match="no rows to fetch"):
        conn.cursor().fetchall()


def test_a_placeholder_takes_a_value(conn: dbapi.Connection) -> None:
    cur = conn.cursor()
    cur.execute("MATCH (p:person) WHERE p.uid = ? RETURN p.name AS name", (20,))
    assert cur.fetchall() == [("grace",)]


def test_placeholders_are_filled_in_the_order_they_are_written(conn: dbapi.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        "MATCH (p:person) WHERE p.uid > ? AND p.uid < ? RETURN p.name AS name",
        (10, 30),
    )
    assert cur.fetchall() == [("grace",)]


def test_the_wrong_number_of_parameters_says_both_numbers(conn: dbapi.Connection) -> None:
    cur = conn.cursor()
    with pytest.raises(dbapi.ProgrammingError, match="2 placeholders and 1 parameters"):
        cur.execute("MATCH (p:person) WHERE p.uid > ? AND p.uid < ? RETURN p.uid AS uid", (10,))


def test_a_question_mark_inside_a_string_is_not_a_placeholder(conn: dbapi.Connection) -> None:
    """It is a character in somebody's data, and rewriting it would put
    a parameter name in the middle of their text."""
    cur = conn.cursor()
    cur.execute("MATCH (p:person) WHERE p.name = 'who?' RETURN p.name AS name")
    assert cur.fetchall() == []
    cur.execute("RETURN 'why? because.' AS asked")
    assert cur.fetchall() == [("why? because.",)]


def test_a_question_mark_in_a_comment_is_not_one_either(conn: dbapi.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        "// which one? this one\n"
        "MATCH (p:person) /* or ? this */ WHERE p.uid = ? RETURN p.name AS name",
        (30,),
    )
    assert cur.fetchall() == [("kay",)]


def test_a_question_mark_in_a_quoted_name_is_not_one_either(conn: dbapi.Connection) -> None:
    cur = conn.cursor()
    cur.execute("MATCH (p:person) WHERE p.uid = ? RETURN p.name AS `who?`", (10,))
    assert cur.description[0][0] == "who?"


def test_an_escaped_quote_does_not_end_the_string(conn: dbapi.Connection) -> None:
    """The scanner follows the lexer's rules or it loses its place and
    every `?` after it is read wrong."""
    cur = conn.cursor()
    cur.execute("RETURN 'it\\'s ?' AS text, ? AS given", (1,))
    assert cur.fetchall() == [("it's ?", 1)]


def test_a_raw_string_doubles_its_quote_rather_than_escaping(conn: dbapi.Connection) -> None:
    cur = conn.cursor()
    cur.execute("RETURN @'it''s ?' AS text, ? AS given", (2,))
    assert cur.fetchall() == [("it's ?", 2)]


def test_a_dict_hands_the_statement_over_untouched(conn: dbapi.Connection) -> None:
    """Which is how a caller who writes zu statements keeps `$name`."""
    cur = conn.cursor()
    cur.execute("MATCH (p:person) WHERE p.uid = $uid RETURN p.name AS name", {"uid": 30})
    assert cur.fetchall() == [("kay",)]


def test_a_statement_that_already_says_the_obvious_name_gets_another(
    conn: dbapi.Connection,
) -> None:
    """A collision nobody would ever find, avoided by looking first.

    The name the marker is rewritten to has to be one the statement is
    not already using, anywhere, for anything: a value bound over the
    top of somebody's own text is a wrong answer with no error in it.
    """
    cur = conn.cursor()
    cur.execute("RETURN 'costs $_1 a row' AS text, ? AS given", (7,))
    assert cur.fetchall() == [("costs $_1 a row", 7)]


def test_a_string_of_parameters_is_refused_rather_than_read_letter_by_letter(
    conn: dbapi.Connection,
) -> None:
    cur = conn.cursor()
    with pytest.raises(dbapi.ProgrammingError, match="not str"):
        cur.execute("MATCH (p:person) WHERE p.name = ? RETURN p.uid AS uid", "ada")


def test_executemany_writes_a_row_for_each_set_of_parameters(conn: dbapi.Connection) -> None:
    cur = conn.cursor()
    cur.executemany(
        "INSERT (p:person {uid: ?, name: ?, score: ?})",
        [(40, "hopper", 1.0), (50, "lovelace", 2.0)],
    )
    conn.commit()
    cur.execute("MATCH (p:person) WHERE p.uid > 39 RETURN p.name AS name")
    assert cur.fetchall() == [("hopper",), ("lovelace",)]


def test_executemany_keeps_no_rows(conn: dbapi.Connection) -> None:
    """PEP 249 leaves it undefined for statements that return rows, so
    what it leaves behind says nothing rather than the last one's."""
    cur = conn.cursor()
    cur.executemany("MATCH (p:person) WHERE p.uid = ? RETURN p.name AS name", [(10,), (20,)])
    assert cur.description is None
    assert cur.rowcount == -1


def test_a_transaction_is_running_from_the_first_statement(conn: dbapi.Connection) -> None:
    """PEP 249 has no `begin`, so the driver is the one that starts it."""
    assert conn.zu.in_transaction is False
    conn.cursor().execute("MATCH (p:person) RETURN p.uid AS uid")
    assert conn.zu.in_transaction is True
    conn.commit()
    assert conn.zu.in_transaction is False


def test_a_rollback_undoes_what_was_not_committed(conn: dbapi.Connection) -> None:
    cur = conn.cursor()
    cur.execute("INSERT (p:person {uid: 40, name: 'hopper', score: 1.0})")
    conn.rollback()
    cur.execute("MATCH (p:person) RETURN count(p) AS people")
    assert cur.fetchall() == [(3,)]


def test_a_commit_keeps_it(conn: dbapi.Connection) -> None:
    cur = conn.cursor()
    cur.execute("INSERT (p:person {uid: 40, name: 'hopper', score: 1.0})")
    conn.commit()
    cur.execute("MATCH (p:person) RETURN count(p) AS people")
    assert cur.fetchall() == [(4,)]


def test_committing_twice_is_allowed(conn: dbapi.Connection) -> None:
    """There is nothing to do the second time, and refusing would only
    make callers keep a flag."""
    conn.cursor().execute("MATCH (p:person) RETURN p.uid AS uid")
    conn.commit()
    conn.commit()


def test_closing_rolls_back_what_was_not_committed(tmp_path: Path) -> None:
    """The one place the two clients disagree about what a program
    meant, and PEP 249 is explicit about it."""
    path = tmp_path / "lost.zu1"
    first = dbapi.connect(path)
    cur = first.cursor()
    cur.execute("INSERT (p:person {uid: 10, name: 'ada'})")
    first.commit()
    cur.execute("INSERT (p:person {uid: 20, name: 'grace'})")
    first.close()
    with dbapi.connect(path) as second:
        rows = second.cursor().execute("MATCH (p:person) RETURN p.name AS name")
        assert rows.fetchall() == [("ada",)]


def test_autocommit_gives_back_the_native_behaviour(tmp_path: Path) -> None:
    path = tmp_path / "each.zu1"
    with dbapi.connect(path, autocommit=True) as conn:
        cur = conn.cursor()
        cur.execute("INSERT (p:person {uid: 10, name: 'ada'})")
        assert conn.zu.in_transaction is False
        conn.close()
    with dbapi.connect(path, read_only=True) as reader:
        rows = reader.cursor().execute("MATCH (p:person) RETURN p.name AS name")
        assert rows.fetchall() == [("ada",)]


def test_a_block_that_finishes_commits_and_one_that_raises_rolls_back(tmp_path: Path) -> None:
    path = tmp_path / "blocks.zu1"
    conn = dbapi.connect(path)
    with conn:
        conn.cursor().execute("INSERT (p:person {uid: 10, name: 'ada'})")
    assert conn.closed is False, "the block is the unit of work, not the connection"
    with pytest.raises(ZeroDivisionError), conn:
        conn.cursor().execute("INSERT (p:person {uid: 20, name: 'grace'})")
        raise ZeroDivisionError
    rows = conn.cursor().execute("MATCH (p:person) RETURN p.name AS name")
    assert rows.fetchall() == [("ada",)]
    conn.close()


def test_a_block_that_closed_the_connection_itself_leaves_quietly(tmp_path: Path) -> None:
    """Closing decided what happened to the transaction already, and
    raising on the way out would bury whatever the block was doing."""
    conn = dbapi.connect(tmp_path / "shut-early.zu1")
    with conn:
        conn.cursor().execute("INSERT (p:person {uid: 10, name: 'ada'})")
        conn.close()
    assert conn.closed is True


def test_a_read_only_connection_reads(tmp_path: Path) -> None:
    """The transaction it opens by itself has to be a read-only one, or
    the first statement would be refused before it ran."""
    path = tmp_path / "readonly.zu1"
    zudb.load(path, nodes="person", columns={"uid": [1, 2, 3]})
    with dbapi.connect(path, read_only=True) as conn:
        cur = conn.cursor()
        cur.execute("MATCH (p:person) RETURN p.uid AS uid")
        assert cur.rowcount == 3


def test_a_cursor_used_after_it_was_closed_says_so(conn: dbapi.Connection) -> None:
    cur = conn.cursor()
    cur.close()
    assert cur.closed is True
    with pytest.raises(dbapi.InterfaceError, match="cursor is closed"):
        cur.execute("MATCH (p:person) RETURN p.uid AS uid")


def test_a_cursor_on_a_closed_connection_says_which_is_closed(tmp_path: Path) -> None:
    conn = dbapi.connect(tmp_path / "shut.zu1")
    cur = conn.cursor()
    conn.close()
    assert cur.closed is True
    with pytest.raises(dbapi.InterfaceError, match="connection this cursor was made from"):
        cur.execute("MATCH (p:person) RETURN p.uid AS uid")


def test_a_connection_used_after_it_was_closed_says_so(tmp_path: Path) -> None:
    conn = dbapi.connect(tmp_path / "gone.zu1")
    conn.close()
    assert conn.closed is True
    for call in (conn.cursor, conn.commit, conn.rollback):
        with pytest.raises(dbapi.InterfaceError, match="connection is closed"):
            call()
    conn.close()


def test_closing_a_cursor_twice_is_allowed(conn: dbapi.Connection) -> None:
    cur = conn.cursor()
    cur.close()
    cur.close()


def test_a_cursor_closes_at_the_end_of_its_block(conn: dbapi.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute("MATCH (p:person) RETURN p.name AS name")
        assert cur.fetchone() == ("ada",)
    assert cur.closed is True


def test_the_hierarchy_is_the_one_pep_249_draws() -> None:
    """Code written against another driver catches these by name and
    expects the classes underneath them to follow."""
    assert issubclass(dbapi.Error, Exception)
    assert issubclass(dbapi.InterfaceError, dbapi.Error)
    assert not issubclass(dbapi.InterfaceError, dbapi.DatabaseError)
    assert issubclass(dbapi.DatabaseError, dbapi.Error)
    for kind in (
        dbapi.DataError,
        dbapi.OperationalError,
        dbapi.IntegrityError,
        dbapi.InternalError,
        dbapi.ProgrammingError,
        dbapi.NotSupportedError,
    ):
        assert issubclass(kind, dbapi.DatabaseError), kind
    assert issubclass(dbapi.Warning, Exception)
    assert not issubclass(dbapi.Warning, dbapi.Error)


def test_a_failure_is_both_classes_at_once(conn: dbapi.Connection) -> None:
    """The whole point of the exception design: code that catches the
    PEP 249 class and code that catches zu's own catch the same object,
    so a program can mix a driver-shaped library with this client."""
    cur = conn.cursor()
    with pytest.raises(dbapi.ProgrammingError) as raised:
        cur.execute("MATCH (p:person RETURN p")
    assert isinstance(raised.value, zudb.SyntaxError)
    assert isinstance(raised.value, dbapi.DatabaseError)
    assert isinstance(raised.value, dbapi.Error)


def test_a_translated_failure_keeps_every_field(conn: dbapi.Connection) -> None:
    """Passing through here costs nothing: the code, the position and
    the link are the ones the engine wrote."""
    with pytest.raises(dbapi.SyntaxError) as raised:
        conn.cursor().execute("MATCH (p:person RETURN p")
    assert raised.value.code == "42001"
    assert raised.value.line == 1
    assert raised.value.column is not None
    assert raised.value.excerpt is not None
    assert raised.value.doc_url is not None
    assert raised.value.caret() is not None


def test_a_failure_is_not_printed_twice(conn: dbapi.Connection) -> None:
    """Re-raised as its PEP 249 class and not chained to itself, since
    the traceback of the original is the traceback this one has."""
    with pytest.raises(dbapi.SyntaxError) as raised:
        conn.cursor().execute("MATCH (p:person RETURN p")
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__ is True


def test_the_exceptions_hang_off_the_connection_and_the_cursor(conn: dbapi.Connection) -> None:
    """PEP 249's extension, for code that was handed a connection and
    does not know which module made it."""
    assert conn.Error is dbapi.Error
    assert conn.cursor().DatabaseError is dbapi.DatabaseError
    assert conn.OperationalError is dbapi.OperationalError


def test_the_type_objects_cover_what_a_column_can_hold(tmp_path: Path) -> None:
    path = tmp_path / "types.zu1"
    with dbapi.connect(path) as conn:
        cur = conn.cursor()
        cur.execute("INSERT (p:person {uid: 1, name: 'ada', score: 1.5, born: DATE '1815-12-10'})")
        conn.commit()
        cur.execute(
            "MATCH (p:person) RETURN p.uid AS uid, p.name AS name, p.score AS score, p.born AS born"
        )
        codes = [column[1] for column in cur.description]
    assert codes == [int, str, float, datetime.date]
    assert [code == dbapi.NUMBER for code in codes] == [True, False, True, False]
    assert [code == dbapi.STRING for code in codes] == [False, True, False, False]
    assert [code == dbapi.DATETIME for code in codes] == [False, False, False, True]
    # Comparing a type object to a type is what these are for, which is
    # the one place `==` on a type is not the mistake ruff takes it for.
    assert dbapi.BINARY == bytes  # noqa: E721
    assert dbapi.ROWID == zudb.Node  # noqa: E721


def test_the_constructors_build_what_a_parameter_takes(conn: dbapi.Connection) -> None:
    """PEP 249 asks for them so a program can build a value without
    knowing what the driver wants. Here it wants Python's own types, so
    they are Python's own types."""
    assert dbapi.Date(1815, 12, 10) == datetime.date(1815, 12, 10)
    assert dbapi.Time(13, 30) == datetime.time(13, 30)
    assert dbapi.Timestamp(1815, 12, 10, 13, 30) == datetime.datetime(1815, 12, 10, 13, 30)
    assert dbapi.Binary(b"raw") == b"raw"
    ticks = datetime.datetime(2026, 8, 19, 9, 15, 30).timestamp()
    assert dbapi.DateFromTicks(ticks) == datetime.date(2026, 8, 19)
    assert dbapi.TimeFromTicks(ticks) == datetime.time(9, 15, 30)
    assert dbapi.TimestampFromTicks(ticks) == datetime.datetime(2026, 8, 19, 9, 15, 30)
    cur = conn.cursor()
    cur.execute("RETURN ? AS born", (dbapi.Date(1815, 12, 10),))
    assert cur.fetchall() == [(datetime.date(1815, 12, 10),)]


def test_the_optional_methods_this_layer_has_not_got_are_absent(conn: dbapi.Connection) -> None:
    """PEP 249 asks for them to be left out rather than defined and
    refused, so that asking whether a method is there is an answer."""
    cur = conn.cursor()
    assert not hasattr(cur, "callproc")
    assert not hasattr(cur, "nextset")
    cur.setinputsizes([1, 2])
    cur.setoutputsizes(1024)


def test_the_native_connection_is_reachable_and_is_the_same_one(conn: dbapi.Connection) -> None:
    """A layer, not a second client: everything this does not have is
    still there, on the connection and inside its transaction."""
    assert isinstance(conn.zu, zudb.Connection)
    cur = conn.cursor()
    cur.execute("INSERT (p:person {uid: 40, name: 'hopper', score: 1.0})")
    assert conn.zu.execute("MATCH (p:person) RETURN count(p) AS people").fetchall() == [(4,)]
    conn.rollback()


def test_the_result_of_the_last_statement_is_reachable(conn: dbapi.Connection) -> None:
    cur = conn.cursor()
    cur.execute("MATCH (p:person) RETURN p.name AS name")
    assert isinstance(cur.result, zudb.Result)
    assert len(cur.result) == 3


def test_what_a_repr_says(conn: dbapi.Connection) -> None:
    cur = conn.cursor()
    assert repr(cur) == "<zudb.dbapi.Cursor no rows>"
    cur.execute("MATCH (p:person) RETURN p.uid AS uid, p.name AS name")
    assert repr(cur) == "<zudb.dbapi.Cursor 2 columns, 3 rows>"
    cur.close()
    assert repr(cur) == "<zudb.dbapi.Cursor closed>"
    assert "social.zu1" in repr(conn)


def test_a_database_that_is_not_there_is_the_class_pep_249_expects(tmp_path: Path) -> None:
    with pytest.raises(dbapi.OperationalError):
        dbapi.connect(tmp_path / "no" / "such" / "place.zu1")


def test_connect_with_no_path_is_a_database_in_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with dbapi.connect() as conn:
        cur = conn.cursor()
        cur.execute("INSERT (p:person {uid: 1, name: 'ada'})")
        cur.execute("MATCH (p:person) RETURN p.name AS n")
        assert cur.fetchall() == [("ada",)]
    assert list(tmp_path.iterdir()) == []


def test_duplicate_is_another_connection_and_cursor_is_not() -> None:
    """The two words this layer has to keep apart: a cursor shares the
    connection it came from, and a duplicate is a connection of its
    own."""
    with dbapi.connect() as conn:
        cur = conn.cursor()
        cur.execute("INSERT (p:person {uid: 1, name: 'ada'})")
        conn.commit()
        other = conn.duplicate()
        assert other is not conn
        assert other.zu is not conn.zu
        theirs = other.cursor()
        theirs.execute("MATCH (p:person) RETURN p.name AS n")
        assert theirs.fetchall() == [("ada",)]
        other.close()
        assert conn.closed is False


def test_a_duplicate_carries_autocommit_and_not_the_transaction() -> None:
    with dbapi.connect(autocommit=True) as conn:
        other = conn.duplicate()
        assert other.autocommit is True
        other.close()


def test_a_closed_connection_duplicates_nothing() -> None:
    conn = dbapi.connect()
    conn.close()
    with pytest.raises(dbapi.InterfaceError, match="closed"):
        conn.duplicate()
