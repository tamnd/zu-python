"""Opening, closing, and what a connection says about itself."""

from __future__ import annotations

from pathlib import Path

import pytest
import zudb


def test_connect_creates_a_database_that_was_not_there(tmp_path: Path) -> None:
    path = tmp_path / "new.zu1"
    assert not path.exists()
    conn = zudb.connect(path)
    assert path.exists()
    assert conn.path == path
    assert conn.read_only is False
    conn.close()


def test_a_written_row_is_there_on_the_next_connection(tmp_path: Path) -> None:
    path = tmp_path / "again.zu1"
    with zudb.connect(path) as conn:
        conn.execute("INSERT (p:person {uid: 1, name: 'ada'})")
    with zudb.connect(path) as conn:
        assert conn.execute("MATCH (p:person) RETURN p.name AS n").fetchall() == [("ada",)]


def test_a_read_only_connection_never_creates(tmp_path: Path) -> None:
    path = tmp_path / "absent.zu1"
    with pytest.raises(zudb.ConnectionError):
        zudb.connect(path, read_only=True)
    assert not path.exists()


def test_a_read_only_connection_reads(tmp_path: Path) -> None:
    path = tmp_path / "frozen.zu1"
    with zudb.connect(path) as conn:
        conn.execute("INSERT (p:person {uid: 1, name: 'ada'})")
    with zudb.connect(path, read_only=True) as conn:
        assert conn.read_only is True
        assert conn.execute("MATCH (p:person) RETURN p.name AS n").fetchall() == [("ada",)]


def test_closing_twice_is_not_an_error(empty: zudb.Connection) -> None:
    assert empty.closed is False
    empty.close()
    assert empty.closed is True
    empty.close()
    assert empty.closed is True


def test_a_statement_after_close_says_so(empty: zudb.Connection) -> None:
    empty.close()
    with pytest.raises(zudb.ProgrammingError, match="closed"):
        empty.execute("RETURN 1 AS one")


def test_the_block_closes_on_the_way_out(tmp_path: Path) -> None:
    with zudb.connect(tmp_path / "block.zu1") as conn:
        assert conn.closed is False
    assert conn.closed is True


def test_an_exception_leaves_the_block_rather_than_being_swallowed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mine"):
        with zudb.connect(tmp_path / "raise.zu1") as conn:
            raise ValueError("mine")
    assert conn.closed is True


def test_repr_names_the_file_and_says_when_it_is_closed(tmp_path: Path) -> None:
    conn = zudb.connect(tmp_path / "shown.zu1")
    assert "shown.zu1" in repr(conn)
    assert "closed" not in repr(conn)
    conn.close()
    assert "closed" in repr(conn)


def test_the_engine_and_abi_versions_are_reported(empty: zudb.Connection) -> None:
    assert zudb.__version__ == "0.0.1"
    assert zudb.__abi_version__.count(".") == 1


def test_a_memory_limit_and_a_thread_count_are_accepted(tmp_path: Path) -> None:
    with zudb.connect(tmp_path / "tuned.zu1", memory_limit=64 << 20, threads=2) as conn:
        assert conn.execute("RETURN 1 AS one").fetchall() == [(1,)]
