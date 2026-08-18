"""Several statements as one unit of work.

A single statement is already atomic, so what these check is the span:
that the work between the two words arrives together, that it goes away
together when the block raises, and that a program can ask which of the
two it is in.

The rollback tests read the database back through the connection that
wrote it and, where it matters, through one opened afterwards, because
a rollback that only convinced the connection that did it would be a
rollback that had not happened.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import zudb


def uids(conn: zudb.Connection) -> list[int]:
    """The people in the database, in the order they were written."""
    return [uid for (uid,) in conn.execute("MATCH (p:person) RETURN p.uid AS uid")]


def test_a_block_that_ends_commits_what_it_wrote(social: zudb.Connection) -> None:
    with social.transaction():
        social.execute("INSERT (p:person {uid: 40, name: 'lynn', score: 31.0})")
        social.execute("INSERT (p:person {uid: 50, name: 'barbara', score: 28.5})")
    assert uids(social) == [10, 20, 30, 40, 50]


def test_a_block_that_raises_rolls_back_and_the_exception_carries_on(
    social: zudb.Connection,
) -> None:
    with pytest.raises(RuntimeError, match="halfway"):
        with social.transaction():
            social.execute("INSERT (p:person {uid: 40, name: 'lynn', score: 31.0})")
            raise RuntimeError("halfway through")
    assert uids(social) == [10, 20, 30]


def test_the_rollback_is_on_the_file_and_not_just_on_the_connection(
    tmp_path: Path,
) -> None:
    """Read back through a connection that was not there for it."""
    path = tmp_path / "rolled.zu1"
    conn = zudb.connect(path)
    conn.execute("INSERT (p:person {uid: 10, name: 'ada'})")
    with pytest.raises(RuntimeError):
        with conn.transaction():
            conn.execute("INSERT (p:person {uid: 20, name: 'grace'})")
            raise RuntimeError("no")
    conn.close()

    again = zudb.connect(path)
    assert uids(again) == [10]
    again.close()


def test_commit_and_rollback_can_be_called_rather_than_waited_for(
    social: zudb.Connection,
) -> None:
    txn = social.transaction()
    social.execute("INSERT (p:person {uid: 40, name: 'lynn', score: 31.0})")
    txn.commit()
    assert uids(social) == [10, 20, 30, 40]

    txn = social.transaction()
    social.execute("INSERT (p:person {uid: 50, name: 'barbara', score: 28.5})")
    txn.rollback()
    assert uids(social) == [10, 20, 30, 40]


def test_a_block_that_committed_early_is_left_alone_on_the_way_out(
    social: zudb.Connection,
) -> None:
    """The rest of the block runs outside the transaction, deliberately."""
    with social.transaction() as txn:
        social.execute("INSERT (p:person {uid: 40, name: 'lynn', score: 31.0})")
        txn.commit()
        assert txn.done
        assert not social.in_transaction
        social.execute("INSERT (p:person {uid: 50, name: 'barbara', score: 28.5})")
    assert uids(social) == [10, 20, 30, 40, 50]


def test_ending_a_transaction_twice_is_refused(social: zudb.Connection) -> None:
    txn = social.transaction()
    txn.commit()
    with pytest.raises(zudb.ProgrammingError, match="already ended"):
        txn.commit()
    with pytest.raises(zudb.ProgrammingError, match="already ended"):
        txn.rollback()


def test_in_transaction_says_which_side_of_the_block_a_program_is_on(
    social: zudb.Connection,
) -> None:
    assert not social.in_transaction
    with social.transaction():
        assert social.in_transaction
    assert not social.in_transaction


def test_a_statement_on_its_own_is_not_a_transaction_this_can_see(
    social: zudb.Connection,
) -> None:
    """Every statement runs in one of its own, and none of them is this."""
    social.execute("INSERT (p:person {uid: 40, name: 'lynn', score: 31.0})")
    assert not social.in_transaction


def test_a_read_only_transaction_reads_and_refuses_to_write(
    social: zudb.Connection,
) -> None:
    with social.transaction(read_only=True) as txn:
        assert txn.read_only
        assert uids(social) == [10, 20, 30]
        with pytest.raises(zudb.TransactionError, match="READ ONLY"):
            social.execute("INSERT (p:person {uid: 40, name: 'lynn', score: 31.0})")


def test_one_transaction_at_a_time(social: zudb.Connection) -> None:
    """Refused rather than nested, because a rollback of an inner one
    would have to invent an answer for what it undoes."""
    with social.transaction():
        with pytest.raises(zudb.TransactionError, match="already running"):
            social.transaction()


def test_the_three_words_still_work_written_out(social: zudb.Connection) -> None:
    """This wraps the statements rather than replacing them."""
    social.execute("START TRANSACTION")
    assert social.in_transaction
    social.execute("INSERT (p:person {uid: 40, name: 'lynn', score: 31.0})")
    social.execute("ROLLBACK")
    assert uids(social) == [10, 20, 30]


def test_a_commit_with_nothing_to_commit_is_refused(social: zudb.Connection) -> None:
    with pytest.raises(zudb.TransactionError, match="no transaction"):
        social.execute("COMMIT")


def test_a_statement_that_failed_leaves_the_transaction_to_its_owner(
    social: zudb.Connection,
) -> None:
    """The engine does not end a transaction on a failed statement, so
    the block that opened it is still the thing that closes it."""
    with pytest.raises(RuntimeError):
        with social.transaction():
            with pytest.raises(zudb.Error):
                social.execute("MATCH (p:person) RETURN")
            assert social.in_transaction
            social.execute("INSERT (p:person {uid: 40, name: 'lynn', score: 31.0})")
            raise RuntimeError("and now unwind all of it")
    assert uids(social) == [10, 20, 30]


def test_an_appender_is_refused_inside_a_transaction(tmp_path: Path) -> None:
    """Its batches are commits of their own, which no rollback reaches."""
    path = tmp_path / "graph.zu1"
    zudb.load(path, nodes="person", columns={"uid": [10, 20, 30]})
    conn = zudb.connect(path)
    with conn.transaction():
        with pytest.raises(zudb.ProgrammingError, match="own commits"):
            conn.appender("person")
    with conn.appender("person") as appender:
        appender.append_row([40])
    assert uids(conn) == [10, 20, 30, 40]
    conn.close()


def test_closing_the_connection_drops_what_was_uncommitted(tmp_path: Path) -> None:
    path = tmp_path / "dropped.zu1"
    conn = zudb.connect(path)
    conn.execute("INSERT (p:person {uid: 10, name: 'ada'})")
    conn.transaction()
    conn.execute("INSERT (p:person {uid: 20, name: 'grace'})")
    conn.close()

    again = zudb.connect(path)
    assert uids(again) == [10]
    again.close()


def test_a_commit_that_cannot_run_raises_out_of_the_block(tmp_path: Path) -> None:
    """Better than a block that ends quietly on a transaction nobody
    committed."""
    conn = zudb.connect(tmp_path / "closed.zu1")
    conn.execute("INSERT (p:person {uid: 10, name: 'ada'})")
    with pytest.raises(zudb.ProgrammingError, match="closed"):
        with conn.transaction():
            conn.close()


def test_a_closed_connection_has_no_transactions(social: zudb.Connection) -> None:
    social.close()
    with pytest.raises(zudb.ProgrammingError, match="closed"):
        social.transaction()
    with pytest.raises(zudb.ProgrammingError, match="closed"):
        assert social.in_transaction is None


def test_the_wrapper_costs_nothing_worth_measuring(social: zudb.Connection) -> None:
    """Two statements and a Python object, which is what it should be.

    The budget is generous by a factor of ten against the 5 microseconds
    this measures on a laptop, because it is here to catch a wrapper
    that started waiting on something rather than to hold a number.
    """
    best = float("inf")
    for _ in range(3):
        started = time.perf_counter()
        for _ in range(200):
            with social.transaction():
                pass
        best = min(best, (time.perf_counter() - started) / 200)
    assert best < 50e-6, f"an empty transaction took {best * 1e6:.0f} us"


def test_repr_says_which_kind_and_whether_it_is_over(social: zudb.Connection) -> None:
    with social.transaction() as txn:
        assert repr(txn) == "<zudb.Transaction>"
    assert repr(txn) == "<zudb.Transaction, done>"

    txn = social.transaction(read_only=True)
    assert repr(txn) == "<zudb.Transaction read only>"
    txn.rollback()
    assert repr(txn) == "<zudb.Transaction read only, done>"
