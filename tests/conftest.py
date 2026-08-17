"""Fixtures shared by the suite.

Every test gets a database of its own under a temporary directory, so a
test that writes cannot change what another test reads and a failure
leaves the file behind for exactly as long as pytest keeps the
directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import zudb

PEOPLE = [(10, "ada", 36.5), (20, "grace", 45.0), (30, "kay", 22.25)]


@pytest.fixture
def empty(tmp_path: Path) -> zudb.Connection:
    """A database with nothing in it."""
    conn = zudb.connect(tmp_path / "empty.zu1")
    yield conn
    conn.close()


@pytest.fixture
def social(tmp_path: Path) -> zudb.Connection:
    """Three people, written the way the engine writes them.

    The property is `uid` rather than `id` because `id` is the name the
    v0 engine gives a node's offset, and a table that carries one of
    its own is a table where the two disagree.

    The first row is written out rather than parameterized, because the
    table does not exist yet and a value that has to be worked out
    first says nothing about the column it would go in. Every row after
    it goes in as parameters, which is the way a program writes rows.
    """
    conn = zudb.connect(tmp_path / "social.zu1")
    first, rest = PEOPLE[0], PEOPLE[1:]
    conn.execute(f"INSERT (p:person {{uid: {first[0]}, name: '{first[1]}', score: {first[2]}}})")
    for uid, name, score in rest:
        conn.execute(
            "INSERT (p:person {uid: $uid, name: $name, score: $score})",
            {"uid": uid, "name": name, "score": score},
        )
    yield conn
    conn.close()


@pytest.fixture
def crowd(tmp_path: Path) -> zudb.Connection:
    """Enough people that a statement over the pairs of them takes long
    enough to time, which is what the threading tests need.
    """
    conn = zudb.connect(tmp_path / "crowd.zu1")
    conn.execute("INSERT (p:person {uid: 0, name: 'seed'})")
    conn.execute(
        "UNWIND $rows AS r INSERT (p:person {uid: r.uid, name: r.name})",
        {"rows": [{"uid": uid, "name": f"p{uid}"} for uid in range(1, 150)]},
    )
    yield conn
    conn.close()
