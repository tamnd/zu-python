"""Building a database out of columns and an edge list.

A load is the only way a Python program makes a graph with edges in it,
so these check both halves: that what went in comes back out through
statements, and that a load which cannot mean anything is refused where
the mistake is rather than written to disk and found later.
"""

from __future__ import annotations

import datetime
import threading
from pathlib import Path

import pytest
import zudb


def test_a_load_says_what_it_wrote(tmp_path: Path) -> None:
    stats = zudb.load(
        tmp_path / "g.zu1",
        nodes="person",
        rels="knows",
        columns={"uid": [1, 2, 3], "name": ["ada", "grace", "kay"]},
        edges=[(0, 1), (1, 2)],
    )
    assert stats == {"nodes": 3, "rels": 2, "columns": 2}


def test_the_rows_read_back_in_the_order_they_went_in(loaded: zudb.Connection) -> None:
    rows = loaded.execute("MATCH (p:person) RETURN p.uid AS uid, p.name AS name")
    assert list(rows) == [(10, "ada"), (20, "grace"), (30, "kay")]


def test_the_edges_are_a_table_a_pattern_can_walk(loaded: zudb.Connection) -> None:
    rows = loaded.execute("MATCH (a:person)-[:knows]->(b:person) RETURN a.name AS a, b.name AS b")
    assert list(rows) == [("ada", "grace"), ("grace", "kay")]


def test_an_edge_comes_back_as_a_rel(loaded: zudb.Connection) -> None:
    rels = [rel for (rel,) in loaded.execute("MATCH ()-[r:knows]->() RETURN r")]
    assert [rel.table for rel in rels] == ["knows", "knows"]
    # The ordinal is the edge's place in the load, which is where its
    # properties sit, so the second edge loaded is the second ordinal.
    assert [(rel.src, rel.dst, rel.ord) for rel in rels] == [(0, 1, 0), (1, 2, 1)]
    assert repr(rels[0]) == "Rel(knows, 0 -> 1)"


def test_two_reads_of_one_edge_are_equal_and_hash_alike(loaded: zudb.Connection) -> None:
    statement = "MATCH (a:person)-[r:knows]->(b:person) WHERE a.uid = 10 RETURN r"
    one = loaded.execute(statement).fetchone()[0]
    two = loaded.execute(statement).fetchone()[0]
    assert one == two
    assert len({one, two}) == 1


def test_a_walk_comes_back_as_a_path(loaded: zudb.Connection) -> None:
    walk = loaded.execute(
        "MATCH q = (a:person)-[:knows]->(b:person) WHERE a.uid = 10 RETURN q"
    ).fetchone()[0]
    assert len(walk) == 1
    assert [node.offset for node in walk.nodes] == [0, 1]
    assert [rel.dst for rel in walk.rels] == [1]
    assert walk.elements == [*walk.nodes[:1], *walk.rels, *walk.nodes[1:]]
    assert repr(walk) == "Path(1 hops)"


def test_a_two_hop_walk_alternates_nodes_and_edges(loaded: zudb.Connection) -> None:
    walk = loaded.execute(
        "MATCH q = (a:person)-[:knows]->()-[:knows]->(c:person) RETURN q"
    ).fetchone()[0]
    assert len(walk) == 2
    assert [node.offset for node in walk.nodes] == [0, 1, 2]
    assert [(rel.src, rel.dst) for rel in walk.rels] == [(0, 1), (1, 2)]


def test_a_load_with_no_edges_is_a_graph_with_none(tmp_path: Path) -> None:
    stats = zudb.load(tmp_path / "g.zu1", nodes="person", rels="knows", columns={"uid": [1, 2]})
    assert stats["rels"] == 0
    with zudb.connect(tmp_path / "g.zu1", read_only=True) as conn:
        assert conn.execute("MATCH ()-[r:knows]->() RETURN count(r) AS n").fetchone() == (0,)


def test_a_load_with_no_columns_still_has_rows(tmp_path: Path) -> None:
    stats = zudb.load(tmp_path / "g.zu1", nodes="person", rels="knows", rows=4, edges=[(0, 3)])
    assert stats == {"nodes": 4, "rels": 1, "columns": 0}
    with zudb.connect(tmp_path / "g.zu1", read_only=True) as conn:
        assert conn.execute("MATCH (p:person) RETURN count(p) AS n").fetchone() == (4,)


def test_the_rel_table_is_called_rel_when_it_is_not_named(tmp_path: Path) -> None:
    zudb.load(tmp_path / "g.zu1", nodes="person", columns={"uid": [1, 2]}, edges=[(0, 1)])
    with zudb.connect(tmp_path / "g.zu1", read_only=True) as conn:
        rel = conn.execute("MATCH ()-[r]->() RETURN r").fetchone()[0]
    assert rel.table == "rel"


def test_the_same_edge_twice_is_one_edge(tmp_path: Path) -> None:
    stats = zudb.load(
        tmp_path / "g.zu1",
        nodes="person",
        rels="knows",
        columns={"uid": [1, 2]},
        edges=[(0, 1), (0, 1), (0, 1)],
    )
    assert stats["rels"] == 1


def test_a_column_of_every_kind_reads_back_as_what_it_was(tmp_path: Path) -> None:
    columns = {
        "count": [1, -2],
        "ratio": [1.5, -0.25],
        "flag": [True, False],
        "name": ["ada", "grace"],
        "born": [datetime.date(1815, 12, 10), datetime.date(1906, 12, 9)],
        "woke": [datetime.time(6, 30), datetime.time(23, 59, 59)],
        "seen": [datetime.datetime(2024, 1, 2, 3, 4, 5, 6), datetime.datetime(1900, 1, 1)],
        "took": [datetime.timedelta(days=1, seconds=2), datetime.timedelta(0)],
        "aged": [zudb.Duration(months=14), zudb.Duration(months=-1)],
    }
    zudb.load(tmp_path / "g.zu1", nodes="person", rels="knows", columns=columns)
    with zudb.connect(tmp_path / "g.zu1", read_only=True) as conn:
        rows = conn.execute(
            "MATCH (p:person) RETURN " + ", ".join(f"p.{name} AS {name}" for name in columns)
        )
        got = list(rows)
    assert got[0] == (
        1,
        1.5,
        True,
        "ada",
        datetime.date(1815, 12, 10),
        datetime.time(6, 30),
        datetime.datetime(2024, 1, 2, 3, 4, 5, 6),
        zudb.Duration(nanoseconds=86_402_000_000_000),
        zudb.Duration(months=14),
    )
    assert got[1][:4] == (-2, -0.25, False, "grace")


def test_a_load_never_writes_over_a_database_that_is_there(tmp_path: Path) -> None:
    path = tmp_path / "g.zu1"
    zudb.load(path, nodes="person", rels="knows", columns={"uid": [1]})
    with pytest.raises(zudb.ConnectionError):
        zudb.load(path, nodes="person", rels="knows", columns={"uid": [2]})
    with zudb.connect(path, read_only=True) as conn:
        assert conn.execute("MATCH (p:person) RETURN p.uid AS uid").fetchone() == (1,)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"nodes": "", "rels": "knows", "rows": 1}, "a table has a name"),
        ({"nodes": "person", "rels": "", "rows": 1}, "a table has a name"),
        ({"nodes": "person", "rels": "knows"}, "has to be told how many"),
        (
            {"nodes": "person", "rels": "knows", "columns": {"a": [1, 2], "b": [3]}},
            "as wide as it is long",
        ),
        ({"nodes": "person", "rels": "knows", "columns": {"a": []}}, "is empty"),
        (
            {"nodes": "person", "rels": "knows", "columns": {"a": [1, 2]}, "rows": 3},
            "against the 3 rows",
        ),
        (
            {"nodes": "person", "rels": "knows", "columns": {"a": [1, 2]}, "edges": [(0, 5)]},
            "row 5 of a table with 2 rows",
        ),
        (
            {"nodes": "person", "rels": "knows", "columns": {"a": [1, 2]}, "edges": [(0, -1)]},
            "row -1 of a table",
        ),
    ],
)
def test_a_load_that_cannot_mean_anything_is_refused(
    tmp_path: Path, kwargs: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        zudb.load(tmp_path / "g.zu1", **kwargs)
    assert not (tmp_path / "g.zu1").exists()


@pytest.mark.parametrize(
    "values,message",
    [
        ([1, True], "holds integers and row 1 is of type 'bool'"),
        ([True, 1], "holds booleans and row 1 is of type 'int'"),
        ([1, "ada"], "holds integers and row 1 is of type 'str'"),
        ([1.5, "ada"], "holds floats and row 1 is of type 'str'"),
        (["ada", 1], "holds strings and row 1 is of type 'int'"),
        (
            [datetime.date(2020, 1, 1), datetime.datetime(2020, 1, 1)],
            "holds dates and row 1 is of type 'datetime'",
        ),
        (
            [datetime.datetime(2020, 1, 1), datetime.date(2020, 1, 1)],
            "holds datetimes and row 1 is of type 'date'",
        ),
        (
            [zudb.Duration(months=1), datetime.timedelta(days=1)],
            "holds year-month durations and row 1 is of type 'timedelta'",
        ),
        (
            [datetime.timedelta(days=1), zudb.Duration(months=1)],
            "holds day-time durations and row 1 is of type 'Duration'",
        ),
        ([object()], "starts at row 0 with a value of type 'object'"),
    ],
)
def test_a_column_holds_one_kind_of_value(tmp_path: Path, values: list, message: str) -> None:
    with pytest.raises(TypeError, match=message):
        zudb.load(tmp_path / "g.zu1", nodes="person", rels="knows", columns={"a": values})


def test_an_edge_that_is_not_a_pair_is_refused(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="edge 1 is not a pair of row numbers"):
        zudb.load(
            tmp_path / "g.zu1",
            nodes="person",
            rels="knows",
            columns={"uid": [1, 2]},
            edges=[(0, 1), 7],
        )


def test_columns_and_edges_may_be_any_iterable(tmp_path: Path) -> None:
    stats = zudb.load(
        tmp_path / "g.zu1",
        nodes="person",
        rels="knows",
        columns={"uid": range(4), "name": (f"p{i}" for i in range(4))},
        edges=((i, i + 1) for i in range(3)),
    )
    assert stats == {"nodes": 4, "rels": 3, "columns": 2}


def test_python_keeps_running_while_a_load_does(tmp_path: Path) -> None:
    # Big enough that the write takes long enough to watch, and shaped
    # so the edges are out of order and have to be sorted, which is the
    # other half of the work the GIL is released for.
    rows = 200_000
    columns = {"uid": list(range(rows)), "name": [f"p{uid}" for uid in range(rows)]}
    edges = [(uid, (uid * 7 + 1) % rows) for uid in range(rows)]
    ticks = 0
    done = threading.Event()
    stats: list[dict] = []

    def run() -> None:
        stats.append(
            zudb.load(
                tmp_path / "big.zu1",
                nodes="person",
                rels="knows",
                columns=columns,
                edges=edges,
            )
        )
        done.set()

    worker = threading.Thread(target=run)
    worker.start()
    while not done.is_set():
        ticks += 1
    worker.join(timeout=120)
    assert not worker.is_alive()
    assert stats[0]["nodes"] == rows
    # A GIL held for the length of the write would leave the main thread
    # nothing but the switch interval, which is a handful of turns.
    assert ticks > 1000, f"the main thread only got {ticks} turns"
