"""Rows on their way into a table that already exists.

An appender is the fast way in and the only way that is not a statement,
so these check the three things that makes it: that the rows arrive and
read back as themselves, that a row which cannot mean anything is
refused where it was appended rather than at the flush, and that a
refusal leaves the appender and the database exactly as they were.

The last of those is the one worth the most tests. A load runs for a
long time, and a caller who is a million rows in and has just been told
that row 999,999 is wrong wants the other 999,998 and a database that
still opens.
"""

from __future__ import annotations

import datetime
import gc
import threading
import time
from pathlib import Path

import pytest
import zudb


@pytest.fixture
def graph(tmp_path: Path) -> zudb.Connection:
    """Three people and the edges between two of them, open for writing.

    Loaded rather than inserted because a rel table is made by a load
    and by nothing else, and half of what an appender is for is adding
    edges to one.
    """
    path = tmp_path / "graph.zu1"
    zudb.load(
        path,
        nodes="person",
        rels="knows",
        columns={"uid": [10, 20, 30], "name": ["ada", "grace", "kay"]},
        edges=[(0, 1)],
    )
    conn = zudb.connect(path)
    yield conn
    conn.close()


def names(conn: zudb.Connection) -> list[str]:
    return [name for (name,) in conn.execute("MATCH (p:person) RETURN p.name AS name")]


def edges(conn: zudb.Connection) -> list[tuple[str, str]]:
    return list(
        conn.execute("MATCH (a:person)-[:knows]->(b:person) RETURN a.name AS a, b.name AS b")
    )


def test_a_row_appended_and_flushed_is_a_row_you_can_query(graph: zudb.Connection) -> None:
    app = graph.appender("person")
    app.append_row([40, "hopper"])
    assert app.flush() == 1
    assert names(graph) == ["ada", "grace", "kay", "hopper"]


def test_nothing_is_written_until_the_flush(graph: zudb.Connection) -> None:
    app = graph.appender("person")
    app.append_row([40, "hopper"])
    assert app.buffered == 1
    assert app.committed == 0
    assert names(graph) == ["ada", "grace", "kay"]
    app.close()


def test_a_flush_empties_the_buffer_and_leaves_the_appender_open(graph: zudb.Connection) -> None:
    app = graph.appender("person")
    app.append_row([40, "hopper"])
    app.flush()
    assert (app.buffered, app.committed, app.closed) == (0, 1, False)
    app.append_row([50, "liskov"])
    assert app.flush() == 2
    assert names(graph) == ["ada", "grace", "kay", "hopper", "liskov"]


def test_append_rows_takes_a_batch_and_says_how_many_it_took(graph: zudb.Connection) -> None:
    app = graph.appender("person")
    assert app.append_rows([[40, "hopper"], [50, "liskov"]]) == 2
    assert app.buffered == 2
    assert app.close() == 2
    assert names(graph) == ["ada", "grace", "kay", "hopper", "liskov"]


def test_append_rows_reads_an_iterator_as_happily_as_a_list(graph: zudb.Connection) -> None:
    rows = ([uid, f"p{uid}"] for uid in (40, 50, 60))
    app = graph.appender("person")
    assert app.append_rows(rows) == 3
    assert app.close() == 3


def test_a_with_block_flushes_on_the_way_out(graph: zudb.Connection) -> None:
    with graph.appender("person") as app:
        app.append_row([40, "hopper"])
    assert app.closed
    assert names(graph) == ["ada", "grace", "kay", "hopper"]


def test_a_block_that_raised_still_writes_the_rows_it_managed(graph: zudb.Connection) -> None:
    # The Rust appender flushes when it is dropped and this is the same
    # answer for the same reason: a load that stopped partway is better
    # served by its rows arriving than by them vanishing, and a caller
    # who wants the other answer has `discard()`.
    with pytest.raises(RuntimeError, match="halfway"):
        with graph.appender("person") as app:
            app.append_row([40, "hopper"])
            raise RuntimeError("stopped halfway")
    assert names(graph) == ["ada", "grace", "kay", "hopper"]


def test_discard_throws_away_what_is_buffered(graph: zudb.Connection) -> None:
    with graph.appender("person") as app:
        app.append_rows([[40, "hopper"], [50, "liskov"]])
        assert app.discard() == 2
        assert app.buffered == 0
    assert names(graph) == ["ada", "grace", "kay"]


def test_discard_does_not_reach_what_a_flush_committed(graph: zudb.Connection) -> None:
    app = graph.appender("person")
    app.append_row([40, "hopper"])
    app.flush()
    app.append_row([50, "liskov"])
    assert app.discard() == 1
    assert app.close() == 1
    assert names(graph) == ["ada", "grace", "kay", "hopper"]


def test_closing_twice_writes_nothing_the_second_time(graph: zudb.Connection) -> None:
    app = graph.appender("person")
    app.append_row([40, "hopper"])
    assert app.close() == 1
    assert app.close() == 1
    assert app.closed


def test_a_closed_appender_takes_no_more_rows(graph: zudb.Connection) -> None:
    app = graph.appender("person")
    app.close()
    with pytest.raises(zudb.ProgrammingError, match="closed appender"):
        app.append_row([40, "hopper"])
    with pytest.raises(zudb.ProgrammingError, match="closed appender"):
        app.flush()
    with pytest.raises(zudb.ProgrammingError, match="closed appender"):
        app.discard()


def test_a_flush_with_nothing_buffered_commits_nothing(graph: zudb.Connection) -> None:
    app = graph.appender("person")
    assert app.flush() == 0
    assert app.flush() == 0
    assert names(graph) == ["ada", "grace", "kay"]
    app.close()


def test_a_row_of_the_wrong_width_is_refused_by_name(graph: zudb.Connection) -> None:
    app = graph.appender("person")
    with pytest.raises(ValueError, match="carries 1 values and 'person' takes 2: uid, name"):
        app.append_row([40])
    with pytest.raises(ValueError, match="more than the 2 values 'person' takes"):
        app.append_row([40, "hopper", "extra"])
    app.close()


def test_a_refused_row_leaves_the_buffer_a_rectangle(graph: zudb.Connection) -> None:
    # The half of the row that went in has to come back out, or the
    # columns are different lengths and the flush is the one that finds
    # out, a long way from the row that did it.
    app = graph.appender("person")
    app.append_row([40, "hopper"])
    with pytest.raises(TypeError):
        app.append_row([50, 50])
    assert app.buffered == 1
    app.append_row([50, "liskov"])
    assert app.close() == 2
    assert names(graph) == ["ada", "grace", "kay", "hopper", "liskov"]


def test_a_value_the_column_does_not_hold_is_refused_where_it_was_appended(
    graph: zudb.Connection,
) -> None:
    app = graph.appender("person")
    with pytest.raises(
        TypeError, match="value 0 of this row is of type 'str' and column 'uid' of 'person'"
    ):
        app.append_row(["forty", "hopper"])
    app.close()


def test_the_table_says_what_a_column_holds_and_not_the_first_row(graph: zudb.Connection) -> None:
    # A first row that was wrong used to settle the shape and then
    # refuse every right row after it. The columns come from the table,
    # so the wrong row is the one refused.
    app = graph.appender("person")
    with pytest.raises(TypeError, match="column 'uid' of 'person' holds integers"):
        app.append_row(["hopper", 40])
    app.append_row([40, "hopper"])
    assert app.close() == 1


def test_an_integer_column_refuses_a_float(graph: zudb.Connection) -> None:
    # A load widens a column of integers when it meets a float, because
    # nothing there has said what the column is. Here the table has
    # said, and 4.5 is not an integer.
    app = graph.appender("person")
    with pytest.raises(TypeError, match="'float' and column 'uid' of 'person' holds integers"):
        app.append_row([4.5, "hopper"])
    app.close()


def test_a_bool_is_not_an_integer(graph: zudb.Connection) -> None:
    app = graph.appender("person")
    with pytest.raises(TypeError, match="'bool' and column 'uid' of 'person' holds integers"):
        app.append_row([True, "hopper"])
    app.close()


def test_every_type_a_column_holds_goes_in_and_comes_back(tmp_path: Path) -> None:
    path = tmp_path / "types.zu1"
    zudb.load(
        path,
        nodes="thing",
        columns={
            "n": [1],
            "f": [1.5],
            "b": [True],
            "s": ["one"],
            "d": [datetime.date(2020, 1, 1)],
            "t": [datetime.time(1, 2, 3)],
            "ts": [datetime.datetime(2020, 1, 1, 1, 2, 3)],
            "dur": [datetime.timedelta(days=1)],
            "ym": [zudb.Duration(months=3)],
        },
    )
    row = [
        2,
        2.5,
        False,
        "two",
        datetime.date(2022, 3, 4),
        datetime.time(5, 6, 7),
        datetime.datetime(2022, 3, 4, 5, 6, 7),
        datetime.timedelta(minutes=30),
        zudb.Duration(months=5),
    ]
    with zudb.connect(path) as conn:
        with conn.appender("thing") as app:
            app.append_row(row)
        got = conn.execute(
            "MATCH (t:thing) WHERE t.n = 2 RETURN t.n, t.f, t.b, t.s, t.d, t.t, t.ts, t.dur, t.ym"
        ).fetchone()
    # A duration comes back as a `Duration`, since Python's own type
    # cannot hold the year-month half of one, so the `timedelta` that
    # went in is the one value that does not read back as itself.
    assert list(got) == [*row[:7], zudb.Duration(nanoseconds=30 * 60 * 1_000_000_000), row[8]]


def test_a_float_column_takes_an_integer(tmp_path: Path) -> None:
    # The one widening the other way round, and the only one: every
    # integer a Python program is likely to hand a float column is a
    # float exactly, and refusing 7 for a column of scores would be
    # pedantry rather than safety.
    path = tmp_path / "scores.zu1"
    zudb.load(path, nodes="person", columns={"score": [36.5]})
    with zudb.connect(path) as conn:
        with conn.appender("person") as app:
            app.append_row([7])
        assert conn.execute("MATCH (p:person) RETURN p.score AS s").fetchall() == [(36.5,), (7.0,)]


def test_an_appender_on_a_rel_table_joins_the_graph(graph: zudb.Connection) -> None:
    with graph.appender("knows") as rels:
        rels.append_row([1, 2])
    assert edges(graph) == [("ada", "grace"), ("grace", "kay")]


def test_a_rel_row_is_two_offsets_and_a_negative_one_is_no_row(graph: zudb.Connection) -> None:
    with graph.appender("knows") as rels:
        with pytest.raises(ValueError, match="row offsets, which count from zero"):
            rels.append_row([0, -1])
        assert rels.buffered == 0


def test_an_edge_to_a_row_that_is_not_there_is_refused_before_it_is_written(
    graph: zudb.Connection,
) -> None:
    # Refused by the flush and not by the fold that comes after it. The
    # fold's refusal arrives once the write is durable, and the frame it
    # leaves behind is refused again by every writer that opens the
    # database afterwards, which is a database nobody can write to over
    # one bad edge.
    rels = graph.appender("knows")
    rels.append_row([0, 99])
    with pytest.raises(ValueError, match="joins row 99 of 'person', which has 3 rows"):
        rels.flush()
    assert rels.buffered == 1
    rels.discard()
    rels.close()
    assert edges(graph) == [("ada", "grace")]


def test_the_database_still_opens_after_an_edge_that_was_refused(tmp_path: Path) -> None:
    path = tmp_path / "graph.zu1"
    zudb.load(path, nodes="person", rels="knows", columns={"uid": [1, 2]}, edges=[(0, 1)])
    with zudb.connect(path) as conn:
        rels = conn.appender("knows")
        rels.append_row([0, 99])
        with pytest.raises(ValueError):
            rels.close()
        rels.discard()
        rels.close()
    with zudb.connect(path) as conn:
        assert conn.execute("MATCH ()-[r:knows]->() RETURN count(r) AS n").fetchone() == (1,)


def test_an_edge_can_name_a_row_a_flush_wrote_a_moment_ago(graph: zudb.Connection) -> None:
    # The row counts are read at the flush and not when the appender
    # opened, so a rel appender held across a load of nodes writes the
    # edges to them rather than refusing every one.
    rels = graph.appender("knows")
    with graph.appender("person") as people:
        people.append_row([40, "hopper"])
    rels.append_row([0, 3])
    rels.close()
    assert edges(graph) == [("ada", "grace"), ("ada", "hopper")]


def test_a_table_nothing_declares_has_no_appender(graph: zudb.Connection) -> None:
    with pytest.raises(zudb.ProgrammingError, match="no node table or rel table 'cities'"):
        graph.appender("cities")


def test_a_read_only_connection_has_no_appender(tmp_path: Path) -> None:
    path = tmp_path / "graph.zu1"
    zudb.load(path, nodes="person", columns={"uid": [1]})
    with zudb.connect(path, read_only=True) as conn:
        with pytest.raises(zudb.ProgrammingError, match="the connection is read-only"):
            conn.appender("person")


def test_a_closed_connection_has_nowhere_to_put_the_rows(tmp_path: Path) -> None:
    path = tmp_path / "graph.zu1"
    zudb.load(path, nodes="person", columns={"uid": [1]})
    conn = zudb.connect(path)
    app = conn.appender("person")
    app.append_row([2])
    conn.close()
    with pytest.raises(zudb.ProgrammingError, match="connection this appender writes through"):
        app.flush()


def test_an_appender_keeps_its_connection_alive(tmp_path: Path) -> None:
    # The connection is held and not borrowed, so an appender handed
    # back by a function that opened one is an appender that still
    # works. A borrowed one would be a buffer with nowhere to go.
    path = tmp_path / "graph.zu1"
    zudb.load(path, nodes="person", columns={"uid": [1]})

    def opened() -> zudb.Appender:
        return zudb.connect(path).appender("person")

    app = opened()
    gc.collect()
    app.append_row([2])
    assert app.close() == 1
    with zudb.connect(path, read_only=True) as conn:
        assert conn.execute("MATCH (p:person) RETURN count(p) AS n").fetchone() == (2,)


def test_the_repr_says_what_it_is_holding(graph: zudb.Connection) -> None:
    app = graph.appender("person")
    assert app.table == "person"
    assert repr(app) == "<zudb.Appender person, 0 buffered, 0 committed>"
    app.append_row([40, "hopper"])
    assert repr(app) == "<zudb.Appender person, 1 buffered, 0 committed>"
    app.flush()
    assert repr(app) == "<zudb.Appender person, 0 buffered, 1 committed>"
    app.close()
    assert repr(app) == "<zudb.Appender person, 0 buffered, 1 committed, closed>"


def test_two_threads_appending_to_one_appender_lose_nothing(graph: zudb.Connection) -> None:
    app = graph.appender("person")

    def run(start: int) -> None:
        for uid in range(start, start + 200):
            app.append_row([uid, f"p{uid}"])

    threads = [threading.Thread(target=run, args=(base,)) for base in (1000, 2000, 3000)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
        assert not thread.is_alive(), "a thread is still waiting for the appender"
    assert app.buffered == 600
    assert app.close() == 600
    assert len(names(graph)) == 603


def test_python_keeps_running_while_a_flush_does(graph: zudb.Connection) -> None:
    ticks = 0
    done = threading.Event()
    app = graph.appender("person")
    app.append_rows([[uid, f"p{uid}"] for uid in range(50_000)])

    def run() -> None:
        app.close()
        done.set()

    worker = threading.Thread(target=run)
    worker.start()
    while not done.is_set():
        ticks += 1
    worker.join(timeout=120)
    assert not worker.is_alive()
    # A GIL held for the length of the flush would leave this loop no
    # turns at all rather than thousands of them.
    assert ticks > 1000, f"the main thread only got {ticks} turns"


#: Rows for the comparison against `INSERT`, and few enough that the
#: `INSERT` half finishes in a few seconds. It is the slow half by three
#: orders of magnitude, and it gets slower as the table grows, because
#: every row of it is a commit and a fold.
COMPARED = 200


@pytest.mark.timing
def test_appending_beats_inserting_by_the_margin_that_makes_it_worth_having(
    tmp_path: Path,
) -> None:
    rows = [(uid, f"p{uid}") for uid in range(1, COMPARED)]

    with zudb.connect(tmp_path / "inserted.zu1") as conn:
        conn.execute("INSERT (p:person {uid: 0, name: 'seed'})")
        started = time.perf_counter()
        for uid, name in rows:
            conn.execute("INSERT (p:person {uid: $u, name: $n})", {"u": uid, "n": name})
        inserting = time.perf_counter() - started

    with zudb.connect(tmp_path / "appended.zu1") as conn:
        conn.execute("INSERT (p:person {uid: 0, name: 'seed'})")
        started = time.perf_counter()
        with conn.appender("person") as app:
            app.append_rows(rows)
        appending = time.perf_counter() - started
        assert conn.execute("MATCH (p:person) RETURN count(p) AS n").fetchone() == (COMPARED,)

    # Measured at about 150 times on this machine at this row count and
    # rising with it, since one commit is one commit however many rows
    # it carries. It is 18 on a shared CI runner, where a commit costs
    # 25 ms of somebody else's disk and the appender's single one is
    # most of what it spends, so the gate is 5: the number that says the
    # appender is still batching rather than the number either machine
    # hits.
    assert inserting > 5 * appending, (
        f"{COMPARED} rows: {inserting * 1000:.0f} ms inserted, {appending * 1000:.0f} ms appended"
    )
