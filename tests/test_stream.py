"""Rows read as the engine makes them.

A result is rows already in memory and a stream is rows on their way, so
what these assert is the difference between the two: that the first row
arrives before the last is made, that no more than a batch or two is
held at once, and that the connection a stream is reading is a
connection nothing else can run on until it ends. The rest is the
behaviour a result already has, checked once through the streaming
spelling to show it survived the crossing.
"""

from __future__ import annotations

import gc
import itertools
import tracemalloc
from pathlib import Path

import pytest
import zudb

# Enough people that a scan over them takes more than one batch, which
# is what makes a batch worth talking about at all.
MANY = 5_000
# Enough people that the pairs of them are more rows than anything here
# reads. A statement that fits in the queue is one the engine can finish
# while the reader is between two rows, so every test about a stream
# still running is written on the pairs rather than on the people.
PAIRED = 2_000
PAIRS = PAIRED * (PAIRED - 1) // 2
PAIRS_OF_PEOPLE = "MATCH (a:person), (b:person) WHERE a.uid < b.uid RETURN a.uid AS a, b.uid AS b"


@pytest.fixture
def many(tmp_path: Path) -> zudb.Connection:
    """Five thousand people, which is a couple of batches and a bit.

    Loaded rather than inserted because a row at a time is a commit at a
    time, and this is scaffolding rather than the thing under test.
    """
    path = tmp_path / "many.zu1"
    zudb.load(
        path,
        nodes="person",
        rels="knows",
        columns={"uid": list(range(MANY)), "name": [f"p{uid}" for uid in range(MANY)]},
        edges=[(0, 1)],
    )
    conn = zudb.connect(path, read_only=True)
    yield conn
    conn.close()


@pytest.fixture
def paired(tmp_path: Path) -> zudb.Connection:
    """Enough people that the pairs of them are millions of rows.

    Two thousand people is two million pairs, which no test here reads
    to the end: it is the statement to open a stream on when what is
    being asserted is about a statement that is still running.
    """
    path = tmp_path / "paired.zu1"
    zudb.load(
        path,
        nodes="person",
        rels="knows",
        columns={"uid": list(range(PAIRED))},
        edges=[(0, 1)],
    )
    conn = zudb.connect(path, read_only=True)
    yield conn
    conn.close()


def test_a_stream_gives_back_every_row_the_statement_made(many: zudb.Connection) -> None:
    rows = list(many.stream("MATCH (p:person) RETURN p.uid AS uid"))

    assert len(rows) == MANY
    assert rows[0] == (0,)
    assert rows[-1] == (MANY - 1,)


def test_a_stream_says_its_columns_before_a_row_is_read(many: zudb.Connection) -> None:
    with many.stream("MATCH (p:person) RETURN p.uid AS uid, p.name AS name") as stream:
        assert stream.columns == ["uid", "name"]
        assert next(iter(stream)) == (0, "p0")


def test_rows_arrive_before_the_statement_is_over(paired: zudb.Connection) -> None:
    """The claim the whole module is for.

    Two million pairs take long enough that a statement which had to
    finish before it gave a row would not have given one yet, so a first
    row in hand while the summary is still `None` is the proof that the
    rows are coming as they are made.
    """
    with paired.stream(PAIRS_OF_PEOPLE) as stream:
        first = next(iter(stream))

        assert first == (0, 1)
        assert stream.summary is None
        assert stream.closed is False


def test_a_stream_holds_a_batch_or_two_and_not_the_result(many: zudb.Connection) -> None:
    """What the memory of a stream looks like against the memory of a
    result.

    Only the Python side is measured, which is the side a caller sees:
    the tuples. A result makes every one of them and holds them, a
    stream makes them a batch at a time and lets each go.
    """
    statement = "MATCH (p:person) RETURN p.uid AS uid, p.name AS name"
    gc.collect()
    tracemalloc.start()
    held = many.execute(statement).fetchall()
    whole = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert len(held) == MANY
    del held
    gc.collect()

    tracemalloc.start()
    counted = 0
    for _ in many.stream(statement):
        counted += 1
    streaming = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    assert counted == MANY
    # A tenth would hold on this machine and half is the assertion that
    # survives a machine where an allocator rounds differently.
    assert streaming < whole / 2


def test_the_batches_are_the_size_that_was_asked_for(many: zudb.Connection) -> None:
    with many.stream("MATCH (p:person) RETURN p.uid AS uid", batch_rows=500) as stream:
        sizes = [len(batch) for batch in stream.batches()]

    assert sum(sizes) == MANY
    assert max(sizes) <= 500
    assert sizes[0] == 500


def test_a_batch_is_a_list_of_the_rows_the_loop_would_have_given(social: zudb.Connection) -> None:
    with social.stream("MATCH (p:person) RETURN p.uid AS uid") as stream:
        batches = list(stream.batches())

    assert batches == [[(10,), (20,), (30,)]]


def test_a_stream_holds_the_connection_until_it_ends(paired: zudb.Connection) -> None:
    """A statement run on a connection a stream is reading is told no
    rather than queued, because queueing it behind a loop that may never
    finish is a program deadlocked on itself.
    """
    stream = paired.stream(PAIRS_OF_PEOPLE)
    next(iter(stream))

    with pytest.raises(zudb.ProgrammingError, match="runs one statement at a time"):
        paired.execute("MATCH (p:person) RETURN count(*) AS n")

    stream.close()
    assert paired.execute("MATCH (p:person) RETURN count(*) AS n").fetchall() == [(PAIRED,)]


def test_a_stream_read_to_the_end_gives_the_connection_back(social: zudb.Connection) -> None:
    assert list(social.stream("MATCH (p:person) RETURN p.uid AS uid")) == [(10,), (20,), (30,)]

    assert social.execute("MATCH (p:person) RETURN count(*) AS n").fetchall() == [(3,)]


def test_a_stream_that_ran_out_says_what_it_did(many: zudb.Connection) -> None:
    stream = many.stream("MATCH (p:person) RETURN p.uid AS uid")
    assert stream.summary is None
    for _ in stream:
        pass
    summary = stream.summary

    assert summary is not None
    assert summary.columns == ["uid"]
    assert summary.rows == MANY
    assert summary.stopped is False
    assert summary.streamed is True
    assert summary.notices == []


def test_a_stream_stopped_early_says_how_much_of_it_was_read(many: zudb.Connection) -> None:
    """The summary of a run cut short is the thing that says how much of
    it happened, so a stream that was closed has one rather than none.
    """
    stream = many.stream("MATCH (p:person) RETURN p.uid AS uid", batch_rows=100)
    next(iter(stream))
    stream.close()
    summary = stream.summary

    assert summary is not None
    assert summary.stopped is True
    assert 0 < summary.rows < MANY


def test_a_statement_that_sorts_reads_the_same_and_says_it_was_buffered(
    many: zudb.Connection,
) -> None:
    """`ORDER BY` cannot give a row before it has seen every row, so the
    engine runs it whole and hands it over in batches afterwards. The
    loop is the loop either way and what differs is what it cost, which
    is why the summary says which of the two happened.
    """
    with many.stream("MATCH (p:person) RETURN p.uid AS uid ORDER BY p.uid DESC") as stream:
        rows = list(itertools.islice(stream, 3))
        stream.close()
        summary = stream.summary

    assert rows == [(MANY - 1,), (MANY - 2,), (MANY - 3,)]
    assert summary is not None
    assert summary.streamed is False


def test_closing_twice_is_not_an_error(social: zudb.Connection) -> None:
    stream = social.stream("MATCH (p:person) RETURN p.uid AS uid")
    stream.close()
    stream.close()

    assert stream.closed is True


def test_a_stream_nobody_read_still_frees_the_connection(many: zudb.Connection) -> None:
    with many.stream("MATCH (p:person) RETURN p.uid AS uid"):
        pass

    assert many.execute("MATCH (p:person) RETURN count(*) AS n").fetchall() == [(MANY,)]


def test_reading_a_closed_stream_gives_nothing_rather_than_rows(many: zudb.Connection) -> None:
    stream = many.stream("MATCH (p:person) RETURN p.uid AS uid")
    stream.close()

    assert list(stream) == []


def test_a_stream_of_a_statement_that_does_not_compile_fails(social: zudb.Connection) -> None:
    """The compile happens on the statement's own thread, so the failure
    arrives at the first row asked for rather than at the call that
    opened the stream, and it is the failure the same statement would
    have raised.
    """
    stream = social.stream("MATCH (")

    with pytest.raises(zudb.SyntaxError):
        next(iter(stream))


def test_the_same_failure_is_raised_at_every_row_asked_for(social: zudb.Connection) -> None:
    """A failure is kept rather than spent, because a loop that asked
    twice would otherwise be told the second time that the rows had run
    out, which is the one thing that did not happen.
    """
    stream = social.stream("MATCH (")
    with pytest.raises(zudb.SyntaxError):
        next(iter(stream))

    with pytest.raises(zudb.SyntaxError):
        next(iter(stream))


def test_a_stream_binds_its_parameters(social: zudb.Connection) -> None:
    statement = "MATCH (p:person) WHERE p.name = $name RETURN p.uid AS uid"

    assert list(social.stream(statement, {"name": "grace"})) == [(20,)]


def test_a_batch_of_no_rows_is_refused_rather_than_looped_on(social: zudb.Connection) -> None:
    with pytest.raises(ValueError, match="starts at one"):
        social.stream("MATCH (p:person) RETURN p.uid AS uid", batch_rows=0)


def test_streaming_on_a_closed_connection_is_refused(tmp_path: Path) -> None:
    conn = zudb.connect(tmp_path / "shut.zu1")
    conn.close()

    with pytest.raises(zudb.ProgrammingError, match="closed"):
        conn.stream("MATCH (p:person) RETURN p.uid AS uid")


def test_closing_the_connection_under_a_stream_stops_the_stream(paired: zudb.Connection) -> None:
    """Closing waits for the statement, so a connection cannot be closed
    out from under a stream that is still reading. What it does instead
    is hang the stream up first, which is why the close returns at all
    rather than waiting for a scan nobody is reading, and why what the
    stream reports afterwards is a run that was stopped.
    """
    stream = paired.stream(PAIRS_OF_PEOPLE)
    next(iter(stream))
    paired.close()

    assert list(stream) != []
    assert stream.closed is True
    summary = stream.summary
    assert summary is not None
    assert summary.stopped is True
    assert summary.rows < PAIRS


def test_a_stream_carries_the_values_a_result_would(loaded: zudb.Connection) -> None:
    """A node and an edge come back as the objects a result gives, named
    for their tables rather than numbered, which is the conversion this
    module had to carry over from the connection it borrowed the
    catalog from.
    """
    statement = "MATCH (a:person)-[r:knows]->(b:person) RETURN a, r, b"
    rows = list(loaded.stream(statement))

    assert rows == loaded.execute(statement).fetchall()
    node, rel, other = rows[0]
    assert isinstance(node, zudb.Node)
    assert isinstance(rel, zudb.Rel)
    assert isinstance(other, zudb.Node)
    assert node.table == "person"
    assert rel.table == "knows"


def test_a_stream_says_what_it_is(social: zudb.Connection) -> None:
    stream = social.stream("MATCH (p:person) RETURN p.uid AS uid")

    assert repr(stream) == '<zudb.Stream "MATCH (p:person) RETURN p.uid AS uid">'
    assert repr(stream.batches()).startswith("<zudb.StreamBatches of <zudb.Stream ")
    list(stream)
    assert repr(stream).endswith(", ended>")
    assert repr(stream.summary) == "<zudb.StreamSummary 3 rows>"
