"""What happens when more than one thread is involved.

A connection is not concurrent and never claims to be: statements on
one run in the order they were asked for, one at a time. What these
check is that the lock which enforces that is taken with the GIL
released, since a thread waiting for the connection while holding the
GIL would stop the thread inside the engine from ever returning.
"""

from __future__ import annotations

import threading
from pathlib import Path

import zudb

# Every pair of people, filtered, which is a statement that runs for
# long enough to watch rather than one that is over before the other
# thread is scheduled.
WORK = "MATCH (a:person), (b:person) WHERE a.uid < b.uid RETURN count(a) AS n"
PAIRS = 150 * 149 // 2


def test_two_threads_sharing_one_connection_both_finish(crowd: zudb.Connection) -> None:
    answers: list[object] = []
    lock = threading.Lock()

    def run() -> None:
        got = crowd.execute(WORK).fetchone()
        with lock:
            answers.append(got)

    threads = [threading.Thread(target=run) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
        assert not thread.is_alive(), "a thread is still waiting for the connection"
    assert answers == [(PAIRS,)] * 4


def test_a_connection_per_thread_reads_the_same_database(tmp_path: Path) -> None:
    path = tmp_path / "shared.zu1"
    with zudb.connect(path) as conn:
        conn.execute("INSERT (p:person {uid: 1, name: 'ada'})")

    answers: list[object] = []
    lock = threading.Lock()

    def run() -> None:
        with zudb.connect(path, read_only=True) as conn:
            got = conn.execute("MATCH (p:person) RETURN p.name AS n").fetchone()
        with lock:
            answers.append(got)

    threads = [threading.Thread(target=run) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
        assert not thread.is_alive(), "a thread is still waiting for the database"
    assert answers == [("ada",)] * 4


def test_python_keeps_running_while_a_statement_does(crowd: zudb.Connection) -> None:
    ticks = 0
    done = threading.Event()

    def run() -> None:
        for _ in range(4):
            crowd.execute(WORK)
        done.set()

    worker = threading.Thread(target=run)
    worker.start()
    while not done.is_set():
        ticks += 1
    worker.join(timeout=120)
    assert not worker.is_alive()
    # A GIL held for the length of a statement would leave this loop
    # the gaps between four of them, which is a handful of turns and
    # not thousands of them.
    assert ticks > 1000, f"the main thread only got {ticks} turns"
