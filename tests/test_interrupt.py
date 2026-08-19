"""Stopping a statement that is already running.

Two ways in, one word underneath. A person presses `Ctrl-C` and the
statement raises `KeyboardInterrupt` on the thread that asked for it; a
program calls `interrupt()` from another thread and the statement raises
`zudb.Interrupted`. Either way the connection is exactly as it was
afterwards, which is the difference between stopping a statement and
closing a connection.

The budget is 50 ms from the press to the exception, and it is asserted
rather than described: a long statement in a notebook that ignores the
keyboard is a kernel somebody kills.
"""

from __future__ import annotations

import _thread
import os
import signal
import sys
import threading
import time

import pytest
import zudb

# Every pair of twelve thousand people, filtered, which is about ten
# seconds of work on the machine this was written on. Nothing here waits
# for it to finish: the point of each test is that it does not.
WORK = "MATCH (a:person), (b:person) WHERE a.uid < b.uid RETURN count(a) AS n"

# What the milestone asks for, from the press to the exception.
BUDGET = 0.05

# Long enough that the statement is certainly inside the executor rather
# than still being parsed when the ask arrives.
SETTLED = 0.2


def press() -> None:
    """`Ctrl-C`, from inside the process that is going to feel it.

    Everywhere but Windows that is the signal a terminal sends, sent the
    way a terminal sends it: to this process, for the main thread to
    answer.

    Windows has no way to send one process a SIGINT. `os.kill` there is
    `TerminateProcess` for every signal except the two console events,
    so `os.kill(os.getpid(), SIGINT)` does not deliver a press, it kills
    the process, which is what it had been doing to this suite on every
    Windows run: five lines of dots and an exit code, no failure and no
    summary, because there was no interpreter left to write one. The two
    console events are no better, since a console event goes to every
    process attached to the console and on a build machine that is the
    build.

    So Windows presses the key the way CPython's own console handler
    does when a real press arrives: `PyErr_SetInterrupt`, which
    `_thread.interrupt_main` is the spelling of. What that leaves
    uncovered is one hop, from the operating system into the C runtime,
    and nothing running inside this process can cover that hop without
    taking the process with it. Everything after the hop is the same on
    both platforms and is all of the code this repository wrote: a
    tripped SIGINT, a statement deep in the executor, and the next
    `PyErr_CheckSignals` turning one into a `KeyboardInterrupt` on the
    main thread inside the budget.
    """
    if sys.platform == "win32":
        _thread.interrupt_main()
    else:
        os.kill(os.getpid(), signal.SIGINT)


def after(delay: float, do) -> threading.Thread:
    """Runs `do` on a thread of its own, `delay` from now, and answers
    the thread so a test can join it."""

    def wait() -> None:
        time.sleep(delay)
        do()

    thread = threading.Thread(target=wait)
    thread.start()
    return thread


@pytest.mark.timing
def test_a_press_raises_keyboard_interrupt_within_the_budget(throng: zudb.Connection) -> None:
    pressed: list[float] = []

    def timed_press() -> None:
        pressed.append(time.perf_counter())
        press()

    presser = after(SETTLED, timed_press)
    with pytest.raises(KeyboardInterrupt):
        throng.execute(WORK)
    felt = time.perf_counter()
    presser.join(timeout=30)
    assert felt - pressed[0] < BUDGET, f"the press took {felt - pressed[0]:.3f}s to arrive"


def test_the_connection_is_the_same_afterwards(throng: zudb.Connection) -> None:
    presser = after(SETTLED, press)
    with pytest.raises(KeyboardInterrupt):
        throng.execute(WORK)
    presser.join(timeout=30)
    # Not reopened, not reconnected: the same connection, which still
    # knows what it knew.
    assert throng.closed is False
    assert throng.execute("MATCH (p:person) RETURN count(p) AS n").fetchone() == (12_000,)


@pytest.mark.timing
def test_interrupt_stops_a_statement_from_another_thread(throng: zudb.Connection) -> None:
    asked: list[float] = []

    def ask() -> None:
        asked.append(time.perf_counter())
        throng.interrupt()

    asker = after(SETTLED, ask)
    with pytest.raises(zudb.Interrupted):
        throng.execute(WORK)
    felt = time.perf_counter()
    asker.join(timeout=30)
    assert felt - asked[0] < BUDGET, f"the ask took {felt - asked[0]:.3f}s to arrive"


def test_a_statement_on_another_thread_is_stopped_too(throng: zudb.Connection) -> None:
    """A worker thread's statement runs on the worker thread, since a
    signal cannot reach one, and `interrupt()` still ends it."""
    raised: list[BaseException] = []
    running = threading.Event()

    def run() -> None:
        running.set()
        try:
            throng.execute(WORK)
        except BaseException as why:  # noqa: BLE001
            raised.append(why)

    worker = threading.Thread(target=run)
    worker.start()
    running.wait(timeout=30)
    time.sleep(SETTLED)
    throng.interrupt()
    worker.join(timeout=30)
    assert not worker.is_alive()
    assert isinstance(raised[0], zudb.Interrupted)


def test_an_ask_with_nothing_running_does_not_end_the_next_statement(
    social: zudb.Connection,
) -> None:
    social.interrupt()
    assert social.execute("MATCH (p:person) RETURN count(p) AS n").fetchone() == (3,)


def test_a_press_with_nothing_running_is_pythons_to_deliver(social: zudb.Connection) -> None:
    with pytest.raises(KeyboardInterrupt):
        press()
        # Python raises the press at the next thing this thread does,
        # which is this call, and the statement it stopped is none.
        for _ in range(1000):
            pass
    assert social.execute("MATCH (p:person) RETURN count(p) AS n").fetchone() == (3,)


def test_interrupt_on_a_closed_connection_says_so(social: zudb.Connection) -> None:
    social.close()
    with pytest.raises(zudb.ProgrammingError, match="closed"):
        social.interrupt()


@pytest.mark.timing
def test_a_connection_answers_while_it_is_busy(throng: zudb.Connection) -> None:
    """Asking a connection how it is going does not queue behind the
    statement it is going through."""
    running = threading.Event()

    def run() -> None:
        running.set()
        with pytest.raises(zudb.Interrupted):
            throng.execute(WORK)

    worker = threading.Thread(target=run)
    worker.start()
    running.wait(timeout=30)
    time.sleep(SETTLED)
    asked = time.perf_counter()
    assert throng.closed is False
    assert throng.rows_read > 0, "a statement that has been running has read something"
    assert time.perf_counter() - asked < BUDGET, "asking waited for the statement"
    throng.interrupt()
    worker.join(timeout=30)
    assert not worker.is_alive()


def test_rows_read_holds_what_the_last_statement_cost(social: zudb.Connection) -> None:
    social.execute("MATCH (p:person) RETURN p.name AS n")
    assert social.rows_read >= 3


@pytest.mark.timing
def test_a_statement_that_finishes_is_not_slowed_by_being_watched(social: zudb.Connection) -> None:
    """The thread a statement runs on is kept rather than made, so a
    small statement on the main thread costs about what it costs on any
    other."""
    query = "MATCH (p:person) RETURN p.name AS n"
    social.execute(query)

    def timed() -> float:
        best = float("inf")
        for _ in range(5):
            start = time.perf_counter()
            for _ in range(200):
                social.execute(query)
            best = min(best, (time.perf_counter() - start) / 200)
        return best

    here = timed()
    elsewhere: list[float] = []
    worker = threading.Thread(target=lambda: elsewhere.append(timed()))
    worker.start()
    worker.join(timeout=120)
    # Generous, because this is a gate against paying for a thread per
    # statement rather than a benchmark: starting one costs some thirty
    # microseconds against a statement that costs ten.
    assert here < elsewhere[0] * 3, f"{here * 1e6:.1f}us here against {elsewhere[0] * 1e6:.1f}us"


def test_a_press_during_a_statement_that_finishes_first_is_still_raised(
    social: zudb.Connection,
) -> None:
    """A press is never swallowed. The statement was over before it
    arrived, so Python raises it at the next thing this thread does."""
    with pytest.raises(KeyboardInterrupt):
        press()
        social.execute("MATCH (p:person) RETURN p.name AS n")
        for _ in range(1000):
            pass
