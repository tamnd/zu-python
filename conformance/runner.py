"""Running the corpus through this client, and saying what happened in
the form the other eight runners are compared against.

Each case gets a database of its own. Cases in a suite are written as if
nothing came before them, and the cheapest way to keep that true is to
make it true: a case that leaked a table into the next one would be a
failure that moves when the file is reordered, which is the worst kind to
be handed.

An outcome is one of three things and not two. Passed and failed are
obvious. Unsupported is the third, and it exists because the corpus is
versioned with the engine and shipped to nine clients that will not all
implement the same subset at the same time: a client that cannot yet
parse a statement should say so, and a report should be able to tell
that apart from an answer that came back wrong.

What this prints is what the Rust runner prints, line for line, so that
a disagreement between two clients is a diff and not a reading exercise.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import zudb

from .cases import Case, Load, Suite, read_dir
from .reader import CorpusError
from .values import same, show, too_fine, truncated

__all__ = ["Ran", "Report", "run", "main"]

PASSED = "passed"
FAILED = "failed"
UNSUPPORTED = "unsupported"

_MARK = {PASSED: "ok", FAILED: "FAILED", UNSUPPORTED: "unsupported"}


@dataclass
class Ran:
    """What one case did, with the account of why when it did not pass."""

    suite: str
    case: str
    line: int
    outcome: str
    #: What went wrong, in enough detail to fix the case or the engine
    #: without running it again.
    detail: str = ""

    def __str__(self) -> str:
        head = f"{self.suite}/{self.case} line {self.line} {_MARK[self.outcome]}"
        return f"{head}: {self.detail}" if self.detail else head


@dataclass
class Report:
    """What a whole run did."""

    ran: list[Ran] = field(default_factory=list)

    def count(self, outcome: str) -> int:
        return sum(1 for r in self.ran if r.outcome == outcome)

    def failures(self) -> list[Ran]:
        return [r for r in self.ran if r.outcome == FAILED]

    def summary(self) -> str:
        """One line saying what the run came to, which is what a CI log
        keeps and what two runs are compared by."""
        return (
            f"{len(self.ran)} cases, {self.count(PASSED)} passed, {self.count(FAILED)} failed, "
            f"{self.count(UNSUPPORTED)} unsupported"
        )


def run(suites: list[Suite], directory: Path) -> Report:
    """Runs every case of every suite, in the order they were written.

    ``directory`` is one the runner may make databases under. Each case
    gets its own file in it, named after the case, so that a failure
    leaves something to open."""
    report = Report()
    for suite in suites:
        for case in suite.cases:
            report.ran.append(_one(suite, case, directory))
    return report


def _one(suite: Suite, case: Case, directory: Path) -> Ran:
    def ran(outcome: str, detail: str = "") -> Ran:
        return Ran(suite.name, case.name, case.line, outcome, detail)

    unheld = _unheld(case)
    if unheld is not None:
        return ran(UNSUPPORTED, unheld)

    path = directory / f"{suite.name}-{case.name}.zu"
    # The load goes in before the connection opens, because it is bulk
    # load and bulk load is the path that builds the file rather than one
    # that goes through a statement. Every case of the suite gets its own
    # copy of it for the same reason every case gets its own database.
    if suite.load is not None:
        try:
            _apply(suite.load, path)
        except zudb.Error as e:
            return ran(FAILED, f"the suite's load: {e}")
    try:
        conn = zudb.connect(path)
    except zudb.Error as e:
        return ran(FAILED, f"opening {path}: {e}")

    try:
        for i, statement in enumerate(case.setup):
            try:
                conn.execute(statement)
            except zudb.Error as e:
                # A setup that fails is not a result about the statement
                # under test, so it is never a pass and never a quiet
                # skip.
                if _unsupported(e):
                    return ran(UNSUPPORTED, f"setup {i + 1}: {_said(e)}")
                return ran(FAILED, f"setup {i + 1} failed: {_said(e)}")

        params = dict(case.params) if case.params else None
        try:
            result = conn.execute(case.query, params)
            rows = result.fetchall()
        except zudb.Error as e:
            if case.raises is not None:
                if e.code is None:
                    return ran(
                        FAILED,
                        f"failed with no GQLSTATUS where the case wants {case.raises}: {_said(e)}",
                    )
                if e.code == case.raises:
                    return ran(PASSED)
                return ran(
                    FAILED,
                    f"raised {e.code} where the case wants {case.raises}: {_said(e)}",
                )
            if _unsupported(e):
                return ran(UNSUPPORTED, _said(e))
            return ran(FAILED, _said(e))

        if case.raises is not None:
            return ran(FAILED, f"returned rows where the case wants {case.raises}")
        detail = _compare(case.columns or [], case.rows or [], result.columns, rows)
        return ran(PASSED) if detail is None else ran(FAILED, detail)
    finally:
        conn.close()


def _apply(load: Load, path: Path) -> None:
    """The suite's load, through this client's own bulk load path, which
    is the strongest form of the corpus question: the value crosses the
    boundary twice and by two different mechanisms.

    A column holding a value finer than this client can carry goes in
    truncated rather than not at all, so that the cases reading the
    other columns of the same suite still run. The cases that read that
    column back say what happened, because their own expectation is a
    value nothing equals."""
    zudb.load(
        path,
        nodes=load.nodes,
        rels=load.edges,
        columns={column.name: truncated(column.values) for column in load.columns},
        edges=load.pairs,
        rows=load.count,
    )


def _unheld(case: Case) -> str | None:
    """Why this client cannot run the case at all, if it cannot.

    The one reason so far is a temporal written finer than a microsecond,
    which the engine keeps and Python's ``datetime`` does not. It is
    unsupported rather than failed because nothing went wrong: the
    engine answered, and the value mapping this client documents is
    where the digits went. Reported before anything runs, because the
    value would otherwise be handed to a binding that has no idea what it
    is. The suite's load is not looked at, because a column this client
    truncates only matters to the cases that read it back, and those
    cases say so on their own."""
    where = [value for _, value in case.params]
    where += [value for row in case.rows or [] for value in row]
    for value in where:
        found = too_fine(value)
        if found is not None:
            return (
                f"this client holds a time to the microsecond and the case writes "
                f"{show(found)}, which is finer"
            )
    return None


def _unsupported(e: zudb.Error) -> bool:
    """Whether a condition means the engine does not implement the
    statement rather than that the statement is wrong.

    The two GQL classes that say so are 42, syntax error or access rule
    violation, and 0A, feature not supported. A case landing on either is
    a case ahead of the engine, which the corpus allows on purpose: the
    cases are the contract and the engine catches up to them."""
    return e.code is not None and (e.code.startswith("42") or e.code.startswith("0A"))


def _said(e: zudb.Error) -> str:
    """What the engine said, which is what the Rust runner prints for the
    same failure. The message a condition carries already opens with its
    own code, so nothing is added to it here and the two reports diff
    clean."""
    return str(e)


def _compare(
    want_columns: list[str],
    want_rows: list[list[object]],
    got_columns: list[str],
    got_rows: list[tuple[object, ...]],
) -> str | None:
    """What differs between what a case wants and what came back, or
    ``None`` if nothing does.

    It reports the first difference rather than all of them, because the
    first is nearly always the cause of the rest, and a report that
    prints a hundred rows is one nobody reads to the end. The order the
    checks run in is the reference runner's, so that two runners looking
    at the same wrong answer say the same thing about it."""
    if want_columns != got_columns:
        return f"columns {_names(got_columns)} where the case wants {_names(want_columns)}"
    for i, (want, got) in enumerate(zip(want_rows, got_rows, strict=False)):
        for j, (a, b) in enumerate(zip(want, got, strict=False)):
            if not same(a, b):
                name = want_columns[j] if j < len(want_columns) else "?"
                return f"row {i + 1} column {name} is {show(b)} where the case wants {show(a)}"
    if len(want_rows) != len(got_rows):
        return f"{len(got_rows)} rows where the case wants {len(want_rows)}"
    return None


def _names(columns: list[str]) -> str:
    """A list of column names the way Rust's ``{:?}`` writes one."""
    quoted = ", ".join(f'"{name}"' for name in columns)
    return f"[{quoted}]"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m conformance",
        description="run the shared corpus cases against this client",
    )
    parser.add_argument("dir", type=Path, help="the directory of case files")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="an unsupported case fails the run, which is what a release branch wants",
    )
    parser.add_argument("--quiet", action="store_true", help="print the summary and nothing else")
    parser.add_argument(
        "--work",
        type=Path,
        default=None,
        help="a directory to make the case databases under, kept rather than removed",
    )
    args = parser.parse_args(argv)

    try:
        suites = read_dir(args.dir)
    except CorpusError as e:
        # One rather than two, because the reference runner exits one for
        # a corpus it cannot read and a report that is compared line for
        # line is worth less if the two disagree about what the run came
        # to.
        print(f"zu corpus: {e}", file=sys.stderr)
        return 1

    if args.work is not None:
        args.work.mkdir(parents=True, exist_ok=True)
        report = run(suites, args.work)
    else:
        with tempfile.TemporaryDirectory(prefix="zudb-corpus-") as work:
            report = run(suites, Path(work))

    if not args.quiet:
        for ran in report.ran:
            if ran.outcome != PASSED:
                print(ran)
    print(report.summary())
    if report.count(FAILED):
        return 1
    if args.strict and report.count(UNSUPPORTED):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
