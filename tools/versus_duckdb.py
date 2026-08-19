"""What a call costs here, against what the same call costs DuckDB.

This is the script behind `docs/clients/duckdb.md` in the engine tree,
kept in the repository so the numbers in that page can be reproduced
rather than believed. A million rows of three columns, an integer, a
double and a short string, in a stored table on both sides, and the
fastest of nine alternating runs for each call, because what is being
compared is the code path and not the scheduler.

One row of the output is not a comparison and is marked as such.
DuckDB's `execute` hands back a result nobody has read yet, so what it
takes is the cost of planning and of nothing else; ours has the whole
answer in memory by the time it returns.

Run: python tools/versus_duckdb.py
Needs: pip install duckdb pyarrow pandas
"""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

import duckdb
import zudb

ROWS = 1_000_000
RUNS = 9


def duel(ours, theirs) -> tuple[float, float]:
    """The fastest of `RUNS` for each, run alternately.

    Alternately rather than one after the other, which matters more than
    the number of runs does. A laptop is not a quiet machine, and
    timing all of one call and then all of the other hands whichever
    went second whatever the machine was doing by then. Interleaving
    gives both of them the same weather.
    """
    fastest = [float("inf"), float("inf")]
    for _ in range(RUNS):
        for at, run in enumerate((ours, theirs)):
            started = time.perf_counter()
            run()
            fastest[at] = min(fastest[at], time.perf_counter() - started)
    return fastest[0], fastest[1]


def read(call):
    """A call that has to have read every row by the time it returns.

    The trap this closes is a handle: a call that hands back a lazy
    reader has done no work, times at nought, and is not the same call
    as one that has the answer. Anything that does not answer with
    `ROWS` did not do what it was being timed for.
    """

    def run() -> None:
        count = call()
        if count != ROWS:
            raise SystemExit(f"read {count} rows and not {ROWS}, so this is not the same call")

    return run


def rows() -> list[tuple[int, float, str]]:
    return [(n, n * 1.5, f"s{n % 10_000:04d}") for n in range(ROWS)]


def build(directory: Path):
    """Both databases, loaded with the same rows."""
    data = rows()

    zudb.load(
        str(directory / "zu"),
        nodes="row",
        columns={
            "id": [row[0] for row in data],
            "f": [row[1] for row in data],
            "s": [row[2] for row in data],
        },
    )
    zu = zudb.connect(str(directory / "zu"))

    duck = duckdb.connect(str(directory / "duck.db"))
    duck.register("staged", _arrow(data))
    duck.execute("CREATE TABLE row AS SELECT * FROM staged")
    duck.unregister("staged")
    return zu, duck


def _arrow(data):
    import pyarrow as pa

    return pa.table(
        {
            "id": pa.array([row[0] for row in data], pa.int64()),
            "f": pa.array([row[1] for row in data], pa.float64()),
            "s": pa.array([row[2] for row in data], pa.string()),
        }
    )


def ms(seconds: float) -> str:
    """Milliseconds, with a decimal only where dropping it would round
    the figure to nothing."""
    return f"{seconds * 1e3:.1f} ms" if seconds < 0.01 else f"{seconds * 1e3:.0f} ms"


def ratio(ours: float, theirs: float) -> str:
    if ours < theirs:
        return f"{theirs / ours:.1f}x faster"
    return f"{ours / theirs:.1f}x slower"


def line(label: str, ours, theirs, note: str = "") -> None:
    ours, theirs = duel(ours, theirs)
    print(f"| {label} | {ms(ours)} | {ms(theirs)} | {note or ratio(ours, theirs)} |")


def micro(label: str, ours, theirs) -> None:
    """The same row, for a call small enough to read in microseconds."""
    ours, theirs = duel(ours, theirs)
    print(f"| {label} | {ours * 1e6:.1f} us | {theirs * 1e6:.1f} us | {ratio(ours, theirs)} |")


def main() -> None:
    directory = Path(tempfile.mkdtemp(prefix="zu-versus-"))
    try:
        zu, duck = build(directory)
        query = "MATCH (r:row) RETURN r.id, r.f, r.s"
        sql = "SELECT id, f, s FROM row"

        print("| call | zu | DuckDB | ratio |")
        print("|---|---|---|---|")
        line(
            "execute, nothing read",
            lambda: zu.execute(query),
            lambda: duck.execute(sql),
            note="not a comparison",
        )
        line(
            "execute and `fetchall`",
            read(lambda: len(zu.execute(query).fetchall())),
            read(lambda: len(duck.execute(sql).fetchall())),
        )
        # DuckDB's `arrow()` hands back a `RecordBatchReader` that has
        # read nothing, which times at nought and is not the same call.
        # `to_arrow_table` is the one that has the table when it
        # returns, which is what ours does.
        line(
            "execute and Arrow table",
            read(lambda: zu.execute(query).to_arrow().num_rows),
            read(lambda: duck.execute(sql).to_arrow_table().num_rows),
        )
        line(
            "execute and pandas",
            read(lambda: len(zu.execute(query).to_pandas())),
            read(lambda: len(duck.execute(sql).df())),
        )

        point = "MATCH (r:row) WHERE r.id = 500000 RETURN r.f"
        point_sql = "SELECT f FROM row WHERE id = 500000"

        def swap(connection, table):
            def run() -> None:
                connection.register("frame", table)
                connection.unregister("frame")

            return run

        table = _arrow(rows())
        numbers = table.select(["id", "f"])

        print()
        print("| call | zu | DuckDB | ratio |")
        print("|---|---|---|---|")
        micro(
            "point read, whole call",
            lambda: zu.execute(point).fetchall(),
            lambda: duck.execute(point_sql).fetchall(),
        )
        micro(
            "`register` and `unregister`, numbers",
            swap(zu, numbers),
            swap(duck, numbers),
        )
        # The same call with a string column in the frame, which is a
        # different answer and is why both are printed: the bytes get
        # a validation pass on the way in and the numbers do not.
        micro(
            "`register` and `unregister`, with strings",
            swap(zu, table),
            swap(duck, table),
        )

        zu.close()
        duck.close()
    finally:
        shutil.rmtree(directory, ignore_errors=True)


if __name__ == "__main__":
    main()
