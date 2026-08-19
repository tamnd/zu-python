"""What a call costs here, against what the same call costs DuckDB.

This is the script behind `docs/clients/duckdb.md` in the engine tree,
kept in the repository so the numbers in that page can be reproduced
rather than believed. A million rows of three columns, an integer, a
double and a short string, in a stored table on both sides, and the
fastest of five runs for each call, because what is being compared is
the code path and not the scheduler.

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
RUNS = 5


def best(run) -> float:
    """The fastest of `RUNS`, in seconds."""
    fastest = float("inf")
    for _ in range(RUNS):
        at = time.perf_counter()
        run()
        fastest = min(fastest, time.perf_counter() - at)
    return fastest


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


def line(label: str, ours: float, theirs: float, note: str = "") -> None:
    if note:
        ratio = note
    elif ours < theirs:
        ratio = f"{theirs / ours:.1f}x faster"
    else:
        ratio = f"{ours / theirs:.1f}x slower"
    print(f"| {label} | {ours * 1e3:.0f} ms | {theirs * 1e3:.0f} ms | {ratio} |")


def micro(label: str, ours: float, theirs: float) -> None:
    """The same row, for a call small enough to read in microseconds."""
    if ours < theirs:
        ratio = f"{theirs / ours:.1f}x faster"
    else:
        ratio = f"{ours / theirs:.1f}x slower"
    print(f"| {label} | {ours * 1e6:.1f} us | {theirs * 1e6:.1f} us | {ratio} |")


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
            best(lambda: zu.execute(query)),
            best(lambda: duck.execute(sql)),
            note="not a comparison",
        )
        line(
            "execute and `fetchall`",
            best(read(lambda: len(zu.execute(query).fetchall()))),
            best(read(lambda: len(duck.execute(sql).fetchall()))),
        )
        # DuckDB's `arrow()` hands back a `RecordBatchReader` that has
        # read nothing, which times at nought and is not the same call.
        # `to_arrow_table` is the one that has the table when it
        # returns, which is what ours does.
        line(
            "execute and Arrow table",
            best(read(lambda: zu.execute(query).to_arrow().num_rows)),
            best(read(lambda: duck.execute(sql).to_arrow_table().num_rows)),
        )
        line(
            "execute and pandas",
            best(read(lambda: len(zu.execute(query).to_pandas()))),
            best(read(lambda: len(duck.execute(sql).df()))),
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
            best(lambda: zu.execute(point).fetchall()),
            best(lambda: duck.execute(point_sql).fetchall()),
        )
        micro(
            "`register` and `unregister`, numbers",
            best(swap(zu, numbers)),
            best(swap(duck, numbers)),
        )
        # The same call with a string column in the frame, which is a
        # different answer and is why both are printed: the bytes get
        # a validation pass on the way in and the numbers do not.
        micro(
            "`register` and `unregister`, with strings",
            best(swap(zu, table)),
            best(swap(duck, table)),
        )

        zu.close()
        duck.close()
    finally:
        shutil.rmtree(directory, ignore_errors=True)


if __name__ == "__main__":
    main()
