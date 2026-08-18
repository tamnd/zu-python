"""Types for the compiled module.

The extension is a shared object, so nothing can read a signature out of
it: mypy, pyright and an editor's completion all read this file
instead. It ships inside the wheel next to `py.typed`, and CI checks it
against the module it describes with griffe, which loads this file
statically and the compiled module by inspection and compares them. A
stub that promises a method the engine does not have is a lie a checker
would believe, so it is worth a gate.

The docstrings here are the first line of each one in the Rust source,
because a stub is what an editor shows and an editor showing nothing is
what a stub is for.
"""

from __future__ import annotations

import datetime
import os
import pathlib
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any

from .types import Value

#: The revision of the C ABI this client answers to.
__abi_version__: str
#: The version of the engine compiled into the wheel.
__engine_version__: str

def connect(
    path: str | os.PathLike[str],
    *,
    read_only: bool = False,
    memory_limit: int | None = None,
    threads: int | None = None,
) -> Connection:
    """Opens the database at `path` and connects to it."""

def load(
    path: str | os.PathLike[str],
    *,
    nodes: str,
    rels: str = "rel",
    columns: Mapping[str, Iterable[Value]] | None = None,
    edges: Iterable[Sequence[int]] | None = None,
    rows: int | None = None,
) -> dict[str, int]:
    """Writes a new database at `path` and answers what went into it."""

class Connection:
    """One connection to one database."""

    @property
    def path(self) -> pathlib.Path:
        """The file this connection was opened on."""

    @property
    def read_only(self) -> bool:
        """Whether it was opened read-only."""

    @property
    def closed(self) -> bool:
        """Whether this connection is still open."""

    def execute(self, statement: str, params: Mapping[str, Value] | None = None) -> Result:
        """Runs one statement and gives back its rows."""

    def sql(self, statement: str, params: Mapping[str, Value] | None = None) -> Result:
        """The same call, named for the way it reads in a notebook."""

    def appender(self, table: str) -> Appender:
        """Opens an appender on `table`, for loading rows into a database that already exists."""

    def close(self) -> None:
        """Closes the connection and frees what it held."""

    def __enter__(self) -> Connection: ...
    def __exit__(self, *_exception: object) -> bool: ...
    def __repr__(self) -> str: ...

class Appender:
    """Rows on their way into a table, buffered until they are flushed."""

    @property
    def table(self) -> str:
        """The table this appender writes to."""

    @property
    def buffered(self) -> int:
        """Rows buffered and not yet written."""

    @property
    def committed(self) -> int:
        """Rows this appender has committed, across every flush."""

    @property
    def closed(self) -> bool:
        """Whether this appender has been closed."""

    def append_row(self, row: Iterable[Value]) -> None:
        """Appends one row, which is a sequence of one value per column of the table."""

    def append_rows(self, rows: Iterable[Iterable[Value]]) -> int:
        """Appends every row of an iterable of rows."""

    def flush(self) -> int:
        """Writes every buffered row and makes it readable."""

    def discard(self) -> int:
        """Throws away what is buffered and answers how many rows that was."""

    def close(self) -> int:
        """Flushes what is left and answers how many rows this appender committed in all."""

    def __enter__(self) -> Appender: ...
    def __exit__(self, *_exception: object) -> bool: ...
    def __repr__(self) -> str: ...

class Result:
    """The rows a statement gave back."""

    @property
    def columns(self) -> list[str]:
        """The column names, in the order the statement projected them."""

    @property
    def notices(self) -> list[dict[str, str]]:
        """The warnings the statement raised, if it raised any."""

    def fetchall(self) -> list[tuple[Value, ...]]:
        """Every row, as a list of tuples."""

    def fetchone(self) -> tuple[Value, ...] | None:
        """The next row, or `None` when there are no more."""

    # `Any` and not `pyarrow.Table`, because the wheel does not depend
    # on pyarrow and a stub that imported it would fail to resolve for
    # every caller who does not have it either.
    def to_arrow(self) -> Any:
        """The rows as a `pyarrow.Table`."""

    def to_pandas(self) -> Any:
        """The rows as a `pandas.DataFrame`, with Arrow-backed dtypes."""

    def to_polars(self) -> Any:
        """The rows as a `polars.DataFrame`."""

    def record_batches(self) -> Any:
        """The rows as a `pyarrow.RecordBatchReader`, a batch at a time."""

    def __arrow_c_stream__(self, requested_schema: object | None = None) -> Any:
        """The rows as an Arrow stream, for anything that speaks Arrow."""

    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[tuple[Value, ...]]: ...
    def __repr__(self) -> str: ...

class Node:
    """One node of the graph."""

    def __init__(self, table: str, offset: int) -> None: ...
    @property
    def table(self) -> str:
        """The name the table was given in the schema."""

    @property
    def offset(self) -> int:
        """The row this node sits at in that table, counting from zero."""

    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...

class Rel:
    """One edge of the graph."""

    def __init__(self, table: str, src: int, dst: int, ord: int) -> None: ...
    @property
    def table(self) -> str:
        """The name the rel table was given in the schema."""

    @property
    def src(self) -> int:
        """The row the edge leaves, in the node table it joins."""

    @property
    def dst(self) -> int:
        """The row the edge arrives at."""

    @property
    def ord(self) -> int:
        """Where the edge's properties sit, which is its place in
        the order the table was loaded in."""

    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...

class Path:
    """A walk: nodes and edges alternating, a node at each end."""

    def __init__(self, elements: list[Node | Rel]) -> None: ...
    @property
    def elements(self) -> list[Node | Rel]:
        """The walk as it is stored, a node and an edge at a time."""

    @property
    def nodes(self) -> list[Node]:
        """The nodes of the walk, in the order it visits them."""

    @property
    def rels(self) -> list[Rel]:
        """The edges of the walk, in the order it crosses them."""

    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...

class Duration:
    """A duration, which Python has no type for."""

    def __init__(self, months: int = 0, nanoseconds: int = 0) -> None: ...
    @property
    def months(self) -> int:
        """Months, for a year-month duration. Zero for a day-time one."""

    @property
    def nanoseconds(self) -> int:
        """Nanoseconds, for a day-time duration. Zero for a year-month one."""

    @property
    def kind(self) -> str:
        """`"year_month"` or `"day_time"`."""

    def to_timedelta(self) -> datetime.timedelta:
        """The same duration as a `datetime.timedelta`, rounded towards zero."""

    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...
