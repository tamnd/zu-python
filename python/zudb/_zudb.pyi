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

    @property
    def rows_read(self) -> int:
        """How many rows the statement running on this connection has read out of storage."""

    def interrupt(self) -> None:
        """Asks the statement running on this connection to stop."""

    def execute(self, statement: str, params: Mapping[str, Value] | None = None) -> Result:
        """Runs one statement and gives back its rows."""

    def sql(self, statement: str, params: Mapping[str, Value] | None = None) -> Result:
        """The same call, named for the way it reads in a notebook."""

    def prepare(self, statement: str) -> Prepared:
        """Compiles a statement now and hands back something that runs it later."""

    def explain(self, statement: str) -> Plan:
        """What the statement would do, without doing it."""

    def profile(self, statement: str, params: Mapping[str, Value] | None = None) -> Profile:
        """Runs the statement with the counters on and answers what its operators really did."""

    def stream(
        self,
        statement: str,
        params: Mapping[str, Value] | None = None,
        *,
        batch_rows: int | None = None,
    ) -> Stream:
        """Runs one statement and hands back its rows as the engine makes them."""

    def transaction(self, *, read_only: bool = False) -> Transaction:
        """Starts a transaction and hands it back for a `with` block."""

    @property
    def in_transaction(self) -> bool:
        """Whether an explicit transaction is running on this connection."""

    def appender(self, table: str) -> Appender:
        """Opens an appender on `table`, for loading rows into a database that already exists."""

    def register(self, name: str, data: Any) -> int:
        """Puts a DataFrame under a name a statement can match on."""

    def unregister(self, name: str) -> None:
        """Takes a registered frame's name away."""

    @property
    def registered(self) -> list[str]:
        """The names this connection has registered frames under, sorted."""

    def close(self) -> None:
        """Closes the connection and frees what it held."""

    def __enter__(self) -> Connection: ...
    def __exit__(self, *_exception: object) -> bool: ...
    def __repr__(self) -> str: ...

class Transaction:
    """A transaction that has been started and not yet ended."""

    @property
    def read_only(self) -> bool:
        """Whether it was started `READ ONLY`."""

    @property
    def done(self) -> bool:
        """Whether this transaction has already been committed or rolled back."""

    def commit(self) -> None:
        """Ends the transaction and keeps what it wrote."""

    def rollback(self) -> None:
        """Ends the transaction and throws away what it wrote."""

    def __enter__(self) -> Transaction: ...
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

class Prepared:
    """A statement the engine has compiled and is holding for you."""

    @property
    def statement(self) -> str:
        """The text it was compiled from."""

    @property
    def params(self) -> list[str]:
        """The names this statement wants bound, in the order it uses them."""

    @property
    def closed(self) -> bool:
        """Whether this prepared statement has been closed."""

    def execute(self, params: Mapping[str, Value] | None = None) -> Result:
        """Runs it with these parameters and gives back its rows."""

    def sql(self, params: Mapping[str, Value] | None = None) -> Result:
        """The same call, named for the way it reads in a notebook."""

    def close(self) -> None:
        """Closes it and gives the statement back to the connection."""

    def __enter__(self) -> Prepared: ...
    def __exit__(self, *_exception: object) -> bool: ...
    def __repr__(self) -> str: ...

class Stream:
    """Rows arriving one batch at a time, while the statement is still running."""

    @property
    def columns(self) -> list[str]:
        """The column names, in the order the statement projects them."""

    @property
    def summary(self) -> StreamSummary | None:
        """What the statement did, once it has done it, and `None` while it is still running."""

    def batches(self) -> StreamBatches:
        """The rows in the batches they arrived in, as lists of tuples."""

    def close(self) -> None:
        """Stops the statement and gives the connection back."""

    @property
    def closed(self) -> bool:
        """Whether the statement is over, by running out of rows or by being closed."""

    def __iter__(self) -> Iterator[tuple[Value, ...]]: ...
    def __next__(self) -> tuple[Value, ...]: ...
    def __enter__(self) -> Stream: ...
    def __exit__(self, *_exception: object) -> bool: ...
    def __repr__(self) -> str: ...

class StreamBatches:
    """The same rows, in the batches they arrived in."""

    def __iter__(self) -> Iterator[list[tuple[Value, ...]]]: ...
    def __next__(self) -> list[tuple[Value, ...]]: ...
    def __repr__(self) -> str: ...

class StreamSummary:
    """What a streamed statement did, known once it has ended."""

    @property
    def columns(self) -> list[str]:
        """The column names, in the order the statement projected them."""

    @property
    def rows(self) -> int:
        """How many rows were handed over."""

    @property
    def stopped(self) -> bool:
        """Whether the reader stopped it before it ran out of rows."""

    @property
    def streamed(self) -> bool:
        """Whether the rows arrived as they were made."""

    @property
    def notices(self) -> list[dict[str, str]]:
        """The warnings the statement raised, in the shape a result reports them."""

    def __repr__(self) -> str: ...

class PlanNode:
    """One operator of a plan."""

    @property
    def op(self) -> str:
        """The kind of operator this is."""

    @property
    def name(self) -> str:
        """What the listing calls it, bracket and all."""

    @property
    def bracket(self) -> str | None:
        """The bracket it sits inside, if it sits inside one."""

    @property
    def detail(self) -> str:
        """What it works on, in the words the listing prints."""

    @property
    def binds(self) -> list[str]:
        """The variables it introduces."""

    @property
    def tables(self) -> list[str]:
        """The tables it reads."""

    @property
    def children(self) -> list[PlanNode]:
        """What it pulls from, in the order the listing prints them."""

    def __repr__(self) -> str: ...

class ScalarPlan:
    """A query written where a value belongs, and the plan it gets."""

    @property
    def reads(self) -> list[str]:
        """The variables it reads from the query it is written inside."""

    @property
    def exists(self) -> bool:
        """Whether it is asking whether there is a row rather than for one."""

    @property
    def plan(self) -> Plan:
        """The plan itself."""

    def __repr__(self) -> str: ...

class Plan:
    """What a statement would do, without doing it."""

    @property
    def root(self) -> PlanNode | None:
        """The operator everything else feeds."""

    @property
    def columns(self) -> list[str]:
        """The columns the statement projects, in order."""

    @property
    def params(self) -> list[str]:
        """The parameters it wants bound."""

    @property
    def notes(self) -> list[str]:
        """What the planner has to say about it, if anything."""

    @property
    def scalars(self) -> list[ScalarPlan]:
        """The plans of the queries written where values belong."""

    @property
    def text(self) -> str:
        """The engine's own listing, which is what `print` gives."""

    def _repr_html_(self) -> str:
        """The listing as a notebook shows it, which is the listing."""

    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class ProfileOp:
    """One operator of a statement that ran, and what it really did."""

    @property
    def op(self) -> str:
        """The kind of operator this is."""

    @property
    def detail(self) -> str:
        """What it worked on, in the words the listing prints."""

    @property
    def pulls(self) -> int:
        """How many times the operator above it asked for rows."""

    @property
    def rows(self) -> int:
        """How many rows it answered with."""

    @property
    def flat(self) -> int:
        """The same count with the vectors unpacked."""

    @property
    def estimate(self) -> float | None:
        """What the optimizer thought it would answer."""

    @property
    def bound(self) -> float | None:
        """The upper bound the optimizer had for it."""

    @property
    def nanos(self) -> int:
        """How long it spent, in nanoseconds."""

    @property
    def qerror(self) -> float | None:
        """The estimate over the truth, or the truth over the estimate,
        whichever is the larger."""

    def __repr__(self) -> str: ...

class ProfileStage:
    """One stage of a statement that ran."""

    @property
    def sink(self) -> str:
        """What the stage feeds."""

    @property
    def rows(self) -> int:
        """How many rows came out of it."""

    @property
    def nanos(self) -> int:
        """How long it took, in nanoseconds."""

    @property
    def ops(self) -> list[ProfileOp]:
        """Its operators, from the one that read to the one that fed the
        sink, which is the order they ran in and the reverse of the order
        the listing prints them.
        """

    def __repr__(self) -> str: ...

class Profile:
    """What a statement did, measured while it did it."""

    @property
    def stages(self) -> list[ProfileStage]:
        """The stages, in the order they ran."""

    @property
    def nanos(self) -> int:
        """Every stage added up, in nanoseconds."""

    @property
    def text(self) -> str:
        """The engine's own listing, which is what `print` gives."""

    def _repr_html_(self) -> str:
        """The listing as a notebook shows it, preformatted for the reason
        a plan's is."""

    def __str__(self) -> str: ...
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

    def _repr_html_(self) -> str:
        """The rows as an HTML table, which is what a notebook shows."""

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

    def _repr_html_(self) -> str:
        """The node as a notebook draws it, which is the pair that names it."""

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

    def _repr_html_(self) -> str:
        """The edge as a notebook draws it, with the rows it joins on either side."""

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

    def _repr_html_(self) -> str:
        """The walk as a notebook draws it, nodes and arrows alternating."""

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
