# zu for Python

The Python client for [zu](https://github.com/tamnd/zu), an embedded property-graph database. In-process, columnar, vectorized, and speaks [ISO/IEC 39075 GQL](https://www.iso.org/standard/76120.html).

```python
import zudb

with zudb.connect("social.zu1") as conn:
    conn.execute("INSERT (p:person {uid: 1, name: 'ada'})")
    conn.execute("INSERT (p:person {uid: $uid, name: $name})", {"uid": 2, "name": "grace"})

    people = conn.execute("MATCH (p:person) RETURN p.name AS name, p.uid AS uid")
    print(people.to_pandas())
```

```
pip install "zudb[pandas]"
```

No compiler, no `pkg-config`, no postinstall script. One wheel per platform with the engine inside it. `pip install zudb` on its own brings nothing else at all; the extra above is pandas, which the last line of the snippet asks for and which a result hands its columns to over Arrow. A test in this repository runs that snippet exactly as it is printed, in a directory of its own, because a quickstart is the most read and least compiled code a project has.

## What this is

Built with PyO3 and maturin, linked against the engine crates rather than against `libzu`'s C ABI. That is ADR 0002 in the engine repository, and it is about mechanism and not about contract: this client and every client that does go through `zu.h` answer the same conformance corpus, which is what says they agree. Linking the crates is what lets a query result reach Python without being flattened through C on the way.

The interesting parts:

- **Arrow all the way down.** `to_arrow()`, `to_pandas()`, `to_polars()`, and `record_batches()` go through the Arrow C Data Interface with no intermediate copy. `to_pandas()` hands back Arrow-backed dtypes, which is what pandas 3 wants anyway.
- **`register()` replacement scans.** A DataFrame in your session becomes a table you can query by name, zero-copy for Arrow-backed frames. This is the DuckDB idea worth copying wholesale, and it deletes the write-to-disk-then-load step from every "load my data" tutorial.
- **The GIL is released** around every query, load, and appender flush, so threads actually parallelize.
- **`Ctrl-C` interrupts a running query** within 50 ms, because a long query in a notebook that looks like a hang gets the kernel killed.
- **Complete `.pyi` stubs inside the wheel**, checked against the runtime in CI, so mypy and pyright and your editor all work with no extra install.
- **`import zudb` costs about 4 ms** on this machine and is gated at 50, and pandas, polars and pyarrow are imported when you ask for one and not before. Importing pandas costs 700 ms, which is most of why none of them is a dependency.
- **Graph values are real classes.** `Node`, `Rel`, and `Path` have `.labels`, `.id`, `.properties`, and an HTML repr. Not dicts, because a dict cannot tell a property named `labels` apart from the label set.

## Building a graph

A statement writes one row at a time, which is the wrong shape for loading data and cannot make a rel table at all. `load` is the other shape: a table's columns whole, the edges between them whole, one file written once.

```python
import zudb

zudb.load(
    "social.zu1",
    nodes="person",
    rels="knows",
    columns={"uid": [1, 2, 3], "name": ["ada", "grace", "kay"]},
    edges=[(0, 1), (1, 2)],
)

with zudb.connect("social.zu1", read_only=True) as conn:
    for a, b in conn.execute(
        "MATCH (a:person)-[:knows]->(b:person) RETURN a.name AS a, b.name AS b"
    ):
        print(a, "knows", b)
```

Edges name rows by position, counting from zero, because at load time a row has no other name. Columns may hold booleans, integers, floats, strings, dates, times, datetimes or durations, one kind to a column, and the GIL is released for the write.

## Adding to one that exists

`load` writes a database and an appender grows one. Rows go into per-column buffers in memory and a flush turns the whole buffer into one commit, so a million rows cost one commit rather than a million.

```python
with conn.appender("person") as rows:
    for uid, name in enumerate(names):
        rows.append_row([uid, name])
```

A row is every column of the table, in the order the table declares them, and the columns come from the table rather than from the first row, so a value that does not belong in one is refused by the call that appended it and the message names the column. A rel table takes rows too, and a row of one is the two ends of an edge as offsets into the tables it runs between, which is how a Python program adds an edge at all.

The `with` block closes, and closing flushes, including on the way out of a block that raised: a load that stopped partway is better served by its rows arriving than by them vanishing, and `discard()` is there for the caller who wants the other answer. A flush that fails keeps its rows, so what did not go in is still there to look at.

An appender nobody closed is the one mistake that cannot be reported where it happens, and it raises a `ResourceWarning` naming the table and the rows when the collector takes it. Flushing from there is the other answer and is not available, because a collector runs whenever it likes, including while another thread is inside a statement on the same connection. The warning is the whole point: a loop that appended a million rows and never closed leaves a database with nothing in it, and going quietly about that is worse than a line on stderr.

On this machine 200,000 rows of an integer and a string take 11 ms to buffer through `append_rows` and 93 ms to flush, which is 1.9 million rows a second including the commit. The same rows one call at a time cost 32 ms of buffering, since a call is a call. Against `INSERT`, 2,000 rows take 32 seconds a row at a time and 28 ms through an appender, and the gap widens with the table because every `INSERT` is a commit and a fold.

## Several statements as one unit of work

Every statement already runs in a transaction of its own, so this is not what makes a write atomic. What it holds is the span: two statements are one unit, and the block rolls back when it raises, which is the failure worth writing it for since it is the one nobody wrote a handler for.

```python
with conn.transaction():
    conn.execute("INSERT (a:account {uid: 1, balance: 100})")
    conn.execute("INSERT (b:account {uid: 2, balance: 0})")
```

It starts at the call rather than at the `with`, so a transaction that cannot start says so at the line that asked for one, and `commit()` and `rollback()` are there for a caller who would rather say when. A block whose transaction has already ended is left alone on the way out, which is what lets a program commit early and carry on, and ending one twice is refused rather than ignored, because the statements between the two are in neither of them. `conn.in_transaction` answers which side of the block a program is on.

`transaction(read_only=True)` starts one that refuses to write, at the statement that writes rather than at the block that would have written. One transaction runs at a time and a second is refused rather than nested, since a rollback of an inner one would have to invent an answer for what it undoes. The three words underneath are `START TRANSACTION`, `COMMIT` and `ROLLBACK`, and they all still work written out.

An appender is refused inside a transaction. Its batches are commits of their own and a rollback does not take them back, so an appender opened in a block would promise a span it is not in: load first, then transact. A statement that failed leaves the transaction running, because the engine does not end one on a failed statement and the block that opened it is still the thing that closes it. A connection closed with work uncommitted drops that work, which is the same answer the block would have given.

The wrapper costs 5 microseconds for an empty transaction, so what it costs is what the engine charges. On this machine that is more rather than less: 200 `INSERT`s cost 2.2 seconds each committing on its own and 3.3 seconds inside one transaction, and reads cost the same either way. A transaction here is worth taking for the span it holds and not for the time it saves, and the v0 write path is where that number has to change.

## Bringing a DataFrame in

A frame a program already has becomes something a statement can match on, under a name the program picks.

```python
conn.register("people", frame)
conn.execute("MATCH (p:people) WHERE p.age > 40 RETURN p.name AS name")
```

Anything that speaks Arrow goes in, which is a pandas or polars DataFrame, a pyarrow table or a reader over one, and a dictionary of lists is there for a caller with none of them installed. Nothing is copied. The frame arrives over the same C Data Interface a result leaves by, and what the engine is told is where each column is, how wide its values are and what they mean; a statement that names it builds vectors pointing straight at the caller's buffers. So registering costs what describing the columns costs and not what the rows cost: 2.6 microseconds for ten rows and 2.3 for ten million.

The one column that is walked is a string column, and it is walked once. Every offset is checked at registration so that reading the frame afterwards cannot fail, which is 362 microseconds for a million strings. Two other things copy and both are said rather than hidden: a stream that arrives as several batches is concatenated into one, because a column of a table is one run of bytes and two batches are two of them, and a dictionary of Python lists is read into buffers of this client's own, because a list holds objects rather than numbers and there is nothing in it to point at.

Because it is not a copy, a registered frame is a view and not a snapshot. Write into the array behind it and the next statement answers what is there now, which is the thing to know about the call and the reason it is worth having. Reading one is as fast as reading a table of the database and faster where the database has to decode: over a million rows on this machine, summing an integer column takes 662 microseconds against a stored table's 833, and finding one row by a string takes 1.8 ms against 3.1.

The frame belongs to the connection it was registered on and goes when that connection does. Nothing is written to the file, so another program opening the same database has never heard of it, and nothing writes to it either: a statement that inserts into or deletes from a registered name is refused with the reason, because that memory is the caller's DataFrame. `unregister(name)` takes the name away and hands the bytes back, which is not always that instant, since a statement still reading the frame holds it until it ends. `conn.registered` says what is registered here.

Registering the same name again replaces what it stands for, columns and all, which is what rerunning a cell means by it. Registering over a table the database already holds is refused, since a statement naming it would mean the stored one. A frame with no rows is a table to match on and answers nothing, because a frame knows its columns without being told by a row. A null anywhere is refused by column and row, since a property that is null is one no row of this engine holds, and registering inside a transaction is refused because a frame is registered on the session, which is the thing the transaction is running on.

## Reading a result as columns

A result is rows to iterate and columns to hand to something else. The columns go out over the Arrow C Data Interface, so pyarrow, pandas and polars each read the same buffers and none of them gets a Python object per cell.

```python
result = conn.execute("MATCH (p:person) RETURN p.name AS name, p.score AS score")
result.to_arrow()  # pyarrow.Table
result.to_pandas()  # DataFrame with Arrow-backed dtypes
result.to_polars()  # polars.DataFrame
result.record_batches()  # a reader, for a result larger than memory
```

`Result` implements `__arrow_c_stream__`, so anything that reads the protocol reads a result directly and none of the four methods above is needed: `pyarrow.table(result)` and `polars.DataFrame(result)` both work. Batches are 65,536 rows. A column holds one type, which the values decide, and integers beside floats are the one mixture that widens rather than being refused. Nodes, rels and paths go across as structs. The copy runs with the GIL released, and on this machine 300,000 rows across three columns take 44 ms as Arrow against 67 ms as Python objects, and a single integer column takes 13.8 ms against 44.5 ms.

## In a notebook

A result in a cell draws itself as a table, because Jupyter asks an object for `_repr_html_` before it falls back to `repr` and a line saying how many rows there are is a strictly worse answer than the rows. Nodes, rels and paths draw themselves too, a path as the walk it is: `(person #0) -[knows]-> (person #1)`.

```python
%load_ext zudb.magic
%gql social.zu1
```

```python
%%gql
MATCH (p:person) WHERE p.score > 40 RETURN p.name AS name, p.score AS score
```

The cell is one statement, it runs on the current connection, and the result is the value of the cell, so `_` is a `zudb.Result` and everything a result can do is still there. `%gql` is about which connection and `%%gql` is about the statement, and neither guesses the other's job. A notebook that already called `zudb.connect` needs no `%gql` at all: if exactly one connection is lying about the namespace `%%gql` uses it, and if there is more than one it names them and asks which. `%%gql --conn other --params args --out rows` says which connection, where the parameters are, and where to put the result instead of showing it, each of them naming a variable because a notebook has the values already.

The markup is a table, a stylesheet and no script, so it survives `nbconvert`, an exported HTML file and a notebook diff, and there is nothing to install for any of it. Colours come from the notebook through `currentColor` and opacity, because a light theme and a dark one are both in the room. Values are escaped, since a string column holding `<script>` is a string column and a client that pasted one into the page would run a caller's data as code in their notebook. The first hundred rows are drawn and the note underneath says how many there were, and a value longer than two hundred characters is cut with a mark where it was cut, because a million rows of markup is a notebook file that will not open again.

IPython is not a dependency. It is what a notebook already has, and nothing here imports it until `%load_ext` does.

## Stopping a statement

A statement that is running can be stopped two ways, and neither of them closes the connection: the session, its plans and its warm readers are all there afterwards, which is the whole difference between stopping a statement and starting again.

```python
conn.execute(long_one)  # Ctrl-C raises KeyboardInterrupt here
conn.interrupt()  # from another thread, raises zudb.Interrupted there
conn.rows_read  # how far the statement running now has got
```

`Ctrl-C` is the one a person presses, and it raises `KeyboardInterrupt` on the thread that called `execute`, measured at 5 ms from the press on this machine against a budget of 50. Python only delivers a signal to the main thread between two bytecodes, so a statement called from the main thread runs on a thread this client keeps for it and the main thread waits and asks for signals while it does. That thread is kept rather than made per statement, because making one costs 30 microseconds against a small statement that costs 10, and a statement called from any other thread runs inline where a signal was never going to arrive anyway.

`interrupt()` is the one a program calls, from a thread that is not the one inside `execute`, and it raises `zudb.Interrupted` there. It is one of the three calls that may be made on a connection while a statement is running, with `rows_read` and `closed`, and none of the three waits for it: a progress bar drawn from `rows_read` is a poll of an atomic, not a queue behind the executor.

## On an event loop

A statement runs inside Rust with the GIL down and comes back when it comes back. Called straight from a coroutine it stops the loop for that whole time, including the tasks answering requests that have nothing to do with the database, so `zudb.aio` gives each connection a thread of its own and hands it every call that could wait.

```python
import asyncio

import zudb.aio


async def main():
    async with zudb.aio.connect("social.zu1") as conn:
        await conn.execute("INSERT (p:person {uid: 1, name: 'ada'})")
        rows = await conn.execute("MATCH (p:person) RETURN p.name AS name")
        for (name,) in rows:
            print(name)


asyncio.run(main())
```

A thread per connection rather than a pool shared between them, because a connection is one lock and statements on it queue anyway: a pool would add no parallelism the engine can use and would let two statements written one after the other run in the other order. Two connections do run at the same time, since the engine puts the GIL down for the work, and two statements together cost 1.09 times what one costs alone on this machine.

What comes back is a `zudb.Result`, which is rows already in memory, so reading them is the call it was: `for row in rows` and `rows.to_arrow()` are not awaited and never were. Only the ways in are, and only the ones that can wait. `path`, `read_only`, `closed`, `rows_read` and `interrupt()` are answered from beside the lock rather than through it, so they stay properties and a progress bar drawn from `rows_read` still reads while the statement it is measuring runs.

Cancelling the task that awaits a statement interrupts the statement. The engine is asked to stop and the coroutine does not return until it has, so the connection is idle again by the time the `CancelledError` reaches the caller rather than busy with work nobody is waiting for, and a statement still queued when the cancellation arrives is dropped without running. A transaction block cancelled partway is a block that raised, so it leaves through the rollback.

`transaction()` and `appender()` are opened with `async with`, `in_transaction()` and `registered()` are methods here because both answers live behind the lock, and everything else is the sync call with an `await` in front of it.

## Through the DB-API

PEP 249 is what a Python program expects a database to look like, and the code that expects it is not always code anyone can change: a dashboard, a test harness, a helper somebody wrote against sqlite3 five years ago. `zudb.dbapi` is that shape, over the same connection.

```python
import zudb.dbapi

with zudb.dbapi.connect("social.zu1") as conn:
    cur = conn.cursor()
    cur.execute("INSERT (p:person {uid: 1, name: 'ada', score: 41.0})")
    cur.execute("MATCH (p:person) WHERE p.score > ? RETURN p.name AS name", (40,))
    print(cur.fetchall())
```

Parameters are `?`, which PEP 249 calls `qmark`, rewritten into the engine's own `$name` before the statement runs. The `named` style cannot work here: `:name` is how a pattern names a label, so `(p:person)` and `WHERE p.uid = :uid` cannot be told apart without parsing the statement, while `?` is a character GQL has no meaning for anywhere. A question mark inside a string, a quoted name or a comment is text somebody wrote and is left alone, and passing a dict instead of a sequence hands the statement over untouched, so `$name` still works for anyone writing zu statements rather than generating them.

Transactions are implicit, which PEP 249 requires and the native client does not do: one opens before the first statement after each `commit` or `rollback`, and closing a connection rolls back what was not committed. `connect(..., autocommit=True)` turns that off and gives back the native behaviour, where every statement stands alone.

The exception classes are both hierarchies at once. `zudb.Error` is the `Error` PEP 249 asks for, and a syntax error is a `zudb.SyntaxError` and a `dbapi.ProgrammingError` and the same object, carrying the same code, position and documentation link, so a driver-shaped library and code written against this client can catch the same failure in the same program. `cur.description` names the columns and gives the Python type of what is in them, read from the first rows, since a result declares no types of its own. `conn.zu` is the connection underneath and `cur.result` is the result the last statement gave back, so appenders, registered frames, `interrupt()` and `to_arrow()` are all still there. It is a layer, not a second client.

## When a program is wrong

Every condition arrives as an exception class carrying its GQLSTATUS code, its position and a link to what the standard says about it, and the class is the class a Python caller would have written: a mistake the program made is a `zudb.ProgrammingError`, a value Python has and zu does not is a `TypeError`, a value of the right type and the wrong shape is a `ValueError`, and a file that is not a database is a `zudb.ConnectionError`, the same as a file that is not there. Both of those are a path that does not lead to a database, and telling a caller who mistyped one to file a bug would be the wrong answer twice.

`tests/test_misuse.py` is twenty-three deliberately wrong programs and what each of them is told, run against the same list the engine runs in `crates/zu/tests/misuse.rs`. A message has to name the thing the caller named, say what was expected instead, and be the engine's own sentence rather than a syscall's, because "failed to fill whole buffer" is a true statement about a read that tells nobody which file was not a database. The suite checks the other two words too: nothing crashes, and nothing leaks, which is five hundred failing connects followed by a database that still opens and no connection left alive behind the collector's back. Half of it is the programs that look wrong and are not, since a parameter nothing reads, a label nothing carries and a second `close()` are all decisions somebody would otherwise reverse by accident.

## Types

The wheel carries `py.typed` and a stub for the compiled module, so mypy, pyright and an editor's completion all work with nothing else installed. `zudb.Value` is the union a row holds and a parameter takes, for code that passes rows around and wants to say so.

The stub is checked against the module it describes in CI: griffe reads the stub as text and the installed extension by inspection, and the two have to agree on every name, every parameter and every default. A stub is a promise no interpreter checks, so something has to.

## What works today

The list above is what this client is for. What it does so far is the core of it: `connect`, `execute` and `sql` with named parameters, results that iterate and fetch, values as Python objects both ways including dates, times, datetimes and durations, `Node`, `Rel` and `Path` as classes, `load` for building a graph with edges in it, an appender for growing one, transactions as a context manager that commits at the end of a block and rolls back when it raises, every condition as an exception class carrying its code, its position and its documentation link, results as Arrow columns and as pandas and polars frames, `register` for putting a frame under a name a statement can match on and reading it where it lies, stubs inside the wheel with a gate that keeps them true, the GIL released around every statement, every load and every copy out, `Ctrl-C` and `interrupt()` stopping a statement without touching the connection under it, `zudb.aio` for the same calls awaited on an event loop, results, nodes, rels and paths that draw themselves in a notebook with `%gql` and `%%gql` to run statements in one, and `zudb.dbapi` for code written against PEP 249. Each one landed with the tests that say it works.

## Wheels

Three per platform, which is more than it sounds like it should be and is not optional. The free-threaded CPython build has no stable ABI until 3.15 and [PEP 803](https://peps.python.org/pep-0803/)'s `abi3t`, so 3.14t needs a version-specific wheel of its own. From 3.15 one wheel serves both builds and carries both ABI tags, which is what PEP 803 is for and the reason this stops at three.

| Tag | Covers |
|---|---|
| `cp311-abi3` | CPython 3.11 through 3.14, GIL-enabled |
| `cp314-cp314t` | free-threaded 3.14 |
| `cp315-abi3.abi3t` | 3.15 and every later 3.x, both builds |

Platforms: manylinux_2_28 and musllinux on x86_64 and aarch64, macOS universal2, Windows x64 and arm64. An `sdist` that builds with only a Rust toolchain is published too, and is built back into a wheel in CI.

That is twenty-one wheels and the release checks all twenty-one, twice. Each build is held to the tag it asked for, and then the grid is checked as a grid: every cell filled and nothing outside it. A build that cannot find the interpreter it wants does not fail, it falls back and produces a version-specific wheel that works on the machine that built it and claims nothing about any other version, which is the kind of thing nobody notices until somebody's install resolves to it.

Optional extras, none required: `zudb[pandas]`, `[polars]`, `[arrow]`, `[viz]`, `[all]`. The base wheel depends on nothing.

## Specification

Spec/2064g/dx/06-python.md in [tamnd/zu](https://github.com/tamnd/zu). Milestones: DX2 (tamnd/zu#168) and DX3 (tamnd/zu#169).

## Status

Pre-1.0 and pre-release. Nothing is published yet. The engine, the C ABI, and this client all move on one version number, so a release here always pairs with the same release of [`tamnd/zu`](https://github.com/tamnd/zu).

## Where things live

| What | Where |
|---|---|
| Engine, Rust SDK, CLI, `zu.h`, conformance corpus | [tamnd/zu](https://github.com/tamnd/zu) |
| Documentation and website | [tamnd/zu-web](https://github.com/tamnd/zu-web) |
| This client | here |

If a bug reproduces through the `zu` CLI, it belongs in [tamnd/zu](https://github.com/tamnd/zu/issues), not here.

## License

Apache-2.0, same as the engine.
