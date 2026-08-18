# zu for Python

The Python client for [zu](https://github.com/tamnd/zu), an embedded property-graph database. In-process, columnar, vectorized, and speaks [ISO/IEC 39075 GQL](https://www.iso.org/standard/76120.html).

```python
import zudb

with zudb.connect("social.zu1") as conn:
    conn.execute("INSERT (p:person {uid: 1, name: 'ada'})")
    conn.execute("INSERT (p:person {uid: $uid, name: $name})", {"uid": 2, "name": "grace"})

    for name, uid in conn.execute("MATCH (p:person) RETURN p.name AS name, p.uid AS uid"):
        print(name, uid)
```

```
pip install zudb
```

No compiler, no `pkg-config`, no postinstall script. One wheel per platform with the engine inside it.

## What this is

Built with PyO3 and maturin, linked against the engine crates rather than against `libzu`'s C ABI. That is ADR 0002 in the engine repository, and it is about mechanism and not about contract: this client and every client that does go through `zu.h` answer the same conformance corpus, which is what says they agree. Linking the crates is what lets a query result reach Python without being flattened through C on the way.

The interesting parts:

- **Arrow all the way down.** `to_arrow()`, `to_pandas()`, `to_polars()`, and `record_batches()` go through the Arrow C Data Interface with no intermediate copy. `to_pandas()` hands back Arrow-backed dtypes, which is what pandas 3 wants anyway.
- **`register()` replacement scans.** A DataFrame in your session becomes a table you can query by name, zero-copy for Arrow-backed frames. This is the DuckDB idea worth copying wholesale, and it deletes the write-to-disk-then-load step from every "load my data" tutorial.
- **The GIL is released** around every query, load, and appender flush, so threads actually parallelize.
- **`Ctrl-C` interrupts a running query** within 50 ms, because a long query in a notebook that looks like a hang gets the kernel killed.
- **Complete `.pyi` stubs inside the wheel**, checked against the runtime in CI, so mypy and pyright and your editor all work with no extra install.
- **Graph values are real classes.** `Node`, `Rel`, and `Path` have `.labels`, `.id`, `.properties`, and an HTML repr. Not dicts, because a dict cannot tell a property named `labels` apart from the label set.

## Building a graph

A statement writes one row at a time, which is the wrong shape for loading data and cannot make a rel table at all. `load` is the other shape: a table's columns whole, the edges between them whole, one file written once.

```python
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

## Types

The wheel carries `py.typed` and a stub for the compiled module, so mypy, pyright and an editor's completion all work with nothing else installed. `zudb.Value` is the union a row holds and a parameter takes, for code that passes rows around and wants to say so.

The stub is checked against the module it describes in CI: griffe reads the stub as text and the installed extension by inspection, and the two have to agree on every name, every parameter and every default. A stub is a promise no interpreter checks, so something has to.

## What works today

The list above is what this client is for. What it does so far is the core of it: `connect`, `execute` and `sql` with named parameters, results that iterate and fetch, values as Python objects both ways including dates, times, datetimes and durations, `Node`, `Rel` and `Path` as classes, `load` for building a graph with edges in it, every condition as an exception class carrying its code, its position and its documentation link, results as Arrow columns and as pandas and polars frames, stubs inside the wheel with a gate that keeps them true, and the GIL released around every statement, every load and every copy out. `register` and the interrupt are next, and each one lands with the tests that say it works.

## Wheels

Three per platform, which is more than it sounds like it should be and is not optional. The free-threaded CPython build has no stable ABI until 3.15 and [PEP 803](https://peps.python.org/pep-0803/)'s `abi3t`, so 3.14t needs a version-specific wheel of its own.

| Tag | Covers |
|---|---|
| `cp311-abi3` | CPython 3.11 through 3.14, GIL-enabled |
| `cp314-cp314t` | free-threaded 3.14 |
| `cp315-abi3t` | 3.15 and every later 3.x, both builds |

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
