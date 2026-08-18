"""What a case is, and how a file of them is read.

A case is a statement and what running it must produce. That is
deliberately the whole of it. Every client in every language can run a
statement and look at the rows that come back, so a corpus written in
those terms is one every client can run, and a corpus written in terms
of a client's own API would be nine corpora.

The expectation is either rows or a condition. A case expecting a
condition names the GQLSTATUS code, not the message, because the code is
the contract and the message is prose that will improve.

A statement may take parameters, which is the other direction the same
values travel: a case with ``params:`` writes a value in the encoding,
hands it to this client's own binding call, and asserts what came back.
A client that decodes a date correctly and encodes it a day early passes
every case that has no parameters in it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import values
from .reader import CorpusError, Node, parse, quote

__all__ = ["SCHEMA", "Case", "Suite", "Column", "Load", "read_dir"]

#: The schema version a file declares. It exists so that a corpus
#: unpacked from an old release tells a new runner what it is instead of
#: failing in the middle.
SCHEMA = 3

_SUITE_KEYS = ("schema", "suite", "doc", "load", "cases")
_CASE_KEYS = ("name", "doc", "setup", "params", "query", "columns", "rows", "raises")
_LOAD_KEYS = ("nodes", "edges", "count", "columns", "pairs")


@dataclass
class Case:
    """One statement and what it owes.

    ``columns`` and ``rows`` are set together or neither is, and
    ``raises`` is set when neither is: a case says what it produces one
    way or the other."""

    name: str
    doc: str
    query: str
    line: int
    setup: list[str] = field(default_factory=list)
    params: list[tuple[str, object]] = field(default_factory=list)
    columns: list[str] | None = None
    rows: list[list[object]] | None = None
    raises: str | None = None


@dataclass
class Column:
    """One column of a load: a name, the type every value in it has, and
    the values in row order."""

    name: str
    ty: str
    values: list[object]


@dataclass
class Load:
    """One node table, its columns, and the edges between its rows.

    Everything else in the corpus is an expression, and an expression
    says what a value means on the way out and nothing about how it got
    in. A load is the other half, and every runner puts it in through
    its own bulk load path, which for this client is ``zudb.load``."""

    nodes: str
    edges: str
    count: int
    columns: list[Column]
    pairs: list[tuple[int, int]]


@dataclass
class Suite:
    """One file of cases."""

    name: str
    doc: str
    load: Load | None
    cases: list[Case]


def read_dir(directory: Path) -> list[Suite]:
    """Every suite in a directory, in the order a sorted listing gives,
    which is the order the reference runner walks them in."""
    suites = []
    for path in sorted(Path(directory).glob("*.yaml")):
        try:
            suite = read(path.read_text(encoding="utf-8"))
        except CorpusError as e:
            raise CorpusError(f"{path}: {e}") from None
        if suite.name != path.stem:
            raise CorpusError(
                f"{path}: the suite calls itself {quote(suite.name)} and the file calls it "
                f"{quote(path.stem)}"
            )
        suites.append(suite)
    if not suites:
        raise CorpusError(f"{directory}: no case files")
    return suites


def read(text: str) -> Suite:
    """A suite, or the first thing in the file that is not one."""
    doc = parse(text)
    unknown = doc.unknown(_SUITE_KEYS)
    if unknown:
        raise CorpusError(f"line {doc.line}: a suite has no key {quote(unknown[0])}")
    schema_node = doc.get("schema")
    schema = schema_node.str_() if schema_node is not None else None
    if schema is None:
        raise CorpusError("the file does not open with `schema:`")
    try:
        version = int(schema)
    except ValueError:
        raise CorpusError(f"{quote(schema)} is not a schema version") from None
    if version != SCHEMA:
        raise CorpusError(f"this is schema {version} and the runner reads schema {SCHEMA}")

    name = _field(doc, "suite")
    doc_text = _field(doc, "doc")
    load_node = doc.get("load")
    load = _load(load_node) if load_node is not None else None

    cases_node = doc.get("cases")
    if cases_node is None:
        raise CorpusError("a suite with no `cases:`")
    items = cases_node.seq()
    if items is None:
        raise CorpusError("`cases:` is a sequence")
    if not items:
        raise CorpusError("a suite with no cases in it")

    cases = [_case(item) for item in items]
    # Names are what a report cites and what a binding's skip list
    # names, so two cases sharing one is a report that says less than it
    # looks like it does.
    seen: set[str] = set()
    for case in cases:
        if case.name in seen:
            raise CorpusError(f"two cases are called {quote(case.name)}")
        seen.add(case.name)
    return Suite(name=name, doc=doc_text, load=load, cases=cases)


def _field(node: Node, key: str) -> str:
    value = node.get(key)
    if value is None:
        raise CorpusError(f"line {node.line}: no `{key}:`")
    text = value.str_()
    if text is None:
        raise CorpusError(f"line {node.line}: `{key}:` is one line of text")
    return text


def _case(node: Node) -> Case:
    line = node.line
    if node.map() is None:
        raise CorpusError(f"line {line}: a case is a mapping, and this is {node.what()}")
    unknown = node.unknown(_CASE_KEYS)
    if unknown:
        raise CorpusError(f"line {line}: a case has no key {quote(unknown[0])}")

    name = _field(node, "name")
    spelled = name.isascii() and all(c.islower() or c.isdigit() or c == "-" for c in name)
    if not name or not spelled:
        raise CorpusError(
            f"line {line}: {quote(name)} is a case name, which is lower case words joined by dashes"
        )
    doc = _field(node, "doc")
    query = _field(node, "query")

    setup: list[str] = []
    setup_node = node.get("setup")
    if setup_node is not None:
        items = setup_node.seq()
        if items is None:
            raise CorpusError(f"line {line}: `setup:` is a sequence of statements")
        for item in items:
            text = item.str_()
            if text is None:
                raise CorpusError(f"line {item.line}: a setup statement is one line")
            setup.append(text)

    params = _params(node)

    raises_node = node.get("raises")
    columns_node = node.get("columns")
    if raises_node is not None and columns_node is not None:
        raise CorpusError(
            f"line {line}: a case that raises has no rows, and one that returns rows does not raise"
        )
    if raises_node is not None:
        code = raises_node.str_()
        if code is None:
            raise CorpusError(f"line {line}: `raises:` is a GQLSTATUS code")
        if len(code) != 5 or not all(c.isdigit() or (c.isupper() and c.isascii()) for c in code):
            raise CorpusError(
                f"line {raises_node.line}: {quote(code)} is not the shape of a GQLSTATUS, which is "
                "five characters of digits and capitals"
            )
        return Case(name, doc, query, line, setup, params, raises=code)
    if columns_node is None:
        raise CorpusError(
            f"line {line}: a case says what it produces, with `columns:` and `rows:` or with "
            "`raises:`"
        )
    names = columns_node.seq()
    if names is None:
        raise CorpusError(f"line {line}: `columns:` is a sequence of names")
    columns = []
    for item in names:
        text = item.str_()
        if text is None:
            raise CorpusError(f"line {item.line}: a column name is one word")
        columns.append(text)
    rows = _rows(node)
    for row in rows:
        if len(row) != len(columns):
            raise CorpusError(f"line {line}: a row of {len(row)} against {len(columns)} columns")
    return Case(name, doc, query, line, setup, params, columns=columns, rows=rows)


def _params(node: Node) -> list[tuple[str, object]]:
    """The parameters a case binds, which is the value encoding with a
    name beside it.

    A name is what the statement spells after the ``$``, so it is checked
    against what a statement may spell: a case whose name is ``n one`` is
    one no client can bind."""
    params_node = node.get("params")
    if params_node is None:
        return []
    items = params_node.seq()
    if items is None:
        raise CorpusError(f"line {params_node.line}: `params:` is a sequence")
    out: list[tuple[str, object]] = []
    for item in items:
        line = item.line
        if item.map() is None:
            raise CorpusError(
                f"line {line}: a parameter is a mapping of `name`, `type` and `value`, and this "
                f"is {item.what()}"
            )
        unknown = item.unknown(("name", "type", "value"))
        if unknown:
            raise CorpusError(f"line {line}: a parameter has no key {quote(unknown[0])}")
        name = _field(item, "name")
        if not name or not all(c.isascii() and (c.isalnum() or c == "_") for c in name):
            raise CorpusError(
                f"line {line}: {quote(name)} is a parameter name, which is what a statement writes "
                "after the `$`"
            )
        if any(n == name for n, _ in out):
            raise CorpusError(f"line {line}: two parameters are called {quote(name)}")
        out.append((name, values.typed(item)))
    return out


def _rows(node: Node) -> list[list[object]]:
    rows_node = node.get("rows")
    if rows_node is None:
        # A statement that returns no rows is a case worth having, and
        # writing it as an absent `rows:` would make it the same shape as
        # one somebody forgot to finish.
        raise CorpusError(
            f"line {node.line}: `columns:` with no `rows:`. A case expecting nothing back writes "
            "`rows:` with an empty sequence under it."
        )
    items = rows_node.seq_or_empty()
    if items is None:
        raise CorpusError(f"line {rows_node.line}: `rows:` is a sequence of rows")
    out: list[list[object]] = []
    for item in items:
        unknown = item.unknown(("values",))
        if unknown:
            raise CorpusError(f"line {item.line}: a row has no key {quote(unknown[0])}")
        cells_node = item.get("values")
        if cells_node is None:
            raise CorpusError(f"line {item.line}: a row is a `values:` and the values under it")
        cells = cells_node.seq_or_empty()
        if cells is None:
            raise CorpusError(f"line {cells_node.line}: `values:` is a sequence of values")
        out.append([values.decode(cell) for cell in cells])
    return out


def _load(node: Node) -> Load:
    line = node.line
    if node.map() is None:
        raise CorpusError(f"line {line}: a load is a mapping, and this is {node.what()}")
    unknown = node.unknown(_LOAD_KEYS)
    if unknown:
        raise CorpusError(f"line {line}: a load has no key {quote(unknown[0])}")
    nodes = _name(node, "nodes")
    edges = _name(node, "edges")
    count_node = node.get("count")
    count_text = count_node.str_() if count_node is not None else None
    if count_text is None:
        raise CorpusError(f"line {line}: a load says how many rows it has, with `count:`")
    try:
        count = int(count_text)
    except ValueError:
        raise CorpusError(f"line {line}: `count:` is a number of rows") from None
    if count == 0:
        raise CorpusError(f"line {line}: a load of no rows is a load nothing can be read back from")

    columns_node = node.get("columns")
    if columns_node is None:
        raise CorpusError(f"line {line}: a load has `columns:`")
    items = columns_node.seq()
    if items is None:
        raise CorpusError(f"line {line}: `columns:` is a sequence")
    columns = [_column(item, count) for item in items]
    if not columns:
        raise CorpusError(f"line {line}: a load with no columns holds no values")
    seen: set[str] = set()
    for column in columns:
        if column.name in seen:
            raise CorpusError(f"line {line}: two columns are called {quote(column.name)}")
        seen.add(column.name)

    pairs: list[tuple[int, int]] = []
    pairs_node = node.get("pairs")
    if pairs_node is not None:
        items = pairs_node.seq_or_empty()
        if items is None:
            raise CorpusError(f"line {line}: `pairs:` is a sequence of edges")
        for item in items:
            pairs.append(_edge(item, count))
    return Load(nodes=nodes, edges=edges, count=count, columns=columns, pairs=pairs)


def _name(node: Node, key: str) -> str:
    text = _field(node, key)
    if not text or not all(c.isascii() and (c.isalnum() or c == "_") for c in text):
        raise CorpusError(f"line {node.line}: {quote(text)} is not a table name")
    return text


def _column(node: Node, count: int) -> Column:
    line = node.line
    unknown = node.unknown(("name", "type", "values"))
    if unknown:
        raise CorpusError(f"line {line}: a column has no key {quote(unknown[0])}")
    name = _name(node, "name")
    ty = _field(node, "type")
    if values.form(ty) is None:
        raise CorpusError(f"line {line}: {ty} is not a type this encoding knows")
    values_node = node.get("values")
    items = values_node.seq() if values_node is not None else None
    if items is None:
        raise CorpusError(f"line {line}: a column holds `values:` in row order")
    if len(items) != count:
        raise CorpusError(
            f"line {line}: column {quote(name)} holds {len(items)} values against the {count} "
            "rows the load declares"
        )
    return Column(name=name, ty=ty, values=[values.payload(ty, item) for item in items])


def _edge(node: Node, count: int) -> tuple[int, int]:
    line = node.line
    unknown = node.unknown(("from", "to"))
    if unknown:
        raise CorpusError(f"line {line}: an edge has no key {quote(unknown[0])}")
    ends = []
    for key in ("from", "to"):
        value = node.get(key)
        text = value.str_() if value is not None else None
        if text is None:
            raise CorpusError(f"line {line}: an edge has a `{key}:` row number")
        try:
            end = int(text)
        except ValueError:
            raise CorpusError(f"line {line}: `{key}:` is a row number") from None
        if not 0 <= end < count:
            raise CorpusError(
                f"line {line}: `{key}: {end}` against a table of {count} rows, which are numbered "
                f"0 to {count - 1}"
            )
        ends.append(end)
    return ends[0], ends[1]
