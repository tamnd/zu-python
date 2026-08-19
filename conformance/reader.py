"""The subset of YAML the corpus is written in.

YAML is a large language and the corpus needs a small corner of it:
block mappings, block sequences, and scalars. Everything else is refused
with a line number. The files are hand written and are read by people in
nine repositories who did not write them, so a construct a reader
quietly reinterpreted would be a case that says one thing to a reviewer
and another to the runner.

So: two space indentation and no tabs, ``- `` with exactly one space,
plain, single quoted and double quoted scalars on one line, and
comments. No flow collections, no block scalars, no anchors, no aliases,
no tags, no document markers, no multi document streams.

This is the third implementation of that subset, after
``crates/zu-corpus/src/yaml.rs`` in the engine and ``conformance/c/yaml.c``
beside it. PyYAML would read these files, and would read a good deal
more besides: it would take a flow sequence, a block scalar and an
anchor, none of which a case may use, and it would hand back ``9223372036854775807``
as a float on the way. What the corpus needs is a reader that refuses,
and the cheapest way to have one is to write it.

Whether a scalar was quoted survives parsing, because the value encoding
turns on it. An INT64 written bare is a number some reader in some
language will round, and refusing it is the whole point of the encoding.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["Node", "CorpusError", "parse", "quote"]


def quote(text: str) -> str:
    """A string the way Rust's ``{:?}`` writes one.

    Every refusal in the corpus is written in three languages and diffed
    across them, so a value quoted one way here and another way there
    would be a difference in the report that is not a difference in the
    answer. Python's ``repr`` reaches for single quotes and Rust never
    does, so the quoting is written out rather than borrowed."""
    out = ['"']
    for c in text:
        if c in '"\\':
            out.append("\\" + c)
        elif c == "\n":
            out.append("\\n")
        elif c == "\r":
            out.append("\\r")
        elif c == "\t":
            out.append("\\t")
        else:
            out.append(c)
    out.append('"')
    return "".join(out)


class CorpusError(Exception):
    """A file the corpus will not read, with the line it gave up on."""


@dataclass(frozen=True)
class Node:
    """One node of a document, with the line it started on.

    Four kinds, told apart by which field is set. ``EMPTY`` is a key
    with nothing under it: it is a node rather than an error because a
    case that expects no rows back writes ``rows:`` and stops, and that
    is a real expectation which needs a spelling. Every accessor says no
    to it, so a ``name:`` left blank is still caught by whoever wanted a
    name.
    """

    kind: str
    line: int
    text: str = ""
    quoted: bool = False
    items: tuple[Node, ...] = ()
    pairs: tuple[tuple[str, Node], ...] = ()

    def what(self) -> str:
        """What kind of node this is, for an error that has to say what
        it found instead of what it wanted."""
        return {
            "scalar": "a scalar",
            "seq": "a sequence",
            "map": "a mapping",
            "empty": "nothing",
        }[self.kind]

    def scalar(self) -> tuple[str, bool] | None:
        return (self.text, self.quoted) if self.kind == "scalar" else None

    def str_(self) -> str | None:
        return self.text if self.kind == "scalar" else None

    def seq(self) -> tuple[Node, ...] | None:
        return self.items if self.kind == "seq" else None

    def seq_or_empty(self) -> tuple[Node, ...] | None:
        """A sequence, counting a key with nothing under it as the empty
        one. Only a caller for whom empty is a meaningful answer should
        reach for this; the rest want :meth:`seq`, so that a list
        somebody left unfinished is refused rather than read as none."""
        if self.kind == "empty":
            return ()
        return self.seq()

    def map(self) -> tuple[tuple[str, Node], ...] | None:
        return self.pairs if self.kind == "map" else None

    def get(self, key: str) -> Node | None:
        if self.kind != "map":
            return None
        for k, v in self.pairs:
            if k == key:
                return v
        return None

    def unknown(self, known: tuple[str, ...]) -> list[str]:
        """The keys that are not in ``known``, so a caller can refuse a
        typo rather than drop the field on the floor."""
        if self.kind != "map":
            return []
        return [k for k, _ in self.pairs if k not in known]


@dataclass
class Line:
    """One meaningful line: its indent, whether a ``- `` opened it, what
    is left after that, and where it was."""

    indent: int
    dash: bool
    text: str
    no: int


def parse(text: str) -> Node:
    """A document, or the first thing in it this reader will not read."""
    lines = _lex(text)
    if not lines:
        raise CorpusError("the file has nothing in it")
    if lines[0].indent != 0:
        raise CorpusError(f"line {lines[0].no}: the first line is indented")
    at = _Cursor(lines)
    node = _node(at, 0)
    if at.i < len(lines):
        raise CorpusError(f"line {lines[at.i].no}: this belongs to nothing above it")
    return node


@dataclass
class _Cursor:
    """Where the parser is, which the recursive calls share."""

    lines: list[Line]
    i: int = field(default=0)

    def at(self, offset: int = 0) -> Line | None:
        j = self.i + offset
        return self.lines[j] if j < len(self.lines) else None


def _lex(text: str) -> list[Line]:
    """Lines, with blanks and comments dropped and every ``- `` split
    into the item it opens and the content that followed it on the same
    line. Splitting here rather than in the parser is what lets
    ``- name: x`` and a ``name: x`` on its own line be the same shape by
    the time anything looks at them."""
    out: list[Line] = []
    for n, raw in enumerate(text.split("\n")):
        no = n + 1
        tab = raw.find("\t")
        if tab >= 0:
            raise CorpusError(
                f"line {no}: a tab at column {tab + 1}, and indentation here is spaces"
            )
        content = _strip_comment(raw).rstrip()
        indent = len(content) - len(content.lstrip())
        rest = content.lstrip()
        if not rest:
            continue
        if rest in ("---", "..."):
            raise CorpusError(
                f"line {no}: {quote(rest)} opens or closes a document, and a file here holds one"
            )
        if indent % 2:
            raise CorpusError(
                f"line {no}: indented {indent}, and indentation here goes two spaces at a time"
            )

        if not (rest == "-" or rest.startswith("- ")):
            out.append(Line(indent, False, rest, no))
            continue
        rest = rest[1:]
        if rest.startswith("  "):
            raise CorpusError(
                f"line {no}: a `- ` takes exactly one space, so that what follows it lines up "
                "with the lines under it"
            )
        rest = rest.lstrip()
        if rest.startswith("- "):
            raise CorpusError(
                f"line {no}: a sequence opening straight into another one, which nothing here needs"
            )
        out.append(Line(indent, True, "", no))
        if rest:
            out.append(Line(indent + 2, False, rest, no))
    return out


def _strip_comment(line: str) -> str:
    """Everything from an unquoted ``` #``` on is a comment.

    Three rules keep this from eating content. A ``#`` starts a comment
    only with whitespace before it, because one inside a word is part of
    the word. A quote opens a quoted run only with whitespace before it,
    because a quote inside a word is part of the word too, which is what
    lets a ``doc:`` say "it's" without opening a run that never closes.
    And a quote that opens nothing that closes was not a run at all,
    which is what lets a ``query:`` hold ``cast('  42  ' AS INT64)``.
    """
    i = 0
    n = len(line)
    while i < n:
        c = line[i]
        opens = i == 0 or line[i - 1].isspace()
        if c == "#" and opens:
            return line[:i]
        if c in "\"'" and opens:
            end = _closing_quote(line[i + 1 :], c)
            if end is not None:
                i += 1 + end
        i += 1
    return line


def _closing_quote(rest: str, mark: str) -> int | None:
    """The offset of the quote that closes a run whose opening quote has
    already been passed, or ``None`` if the line ends first.

    The two styles hide a quote differently: a double quoted run escapes
    with a backslash, and a single quoted run doubles the quote, which
    is the only escape it has."""
    i = 0
    n = len(rest)
    while i < n:
        if rest[i] == "\\" and mark == '"':
            i += 2
            continue
        if rest[i] == mark:
            if mark == "'" and rest[i + 1 : i + 2] == "'":
                i += 2
                continue
            return i
        i += 1
    return None


def _node(at: _Cursor, indent: int) -> Node:
    """The node that starts where the cursor is and is indented
    ``indent``, leaving the cursor on the first line that is not part of
    it."""
    line = at.at()
    assert line is not None
    if line.dash:
        return _seq(at, indent)
    # A mapping key is a bare word and a `:`. Anything else at this
    # position is a scalar standing on its own, which is what the items
    # of a sequence of scalars are.
    if _split_key(line.text) is not None:
        return _map(at, indent)
    at.i += 1
    return _scalar(line.text, line.no)


def _seq(at: _Cursor, indent: int) -> Node:
    line = at.at()
    assert line is not None
    start = line.no
    items: list[Node] = []
    while True:
        here = at.at()
        if here is None or not here.dash or here.indent != indent:
            break
        opened = here.no
        at.i += 1
        nxt = at.at()
        if nxt is not None and nxt.indent == indent + 2:
            items.append(_node(at, indent + 2))
        elif nxt is not None and nxt.indent > indent:
            raise CorpusError(
                f"line {nxt.no}: indented {nxt.indent}, where an item of the sequence on line "
                f"{opened} is indented {indent + 2}"
            )
        else:
            raise CorpusError(f"line {opened}: a `-` with nothing after it")
    return Node("seq", start, items=tuple(items))


def _map(at: _Cursor, indent: int) -> Node:
    line = at.at()
    assert line is not None
    start = line.no
    pairs: list[tuple[str, Node]] = []
    while True:
        here = at.at()
        if here is None or here.dash or here.indent != indent:
            break
        split = _split_key(here.text)
        if split is None:
            break
        key, rest = split
        opened = here.no
        at.i += 1
        if rest:
            value = _scalar(rest, opened)
        else:
            nxt = at.at()
            if nxt is not None and nxt.indent == indent + 2:
                value = _node(at, indent + 2)
            elif nxt is not None and nxt.indent > indent:
                raise CorpusError(
                    f"line {nxt.no}: indented {nxt.indent}, where what is under `{key}:` on line "
                    f"{opened} is indented {indent + 2}"
                )
            else:
                value = Node("empty", opened)
        if any(k == key for k, _ in pairs):
            raise CorpusError(f"line {opened}: {key} is set twice in one mapping")
        pairs.append((key, value))
    return Node("map", start, pairs=tuple(pairs))


def _split_key(text: str) -> tuple[str, str] | None:
    """The key and the rest of the line, when the line opens a mapping
    entry. A key is a bare word, and the ``:`` after it ends the line or
    has a space after it, so that a plain scalar holding a colon is still
    a scalar."""
    key, sep, rest = text.partition(": ")
    if sep:
        rest = rest.lstrip()
    else:
        if not text.endswith(":"):
            return None
        key, rest = text[:-1], ""
    if not key or not all(c.isascii() and (c.isalnum() or c in "_-") for c in key):
        return None
    return key, rest


def _scalar(text: str, line: int) -> Node:
    for mark in "\"'":
        if not text.startswith(mark):
            continue
        body = text[1:]
        # The closing quote is found by scanning rather than by taking
        # the last one on the line, so that `"a" and "b"` is refused
        # instead of read as one scalar with quotes in the middle.
        end = _closing_quote(body, mark)
        if end is None:
            raise CorpusError(f"line {line}: a {mark} that opens and does not close on its line")
        if end + 1 != len(body):
            raise CorpusError(f"line {line}: {quote(body[end + 1 :])} after the scalar ends")
        inner = body[:end]
        # A single quoted run has one escape, the doubled quote, and a
        # backslash in it is a backslash.
        value = _unescape(inner, line) if mark == '"' else inner.replace("''", "'")
        return Node("scalar", line, text=value, quoted=True)
    if text and text[0] in "[]{}&*!|>%@`":
        raise CorpusError(
            f"line {line}: a plain scalar opening with '{text[0]}', which is a construct this "
            "reader does not read"
        )
    return Node("scalar", line, text=text, quoted=False)


# The escapes the corpus uses, which is a subset of YAML's. The ones
# that name a code point by its digits are not here, because the corpus
# writes those as the character itself and a case that wants the digits
# is testing the engine's own escapes inside a query rather than the
# file's.
_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "0": "\0",
    "b": "\b",
    "f": "\f",
}


def _unescape(body: str, line: int) -> str:
    out: list[str] = []
    i = 0
    while i < len(body):
        c = body[i]
        if c != "\\":
            out.append(c)
            i += 1
            continue
        if i + 1 >= len(body):
            raise CorpusError(f"line {line}: a scalar ending in a backslash")
        nxt = body[i + 1]
        if nxt not in _ESCAPES:
            raise CorpusError(f"line {line}: \\{nxt} is not an escape")
        out.append(_ESCAPES[nxt])
        i += 2
    return "".join(out)
