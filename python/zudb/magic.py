"""`%%gql` in a notebook.

    %load_ext zudb.magic
    %gql social.zu1

    %%gql
    MATCH (p:person) WHERE p.score > 40 RETURN p.name AS name, p.score AS score

The cell is one statement, it runs on the current connection, and what
comes back is a `zudb.Result`, which the notebook draws as a table
because a result knows how to draw itself. It is the value of the cell
as well, so `_` is the result and everything a result can do is still
there.

Two magics and no more. `%gql` is about which connection, `%%gql` is
about the statement, and neither of them tries to guess the other's
job: a line magic that took a path or a statement depending on what it
looked like would guess wrong on the day somebody named a file
`MATCH`.

`%gql` with a path opens a connection and makes it the current one.
`%gql` with nothing says which connection that is. `%gql --close` shuts
it. A notebook that made its own connection does not need any of that:
if exactly one `zudb.Connection` is lying about the namespace, `%%gql`
uses it, and if there is more than one it says so and asks which,
because picking one of them would be picking somebody's read-only
connection half the time.

The cell magic takes three options, all of them naming variables rather
than holding values, since a notebook has the values already:

    %%gql --conn other --params args --out rows
    MATCH (p:person) WHERE p.uid = $uid RETURN p.name AS name

`--conn` is the connection to run on, `--params` is a dict of
parameters, and `--out` is where to put the result instead of showing
it.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from IPython.core.error import UsageError
from IPython.core.magic import Magics, cell_magic, line_magic, magics_class

from ._zudb import Connection, connect

__all__ = ["ZuMagics", "load_ipython_extension"]


@magics_class
class ZuMagics(Magics):
    """The two magics, and the connection `%gql` opened if it opened one."""

    def __init__(self, shell: Any) -> None:
        super().__init__(shell)
        self.connection: Connection | None = None

    @line_magic("gql")
    def gql(self, line: str) -> Connection | None:
        """Opens a connection and makes it the one `%%gql` runs on.

            %gql social.zu1
            %gql --read-only social.zu1
            %gql
            %gql --close

        The connection comes back so that a cell can keep it, and it is
        closed at the end of the session like any other.
        """
        words = shlex.split(line)
        read_only = "--read-only" in words
        closing = "--close" in words
        paths = [word for word in words if not word.startswith("-")]
        unknown = [word for word in words if word.startswith("-")]
        for word in unknown:
            if word not in ("--read-only", "--close"):
                raise UsageError(f"%gql does not take {word}")
        if closing:
            if self.connection is not None:
                self.connection.close()
                self.connection = None
            return None
        if not paths:
            if self.connection is None:
                raise UsageError("no connection is open here: %gql <path> opens one")
            return self.connection
        if len(paths) > 1:
            raise UsageError("%gql opens one database, and this named more than one")
        # The old one is closed rather than left holding a file, since
        # a notebook that opens a second database has finished with the
        # first and nothing else here has a name for it.
        if self.connection is not None:
            self.connection.close()
        self.connection = connect(Path(paths[0]).expanduser(), read_only=read_only)
        return self.connection

    @cell_magic("gql")
    def gql_cell(self, line: str, cell: str) -> Any:
        """Runs the cell as one statement on the current connection."""
        options = self.options(line)
        conn = self.current(options.get("conn"))
        statement = cell.strip()
        # A person types the semicolon out of habit and a statement
        # does not take one, which is a syntax error about a character
        # nobody meant to type.
        while statement.endswith(";"):
            statement = statement[:-1].rstrip()
        if not statement:
            raise UsageError("this cell holds no statement")
        result = conn.execute(statement, self.params(options.get("params")))
        out = options.get("out")
        if out is None:
            return result
        self.namespace()[out] = result
        return None

    def options(self, line: str) -> dict[str, str]:
        """`--conn`, `--params` and `--out`, each naming a variable."""
        words = shlex.split(line)
        taken: dict[str, str] = {}
        while words:
            word = words.pop(0)
            name = word.removeprefix("--")
            if name == word or name not in ("conn", "params", "out"):
                raise UsageError(f"%%gql takes --conn, --params and --out, and not {word}")
            if not words:
                raise UsageError(f"--{name} names a variable, and this named none")
            taken[name] = words.pop(0)
        return taken

    def namespace(self) -> dict[str, Any]:
        """Where the notebook's own names live."""
        if self.shell is None:
            raise UsageError("there is no notebook here to read names out of")
        return self.shell.user_ns

    def current(self, named: str | None) -> Connection:
        """The connection to run on: the one named, the one `%gql`
        opened, or the one the notebook made if it made exactly one.
        """
        if named is not None:
            found = self.namespace().get(named)
            if not isinstance(found, Connection):
                raise UsageError(f"'{named}' is not a zudb.Connection in this notebook")
            return found
        if self.connection is not None:
            return self.connection
        theirs = sorted(
            name
            for name, value in self.namespace().items()
            if isinstance(value, Connection) and not name.startswith("_")
        )
        if len(theirs) == 1:
            return self.namespace()[theirs[0]]
        if not theirs:
            raise UsageError(
                "no connection is open here: %gql <path> opens one, or make one with "
                "zudb.connect and %%gql will find it"
            )
        raise UsageError(
            "this notebook holds more than one connection ("
            + ", ".join(theirs)
            + "), so %%gql --conn <name> has to say which"
        )

    def params(self, named: str | None) -> dict[str, Any] | None:
        """The parameters, out of the variable `--params` named."""
        if named is None:
            return None
        found = self.namespace().get(named)
        if not isinstance(found, dict):
            raise UsageError(f"'{named}' is not a dict of parameters in this notebook")
        return found


def load_ipython_extension(ipython: Any) -> None:
    """Registers `%gql` and `%%gql`, which is what `%load_ext` calls.

    Registered by hand rather than on import, because a library that
    reaches into the interpreter it was imported into is a library that
    surprises somebody, and `%load_ext zudb.magic` is one line.
    """
    ipython.register_magics(ZuMagics)
