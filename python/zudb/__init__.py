"""zu for Python: an embedded property-graph database, in your process.

    import zudb

    with zudb.connect("social.zu1") as conn:
        rows = conn.execute("MATCH (p:person) RETURN p.name AS name")
        for (name,) in rows:
            print(name)

The engine is compiled into the wheel, so there is nothing to install,
nothing to run, and no server to connect to. Statements are ISO/IEC
39075 GQL.
"""

from __future__ import annotations

from ._zudb import (
    Connection,
    Duration,
    Node,
    Path,
    Rel,
    Result,
    __abi_version__,
    connect,
    load,
)
from .errors import (
    ConnectionError,
    DataError,
    Error,
    InternalError,
    Interrupted,
    ProgrammingError,
    SyntaxError,
    TransactionError,
)

__version__ = "0.0.1"

__all__ = [
    "connect",
    "load",
    "Connection",
    "Result",
    "Node",
    "Rel",
    "Path",
    "Duration",
    "Error",
    "ConnectionError",
    "DataError",
    "TransactionError",
    "SyntaxError",
    "ProgrammingError",
    "InternalError",
    "Interrupted",
    "__version__",
    "__abi_version__",
]
