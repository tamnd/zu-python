"""The shared conformance corpus, run through this client.

The cases live in the engine repository, versioned with the engine and
shipped as a release artifact every client consumes. Nothing here holds
a case: a case that lived beside one runner would be a case that runner
cannot fail.

What is here is the third implementation of the reader, the second of
the value encoding, and the second runner, all of which exist so that a
value put in through this client and taken out through another means the
same thing.
"""

from __future__ import annotations

from .cases import Case, Suite, read_dir
from .reader import CorpusError
from .runner import Report, run

__all__ = ["Case", "CorpusError", "Report", "Suite", "read_dir", "run"]
