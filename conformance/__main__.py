"""``python -m conformance <dir>``, which is how CI runs it."""

from __future__ import annotations

from .runner import main

raise SystemExit(main())
