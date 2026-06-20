"""Deprecated alias for :mod:`ponyexl3.cli.convert` (the one-command pipeline).

Kept so ``ponyexl3-convert-e2e`` and existing scripts keep working. Use
``ponyexl3-convert`` instead — it now *is* the end-to-end convert.
"""

from __future__ import annotations

from ponyexl3.cli.convert import (  # noqa: F401
    _default_candidate_bits,
    main,
)

if __name__ == "__main__":
    raise SystemExit(main())
