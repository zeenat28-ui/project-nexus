"""Module entry point for `python -m project_nexus`."""

from __future__ import annotations

import sys

from project_nexus.cli import main

if __name__ == "__main__":
    sys.exit(main())
