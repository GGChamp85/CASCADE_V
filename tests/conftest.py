"""Shared pytest fixtures.

By default tests run against real dependencies (torch + z3) since the venv
should have them installed. The legacy stubbed-test files (test_imports.py,
test_core_math.py) install their own stubs at module-import time and remain
runnable as standalone scripts.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def project_root():
    from pathlib import Path
    return Path(__file__).resolve().parent.parent
