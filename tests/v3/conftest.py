from __future__ import annotations

import logging

import pytest


@pytest.fixture(autouse=True)
def _enable_v3_log_propagation_for_caplog() -> None:
    """Restore propagation on the app.v3 logger so pytest caplog can capture records.

    F15 install_observability sets app.v3.propagate=False to route records only
    through its JSON handler. Once installed, the flag persists across tests and
    blocks pytest's caplog (which listens on root) from seeing child-logger
    records. This autouse fixture restores propagation per test and rolls back
    afterwards, without touching the product code.
    """
    logger = logging.getLogger("app.v3")
    original_propagate = logger.propagate
    logger.propagate = True
    try:
        yield
    finally:
        logger.propagate = original_propagate
