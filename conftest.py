"""Root conftest.py -- resets shared registry state between test modules.

Each test module gets a fresh in-memory store so tests do not bleed into
each other through the global _agents dict in the registry.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def reset_registry_state():
    """Reset the registry's in-memory stores before each test.

    This hooks into the db singleton so tests using the in-memory fallback
    start with a clean slate.
    """
    import mesh.db as db_module
    import mesh.router as router_module

    # Reset db singleton
    db_module._db_instance = None
    db_module._pool = None

    # Reset the router index
    router_module_router = getattr(router_module, "TaskRouter", None)
    # Patch the registry's router_engine index
    try:
        import mesh.registry as registry_module
        registry_module.router_engine._index.clear()
    except Exception:
        pass

    yield

    # Teardown: reset again
    db_module._db_instance = None
    db_module._pool = None
