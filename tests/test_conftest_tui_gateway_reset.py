"""conftest's tui_gateway teardown must tolerate a partially initialized module.

``sys.modules['tui_gateway.server']`` can hold an entry whose module body has
not finished executing — a concurrent import of it is still in flight, or one
raised after the entry was installed. ``_reset_tui_gateway_server_state`` then
used to read ``mod._sessions`` unconditionally and raise
``AttributeError: partially initialized module ...`` from the *finalizer*,
which pytest reports as an ERROR on tests that all passed (and so as a
non-zero exit for a green file).

The fixture finalizer runs after the test body, so this is reproduced by
driving the fixture generator directly: setup while the module is absent,
stub entry installed, then teardown.
"""
from __future__ import annotations

import sys
import types

import pytest

from tests import conftest as suite_conftest

_MODNAME = "tui_gateway.server"


def _drive_reset_fixture(stub: types.ModuleType) -> None:
    fixture = getattr(
        suite_conftest._reset_tui_gateway_server_state, "__wrapped__", None
    ) or suite_conftest._reset_tui_gateway_server_state
    gen = fixture()
    next(gen)  # setup: module absent, so no snapshot is taken

    saved = sys.modules.get(_MODNAME)
    sys.modules[_MODNAME] = stub
    try:
        with pytest.raises(StopIteration):
            next(gen)  # teardown must run to completion without raising
    finally:
        if saved is None:
            sys.modules.pop(_MODNAME, None)
        else:
            sys.modules[_MODNAME] = saved


def test_teardown_skips_module_without_sessions_global():
    """A module object with no ``_sessions`` (mid-import) is skipped, not fatal."""
    _drive_reset_fixture(types.ModuleType(_MODNAME))


def test_teardown_still_closes_sessions_on_initialized_module():
    """The tolerant read must not turn the session teardown into a no-op."""
    closed: list[str] = []
    stub = types.ModuleType(_MODNAME)
    stub._sessions = {"sid-1": object()}
    stub._close_session_by_id = lambda sid, end_reason=None: closed.append(sid)

    _drive_reset_fixture(stub)

    assert closed == ["sid-1"]
