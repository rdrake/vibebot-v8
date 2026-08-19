"""Shared plugin fixture for draw-callback tests.

`TestDrawForMeta` in test_assistant.py builds the same object from a pytest
fixture; this is the importable form, for suites that need it outside that
class.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from llm.plugin import LLM

from .conftest import make_registry_side_effect, plugin_init_patches

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def make_draw_plugin(mocker: MockerFixture):  # type: ignore[no-untyped-def]
    """An LLM plugin with the service and db mocked, ready for _draw_for_assistant."""
    plugin_init_patches(mocker)
    plugin = LLM(mocker.MagicMock())
    plugin.registryValue = mocker.Mock(side_effect=make_registry_side_effect())
    plugin.llm_service = mocker.MagicMock()
    plugin.db = mocker.MagicMock()
    plugin._MetaSynchronized_rlock = threading.RLock()
    return plugin
