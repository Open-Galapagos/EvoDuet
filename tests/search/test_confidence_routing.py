"""Reject unsupported confidence controllers before their constructors run."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from skydiscover.config import Config
from skydiscover.search.default_discovery_controller import DiscoveryController
from skydiscover.search.registry import _CONTROLLER_REGISTRY
from skydiscover.search.route import get_discovery_controller


def _input(search_type, *, enabled=True):
    config = Config()
    config.search.type = search_type
    config.solution_confidence.enabled = enabled
    return SimpleNamespace(config=config)


@pytest.mark.parametrize(
    "search_type", ["claude_code", "claude_code_local", "gepa_native", "adaevolve", "evox"]
)
def test_confidence_rejects_custom_controller_before_initialization(search_type):
    controller_class = _CONTROLLER_REGISTRY[search_type]
    with patch.object(controller_class, "__init__") as initialize:
        with pytest.raises(ValueError, match="solution_confidence is not supported"):
            get_discovery_controller(_input(search_type))
    initialize.assert_not_called()


@pytest.mark.parametrize(
    "search_type", ["openevolve_native", "topk", "best_of_n", "beam_search", "future_database"]
)
def test_confidence_allows_databases_using_the_standard_controller(search_type):
    controller_input = _input(search_type)
    with patch.object(DiscoveryController, "__init__", return_value=None) as initialize:
        controller = get_discovery_controller(controller_input)

    assert type(controller) is DiscoveryController
    initialize.assert_called_once_with(controller_input)


def test_confidence_disabled_preserves_autonomous_controller_routing():
    controller_input = _input("claude_code_local", enabled=False)
    controller_class = _CONTROLLER_REGISTRY["claude_code_local"]
    with patch.object(controller_class, "__init__", return_value=None) as initialize:
        controller = get_discovery_controller(controller_input)

    assert type(controller) is controller_class
    initialize.assert_called_once_with(controller_input)
