"""Opt-in candidate counts survive config loading and CLI overrides."""

from unittest.mock import patch

import pytest

from skydiscover.cli import _parse_dot_overrides
from skydiscover.config import Config, apply_dot_overrides, apply_overrides
from skydiscover.search.default_discovery_controller import (
    DiscoveryController,
    DiscoveryControllerInput,
)
from skydiscover.search.route import get_discovery_controller


def test_defaults_and_yaml_roundtrip(tmp_path):
    config = Config()
    assert config.num_generations == config.max_parallel_evaluations == 1
    assert config.max_parallel_generations is None
    values = {"num_generations": 4, "max_parallel_generations": 2, "max_parallel_evaluations": 3}
    config = Config.from_dict(values)
    path = tmp_path / "config.yaml"
    config.to_yaml(path)
    loaded = Config.from_yaml(path)
    assert all(getattr(loaded, key) == value for key, value in values.items())
    assert all(loaded.to_dict()[key] == value for key, value in values.items())


@pytest.mark.parametrize(
    "name", ["num_generations", "max_parallel_generations", "max_parallel_evaluations"]
)
@pytest.mark.parametrize("value", [0, -1, True, False, 2.5, "4", "bad"])
def test_invalid_counts_fail_on_construction_and_load(name, value):
    with pytest.raises(ValueError, match=name):
        Config(**{name: value})
    with pytest.raises(ValueError, match=name):
        Config.from_dict({name: value})


def test_cli_uses_top_level_counts_and_can_reset_optional_cap():
    config = Config()
    apply_dot_overrides(
        config,
        _parse_dot_overrides(
            [
                "--num_generations",
                "4",
                "--max_parallel_generations",
                "2",
                "--max_parallel_evaluations",
                "3",
            ]
        ),
    )
    assert (
        config.num_generations,
        config.max_parallel_generations,
        config.max_parallel_evaluations,
    ) == (4, 2, 3)
    apply_dot_overrides(config, {"max_parallel_generations": "null"})
    assert config.max_parallel_generations is None


@pytest.mark.parametrize(
    "name", ["num_generations", "max_parallel_generations", "max_parallel_evaluations"]
)
@pytest.mark.parametrize("raw", ["0", "-1", "true", "1.5", "invalid"])
def test_invalid_cli_counts_are_rejected(name, raw):
    with pytest.raises(ValueError, match=name):
        apply_dot_overrides(Config(), {name: raw})


@pytest.mark.parametrize("search", ["openevolve_native", "topk", "best_of_n", "beam_search"])
def test_standard_controller_supports_batches(search):
    config = Config(num_generations=4)
    config.search.type = search
    with patch.object(DiscoveryController, "__init__", return_value=None):
        assert (
            type(get_discovery_controller(DiscoveryControllerInput(config, "unused.py", None)))
            is DiscoveryController
        )


@pytest.mark.parametrize(
    "search", ["evox", "adaevolve", "gepa_native", "openevolve", "alphaevolve", "claude_code_local"]
)
def test_unsupported_search_fails_before_initializing_clients(search):
    config = Config(num_generations=4)
    config.search.type = search
    with patch("skydiscover.search.default_discovery_controller.LLMPool") as pool:
        with pytest.raises(ValueError, match="standard discovery controller"):
            get_discovery_controller(DiscoveryControllerInput(config, "unused.py", None))
        pool.assert_not_called()


@pytest.mark.parametrize("mode", ["agentic", "image"])
def test_unsupported_generation_modes_fail_before_initializing_clients(mode):
    config = Config(num_generations=4)
    if mode == "agentic":
        config.agentic.enabled = True
    else:
        config.language = "image"
    with patch("skydiscover.search.default_discovery_controller.LLMPool") as pool:
        with pytest.raises(ValueError, match="non-agentic text/code"):
            DiscoveryController(DiscoveryControllerInput(config, "unused.py", None))
        pool.assert_not_called()


def test_cli_can_disable_batches_before_switching_to_an_external_backend():
    config = Config(num_generations=4)
    apply_overrides(config, search="openevolve")
    apply_dot_overrides(config, {"num_generations": "1"})
    config.validate_generation_mode()
    assert config.search.type == "openevolve"
