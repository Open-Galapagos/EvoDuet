"""Configuration contracts for candidate improvement confidence."""

from dataclasses import asdict

import pytest

from skydiscover.config import Config, SolutionConfidenceConfig, apply_dot_overrides


def test_confidence_is_opt_in_and_defaults_to_strict_combined_score_improvement():
    config = Config()

    assert asdict(config.solution_confidence) == {
        "enabled": False,
        "metric": "combined_score",
        "higher_is_better": True,
        "improvement_epsilon": 0.0,
        "timeout": 120.0,
        "top_logprobs": 20,
        "openrouter_provider": None,
    }
    config.solution_confidence.enabled = True
    assert Config().solution_confidence.enabled is False


def test_confidence_config_dict_and_yaml_roundtrip(tmp_path):
    values = {
        "enabled": True,
        "metric": "loss",
        "higher_is_better": False,
        "improvement_epsilon": 1e-6,
        "timeout": 15.5,
        "top_logprobs": 5,
        "openrouter_provider": "alibaba",
    }
    config = Config.from_dict({"solution_confidence": values})

    assert isinstance(config.solution_confidence, SolutionConfidenceConfig)
    assert config.to_dict()["solution_confidence"] == values
    assert asdict(Config.from_dict(config.to_dict()).solution_confidence) == values
    path = tmp_path / "config.yaml"
    config.to_yaml(path)
    assert asdict(Config.from_yaml(path).solution_confidence) == values


def test_confidence_normalizes_numeric_yaml_values():
    config = Config.from_dict(
        {
            "solution_confidence": {
                "metric": " combined_score ",
                "improvement_epsilon": "1e-6",
                "timeout": "2.5",
                "top_logprobs": "5",
                "openrouter_provider": " alibaba ",
            }
        }
    )

    assert config.solution_confidence.metric == "combined_score"
    assert config.solution_confidence.improvement_epsilon == 1e-6
    assert config.solution_confidence.timeout == 2.5
    assert config.solution_confidence.top_logprobs == 5
    assert isinstance(config.solution_confidence.top_logprobs, int)
    assert config.solution_confidence.openrouter_provider == "alibaba"


@pytest.mark.parametrize("name", ["enabled", "higher_is_better"])
@pytest.mark.parametrize("value", ["true", "false", 1, 0, None])
def test_confidence_rejects_non_boolean_config_flags(name, value):
    with pytest.raises(ValueError, match=rf"solution_confidence\.{name} must be boolean"):
        Config.from_dict({"solution_confidence": {name: value}})


@pytest.mark.parametrize("value", [None, "", " \t", 3, False])
def test_confidence_metric_must_be_named(value):
    with pytest.raises(ValueError, match=r"solution_confidence\.metric"):
        SolutionConfidenceConfig(metric=value)


@pytest.mark.parametrize("name", ["improvement_epsilon", "timeout"])
@pytest.mark.parametrize("value", [-1, "nan", "inf", "-inf", "invalid", None, True])
def test_confidence_rejects_invalid_numeric_settings(name, value):
    with pytest.raises(ValueError, match=rf"solution_confidence\.{name}"):
        Config.from_dict({"solution_confidence": {name: value}})


def test_zero_epsilon_is_valid_but_zero_timeout_is_not():
    assert SolutionConfidenceConfig(improvement_epsilon=0).improvement_epsilon == 0.0
    with pytest.raises(ValueError, match=r"solution_confidence\.timeout"):
        SolutionConfidenceConfig(timeout=0)


@pytest.mark.parametrize("value", [1, "5", 5.0, "5.0", 20])
def test_confidence_top_logprobs_normalizes_integral_values(value):
    config = SolutionConfidenceConfig(top_logprobs=value)

    assert config.top_logprobs == int(float(value))
    assert isinstance(config.top_logprobs, int)


@pytest.mark.parametrize(
    "value", [None, True, False, 0, 21, -1, 5.5, "5.5", "nan", "inf", "invalid"]
)
def test_confidence_rejects_invalid_top_logprobs(value):
    with pytest.raises(ValueError, match=r"solution_confidence\.top_logprobs"):
        Config.from_dict({"solution_confidence": {"top_logprobs": value}})


@pytest.mark.parametrize("value", ["", " \t", 3, False, []])
def test_confidence_rejects_invalid_openrouter_provider(value):
    with pytest.raises(ValueError, match=r"solution_confidence\.openrouter_provider"):
        SolutionConfidenceConfig(openrouter_provider=value)


def test_confidence_default_provider_roundtrips_as_none(tmp_path):
    config = Config()
    path = tmp_path / "default.yaml"
    config.to_yaml(path)

    assert Config.from_dict(config.to_dict()).solution_confidence.openrouter_provider is None
    assert Config.from_yaml(path).solution_confidence.openrouter_provider is None


def test_confidence_cli_overrides_keep_types_and_validate():
    config = Config()
    apply_dot_overrides(
        config,
        {
            "solution_confidence.enabled": "true",
            "solution_confidence.metric": "loss",
            "solution_confidence.higher_is_better": "false",
            "solution_confidence.improvement_epsilon": "1e-4",
            "solution_confidence.timeout": "25",
            "solution_confidence.top_logprobs": "5",
            "solution_confidence.openrouter_provider": " alibaba ",
        },
    )

    assert config.to_dict()["solution_confidence"] == {
        "enabled": True,
        "metric": "loss",
        "higher_is_better": False,
        "improvement_epsilon": 1e-4,
        "timeout": 25.0,
        "top_logprobs": 5,
        "openrouter_provider": "alibaba",
    }


@pytest.mark.parametrize("provider", ["alibaba", "novita/bf16", "123", "true"])
def test_confidence_cli_provider_remains_string_with_none_default(provider):
    config = Config()
    apply_dot_overrides(config, {"solution_confidence.openrouter_provider": provider})

    assert config.solution_confidence.openrouter_provider == provider


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("enabled", "sometimes"),
        ("higher_is_better", "sometimes"),
        ("metric", " "),
        ("improvement_epsilon", "-0.1"),
        ("improvement_epsilon", "nan"),
        ("timeout", "0"),
        ("timeout", "inf"),
        ("top_logprobs", "0"),
        ("top_logprobs", "21"),
        ("top_logprobs", "5.5"),
        ("top_logprobs", "true"),
        ("openrouter_provider", " "),
    ],
)
def test_invalid_confidence_cli_overrides_fail_before_running(name, value):
    with pytest.raises(ValueError, match=rf"solution_confidence\.{name}"):
        apply_dot_overrides(Config(), {f"solution_confidence.{name}": value})
