"""Validate current EvoDuet settings and reject removed configuration formats."""

import pytest

from skydiscover.config import (
    Config,
    EvoDuetConfig,
    SearchSelectionConfig,
    TavilyRetrievalConfig,
    apply_dot_overrides,
)


@pytest.mark.parametrize("source", ["yaml", "cli"])
@pytest.mark.parametrize(
    "name,value",
    [
        ("retrieval_backend_type", "tavily"),
        ("retrieval_backend_type", "oracle"),
        ("retrieval_policy", "query"),
        ("retrieval_policy_optimization", False),
        ("query_optimization_top_k", 1),
        ("summarize_documents", False),
        ("oracle_retrieval", {}),
        ("claude_code_retrieval", {}),
        ("retrieval_model", {}),
        ("retrieval_gating_model", {}),
        ("query_construction_model", {}),
        ("query_construction_prompt_template_name", "new_query_construction"),
        ("rag_file_path", "seed.jsonl"),
        ("offline_rag_on", False),
        ("online_rag_on", False),
        ("analysis_use_evolution_model", True),
        ("document_summary_max_tokens", 32768),
        ("estimated_score_prompt_template_name", "reranking"),
        ("query_optimization_fallback", False),
        ("query_optimisation_max_rounds", 3),
    ],
)
def test_removed_settings_fail_even_with_inactive_values(tmp_path, source, name, value):
    import yaml

    assert name not in Config().to_dict()["evoduet"]
    with pytest.raises((TypeError, ValueError), match=name):
        if source == "yaml":
            path = tmp_path / "config.yaml"
            path.write_text(yaml.safe_dump({"evoduet": {name: value}}))
            Config.from_yaml(path)
        else:
            apply_dot_overrides(Config(), {f"evoduet.{name}": str(value)})


@pytest.mark.parametrize("enabled", [False, True])
def test_world_knowledge_section_is_rejected(enabled):
    with pytest.raises(ValueError, match="world_knowledge settings are unsupported"):
        Config.from_dict({"world_knowledge": {"enabled": enabled}})
    with pytest.raises(ValueError, match="world_knowledge"):
        apply_dot_overrides(Config(), {"world_knowledge.enabled": str(enabled)})


@pytest.mark.parametrize("section", ["tavily_retrieval", "search_selection"])
def test_unknown_nested_evoduet_setting_is_rejected(section):
    with pytest.raises(TypeError, match="unknown_option"):
        Config.from_dict({"evoduet": {section: {"unknown_option": 1}}})
    with pytest.raises(ValueError, match="unknown_option"):
        apply_dot_overrides(Config(), {f"evoduet.{section}.unknown_option": "1"})


def test_evoduet_context_and_generation_budget_defaults_roundtrip(tmp_path):
    from skydiscover.config import Config

    config = Config()
    saved = tmp_path / "evoduet_budgets.yaml"
    config.to_yaml(saved)
    restored = Config.from_yaml(saved)

    for wk in (config.evoduet, restored.evoduet):
        assert wk.analysis_max_chars == 32_768
        assert wk.search_database_analysis_max_chars is None
        assert wk.query_construction_max_tokens == 32_768


@pytest.mark.parametrize("source_kind", ["yaml", "cli"])
def test_evoduet_context_and_generation_budgets_normalize_and_roundtrip(tmp_path, source_kind):
    from skydiscover.config import Config, apply_dot_overrides

    values = {
        "analysis_max_chars": 40_000,
        "search_database_analysis_max_chars": 70_000,
        "query_construction_max_tokens": 16_384,
    }
    saved = tmp_path / "evoduet_budgets.yaml"
    if source_kind == "yaml":
        saved.write_text(
            "evoduet:\n" + "".join(f"  {name}: '{value}'\n" for name, value in values.items()),
            encoding="utf-8",
        )
        config = Config.from_yaml(saved)
    else:
        config = Config()
        apply_dot_overrides(
            config, {f"evoduet.{name}": str(value) for name, value in values.items()}
        )
    config.to_yaml(saved)
    restored = Config.from_yaml(saved)

    for wk in (config.evoduet, restored.evoduet):
        for name, value in values.items():
            assert getattr(wk, name) == value
            assert isinstance(getattr(wk, name), int)


def test_search_analysis_input_cap_cli_can_reset_to_unbounded(tmp_path):
    from skydiscover.config import Config, apply_dot_overrides

    config = Config.from_dict({"evoduet": {"search_database_analysis_max_chars": 60_000}})
    apply_dot_overrides(config, {"evoduet.search_database_analysis_max_chars": "null"})

    assert config.evoduet.search_database_analysis_max_chars is None
    saved = tmp_path / "evoduet_budgets.yaml"
    config.to_yaml(saved)
    assert Config.from_yaml(saved).evoduet.search_database_analysis_max_chars is None


@pytest.mark.parametrize(
    "name",
    [
        "analysis_max_chars",
        "search_database_analysis_max_chars",
        "query_construction_max_tokens",
    ],
)
@pytest.mark.parametrize("value", [0, -1, 1.5, "1.5", "invalid", "nan", "inf", True])
def test_evoduet_budgets_reject_invalid_values(name, value):
    with pytest.raises(ValueError, match=rf"{name} must be an integer"):
        EvoDuetConfig(**{name: value})


@pytest.mark.parametrize("name", ["analysis_max_chars", "query_construction_max_tokens"])
def test_required_evoduet_budgets_reject_null(name):
    with pytest.raises(ValueError, match=rf"{name} must be an integer"):
        EvoDuetConfig(**{name: None})


@pytest.mark.parametrize(
    "name",
    [
        "analysis_max_chars",
        "search_database_analysis_max_chars",
        "query_construction_max_tokens",
    ],
)
@pytest.mark.parametrize("value", ["0", "-1", "1.5", "true", "invalid"])
def test_evoduet_budget_cli_rejects_invalid_values(name, value):
    from skydiscover.config import Config, apply_dot_overrides

    with pytest.raises(ValueError, match=name):
        apply_dot_overrides(Config(), {f"evoduet.{name}": value})


@pytest.mark.parametrize(
    "name",
    [
        "query_evolution_steps",
        "query_self_reflection_prompt_template_name",
        "retrieval_policy_steps",
        "retrieval_policy_calibration",
        "retrieval_policy_calibration_decay",
        "retrieval_policy_calibration_ridge",
    ],
)
def test_removed_evoduet_settings_are_rejected(name):
    from skydiscover.config import Config

    assert not hasattr(EvoDuetConfig(), name)
    assert name not in Config().to_dict()["evoduet"]
    with pytest.raises(TypeError, match=name):
        Config.from_dict({"evoduet": {name: 0}})


@pytest.mark.parametrize(
    "name",
    [
        "query_optimization_max_rounds",
        "query_optimization_queries_per_round",
        "search_result_top_k",
    ],
)
@pytest.mark.parametrize("value", [0, -1, 0.5, "1.5", "invalid", "nan", "inf", None, True])
def test_search_counts_reject_invalid_integers(name, value):
    with pytest.raises(ValueError, match=rf"{name} must be an integer"):
        EvoDuetConfig(**{name: value})


def test_population_analysis_input_is_unbounded_by_default():
    assert EvoDuetConfig().population_state_max_chars is None
    assert EvoDuetConfig(population_state_max_chars=None).population_state_max_chars is None


def test_population_analysis_reads_the_twenty_most_recent_programs_by_default():
    assert EvoDuetConfig().population_state_recent_k == 20
    assert EvoDuetConfig(population_state_recent_k="5").population_state_recent_k == 5
    assert EvoDuetConfig(population_state_recent_k=None).population_state_recent_k is None
    with pytest.raises(ValueError, match="population_state_recent_k"):
        EvoDuetConfig(population_state_recent_k=0)


@pytest.mark.parametrize("top_k", [0, -1])
def test_top_k_for_retrieval_must_be_positive(top_k):
    with pytest.raises(ValueError, match="top_k_for_retrieval must be positive"):
        EvoDuetConfig(top_k_for_retrieval=top_k)


def test_shared_top_k_for_tavily_must_not_exceed_provider_limit():
    with pytest.raises(ValueError, match="at most 20 when used by Tavily"):
        EvoDuetConfig(
            top_k_for_retrieval=21,
            tavily_retrieval=TavilyRetrievalConfig(max_results=None),
        )


def test_tavily_limit_does_not_apply_when_explicit_max_results_wins():
    config = EvoDuetConfig(
        top_k_for_retrieval=21,
        tavily_retrieval=TavilyRetrievalConfig(max_results=5),
    )

    assert config.top_k_for_retrieval == 21


def test_tavily_numeric_fields_are_normalized():
    config = EvoDuetConfig(
        tavily_retrieval=TavilyRetrievalConfig(
            max_results="7", timeout="2.5", chunks_per_source="2"
        )
    )

    assert config.tavily_retrieval.max_results == 7
    assert config.tavily_retrieval.timeout == 2.5
    assert config.tavily_retrieval.chunks_per_source == 2


def test_tavily_defaults_keep_each_search_small():
    config = EvoDuetConfig().tavily_retrieval

    assert config.max_results == 5
    assert config.include_raw_content is False


@pytest.mark.parametrize("max_results", [0, 21])
def test_tavily_max_results_must_be_within_provider_limit(max_results):
    with pytest.raises(ValueError, match="max_results must be between 1 and 20"):
        EvoDuetConfig(tavily_retrieval=TavilyRetrievalConfig(max_results=max_results))


@pytest.mark.parametrize("timeout", [0, -1, "nan", "inf"])
def test_tavily_timeout_must_be_positive_and_finite(timeout):
    with pytest.raises(ValueError, match="tavily_retrieval.timeout must be positive"):
        EvoDuetConfig(tavily_retrieval=TavilyRetrievalConfig(timeout=timeout))


@pytest.mark.parametrize("chunks", [0, 4])
def test_tavily_chunks_per_source_must_be_between_one_and_three(chunks):
    with pytest.raises(ValueError, match="chunks_per_source must be between 1 and 3"):
        EvoDuetConfig(tavily_retrieval=TavilyRetrievalConfig(chunks_per_source=chunks))


@pytest.mark.parametrize("num", [0, -1])
def test_search_selection_num_must_be_positive_when_set(num):
    with pytest.raises(ValueError, match="search_selection.num must be positive"):
        EvoDuetConfig(search_selection=SearchSelectionConfig(num=num))


def test_search_selection_num_can_be_none():
    config = EvoDuetConfig(search_selection=SearchSelectionConfig(num=None))

    assert config.search_selection.num is None


def test_current_config_roundtrip_preserves_active_budgets(tmp_path):
    config = Config.from_dict(
        {
            "evoduet": {
                "enabled": True,
                "query_optimization_max_rounds": "4",
                "query_optimization_queries_per_round": "3",
                "search_result_top_k": "2",
            }
        }
    )
    apply_dot_overrides(config, {"evoduet.query_optimization_max_rounds": "7"})
    path = tmp_path / "config.yaml"
    config.to_yaml(path)
    restored = Config.from_yaml(path).evoduet
    assert restored.query_optimization_max_rounds == 7
    assert restored.query_optimization_queries_per_round == 3
    assert restored.search_result_top_k == 2


def test_unknown_saved_setting_is_not_discarded_as_legacy():
    with pytest.raises(TypeError, match="query_optimisation_max_rounds"):
        Config.from_dict({"evoduet": {"query_optimisation_max_rounds": 3}})


def test_active_integer_fields_are_normalized():
    config = EvoDuetConfig(
        top_k_for_retrieval="21",
        population_state_max_chars="1024",
        max_document_chars="512",
        search_selection=SearchSelectionConfig(num="4"),
    )
    assert config.top_k_for_retrieval == 21
    assert config.population_state_max_chars == 1024
    assert config.max_document_chars == 512
    assert config.search_selection.num == 4
