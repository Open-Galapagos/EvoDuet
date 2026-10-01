"""
Configuration handling for SkyDiscover
"""

from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from urllib.parse import urlparse

import yaml

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Internal — provider resolution helpers
# ═══════════════════════════════════════════════════════════════════════

# No OPENAI_API_KEY fallback: an OpenAI key cannot authenticate to Foundry, and sending
# one there would hand a credential to a host that is not its issuer.
_AZURE_KEY_ENV = ["AZURE_OPENAI_API_KEY", "AZURE_API_KEY"]
_AZURE_DEFAULT_BASE_URL = "https://galapagos.services.ai.azure.com/openai/v1"
_OPENROUTER_KEY_ENV = ["OPENROUTER_API_KEY"]
_OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

_PROVIDERS: Dict[str, tuple] = {
    "openai": ("https://api.openai.com/v1", ["OPENAI_API_KEY"]),
    "openrouter": (_OPENROUTER_DEFAULT_BASE_URL, _OPENROUTER_KEY_ENV),
    # Azure AI Foundry. The default resource is overridable through the environment
    # or an explicit api_base — see _provider_defaults().
    "azure": (_AZURE_DEFAULT_BASE_URL, _AZURE_KEY_ENV),
    "gemini": (
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    ),
    "anthropic": ("https://api.anthropic.com/v1/", ["ANTHROPIC_API_KEY"]),
    "deepseek": ("https://api.deepseek.com/v1", ["DEEPSEEK_API_KEY"]),
    "mistral": ("https://api.mistral.ai/v1", ["MISTRAL_API_KEY"]),
    "cohere": ("https://api.cohere.com/v1", ["CO_API_KEY", "COHERE_API_KEY"]),
    "huggingface": (None, ["HF_TOKEN", "HUGGINGFACE_API_KEY"]),
    "ollama": (None, []),
    "vllm": (None, []),
}

# Bare model-name prefixes → provider  (backwards compat for --model gpt-5, etc.)
_BARE_PREFIX_MAP: Dict[str, str] = {
    "gpt-": "openai",
    "o1": "openai",
    "o3": "openai",
    "o4": "openai",
    "gemini-": "gemini",
    "claude-": "anthropic",
    "deepseek-": "deepseek",
    "mistral-": "mistral",
    "command-": "cohere",
}


# A different Foundry resource is selected through these, ahead of _AZURE_DEFAULT_BASE_URL.
_AZURE_BASE_URL_ENV = ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_BASE_URL")


def is_azure_endpoint(api_base: Optional[str]) -> bool:
    """Return whether *api_base* points at Azure AI Foundry."""
    if not api_base or "://" not in api_base:
        return False
    return (urlparse(api_base).hostname or "").lower().endswith(".services.ai.azure.com")


def is_openrouter_endpoint(api_base: Optional[str]) -> bool:
    """Return whether *api_base* points at OpenRouter's API."""
    if not api_base or "://" not in api_base:
        return False
    return (urlparse(api_base).hostname or "").lower() == "openrouter.ai"


def _env_vars_for_base_url(api_base: Optional[str]) -> Optional[List[str]]:
    """Return the API-key env vars that belong to the provider serving *api_base*."""
    if is_azure_endpoint(api_base):
        return _AZURE_KEY_ENV
    for _name, (base_url, env_list) in _PROVIDERS.items():
        if base_url and api_base and api_base.startswith(base_url.rstrip("/")):
            return env_list
    return None


def resolve_azure_api_key() -> Optional[str]:
    """Return the first Azure API key found in the environment."""
    return _resolve_api_key_from_env(_AZURE_KEY_ENV)


def _provider_defaults(provider: str) -> tuple:
    """Return ``(default_api_base, env_vars)`` for *provider*, resolving env-driven hosts."""
    api_base, env_vars = _PROVIDERS[provider]
    if provider == "azure":
        for var in _AZURE_BASE_URL_ENV:
            configured = os.environ.get(var)
            if configured:
                api_base = configured.rstrip("/")
                break
    return api_base, env_vars


def _parse_model_spec(model_str: str, api_base: Optional[str] = None) -> tuple:
    """Parse a model string into ``(provider, model_name, default_api_base, env_vars)``.

    Supports:
      - OpenRouter model IDs with an explicit OpenRouter endpoint; the full
        ``author/model`` ID is preserved (e.g. ``qwen/qwen3.5-27b``)
      - ``provider/model``  (e.g. ``gemini/gemini-3-pro``)
      - bare names with known prefix (e.g. ``gemini-3-pro`` → gemini)
      - unknown bare names return ``(None, model_str, None, [])`` with a warning
    """
    if is_openrouter_endpoint(api_base):
        model_name = model_str
        if model_str.lower().startswith("openrouter/"):
            _, _, model_name = model_str.partition("/")
        return "openrouter", model_name, api_base, _OPENROUTER_KEY_ENV

    if "/" in model_str:
        provider, _, model_name = model_str.partition("/")
        provider_lower = provider.lower()
        if provider_lower in _PROVIDERS:
            api_base, env_vars = _provider_defaults(provider_lower)
            return provider_lower, model_name, api_base, env_vars

    for prefix, provider in _BARE_PREFIX_MAP.items():
        if model_str.startswith(prefix):
            api_base, env_vars = _provider_defaults(provider)
            return provider, model_str, api_base, env_vars

    logger.warning(
        "Unknown model '%s': no provider matched. "
        "Use 'provider/model' format (e.g. 'openai/my-model') or set api_base explicitly.",
        model_str,
    )
    return None, model_str, None, []


def _resolve_api_key_from_env(env_vars: Optional[List[str]] = None) -> Optional[str]:
    """Return the first API key found in *env_vars*.

    *env_vars* typically comes from ``_parse_model_spec()``.
    Only returns a key if it matches the provider's own env vars.
    """
    for var in env_vars or []:
        key = os.environ.get(var)
        if key:
            return key
    return None


def _expand_env_vars(text: str) -> str:
    """Expand ${VAR} patterns in text with environment variable values."""

    def _replacer(match):
        return os.environ.get(match.group(1), match.group(0))

    return re.sub(r"\$\{(\w+)\}", _replacer, text)


def _config_dict_without_api_keys(value: Any) -> Dict[str, Any]:
    """Serialize one dataclass config while keeping credentials out of snapshots."""

    def clean(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: clean(child) for key, child in item.items() if key != "api_key"}
        if isinstance(item, (list, tuple)):
            return [clean(child) for child in item]
        return item

    return clean(asdict(value))


# ═══════════════════════════════════════════════════════════════════════
# 1. Context Builder — assembles LLM prompts (prompt/)
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class ContextBuilderConfig:
    """Configuration for prompt generation"""

    template: str = "default"  # "default", "evox"
    template_dir: Optional[str] = None
    system_message: str = "system_message"
    evaluator_system_message: str = "evaluator_system_message"

    suggest_simplification_after_chars: Optional[int] = 500


# ═══════════════════════════════════════════════════════════════════════
# 2. Solution Generator — produces candidates via LLM calls (llm/)
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class ToolExecutionConfig:
    """SkyDiscover-owned controls for executing LLM function tools."""

    # Optional serialized-result limit. None returns the complete tool result.
    max_output_chars: Optional[int] = None


@dataclass
class TavilyToolConfig:
    """Configuration for the ``tavily`` LLM function tool.

    The tool is opt-in through ``llm.tools: [tavily]``. Its API key
    falls back to ``TAVILY_API_KEY`` so credentials do not need to be stored
    in YAML. Search request defaults mirror Tavily's documented ``/search``
    defaults; ``query`` is supplied by the model at call time.
    """

    # Auth / transport
    api_key: Optional[str] = None
    base_url: str = "https://api.tavily.com"
    timeout: float = 60.0

    # Tavily /search body parameters, in the order used by the API reference:
    # https://docs.tavily.com/documentation/api-reference/endpoint/search
    # The model supplies query and may override max_results per call; all other
    # settings remain under caller control.
    search_depth: str = "basic"
    chunks_per_source: int = 3
    max_results: int = 5
    topic: str = "general"
    time_range: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    # Response content
    include_answer: Union[bool, str] = False
    include_raw_content: Union[bool, str] = False
    include_images: bool = False
    include_image_descriptions: bool = False
    include_favicon: bool = False

    # Domain / geo / language filtering
    include_domains: List[str] = field(default_factory=list)
    exclude_domains: List[str] = field(default_factory=list)
    country: Optional[str] = None
    language: Optional[str] = None
    filter_by_language: bool = False

    # Search behavior / response metadata
    auto_parameters: bool = False
    exact_match: bool = False
    include_usage: bool = False
    safe_search: bool = False


@dataclass
class LLMModelConfig:
    """Configuration for a single LLM model"""

    # API configuration
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    name: Optional[str] = None

    # Custom LLM client
    init_client: Optional[Callable] = None

    # Weight for model in pool, default to random sampling model based on weight
    weight: float = 1.0

    # Generation parameters
    system_message: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None

    # Request parameters
    timeout: Optional[int] = None
    retries: Optional[int] = None
    retry_delay: Optional[int] = None

    # Reasoning parameters
    reasoning_effort: Optional[str] = None

    # Built-in function tools. ``None`` inherits the shared LLMConfig value;
    # an explicit empty list disables shared tools for this model.
    tools: Optional[List[str]] = None
    tool_choice: Optional[str] = None
    max_tool_rounds: Optional[int] = None
    tool_execution: Optional[ToolExecutionConfig] = None
    tavily_tool: Optional[TavilyToolConfig] = None

    # Preserve the request dialect when a provider uses a custom proxy endpoint.
    # None retains endpoint-based automatic detection.
    api_provider: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.tool_execution, dict):
            self.tool_execution = ToolExecutionConfig(**self.tool_execution)
        if isinstance(self.tavily_tool, dict):
            self.tavily_tool = TavilyToolConfig(**self.tavily_tool)


@dataclass
class LLMConfig(LLMModelConfig):
    """Configuration for LLM models"""

    # API configuration
    api_base: Optional[str] = None

    # Generation parameters
    system_message: Optional[str] = "system_message"
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = None
    max_tokens: int = 32000

    # Request parameters
    timeout: int = 600
    retries: int = 3
    retry_delay: int = 5

    # model(s) for solution discovery
    models: List[LLMModelConfig] = field(default_factory=list)

    # model(s) for evaluator
    evaluator_models: List[LLMModelConfig] = field(default_factory=lambda: [])

    # model(s) for guide tasks (idea generation, paradigm breakthroughs, etc.)
    # If not specified, falls back to using the main 'models' list
    guide_models: List[LLMModelConfig] = field(default_factory=lambda: [])

    # Reasoning parameters (inherited from LLMModelConfig but can be overridden)
    reasoning_effort: Optional[str] = None

    # Optional built-in function tools sent on ordinary LLM calls.
    # Supported names: "tavily".
    tools: List[str] = field(default_factory=list)
    tool_choice: str = "auto"
    max_tool_rounds: int = 3
    tool_execution: ToolExecutionConfig = field(default_factory=ToolExecutionConfig)
    tavily_tool: TavilyToolConfig = field(default_factory=TavilyToolConfig)

    def __post_init__(self):
        """Post-initialization to set up model configurations"""
        super().__post_init__()

        # If no evaluator models are defined, use the same models as for solution discovery
        if not self.evaluator_models:
            self.evaluator_models = self.models.copy()

        # If no guide models are defined, use the same models as for solution discovery
        if not self.guide_models:
            self.guide_models = self.models.copy()

        # Resolve per-model api_base, api_key, and bare name from provider prefix
        user_set_api_base = self.api_base is not None
        for model in self.models + self.evaluator_models + self.guide_models:
            if model.name and model.api_base is None:
                provider, bare_name, provider_base, env_vars = _parse_model_spec(
                    model.name, self.api_base
                )
                # A shared api_base is an OpenAI-compatible endpoint (proxy, gateway,
                # local server) that may serve gemini-*, deepseek-*, ... next to gpt-*,
                # so every bare name follows it; only an explicit provider prefix such
                # as gemini/... still routes to that provider's own endpoint.
                follows_shared_base = user_set_api_base and (
                    provider in ("openai", "azure") or "/" not in model.name
                )
                if provider_base and not follows_shared_base:
                    model.api_base = provider_base
                if model.api_key is None:
                    if follows_shared_base:
                        # The endpoint decides the credential: a vendor key must not
                        # be sent to a host that did not issue it.
                        env_vars = _env_vars_for_base_url(self.api_base)
                    model.api_key = _resolve_api_key_from_env(env_vars)
                # Strip provider prefix so the API receives the bare model name
                if "/" in model.name and provider != "openai":
                    model.name = bare_name

        # Update models with shared configuration values
        shared_config = {
            "api_base": self.api_base,
            "api_key": self.api_key,
            "api_provider": self.api_provider,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "retries": self.retries,
            "retry_delay": self.retry_delay,
            "reasoning_effort": self.reasoning_effort,
            "tools": self.tools,
            "tool_choice": self.tool_choice,
            "max_tool_rounds": self.max_tool_rounds,
            "tool_execution": self.tool_execution,
            "tavily_tool": self.tavily_tool,
        }
        self.update_model_params(shared_config)

    def update_model_params(self, args: Dict[str, Any], overwrite: bool = False) -> None:
        """Update model parameters for all models (including guide_models)."""
        all_models = self.models + self.evaluator_models + self.guide_models
        for model in all_models:
            for key, value in args.items():
                if overwrite or getattr(model, key, None) is None:
                    setattr(model, key, value)


@dataclass
class AgenticConfig:
    """Configuration for agentic solution generation.

    When enabled, replaces the single-shot LLM call with a multi-turn
    tool-calling agent loop that can read files and search the codebase
    before outputting the discovered solution.
    """

    enabled: bool = False
    codebase_root: Optional[str] = None

    # Agent loop limits
    max_steps: int = 5

    # Timeouts (seconds)
    per_step_timeout: float = 60.0
    overall_timeout: float = 300.0

    # Context management
    max_context_chars: int = 400_000
    max_file_chars: int = 50_000
    max_search_results: int = 50
    max_files_read: int = 20

    # Regex safety
    regex_timeout: float = 2.0
    max_regex_length: int = 200

    # Repo map — a depth-limited directory tree injected into the agent's first
    # message so it knows what files are available to read_file/search.
    repo_map_max_depth: int = 4

    # File access
    allowed_extensions: tuple = (
        ".py",
        ".txt",
        ".md",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".cfg",
        ".ini",
        ".js",
        ".ts",
        ".java",
        ".cpp",
        ".c",
        ".h",
        ".rs",
        ".go",
    )
    excluded_dirs: tuple = (
        ".git",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        ".mypy_cache",
        ".pytest_cache",
        "dist",
        "build",
    )


# ═══════════════════════════════════════════════════════════════════════
# 3. Evaluator — scores candidates and logs metadata (evaluation/)
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class EvaluatorConfig:
    """Configuration for program evaluation"""

    evaluation_file: Optional[str] = None
    file_suffix: str = ".py"
    is_image_mode: bool = False

    timeout: int = 360
    max_retries: int = 3

    # Final (test-mode) evaluation of the best program after the search ends.
    # When the evaluator module defines evaluate_final(), it replaces evaluate()
    # in test mode — that is where held-out / private scoring belongs.  Its
    # metrics are reported under a test_ prefix.  Held-out sets are usually far
    # larger than the training slice, so it gets its own timeout.
    final_evaluation: bool = True
    final_timeout: int = 7200

    # Evaluation strategies
    cascade_evaluation: bool = True
    cascade_thresholds: List[float] = field(default_factory=lambda: [0.3, 0.6])

    # When True, the evaluator source code (or instruction.md for Harbor
    # tasks) is prepended to the LLM system message so the model can see
    # exactly how solutions are scored.  Disabled by default to avoid
    # leaking implementation details that may introduce noise.
    inject_evaluator_context: bool = False

    # LLM-as-a-judge: when True, an LLMJudge scores programs alongside the
    # evaluator and appends llm_* metrics to the result.
    # This will read from prompt.evaluator_system_message if provided, otherwise use the default system prompt.
    llm_as_judge: bool = False

    # Harbor / Frontier-Bench adapter.  These are ignored by ordinary Python
    # and containerized evaluators.
    harbor_solution_path: Optional[str] = None
    harbor_run_command: Optional[str] = None
    harbor_candidate_timeout: Optional[int] = None
    harbor_reward_mode: str = "official"  # "official" | "ctrf" | "metric"
    harbor_partial_credit_cap: float = 0.99
    # Optional numeric verifier report used for continuous discovery rewards.
    # Keys in ``harbor_metric_upper_limits`` are dot-separated JSON paths;
    # each value is the corresponding pass threshold (lower is better).
    harbor_metric_path: Optional[str] = None
    harbor_metric_upper_limits: Dict[str, float] = field(default_factory=dict)
    harbor_expose_verifier_output: bool = True
    harbor_network_mode: str = "default"  # "default" | "none"


# ═════════════════════════════════════════════════════════════════════════════════════════════
# 4. Solution Selector — maintains database and strategy to pick prior programs (search/)
# ═════════════════════════════════════════════════════════════════════════════════════════════


@dataclass
class DatabaseConfig:
    """Base configuration shared by all database types."""

    db_path: Optional[str] = None
    log_prompts: bool = True


@dataclass
class EvolveDatabaseConfig(DatabaseConfig):
    """Read database from a file."""

    database_file_path: Optional[str] = None


@dataclass
class EvoxDatabaseConfig(EvolveDatabaseConfig):
    """Evox (co-evolution) database config with built-in defaults.

    Multiobjective fields mirror AdaEvolve (#39 / issue #42): when
    ``pareto_objectives`` is non-empty, the solution database maintains a
    Pareto front and uses a scalar proxy only for tie-breaking / search
    scoring. Leave empty for scalar ``combined_score`` behaviour.
    """

    evaluation_file: Optional[str] = None
    config_path: Optional[str] = None
    auto_generate_variation_operators: bool = True

    # Metric direction / multiobjective (opt-in)
    higher_is_better: Dict[str, bool] = field(default_factory=dict)
    fitness_key: Optional[str] = None
    pareto_objectives: List[str] = field(default_factory=list)
    pareto_objectives_weight: float = 0.0

    _evox_config_dir = Path(__file__).parent / "search" / "evox" / "config"
    _evox_database_dir = Path(__file__).parent / "search" / "evox" / "database"

    def __post_init__(self):
        if self.database_file_path is None:
            # Initial guide strategy for the solution discovery
            self.database_file_path = str(self._evox_database_dir / "initial_search_strategy.py")
        if self.evaluation_file is None:
            # Dummy evaluator for the guide strategy
            self.evaluation_file = str(self._evox_database_dir / "search_strategy_evaluator.py")
        if self.config_path is None:
            # Default config for the guide strategy
            self.config_path = str(self._evox_config_dir / "search.yaml")


@dataclass
class BeamSearchDatabaseConfig(DatabaseConfig):
    """Beam search database config."""

    beam_width: int = 5
    beam_selection_strategy: str = "diversity_weighted"
    beam_diversity_weight: float = 0.3
    beam_temperature: float = 1.0
    beam_depth_penalty: float = 0.0


@dataclass
class BestOfNDatabaseConfig(DatabaseConfig):
    """Best-of-N database config."""

    best_of_n: int = 5


@dataclass
class AdaEvolveDatabaseConfig(DatabaseConfig):
    """AdaEvolve adaptive multi-island database config."""

    population_size: int = 20
    num_islands: int = 2
    decay: float = 0.9
    intensity_min: float = 0.15
    intensity_max: float = 0.5
    use_adaptive_search: bool = True
    use_ucb_selection: bool = True
    use_migration: bool = True
    use_unified_archive: bool = True
    fixed_intensity: float = 0.4
    migration_interval: int = 15
    migration_count: int = 5
    local_context_program_ratio: float = 0.6
    archive_elite_ratio: float = 0.2
    pareto_weight: float = 0.4
    fitness_weight: float = 1.0
    novelty_weight: float = 0.0
    k_neighbors: int = 5
    diversity_strategy: str = "code"
    use_dynamic_islands: bool = True
    max_islands: int = 5
    spawn_productivity_threshold: float = 0.015
    spawn_cooldown_iterations: int = 30
    use_paradigm_breakthrough: bool = True
    paradigm_window_size: int = 10
    paradigm_improvement_threshold: float = 0.12
    paradigm_max_uses: int = 2
    paradigm_num_to_generate: int = 3
    paradigm_max_tried: int = 10

    # Stagnation handling
    stagnation_threshold: int = 10
    stagnation_multi_child_count: int = 3

    # Sibling context
    sibling_context_limit: int = 5

    # Error retry
    enable_error_retry: bool = True
    max_error_retries: int = 2

    # Archive
    archive_size: int = 100

    # Metric direction
    higher_is_better: Dict[str, bool] = field(default_factory=dict)
    fitness_key: Optional[str] = None
    pareto_objectives: List[str] = field(default_factory=list)
    pareto_objectives_weight: float = 0.0


@dataclass
class OpenEvolveNativeDatabaseConfig(DatabaseConfig):
    """OpenEvolve Native: MAP-Elites + island-based search config."""

    num_islands: int = 5
    population_size: int = 40
    archive_size: int = 100
    exploration_ratio: float = 0.2
    exploitation_ratio: float = 0.7
    elite_selection_ratio: float = 0.1
    feature_dimensions: List[str] = field(default_factory=lambda: ["complexity", "diversity"])
    feature_bins: int = 10
    diversity_reference_size: int = 20
    migration_interval: int = 10
    migration_rate: float = 0.1
    random_seed: Optional[int] = 42


@dataclass
class ClaudeCodeConfig(DatabaseConfig):
    """Configuration for the Claude Code baseline.

    Claude Code runs autonomously inside a Docker container, iterating on
    the solution using the evaluator directly.  max_turns maps to the
    --max-turns flag passed to the claude CLI.
    """

    max_turns: int = 50
    docker_image: str = "skydiscover-claude-code:latest"


@dataclass
class ClaudeCodeLocalConfig(DatabaseConfig):
    """Configuration for the local Claude Code baseline.

    Same as the Claude Code baseline but runs the claude CLI locally (no
    Docker) using the user's subscription login.  max_turns maps to the
    --max-turns flag passed to the claude CLI.
    """

    max_turns: int = 50


@dataclass
class GEPANativeDatabaseConfig(DatabaseConfig):
    """Configuration for GEPA Native search database.

    GEPA (Guided Evolution for Program Adaptation) uses an elite pool with
    epsilon-greedy selection, acceptance gating, and LLM-mediated merge.
    """

    population_size: int = 40
    candidate_selection_strategy: str = "epsilon_greedy"  # "epsilon_greedy", "best", "pareto"
    epsilon: float = 0.1
    max_rejection_history: int = 20

    # Controller-read settings (stored here for single config source)
    acceptance_gating: bool = True
    use_merge: bool = True
    merge_after_stagnation: int = 15
    max_merge_attempts: int = 10
    max_recent_failures: int = 5
    random_seed: Optional[int] = 42


_DB_CONFIG_BY_TYPE: Dict[str, type] = {
    "evox": EvoxDatabaseConfig,
    "beam_search": BeamSearchDatabaseConfig,
    "best_of_n": BestOfNDatabaseConfig,
    "topk": DatabaseConfig,
    "adaevolve": AdaEvolveDatabaseConfig,
    "openevolve_native": OpenEvolveNativeDatabaseConfig,
    "gepa_native": GEPANativeDatabaseConfig,
    "claude_code": ClaudeCodeConfig,
    "claude_code_local": ClaudeCodeLocalConfig,
}


@dataclass
class SearchConfig:
    """General Configuration for All Search Algorithms"""

    type: str = "topk"
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    num_context_programs: int = 4
    output_dir: Optional[str] = None
    switch_interval: Optional[int] = (
        None  # EvoX: stagnation iters before strategy switch. Auto-calculated if None.
    )
    share_llm: bool = (
        False  # EvoX: if True, meta-level search evolution uses the same LLM as the main discovery process.
    )


# ═══════════════════════════════════════════════════════════════════════
# Extras — live monitor dashboard
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class MonitorConfig:
    """Configuration for the live run monitor dashboard"""

    enabled: bool = False
    port: int = 8765
    host: str = "127.0.0.1"
    max_solution_length: int = 10000

    # AI summary settings
    summary_model: Optional[str] = None
    summary_api_key: Optional[str] = None
    summary_api_base: Optional[str] = None
    summary_top_k: int = 3
    summary_interval: int = 0  # Auto-generate every N programs (0 = manual)


# ═══════════════════════════════════════════════════════════════════════
# Benchmark Loader
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class BenchmarkConfig:
    """Configuration for loading problems from external benchmark datasets.

    When enabled, allows SkyDiscover to fetch problems from external
    benchmark datasets (e.g., KernelBench, Frontier-CS) without requiring
    explicit initial_program paths.

    Benchmark specification and evaluation parameters (e.g., target problem)
    are stored in a `params` dictionary.
    """

    enabled: bool = False
    name: Optional[str] = None
    resolver: Optional[str] = (
        None  # Python import path to resolver module (e.g., 'benchmarks.kernelbench.resolver')
    )
    params: Dict[str, Any] = field(default_factory=dict)  # Benchmark-specific parameters


@dataclass
class TavilyRetrievalConfig:
    """tavily retrieve backend — one Tavily REST ``/search`` call per retrieve.

    Mirrors the full Tavily ``/search`` request body. Only ``/search`` is used: it
    returns query-ranked documents (title/url/content/score) in a single call.
    ``query`` is not here — it is the per-retrieve argument. Cost is set solely by
    ``search_depth`` ("basic"/"fast"/"ultra-fast" = 1 credit, "advanced" = 2);
    ``include_domains`` can pin retrieval to e.g. arxiv.org, github.com, oeis.org,
    mathoverflow.net (empty = unrestricted).
    """

    # ── auth / transport (not Tavily body params) ──
    api_key: Optional[str] = None  # falls back to the TAVILY_API_KEY env var
    base_url: str = "https://api.tavily.com"
    timeout: float = 60.0

    # ── search behaviour ──
    search_depth: str = "advanced"  # "basic" | "advanced" | "fast" | "ultra-fast"
    topic: str = "general"  # "general" | "news" | "finance"
    max_results: Optional[int] = 5  # 1-20; None → evoduet.top_k_for_retrieval
    chunks_per_source: int = 3  # 1-3; advanced search only
    auto_parameters: bool = False  # auto-tune params (can force advanced → 2 credits)
    exact_match: bool = False  # only results containing the exact quoted phrase(s)

    # ── date filtering ──
    time_range: Optional[str] = None  # "day"|"week"|"month"|"year" (or d|w|m|y)
    start_date: Optional[str] = None  # YYYY-MM-DD; published/updated on or after
    end_date: Optional[str] = None  # YYYY-MM-DD; published/updated on or before

    # ── domain / geo filtering ──
    include_domains: List[str] = field(default_factory=list)  # max 300
    exclude_domains: List[str] = field(default_factory=list)  # max 150
    country: Optional[str] = None  # boost a country's results; general topic only

    # ── response content ──
    # May request Tavily's response-level answer, but EvoDuet deliberately
    # ranks only source-backed results[] documents.
    include_answer: Union[bool, str] = False
    include_raw_content: Union[bool, str] = False  # True | "markdown" | "text" | False
    include_images: bool = False  # add query-related images
    include_image_descriptions: bool = False  # describe images (requires include_images)
    include_favicon: bool = False  # add each result's favicon URL
    include_usage: bool = True  # return credit-usage in the response (for cost tracking)
    safe_search: bool = False  # enterprise only; not for fast/ultra-fast depths


@dataclass
class SearchSelectionConfig:
    """How prior search records are selected for the next EvoDuet step.

    ``delta`` keeps both the best and worst measured directions; ``estimated_score``
    keeps the model-estimated best; ``recency``/``full`` are direct controls.
    ``top_k`` selects individual results to include in the search-history context.
    """

    policy: str = "delta"
    num: Optional[int] = 6
    criterion: str = "relevance_score"  # SearchResult field to rank by when policy == "top_k"


@dataclass
class EvoDuetConfig:
    """Observed web search and retrieval gating."""

    enabled: bool = False
    tavily_retrieval: TavilyRetrievalConfig = field(default_factory=TavilyRetrievalConfig)

    # Generative stages use the solution model selected for the current iteration.
    retrieval_gating_backend_type: str = "llm"  # "llm" | "heuristic" | "always" | "stagnation"
    gating_retrieve_probability: float = 0.1
    retrieval_gating_prompt_template_name: str = "retrieval_gating"
    query_construction_max_tokens: int = 32_768

    query_optimization_max_rounds: int = 3
    query_optimization_queries_per_round: int = 1
    search_result_top_k: int = 10
    documents_per_entry: int = 3

    population_state_prompt_template_name: str = "population_analysis"
    knowledge_state_analysis_prompt_template_name: str = "knowledge_analysis"
    analysis_max_chars: int = 32_768
    # Optional input caps; None preserves the full input.
    search_database_analysis_max_chars: Optional[int] = None
    population_state_max_chars: Optional[int] = None
    population_state_recent_k: Optional[int] = 20
    max_document_chars: int = 10_000
    random_seed: int = 0
    top_k_for_retrieval: int = 10
    search_selection: SearchSelectionConfig = field(default_factory=SearchSelectionConfig)

    @classmethod
    def from_dict(cls, settings: Dict[str, Any]) -> "EvoDuetConfig":
        """Load current settings; unknown and removed options are rejected."""
        values = dict(settings)
        for name, config_type in (
            ("tavily_retrieval", TavilyRetrievalConfig),
            ("search_selection", SearchSelectionConfig),
        ):
            if isinstance(values.get(name), dict):
                values[name] = config_type(**values[name])
        return cls(**values)

    def normalize_context_and_generation_budgets(self) -> None:
        """Validate the input and output budgets used by observed search."""
        for path in (
            "analysis_max_chars",
            "search_database_analysis_max_chars",
            "query_construction_max_tokens",
        ):
            value = getattr(self, path)
            if path == "search_database_analysis_max_chars" and value is None:
                continue
            message = f"evoduet.{path} must be an integer >= 1"
            if isinstance(value, bool):
                raise ValueError(message)
            try:
                normalized = int(value)
                if float(value) != normalized or normalized < 1:
                    raise ValueError(message)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(message) from exc
            setattr(self, path, normalized)

    def __post_init__(self):
        self.normalize_context_and_generation_budgets()
        if isinstance(self.search_selection, dict):
            self.search_selection = SearchSelectionConfig(**self.search_selection)
        for path in (
            "query_optimization_max_rounds",
            "query_optimization_queries_per_round",
            "search_result_top_k",
        ):
            value = getattr(self, path)
            message = f"evoduet.{path} must be an integer >= 1"
            if isinstance(value, bool):
                raise ValueError(message)
            try:
                normalized = int(value)
                if float(value) != normalized or normalized < 1:
                    raise ValueError(message)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(message) from exc
            setattr(self, path, normalized)
        self.documents_per_entry = int(self.documents_per_entry)
        self.top_k_for_retrieval = int(self.top_k_for_retrieval)
        if self.population_state_max_chars is not None:
            self.population_state_max_chars = int(self.population_state_max_chars)
        if self.population_state_recent_k is not None:
            self.population_state_recent_k = int(self.population_state_recent_k)
        self.max_document_chars = int(self.max_document_chars)
        if self.tavily_retrieval.max_results is not None:
            self.tavily_retrieval.max_results = int(self.tavily_retrieval.max_results)
        self.tavily_retrieval.timeout = float(self.tavily_retrieval.timeout)
        self.tavily_retrieval.chunks_per_source = int(self.tavily_retrieval.chunks_per_source)
        if self.search_selection.num is not None:
            self.search_selection.num = int(self.search_selection.num)
            if self.search_selection.num <= 0:
                raise ValueError("evoduet.search_selection.num must be positive")
        if self.search_result_top_k <= 0:
            raise ValueError("evoduet.search_result_top_k must be positive")
        if self.documents_per_entry <= 0:
            raise ValueError("evoduet.documents_per_entry must be positive")
        if self.top_k_for_retrieval <= 0:
            raise ValueError("evoduet.top_k_for_retrieval must be positive")
        if self.tavily_retrieval.max_results is not None and not (
            1 <= self.tavily_retrieval.max_results <= 20
        ):
            raise ValueError("evoduet.tavily_retrieval.max_results must be between 1 and 20")
        if not math.isfinite(self.tavily_retrieval.timeout) or self.tavily_retrieval.timeout <= 0:
            raise ValueError("evoduet.tavily_retrieval.timeout must be positive")
        if not 1 <= self.tavily_retrieval.chunks_per_source <= 3:
            raise ValueError("evoduet.tavily_retrieval.chunks_per_source must be between 1 and 3")
        if self.tavily_retrieval.max_results is None and self.top_k_for_retrieval > 20:
            raise ValueError("evoduet.top_k_for_retrieval must be at most 20 when used by Tavily")
        if not 0.0 <= float(self.gating_retrieve_probability) <= 1.0:
            raise ValueError("evoduet.gating_retrieve_probability must be between 0 and 1")
        if self.search_selection.policy not in {
            "delta",
            "estimated_score",
            "recency",
            "full",
            "top_k",
        }:
            raise ValueError(
                "evoduet.search_selection.policy must be delta, estimated_score, recency, full, or top_k"
            )
        if self.population_state_max_chars is not None and self.population_state_max_chars < 512:
            raise ValueError("evoduet.population_state_max_chars must be at least 512 (or None)")
        if self.population_state_recent_k is not None and self.population_state_recent_k <= 0:
            raise ValueError("evoduet.population_state_recent_k must be positive (or None)")
        if self.max_document_chars <= 0:
            raise ValueError("evoduet.max_document_chars must be positive")


# ═══════════════════════════════════════════════════════════════════════
# Solution confidence — predicts whether a candidate improves on its parent
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class SolutionConfidenceConfig:
    """Optional logprob and verbalized assessments before candidate evaluation.

    Confidence estimates the probability of strictly improving ``metric`` over
    the parent by more than ``improvement_epsilon``. The evaluation outcome is
    shared by both assessments and recorded separately from search metrics,
    together with a Brier error for each prediction. Enabling this runs both.
    """

    enabled: bool = False
    metric: str = "combined_score"
    higher_is_better: bool = True
    improvement_epsilon: float = 0.0
    timeout: float = 120.0
    top_logprobs: int = 20
    # Pin both confidence assessments to an OpenRouter provider slug.
    openrouter_provider: Optional[str] = None

    def __post_init__(self):
        for name in ("enabled", "higher_is_better"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"solution_confidence.{name} must be boolean")
        if not isinstance(self.metric, str) or not self.metric.strip():
            raise ValueError("solution_confidence.metric must be a nonempty string")
        self.metric = self.metric.strip()
        for name, allow_zero in (("improvement_epsilon", True), ("timeout", False)):
            value = getattr(self, name)
            requirement = "nonnegative and finite" if allow_zero else "positive and finite"
            message = f"solution_confidence.{name} must be {requirement}"
            if isinstance(value, bool):
                raise ValueError(message)
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(message) from exc
            if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
                raise ValueError(message)
            setattr(self, name, value)
        message = "solution_confidence.top_logprobs must be an integer from 1 to 20"
        if isinstance(self.top_logprobs, bool):
            raise ValueError(message)
        try:
            top_logprobs = float(self.top_logprobs)
        except (TypeError, ValueError) as exc:
            raise ValueError(message) from exc
        if (
            not math.isfinite(top_logprobs)
            or not top_logprobs.is_integer()
            or not 1 <= top_logprobs <= 20
        ):
            raise ValueError(message)
        self.top_logprobs = int(top_logprobs)
        if self.openrouter_provider is not None:
            if (
                not isinstance(self.openrouter_provider, str)
                or not self.openrouter_provider.strip()
            ):
                raise ValueError(
                    "solution_confidence.openrouter_provider must be None or a nonempty string"
                )
            self.openrouter_provider = self.openrouter_provider.strip()


# ═══════════════════════════════════════════════════════════════════════
# Master Configuration
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class Config:
    """Master configuration for SkyDiscover"""

    # General settings
    max_iterations: int = 100
    checkpoint_interval: int = 10
    log_level: str = "INFO"
    log_dir: Optional[str] = None
    language: Optional[str] = None
    file_suffix: str = ".py"

    # Component configurations
    llm: LLMConfig = field(default_factory=LLMConfig)
    context_builder: ContextBuilderConfig = field(default_factory=ContextBuilderConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    agentic: AgenticConfig = field(default_factory=AgenticConfig)
    evoduet: EvoDuetConfig = field(default_factory=EvoDuetConfig)
    solution_confidence: SolutionConfidenceConfig = field(default_factory=SolutionConfidenceConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    # Live monitor dashboard
    monitor: MonitorConfig = field(default_factory=MonitorConfig)

    # Human feedback settings
    human_feedback_enabled: bool = False
    human_feedback_file: Optional[str] = None
    human_feedback_mode: str = "append"  # "append" or "replace"

    # Generation settings
    diff_based_generation: bool = True
    max_solution_length: int = 60000
    # Best of N within one iteration; only the selected candidate enters the database.
    num_generations: int = 1
    # None allows all N generation requests in flight. Limits span concurrent iterations.
    max_parallel_generations: Optional[int] = None
    # Candidate evaluators may share GPU memory or mutable state; opt into parallel evaluation.
    max_parallel_evaluations: int = 1

    # Parallelism — how many iterations run concurrently.
    # 1 = sequential (default, current behaviour).
    # >1 = N iterations overlap via asyncio tasks: while one evaluates,
    #       others can sample/generate, giving near-linear speedup.
    max_parallel_iterations: int = 1

    # Runtime-only: system prompt override (set by apply_overrides, read by external backends)
    system_prompt_override: Optional[str] = None

    # Backend-specific section for the AlphaEvolve external backend
    # (project_id, engine_id, location, ...). Read by
    # skydiscover/extras/external/alphaevolve_backend.py.
    alphaevolve: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate_generation()

    def validate_generation(self) -> None:
        """Validate candidate counts after construction, YAML loading or CLI overrides."""
        for name in ("num_generations", "max_parallel_generations", "max_parallel_evaluations"):
            value = getattr(self, name)
            if name == "max_parallel_generations" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    def validate_generation_mode(self) -> None:
        """Reject unsupported execution paths instead of silently dropping N."""
        self.validate_generation()
        if self.num_generations == 1:
            return
        if self.search.type not in {"openevolve_native", "topk", "best_of_n", "beam_search"}:
            raise ValueError(
                "num_generations > 1 requires a standard discovery controller "
                "(openevolve_native, topk, best_of_n, beam_search)"
            )
        if self.agentic.enabled or self.language == "image":
            raise ValueError("num_generations > 1 requires non-agentic text/code generation")

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> Config:
        """Load configuration from a YAML file"""
        config_path = Path(path)
        config_dir = config_path.parent

        with open(path, "r") as f:
            raw = f.read()
        config_dict = yaml.safe_load(_expand_env_vars(raw))

        # Handle file references for system_message
        if "prompt" in config_dict and "system_message" in config_dict["prompt"]:
            system_message = config_dict["prompt"]["system_message"]
            if (
                isinstance(system_message, str)
                and "\n" not in system_message.strip()
                and len(system_message.strip()) < 256
            ):
                file_path = config_dir / system_message
                try:
                    if file_path.exists() and file_path.is_file():
                        with open(file_path, "r") as f:
                            config_dict["prompt"]["system_message"] = f.read()
                except OSError:
                    logger.debug("Could not read system_message from %s", file_path, exc_info=True)

        return cls.from_dict(config_dict)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to a YAML file"""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> Config:
        """Create configuration from a dictionary"""
        if "world_knowledge" in config_dict:
            raise ValueError(
                "world_knowledge settings are unsupported; use a current evoduet configuration"
            )
        # Handle nested configurations
        config = Config()

        # Update top-level fields
        for key, value in config_dict.items():
            if key not in [
                "llm",
                "prompt",
                "database",
                "search",
                "evaluator",
                "agentic",
                "evoduet",
                "solution_confidence",
                "benchmark",
                "monitor",
            ] and hasattr(config, key):
                setattr(config, key, value)

        # Update nested configs
        if "llm" in config_dict:
            llm_dict = config_dict["llm"]
            if "models" in llm_dict:
                llm_dict["models"] = [LLMModelConfig(**m) for m in llm_dict["models"]]
            if "evaluator_models" in llm_dict:
                llm_dict["evaluator_models"] = [
                    LLMModelConfig(**m) for m in llm_dict["evaluator_models"]
                ]
            if "guide_models" in llm_dict:
                llm_dict["guide_models"] = [LLMModelConfig(**m) for m in llm_dict["guide_models"]]
            config.llm = LLMConfig(**llm_dict)
        if "prompt" in config_dict:
            config.context_builder = ContextBuilderConfig(**config_dict["prompt"])

        if "search" in config_dict:
            search_dict = config_dict["search"]
            search_type = search_dict.get("type", "topk")
            db_config_cls = _DB_CONFIG_BY_TYPE.get(search_type, DatabaseConfig)
            if "database" in search_dict:
                db_dict = search_dict["database"]
                # Separate known fields from algorithm-specific extras
                # (e.g., adaevolve's decay, intensity_min, use_adaptive_search, etc.)
                known_fields = {f.name for f in fields(db_config_cls)}
                db_known = {k: v for k, v in db_dict.items() if k in known_fields}
                db_extras = {k: v for k, v in db_dict.items() if k not in known_fields}
                db_config = db_config_cls(**db_known)
                for k, v in db_extras.items():
                    setattr(db_config, k, v)
                search_dict["database"] = db_config
            else:
                search_dict["database"] = db_config_cls()
            config.search = SearchConfig(**search_dict)

        if "evaluator" in config_dict:
            config.evaluator = EvaluatorConfig(**config_dict["evaluator"])
        if "agentic" in config_dict:
            agentic_dict = dict(config_dict["agentic"])  # copy to avoid mutating input
            # Convert list fields to tuples for the dataclass
            for tuple_field in ("allowed_extensions", "excluded_dirs"):
                if tuple_field in agentic_dict and isinstance(agentic_dict[tuple_field], list):
                    agentic_dict[tuple_field] = tuple(agentic_dict[tuple_field])
            config.agentic = AgenticConfig(**agentic_dict)
        if "evoduet" in config_dict:
            config.evoduet = EvoDuetConfig.from_dict(config_dict["evoduet"])
        if "solution_confidence" in config_dict:
            config.solution_confidence = SolutionConfidenceConfig(
                **config_dict["solution_confidence"]
            )
        if "benchmark" in config_dict:
            benchmark_dict = config_dict["benchmark"]
            # Separate known dataclass fields from benchmark-specific parameters
            known_fields = {f.name for f in fields(BenchmarkConfig) if f.name != "params"}
            benchmark_known = {k: v for k, v in benchmark_dict.items() if k in known_fields}
            benchmark_params = {k: v for k, v in benchmark_dict.items() if k not in known_fields}
            config.benchmark = BenchmarkConfig(**benchmark_known, params=benchmark_params)
        if "monitor" in config_dict:
            config.monitor = MonitorConfig(**config_dict["monitor"])

        config.validate_generation()
        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary"""
        return {
            # General settings
            "max_iterations": self.max_iterations,
            "checkpoint_interval": self.checkpoint_interval,
            "log_level": self.log_level,
            "log_dir": self.log_dir,
            # Component configurations
            "llm": {
                "models": self.llm.models,
                "evaluator_models": self.llm.evaluator_models,
                "api_base": self.llm.api_base,
                "temperature": self.llm.temperature,
                "top_p": self.llm.top_p,
                "max_tokens": self.llm.max_tokens,
                "timeout": self.llm.timeout,
                "retries": self.llm.retries,
                "retry_delay": self.llm.retry_delay,
                "tools": list(self.llm.tools),
                "tool_choice": self.llm.tool_choice,
                "max_tool_rounds": self.llm.max_tool_rounds,
                "tool_execution": {
                    f.name: getattr(self.llm.tool_execution, f.name)
                    for f in fields(self.llm.tool_execution)
                },
                "tavily_tool": {
                    f.name: getattr(self.llm.tavily_tool, f.name)
                    for f in fields(self.llm.tavily_tool)
                    if f.name != "api_key"
                },
            },
            "prompt": {
                "template": self.context_builder.template,
                "template_dir": self.context_builder.template_dir,
                "system_message": self.context_builder.system_message,
                "evaluator_system_message": self.context_builder.evaluator_system_message,
            },
            "search": {
                "type": self.search.type,
                "num_context_programs": self.search.num_context_programs,
                "database": {
                    f.name: getattr(self.search.database, f.name)
                    for f in fields(self.search.database)
                },
            },
            "evaluator": {
                "evaluation_file": self.evaluator.evaluation_file,
                "file_suffix": self.evaluator.file_suffix,
                "is_image_mode": self.evaluator.is_image_mode,
                "timeout": self.evaluator.timeout,
                "max_retries": self.evaluator.max_retries,
                "final_evaluation": self.evaluator.final_evaluation,
                "final_timeout": self.evaluator.final_timeout,
                "cascade_evaluation": self.evaluator.cascade_evaluation,
                "cascade_thresholds": self.evaluator.cascade_thresholds,
                "inject_evaluator_context": self.evaluator.inject_evaluator_context,
                "llm_as_judge": self.evaluator.llm_as_judge,
                "harbor_solution_path": self.evaluator.harbor_solution_path,
                "harbor_run_command": self.evaluator.harbor_run_command,
                "harbor_candidate_timeout": self.evaluator.harbor_candidate_timeout,
                "harbor_reward_mode": self.evaluator.harbor_reward_mode,
                "harbor_partial_credit_cap": self.evaluator.harbor_partial_credit_cap,
                "harbor_metric_path": self.evaluator.harbor_metric_path,
                "harbor_metric_upper_limits": dict(self.evaluator.harbor_metric_upper_limits),
                "harbor_expose_verifier_output": (self.evaluator.harbor_expose_verifier_output),
                "harbor_network_mode": self.evaluator.harbor_network_mode,
            },
            # Agentic generation
            "agentic": {
                "enabled": self.agentic.enabled,
                "codebase_root": self.agentic.codebase_root,
                "max_steps": self.agentic.max_steps,
                "per_step_timeout": self.agentic.per_step_timeout,
                "overall_timeout": self.agentic.overall_timeout,
                "max_context_chars": self.agentic.max_context_chars,
                "max_file_chars": self.agentic.max_file_chars,
                "max_search_results": self.agentic.max_search_results,
                "max_files_read": self.agentic.max_files_read,
                "regex_timeout": self.agentic.regex_timeout,
                "max_regex_length": self.agentic.max_regex_length,
                "repo_map_max_depth": self.agentic.repo_map_max_depth,
                "allowed_extensions": list(self.agentic.allowed_extensions),
                "excluded_dirs": list(self.agentic.excluded_dirs),
            },
            "evoduet": _config_dict_without_api_keys(self.evoduet),
            "solution_confidence": asdict(self.solution_confidence),
            # Live monitor
            "monitor": {
                "enabled": self.monitor.enabled,
                "port": self.monitor.port,
                "host": self.monitor.host,
                "max_solution_length": self.monitor.max_solution_length,
                "summary_model": self.monitor.summary_model,
                "summary_top_k": self.monitor.summary_top_k,
                "summary_interval": self.monitor.summary_interval,
            },
            # Human-in-the-loop
            "human_feedback_enabled": self.human_feedback_enabled,
            "human_feedback_file": self.human_feedback_file,
            # Generation settings
            "diff_based_generation": self.diff_based_generation,
            "max_solution_length": self.max_solution_length,
            "num_generations": self.num_generations,
            "max_parallel_generations": self.max_parallel_generations,
            "max_parallel_evaluations": self.max_parallel_evaluations,
            # Parallelism
            "max_parallel_iterations": self.max_parallel_iterations,
        }


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Load configuration from a YAML file or use defaults"""
    if config_path:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        config = Config.from_yaml(config_path)
    else:
        config = Config()

    # Update api_base from environment if provided — use overwrite=True
    # because __post_init__ already pushed the hardcoded default to all models.
    api_base = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")
    if api_base:
        config.llm.api_base = api_base
        config.llm.update_model_params({"api_base": api_base}, overwrite=True)

    # Determine which API key to use (provider-aware)
    if not config.llm.api_key:
        env_vars = None
        if config.llm.models:
            first_model_name = config.llm.models[0].name
            if first_model_name:
                _, _, _, env_vars = _parse_model_spec(
                    first_model_name,
                    config.llm.models[0].api_base or config.llm.api_base,
                )
        api_key = _resolve_api_key_from_env(env_vars)
        if api_key:
            config.llm.api_key = api_key
            config.llm.update_model_params({"api_key": api_key})

    # Make the system message available to the individual models, in case it is not provided from the prompt sampler
    config.llm.update_model_params({"system_message": config.context_builder.system_message})

    # Bridge provider env vars so that downstream configs (e.g. evox search.yaml)
    # can resolve ${OPENAI_API_KEY} from the environment.
    bridge_provider_env(config)

    return config


def bridge_provider_env(config: Config) -> None:
    """
    Set provider-specific env vars from resolved config.

    External backends read credentials from environment variables directly.
    """
    if not config.llm.models:
        return
    model = config.llm.models[0]
    if not model.api_key:
        return

    # Use _parse_model_spec to get the right env vars for this model
    _, _, _, env_vars = _parse_model_spec(model.name or "", model.api_base)
    for var in env_vars:
        os.environ.setdefault(var, model.api_key)

    # Always ensure OPENAI_API_KEY is set — many tools (ShinkaEvolve, etc.) expect it
    os.environ.setdefault("OPENAI_API_KEY", model.api_key)

    # Set OPENAI_API_BASE so backends that check it can find the endpoint
    if model.api_base:
        os.environ.setdefault("OPENAI_API_BASE", model.api_base)


def build_output_dir(search_type: str, initial_program_path: str, base_dir: str = "outputs") -> str:
    """Build a standardized output directory: outputs/<search_type>/<problem_name>_<MMDD_HHMM>/"""
    from datetime import datetime

    problem_name = (
        os.path.basename(os.path.dirname(os.path.abspath(initial_program_path))) or "unknown"
    )
    timestamp = datetime.now().strftime("%m%d_%H%M")
    return os.path.join(base_dir, search_type, f"{problem_name}_{timestamp}")


# ═══════════════════════════════════════════════════════════════════════
# Runtime overrides — shared by the public API and CLI
# ═══════════════════════════════════════════════════════════════════════


def apply_overrides(
    config: Config,
    *,
    model: Optional[str] = None,
    api_base: Optional[str] = None,
    agentic: bool = False,
    search: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> None:
    """Apply runtime overrides (model, api_base, etc.) to a loaded Config in place."""
    if model:
        # Parse the model string into a list of model specifications
        specs = [s.strip() for s in model.split(",")]
        models: List[LLMModelConfig] = []
        for spec in specs:
            provider, model_name, default_api_base, env_vars = _parse_model_spec(spec, api_base)
            effective_base = api_base or default_api_base
            if effective_base is None:
                raise ValueError(
                    f"Provider '{provider}' requires an explicit api_base.\n"
                    f"Example: model='{spec}', api_base='http://localhost:8000/v1'"
                )
            if is_azure_endpoint(effective_base):
                env_vars = _AZURE_KEY_ENV
            resolved_key = _resolve_api_key_from_env(env_vars)
            models.append(
                LLMModelConfig(
                    name=model_name,
                    api_base=effective_base,
                    api_key=resolved_key,
                    api_provider=provider,
                )
            )

        config.llm.api_base = models[0].api_base
        if models[0].api_key:
            config.llm.api_key = models[0].api_key
        config.llm.models = models
        config.llm.evaluator_models = [
            LLMModelConfig(
                name=m.name, api_base=m.api_base, api_key=m.api_key, api_provider=m.api_provider
            )
            for m in models
        ]
        config.llm.guide_models = [
            LLMModelConfig(
                name=m.name, api_base=m.api_base, api_key=m.api_key, api_provider=m.api_provider
            )
            for m in models
        ]
    elif api_base:
        # A transport-only override keeps the provider dialect. A model override
        # above recomputes it when switching providers.
        config.llm.api_base = api_base
        config.llm.update_model_params({"api_base": api_base}, overwrite=True)

    # API key (api_base-only; multi-model already resolved above)
    if not model and api_base:
        parsed_env_vars = _env_vars_for_base_url(config.llm.api_base)
        resolved_key = _resolve_api_key_from_env(parsed_env_vars)
        if resolved_key:
            config.llm.api_key = resolved_key
            config.llm.update_model_params({"api_key": resolved_key}, overwrite=True)

    # Propagate shared generation/request settings
    if model or api_base:
        config.llm.update_model_params(
            {
                "temperature": config.llm.temperature,
                "top_p": config.llm.top_p,
                "max_tokens": config.llm.max_tokens,
                "timeout": config.llm.timeout,
                "retries": config.llm.retries,
                "retry_delay": config.llm.retry_delay,
                "reasoning_effort": config.llm.reasoning_effort,
                "tools": config.llm.tools,
                "tool_choice": config.llm.tool_choice,
                "max_tool_rounds": config.llm.max_tool_rounds,
                "tool_execution": config.llm.tool_execution,
                "tavily_tool": config.llm.tavily_tool,
            },
            overwrite=True,
        )
        # Fill api_base/api_key only where a model doesn't already have them
        config.llm.update_model_params(
            {"api_base": config.llm.api_base, "api_key": config.llm.api_key},
            overwrite=False,
        )

    if agentic:
        config.agentic.enabled = True

    if search:
        if not hasattr(config, "search"):
            config.search = SearchConfig()
        config.search.type = search
        new_db_cls = _DB_CONFIG_BY_TYPE.get(search)
        if new_db_cls and not isinstance(config.search.database, new_db_cls):
            config.search.database = new_db_cls()

    if system_prompt:
        config.context_builder.system_message = system_prompt
        config.system_prompt_override = system_prompt

    config.validate_generation()


# YAML section names whose Config attribute differs from the section key.
_SECTION_ALIASES = {"prompt": "context_builder"}

# Shared llm generation/request params that must be propagated to per-model
# configs (mirrors the propagation apply_overrides does).
_LLM_PROPAGATED = (
    "temperature",
    "top_p",
    "max_tokens",
    "timeout",
    "retries",
    "retry_delay",
    "reasoning_effort",
    "api_base",
    "api_key",
    "api_provider",
    "tools",
    "tool_choice",
    "max_tool_rounds",
    "tool_execution",
    "tavily_tool",
)


def _infer_scalar(raw: str) -> Any:
    """Best-effort scalar coercion for a brand-new field with no known type."""
    low = raw.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _coerce_override(raw: str, current: Any) -> Any:
    """Coerce a CLI string value to match the type of the existing field."""
    if isinstance(current, bool):
        low = raw.strip().lower()
        if low in ("true", "1", "yes", "on"):
            return True
        if low in ("false", "0", "no", "off"):
            return False
        raise ValueError(f"expected a boolean, got {raw!r}")
    if isinstance(current, int):  # bool already handled above
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, (list, tuple)):
        # Comma-separated CLI value -> list (e.g. --evaluator.cascade_thresholds 0.5,0.8)
        return [x.strip() for x in raw.split(",") if x.strip()]
    if current is None:
        return _infer_scalar(raw)
    return raw


def apply_dot_overrides(config: Config, overrides: Dict[str, str]) -> None:
    """Apply dotted-path overrides onto a loaded Config in place.

    Keys are dotted paths into the Config object (e.g. ``llm.temperature``,
    ``evaluator.timeout``, ``checkpoint_interval``); the YAML section name
    ``prompt`` maps to the ``context_builder`` attribute.  Values are strings
    from the CLI and are coerced to the type of the existing field.  A leaf
    field that does not already exist is rejected for EvoDuet. Other sections
    allow extension fields with a warning.
    """
    if not overrides:
        return

    touched_llm_fields = set()
    touched_solution_confidence = False
    touched_evoduet = False
    for dotted, raw in overrides.items():
        parts = dotted.split(".")
        if parts[0] in _SECTION_ALIASES:
            parts = [_SECTION_ALIASES[parts[0]]] + parts[1:]

        obj = config
        for p in parts[:-1]:
            if not hasattr(obj, p):
                raise ValueError(f"unknown config section in override '--{dotted}'")
            obj = getattr(obj, p)

        leaf = parts[-1]
        if hasattr(obj, leaf):
            current = getattr(obj, leaf)
        else:
            if parts[0] == "evoduet":
                raise ValueError(f"unknown EvoDuet config field in override '--{dotted}'")
            logger.warning(
                "Override '--%s' targets an unknown config field; it will be "
                "stored but is likely ignored.",
                dotted,
            )
            current = None

        try:
            if obj is config and leaf == "max_parallel_generations":
                value = None if raw.strip().lower() == "null" else _infer_scalar(raw)
            elif isinstance(obj, SolutionConfidenceConfig) and leaf == "openrouter_provider":
                # Its None default must not turn a string provider slug into a scalar.
                value = raw
            elif isinstance(obj, EvoDuetConfig) and leaf == "search_database_analysis_max_chars":
                # Allow resetting this optional input cap to its unbounded default.
                value = None if raw.strip().lower() == "null" else _infer_scalar(raw)
            else:
                value = _coerce_override(raw, current)
        except ValueError as exc:
            raise ValueError(f"invalid value for override '--{dotted}': {exc}")
        setattr(obj, leaf, value)

        if parts[0] == "llm" and leaf in _LLM_PROPAGATED:
            touched_llm_fields.add(leaf)
        if parts[0] == "solution_confidence":
            touched_solution_confidence = True
        if parts[0] == "evoduet":
            touched_evoduet = True

    config.validate_generation_mode()
    if touched_solution_confidence:
        config.solution_confidence.__post_init__()
    if touched_evoduet:
        config.evoduet.__post_init__()

    # An overridden endpoint must not keep the previous provider's credential. Only adopt a
    # key the new endpoint actually has, so an explicitly configured api_key is not wiped.
    if "api_base" in touched_llm_fields and "api_key" not in touched_llm_fields:
        resolved_key = _resolve_api_key_from_env(_env_vars_for_base_url(config.llm.api_base))
        if resolved_key:
            config.llm.api_key = resolved_key
            touched_llm_fields.add("api_key")

    # Re-propagate the overridden llm params onto per-model configs so generation uses
    # them (apply_overrides already ran by this point). Only the fields actually
    # overridden: re-propagating all of _LLM_PROPAGATED would stamp the shared
    # api_base/api_key over the per-model endpoints of a multi-provider pool.
    if touched_llm_fields:
        config.llm.update_model_params(
            {k: getattr(config.llm, k) for k in touched_llm_fields},
            overwrite=True,
        )
