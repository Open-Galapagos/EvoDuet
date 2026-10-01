"""Release entry points resolve tasks without network access or credentials."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from skydiscover import cli as framework_cli
from skydiscover import evoduet
from skydiscover.config import (
    Config,
    EvoDuetConfig,
    apply_dot_overrides,
    apply_overrides,
    load_config,
)
from skydiscover.evoduet import EvoDuet
from skydiscover.evoduet.layer import EVODUET_STATE_VERSION
from skydiscover.evoduet.retrieval import SearchStore
from skydiscover.evoduet_cli import main, prepare_run
from skydiscover.runner import CHECKPOINT_COMPLETE_FILE, CHECKPOINT_SCHEMA_VERSION
from skydiscover.search.default_discovery_controller import DiscoveryController


@pytest.fixture
def task(tmp_path, monkeypatch):
    def forbid_network(*args, **kwargs):
        raise AssertionError("A preview must not access the network")

    monkeypatch.setattr("socket.socket.connect", forbid_network)
    path = tmp_path / "task with spaces"
    path.mkdir()
    (path / "initial_program.py").write_text("value = 0\n")
    (path / "skydiscover_evaluator.py").write_text(
        "raise AssertionError('preview executed evaluator')\n"
    )
    (path / "config.yaml").write_text(
        "max_iterations: 12\nllm:\n  models:\n    - name: offline-model\n"
        "prompt:\n  system_message: Preserve the task instructions.\n"
        "evaluator:\n  timeout: 123\n"
    )
    return path


def test_task_settings_and_advanced_overrides_survive(task):
    _, config, forwarded = prepare_run(
        [
            str(task),
            "--num-generations",
            "8",
            "--seed",
            "17",
            "--edit",
            "full",
            "--inner-rounds",
            "2",
            "--query-count",
            "3",
            "--llm.temperature",
            "0.25",
        ]
    )
    assert config.context_builder.system_message == "Preserve the task instructions."
    assert config.evaluator.timeout == 123
    assert config.num_generations == config.max_parallel_evaluations == 8
    assert config.search.database.random_seed == config.evoduet.random_seed == 17
    assert not config.diff_based_generation
    assert config.evoduet.query_optimization_max_rounds == 2
    assert config.evoduet.query_optimization_queries_per_round == 3
    assert config.evoduet.retrieval_gating_prompt_template_name == "retrieval_gating"
    assert config.evoduet.knowledge_state_analysis_prompt_template_name == "knowledge_analysis"
    assert config.evoduet.search_selection.policy == "recency"
    assert config.evoduet.search_selection.num == 10
    assert config.llm.temperature == 0.25
    assert forwarded[:2] == [
        str(task / "initial_program.py"),
        str(task / "skydiscover_evaluator.py"),
    ]


@pytest.mark.parametrize(
    "flag,value",
    [
        ("evoduet.retrieval_backend_type", "oracle"),
        ("evoduet.oracle_retrieval.file_path", "web_search.json"),
    ],
)
def test_oracle_options_are_rejected_before_running(task, capsys, flag, value):
    with pytest.raises(SystemExit) as error:
        prepare_run([str(task), "--" + flag, value])
    assert error.value.code == 2
    assert flag in capsys.readouterr().err


@pytest.mark.parametrize("baseline", [False, True])
@pytest.mark.parametrize("generations", [1, 8])
def test_experiment_defaults_reach_the_framework_and_model(
    task, monkeypatch, baseline, generations
):
    _, preview, forwarded = prepare_run(
        [str(task), "--num-generations", str(generations)] + (["--baseline"] if baseline else [])
    )
    monkeypatch.setattr(sys, "argv", ["evoduet-run", *forwarded])
    parsed = framework_cli.parse_args()
    actual = load_config(parsed.config)
    apply_overrides(actual, search=parsed.search)
    apply_dot_overrides(actual, parsed._dot_overrides)

    for config in (preview, actual):
        for model in (config.llm, *config.llm.models):
            assert model.max_tokens == 32768
            assert model.temperature == 0.7
            assert model.top_p == 0.95
            assert model.reasoning_effort == "medium"
            assert model.timeout == 1800
            assert model.retries == 3
            assert not model.tools
        assert config.checkpoint_interval == 1
        assert not config.solution_confidence.enabled
        assert not config.evaluator.cascade_evaluation
        assert not config.evaluator.inject_evaluator_context
        assert config.evaluator.final_evaluation  # best evaluation, no NDG postprocessing
        assert config.num_generations == config.max_parallel_generations == generations
        assert config.max_parallel_evaluations == (1 if baseline else generations)
        assert config.evaluator.timeout == 123
        assert config.llm.models[0].name == "offline-model"


def test_explicit_flags_override_experiment_defaults(task):
    _, config, _ = prepare_run(
        [
            str(task),
            "--baseline",
            "--num-generations",
            "8",
            "--llm.max_tokens",
            "4096",
            "--llm.top_p",
            "0.8",
            "--llm.reasoning_effort",
            "high",
            "--llm.timeout",
            "90",
            "--checkpoint_interval",
            "5",
            "--max_parallel_evaluations",
            "2",
        ]
    )
    assert config.llm.models[0].max_tokens == 4096
    assert config.llm.models[0].top_p == 0.8
    assert config.llm.models[0].reasoning_effort == "high"
    assert config.llm.models[0].timeout == 90
    assert config.checkpoint_interval == 5
    assert config.max_parallel_evaluations == 2


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "existing-run/checkpoints/checkpoint_4"
    path.mkdir(parents=True)
    (path / "metadata.json").write_text(json.dumps({"last_iteration": 4}))
    (path / CHECKPOINT_COMPLETE_FILE).write_text(
        json.dumps({"schema_version": CHECKPOINT_SCHEMA_VERSION, "iteration": 4})
    )
    (path / "evoduet.json").write_text(
        json.dumps(
            {
                "schema_version": EVODUET_STATE_VERSION,
                "store": SearchStore().state_dict(),
                "query_optimization_history": [],
            }
        )
    )
    return path


def test_resume_uses_total_iteration_target_and_public_state(task, checkpoint, monkeypatch, capsys):
    before = (checkpoint / "evoduet.json").read_bytes()
    arguments = [str(task), "--checkpoint", str(checkpoint), "--iterations", "10"]
    assert main([*arguments, "--dry-run"]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["checkpoint_state_file"] == "evoduet.json"
    assert preview["iterations"] == 10
    assert preview["resume_iteration"] == 4
    assert preview["remaining_iterations"] == 6
    assert preview["output"] == str(checkpoint.parent.parent)

    def run():
        parsed = framework_cli.parse_args()
        assert parsed.iterations == 6
        assert parsed.checkpoint == str(checkpoint)
        assert parsed.output == str(checkpoint.parent.parent)
        return 0

    monkeypatch.setattr(framework_cli, "main", run)
    assert main(arguments) == 0
    assert (checkpoint / "evoduet.json").read_bytes() == before


def test_resume_requires_evoduet_json_instead_of_starting_empty(task, checkpoint, capsys):
    (checkpoint / "evoduet.json").rename(checkpoint / "world_knowledge.json")
    with pytest.raises(SystemExit) as error:
        prepare_run([str(task), "--checkpoint", str(checkpoint)])
    assert error.value.code == 2
    assert "evoduet.json" in capsys.readouterr().err
    _, _, forwarded = prepare_run([str(task), "--checkpoint", str(checkpoint), "--baseline"])
    assert forwarded[forwarded.index("--iterations") + 1] == "8"


def test_resume_does_not_add_iterations_after_reaching_target(task, checkpoint):
    args, _, forwarded = prepare_run(
        [str(task), "--checkpoint", str(checkpoint), "--iterations", "4"]
    )
    assert args.remaining_iterations == 0
    assert forwarded[forwarded.index("--iterations") + 1] == "0"


def test_resume_rejects_iteration_target_before_checkpoint(task, checkpoint, capsys):
    with pytest.raises(SystemExit):
        prepare_run([str(task), "--checkpoint", str(checkpoint), "--iterations", "3"])
    assert "exceeds the requested total" in capsys.readouterr().err


@pytest.mark.parametrize("missing", ["metadata.json", CHECKPOINT_COMPLETE_FILE])
def test_resume_rejects_incomplete_checkpoint(task, checkpoint, missing, capsys):
    (checkpoint / missing).unlink()
    with pytest.raises(SystemExit):
        prepare_run([str(task), "--checkpoint", str(checkpoint)])
    assert missing in capsys.readouterr().err


def test_resume_rejects_mismatched_completion_marker(task, checkpoint, capsys):
    (checkpoint / CHECKPOINT_COMPLETE_FILE).write_text(
        json.dumps({"schema_version": CHECKPOINT_SCHEMA_VERSION, "iteration": 3})
    )
    with pytest.raises(SystemExit):
        prepare_run([str(task), "--checkpoint", str(checkpoint)])
    assert "does not match its metadata" in capsys.readouterr().err


@pytest.mark.parametrize("baseline", [False, True])
def test_resume_rejects_older_checkpoint_version(task, checkpoint, capsys, baseline):
    (checkpoint / CHECKPOINT_COMPLETE_FILE).write_text(
        json.dumps({"schema_version": 1, "iteration": 4})
    )
    args = [str(task), "--checkpoint", str(checkpoint)]
    if baseline:
        args.append("--baseline")
    with pytest.raises(SystemExit):
        prepare_run(args)
    assert "unsupported checkpoint format" in capsys.readouterr().err


def test_preview_hides_credentials_and_creates_no_outputs(task, tmp_path, capsys):
    output = tmp_path / "outputs"
    assert (
        main(
            [
                str(task),
                "--dry-run",
                "--output",
                str(output),
                "--llm.api_key",
                "private-value-must-not-appear",
            ]
        )
        == 0
    )
    preview = capsys.readouterr().out
    assert "private-value" not in preview
    assert json.loads(preview)["method"] == "evoduet"
    assert not output.exists()


def test_container_evaluator_directory_is_accepted(task):
    (task / "skydiscover_evaluator.py").unlink()
    (task / "evaluator").mkdir()
    _, _, forwarded = prepare_run([str(task), "--dry-run"])
    assert forwarded[1] == str(task / "evaluator")


@pytest.mark.parametrize(
    "bad_args",
    [
        ["--num-generations", "0"],
        ["--inner-rounds", "-1"],
        ["--documents-per-query", "21"],
        ["--evoduet.retrieval_policy", "prompt"],
        ["--evoduet.summarize_documents", "true"],
        ["--unknown-option", "x"],
    ],
)
def test_invalid_settings_fail_before_execution(task, bad_args):
    with pytest.raises(SystemExit) as error:
        prepare_run([str(task), *bad_args])
    assert error.value.code == 2


@pytest.mark.parametrize("baseline", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_dispatch_and_exception_restore_selective_scope(task, monkeypatch, baseline, failure):
    original_argv = sys.argv
    original_iteration = DiscoveryController._run_iteration
    original_layer = evoduet.EvoDuet

    def run():
        parsed = framework_cli.parse_args()
        assert parsed._dot_overrides["num_generations"] == "8"
        assert parsed._dot_overrides["evoduet.enabled"] == str(not baseline).lower()
        assert (DiscoveryController._run_iteration is not original_iteration) == (not baseline)
        assert evoduet.EvoDuet is (original_layer if baseline else EvoDuet)
        if failure:
            raise RuntimeError("mock runner failure")
        return 17

    monkeypatch.setattr(framework_cli, "main", run)
    args = [str(task), "--num-generations", "8"] + (["--baseline"] if baseline else [])
    if failure:
        with pytest.raises(RuntimeError, match="mock runner failure"):
            main(args)
    else:
        assert main(args) == 17
    assert sys.argv is original_argv
    assert DiscoveryController._run_iteration is original_iteration
    assert evoduet.EvoDuet is original_layer


def test_config_roundtrip_uses_public_namespace():
    config = Config(evoduet=EvoDuetConfig(enabled=True))
    serialized = config.to_dict()
    assert "world_knowledge" not in serialized
    assert "retrieval_policy" not in serialized["evoduet"]
    assert "retrieval_policy_optimization" not in serialized["evoduet"]
    assert Config.from_dict(serialized).evoduet == config.evoduet
    assert EvoDuet.__module__ == "skydiscover.evoduet.layer"


@pytest.mark.parametrize(
    "script,method,generations",
    [
        ("run_evoduet.sh", "evoduet", 1),
        ("run_evoduet_parallel.sh", "evoduet", 8),
        ("run_baseline.sh", "baseline", 1),
        ("run_baseline_parallel.sh", "baseline", 8),
    ],
)
def test_shell_launchers_accept_arbitrary_working_directory(
    task, tmp_path, script, method, generations
):
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["bash", str(root / "scripts" / script), str(task), "--dry-run"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    preview = json.loads(result.stdout)
    assert preview["method"] == method
    assert preview["num_generations"] == generations
    assert preview["llm"]["max_tokens"] == 32768
    assert preview["checkpoint_interval"] == 1
    assert preview["max_parallel_evaluations"] == (1 if method == "baseline" else generations)
