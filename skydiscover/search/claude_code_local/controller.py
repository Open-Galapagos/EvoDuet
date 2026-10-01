"""Claude Code (local subscription) baseline controller.

Runs the Claude Code CLI directly on the local machine -- no Docker -- as a
single-agent baseline. Claude iterates on the solution using the evaluator
directly; the framework scores the final result and records intermediate
checkpoints. This mirrors the docker-based ``claude_code`` baseline but invokes
``claude`` locally using the user's subscription login (``~/.claude``) instead
of an ``ANTHROPIC_API_KEY``.

Only python evaluators are supported (directory/Docker evaluators are rejected).
"""

import asyncio
import json
import logging
import multiprocessing as mp
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from skydiscover.evaluation import create_evaluator
from skydiscover.evaluation.evaluation_result import EvaluationResult
from skydiscover.search.base_database import Program
from skydiscover.search.default_discovery_controller import (
    DiscoveryController,
    DiscoveryControllerInput,
)

logger = logging.getLogger(__name__)

_CLAUDE_MODEL_PREFIXES = ("claude-", "sonnet", "opus", "haiku")

# Reasoning effort levels accepted by `claude --effort`.
_EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")

# Optional task-prompt nudge: tell Claude to fall back to web search for
# inspiration when its score stalls. WebSearch/WebFetch are always available to
# the session (they are not in --disallowedTools); this only changes whether the
# prompt *encourages* using them. Off by default; enable with WEB_SEARCH_HINT=1
# so the baseline behaviour is preserved and runs stay A/B-comparable.
_WEB_SEARCH_HINT = (
    "- You have live web access via the `WebSearch` and `WebFetch` tools. If your "
    "`combined_score` stalls -- no meaningful improvement over several turns even "
    "though you keep changing the code -- treat that as a signal to stop making small "
    "local tweaks. Step back and **search the web** for known results, record bounds, "
    "constructions, or numerical techniques relevant to this problem, then use what you "
    "learn to attempt a substantially different approach.\n"
)


# "Scholarly" variant of the web-search hint that forbids "hacking" the answer:
# Claude is REQUIRED to actually read other scientists'/mathematicians'
# publications on *different* problems and bring cross-domain knowledge to bear,
# but must not look up the previously published SOTA / known construction for
# THIS task. Enable with WEB_SEARCH_HINT_MODE=scholarly.
_WEB_SEARCH_HINT_SCHOLARLY = (
    "- You have live web access via the `WebSearch` and `WebFetch` tools, and in this run **actually "
    "using them is a required part of your method, not optional**. Early on -- before you have converged "
    "-- and again whenever your `combined_score` stops improving, you MUST run real `WebSearch`/`WebFetch` "
    "queries to **study how other mathematicians and scientists approached *different* problems**: read "
    "their papers, writeups, and constructions, and pull in cross-domain knowledge (optimization methods, "
    "numerical techniques, constructions, proof ideas, or approaches from unrelated areas), then adapt "
    "those ideas to this problem yourself. Aim for at least a few substantive searches over the run and "
    "explicitly try to transfer in ideas you would not have invented unaided.\n"
    "- Strict no-hacking rule: you must still solve THIS problem yourself. **Do NOT search for, read, or "
    "copy any previously published solution, state-of-the-art result, optimal construction, record bound, "
    "or reported best score for this specific problem** (including AlphaEvolve, arXiv papers, or GitHub "
    "repos that target this exact task). Use the web for transferable ideas from *other* problems and "
    "fields, never to look up the answer to this one.\n"
)


# Optional persistence nudge: keep improving even after beating the benchmark,
# spending the remaining turn budget instead of stopping at combined_score >= 1.
# Enable with KEEP_IMPROVING=1.
_PERSIST_HINT = (
    "- **Do not stop when you beat the benchmark or reach `combined_score >= 1`.** A passing score is "
    "not the goal -- maximizing the score is. As long as you have turns left in your budget, keep trying "
    "distinct, more ambitious approaches to push `combined_score` strictly higher. Only conclude when you "
    "have genuinely exhausted your turn budget, or have tried many materially different strategies that all "
    "fail to improve on your current best.\n"
)


def _env_true(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _web_search_hint_enabled() -> bool:
    """Whether the (SOTA-seeking) stall -> web-search instruction is requested."""
    return _env_true("WEB_SEARCH_HINT")


def _web_tools_disabled() -> bool:
    """Whether to remove WebSearch/WebFetch from the session entirely."""
    return _env_true("CLAUDE_DISABLE_WEB_TOOLS")


def _web_search_mode() -> str:
    """Resolved web-research mode: 'disabled' | 'scholarly' | 'sota' | 'off'."""
    if _web_tools_disabled():
        return "disabled"
    # 'transfer' kept as a legacy alias for the renamed 'scholarly' mode.
    if os.environ.get("WEB_SEARCH_HINT_MODE", "").strip().lower() in ("scholarly", "transfer"):
        return "scholarly"
    if _web_search_hint_enabled():
        return "sota"
    return "off"


def _web_search_hint_text() -> str:
    """The hint paragraph to append to the task prompt for the active mode."""
    mode = _web_search_mode()
    if mode == "scholarly":
        return _WEB_SEARCH_HINT_SCHOLARLY
    if mode == "sota":
        return _WEB_SEARCH_HINT
    return ""


def _keep_improving_enabled() -> bool:
    """Whether to tell the agent to keep improving past the benchmark."""
    return _env_true("KEEP_IMPROVING")


def _persist_hint_text() -> str:
    """The persistence paragraph to append to the task prompt, if enabled."""
    return _PERSIST_HINT if _keep_improving_enabled() else ""


_EMPTY_RESULT = EvaluationResult(metrics={}, artifacts={})

# Full untrimmed event log of the Claude Code session, written next to the
# other run artifacts. Mirrors the science_laboratory trajectory format.
_TRAJECTORY_FILENAME = "ai_scientist_trajectory.json"


def _resolve_claude_bin() -> str:
    """Locate the local ``claude`` CLI binary."""
    claude_bin = os.environ.get("CLAUDE_BIN") or shutil.which("claude")
    if not claude_bin:
        raise ValueError(
            "claude CLI not found. Install Claude Code and ensure `claude` is on "
            "PATH, or set CLAUDE_BIN to its absolute path."
        )
    return claude_bin


def _claude_version(claude_bin: str) -> Optional[str]:
    """Best-effort ``claude --version`` string (e.g. '2.1.133'); None on failure."""
    try:
        result = subprocess.run(
            [claude_bin, "--version"], capture_output=True, text=True, check=False
        )
    except OSError:
        return None
    import re

    match = re.search(r"\d+\.\d+\.\d+", result.stdout or "")
    return match.group(0) if match else None


def _format_elapsed(seconds: float) -> str:
    """Format elapsed seconds as ``Hh Mm Ss``."""
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _save_trajectory(path: Path, metadata: dict, events: list) -> None:
    """Write ``{"metadata": ..., "events": ...}`` to ``path``.

    Written even on timeout/crash so a partial trajectory is recoverable.
    Falls back to ``default=str`` coercion if some embedded value is not
    JSON-serializable.
    """
    payload = {"metadata": metadata, "events": events}
    try:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    except (OSError, TypeError):
        path.write_text(json.dumps(payload, ensure_ascii=False, default=str))


class ClaudeCodeLocalController(DiscoveryController):
    """Discovery controller that delegates iteration to a local Claude Code CLI."""

    def __init__(self, controller_input: DiscoveryControllerInput):
        self.config = controller_input.config
        self.evaluation_file = controller_input.evaluation_file
        self.database = controller_input.database
        self.file_suffix = controller_input.file_suffix
        self.output_dir = controller_input.output_dir

        self.config.evaluator.evaluation_file = self.evaluation_file
        self.config.evaluator.file_suffix = self.file_suffix
        self.config.evaluator.is_image_mode = self.config.language == "image"

        self.evaluator = create_evaluator(self.config.evaluator)
        self._inject_evaluator_context()

        self.monitor_callback = None
        self.feedback_reader = None
        self.early_stopping_triggered = False
        self.shutdown_event = mp.Event()

    # ------------------------------------------------------------------
    # Workspace setup
    # ------------------------------------------------------------------

    def _write_eval_script(self, workspace: Path, timeout: int = 360) -> None:
        """Write run_eval.sh that Claude Code calls to score a candidate."""
        script = (
            "#!/bin/bash\nset -euo pipefail\n"
            f"timeout {timeout} python3 - \"$1\" <<'PYEOF'\n"
            "import sys, json\n"
            f"sys.path.insert(0, {str(workspace)!r})\n"
            "import evaluator\n"
            "result = evaluator.evaluate(sys.argv[1])\n"
            "print(json.dumps(result))\n"
            "PYEOF\n"
        )
        path = workspace / "run_eval.sh"
        path.write_text(script)
        path.chmod(0o755)

    def _write_task_prompt(self, workspace: Path, suffix: str, max_turns: int) -> str:
        """Write TASK.md and return its content for piping to the CLI."""
        system_msg = getattr(self.config.context_builder, "system_message", "") or ""
        eval_timeout = self.config.evaluator.timeout
        solution_path = workspace / f"solution{suffix}"
        eval_script = workspace / "run_eval.sh"
        instructions = (
            "- Run the evaluator once to confirm the baseline score, then start improving.\n"
            "- After each change, evaluate and decide whether to keep or revert.\n"
            f"- Always keep `{solution_path}` set to your best solution.\n"
            "- Aim to try several distinct approaches within your turn budget.\n"
        )
        instructions += _web_search_hint_text()
        instructions += _persist_hint_text()
        content = (
            "# SkyDiscover: Optimization Task\n\n"
            "You are an AI assistant iteratively improving a program to maximize "
            f"its evaluation score. You have **{max_turns} turns** total.\n\n"
            "## Current solution\n\n"
            f"`{solution_path}` -- read it, understand it, modify it freely.\n\n"
            "## How to evaluate\n\n"
            "```bash\n"
            f"bash {eval_script} {solution_path}\n"
            "```\n\n"
            "Output is JSON. The `combined_score` field is what you want to maximize "
            f"(higher is better). The evaluator has a **{eval_timeout}s timeout**.\n\n"
            "## Task description\n\n"
            f"{system_msg}\n\n"
            "## Instructions\n\n"
            f"{instructions}"
        )
        (workspace / "TASK.md").write_text(content)
        return content

    # ------------------------------------------------------------------
    # Main discovery loop
    # ------------------------------------------------------------------

    async def run_discovery(
        self,
        start_iteration: int,
        max_iterations: int,
        checkpoint_callback: Optional[Callable] = None,
        **kwargs,
    ) -> Optional[Program]:
        max_turns = max_iterations

        model = self.config.llm.models[0].name if self.config.llm.models else None
        if model and not any(model.startswith(p) for p in _CLAUDE_MODEL_PREFIXES):
            raise ValueError(
                f"claude_code_local only supports Claude models, got: {model!r}. "
                f"Use a claude-* model name (e.g. claude-opus-4-8)."
            )

        # Reasoning effort for `claude --effort` (e.g. "high"). Read from the
        # shared LLM config, falling back to the first model's setting.
        effort = self.config.llm.reasoning_effort
        if effort is None and self.config.llm.models:
            effort = self.config.llm.models[0].reasoning_effort
        if effort is not None and effort not in _EFFORT_LEVELS:
            raise ValueError(
                f"claude_code_local: invalid reasoning_effort {effort!r}. "
                f"Use one of {', '.join(_EFFORT_LEVELS)}."
            )

        claude_bin = _resolve_claude_bin()

        # Subscription auth: require a local login, and drop ANTHROPIC_API_KEY
        # from the child env so claude uses the subscription credentials.
        creds = Path.home() / ".claude" / ".credentials.json"
        if not creds.exists():
            raise ValueError(
                "Claude Code subscription login not found at "
                "~/.claude/.credentials.json. Run `claude /login` once to "
                "authenticate via your subscription."
            )
        child_env = os.environ.copy()
        child_env.pop("ANTHROPIC_API_KEY", None)

        loop = asyncio.get_running_loop()

        initial = self.database.get_best_program()
        initial_code = initial.solution if initial else ""

        tmp_base = os.path.expanduser("~/.tmp")
        os.makedirs(tmp_base, exist_ok=True)
        workspace = Path(tempfile.mkdtemp(dir=tmp_base))

        # Holder so the polling loop / finally block can reach the live process.
        proc_holder: dict = {}

        def _kill_proc_group(p) -> None:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    p.kill()
                except OSError:
                    pass

        # Full stream-json event log of the Claude Code session. Each entry is
        # {t_offset, parsed, raw_line}; written to ai_scientist_trajectory.json
        # in the finally block (even on crash/timeout).
        trajectory_events: list = []
        run_meta: dict = {
            "return_code": None,
            "timed_out": False,
            "session_id": None,
            "wall_timeout": None,
        }
        start_iso = datetime.now(timezone.utc).isoformat()
        run_start = time.monotonic()

        try:
            suffix = self.file_suffix
            solution_path = workspace / f"solution{suffix}"
            solution_path.write_text(initial_code)

            eval_path = Path(self.evaluation_file)
            if eval_path.is_dir():
                raise ValueError(
                    "claude_code_local supports python evaluators only; got a "
                    f"directory (docker) evaluator: {eval_path}. Use the "
                    "docker-based 'claude_code' search for containerized evaluators."
                )
            eval_timeout = self.config.evaluator.timeout
            shutil.copy(eval_path, workspace / "evaluator.py")
            self._write_eval_script(workspace, timeout=eval_timeout)

            task_content = self._write_task_prompt(workspace, suffix, max_turns)

            # Prompt file -- avoids shell quoting issues with backticks in task.
            prompt_path = workspace / ".prompt.txt"
            prompt_path.write_text(task_content)

            cmd = [
                claude_bin,
                "-p",
                "-",
                "--max-turns",
                str(max_turns),
                "--dangerously-skip-permissions",
                "--disallowedTools",
                ",".join(
                    ["AskUserQuestion", "EnterPlanMode", "ExitPlanMode"]
                    + (["WebSearch", "WebFetch"] if _web_tools_disabled() else [])
                ),
                "--output-format",
                "stream-json",
                "--verbose",
            ]
            if model:
                cmd += ["--model", model]
            if effort:
                cmd += ["--effort", effort]

            # Wall-clock safety net: full eval timeout + 2 min thinking per turn.
            wall_timeout = max(max_turns * (120 + eval_timeout), 600)
            run_meta["wall_timeout"] = wall_timeout

            out = Path(self.output_dir) if self.output_dir else None
            progress_log = (out / "progress.log") if out else None
            if out:
                out.mkdir(parents=True, exist_ok=True)

            log_path = workspace / "claude.log"
            _progress_lock = threading.Lock()

            def _write_progress(line: str) -> None:
                ts = time.strftime("%H:%M:%S")
                entry = f"[{ts}] {line}"
                logger.info(entry)
                if progress_log:
                    with _progress_lock:
                        with open(progress_log, "a") as f:
                            f.write(entry + "\n")

            _write_progress(
                f"Run started -- model={model or 'default'}, "
                f"effort={effort or 'default'}, "
                f"max_turns={max_turns}, wall_timeout={wall_timeout}s"
            )

            # Shared state, modified only from the executor thread.
            cumulative_turns = 0
            total_cost_usd = 0.0
            stream_turns = 0

            def _run_with_turn_limit() -> None:
                nonlocal cumulative_turns, total_cost_usd, stream_turns
                start = time.monotonic()
                hard_stop_at = 0.0

                with open(log_path, "w") as log_file:
                    with open(prompt_path, "rb") as prompt_fh:
                        proc = subprocess.Popen(
                            cmd,
                            stdin=prompt_fh,
                            stdout=subprocess.PIPE,
                            stderr=log_file,
                            cwd=str(workspace),
                            env=child_env,
                            start_new_session=True,
                        )
                    proc_holder["proc"] = proc
                    try:
                        for raw_line in proc.stdout:
                            decoded = raw_line.decode("utf-8", errors="replace")
                            log_file.write(decoded)
                            log_file.flush()
                            t_offset = round(time.monotonic() - start, 3)

                            try:
                                evt = json.loads(raw_line)
                            except (json.JSONDecodeError, ValueError):
                                stripped = decoded.strip()
                                if stripped:
                                    trajectory_events.append(
                                        {"t_offset": t_offset, "parsed": None, "raw_line": stripped}
                                    )
                                continue

                            trajectory_events.append(
                                {"t_offset": t_offset, "parsed": evt, "raw_line": None}
                            )
                            if run_meta["session_id"] is None and isinstance(evt, dict):
                                sid = evt.get("session_id")
                                if sid:
                                    run_meta["session_id"] = sid

                            evt_type = evt.get("type")

                            if evt_type == "assistant":
                                tool_names = [
                                    c.get("name", "")
                                    for c in evt.get("message", {}).get("content", [])
                                    if c.get("type") == "tool_use"
                                ]
                                if tool_names:
                                    stream_turns += 1
                                    elapsed = time.monotonic() - start
                                    _write_progress(
                                        f"Active: {', '.join(tool_names)}"
                                        f" (elapsed {elapsed:.0f}s,"
                                        f" turn {stream_turns}/{max_turns})"
                                    )
                                    if stream_turns > max_turns and not hard_stop_at:
                                        hard_stop_at = time.monotonic()
                                        _write_progress(
                                            f"Hard stop: stream turn {stream_turns}"
                                            f" exceeded {max_turns} -- waiting for result"
                                        )

                            elif evt_type == "result":
                                seg_turns = evt.get("num_turns", 0)
                                cumulative_turns += seg_turns
                                seg_cost = evt.get("total_cost_usd", 0) or 0
                                if seg_cost > total_cost_usd:
                                    total_cost_usd = seg_cost
                                _write_progress(
                                    f"Segment done ({evt.get('subtype', '')}): "
                                    f"+{seg_turns} turns, "
                                    f"{cumulative_turns}/{max_turns} cumulative, "
                                    f"cost=${total_cost_usd:.4f}"
                                )
                                if cumulative_turns >= max_turns or hard_stop_at:
                                    _write_progress("Turn budget reached -- stopping")
                                    _kill_proc_group(proc)
                                    break

                            if hard_stop_at and time.monotonic() - hard_stop_at > 30:
                                _write_progress("Hard stop grace period elapsed -- force killing")
                                _kill_proc_group(proc)
                                break

                            if time.monotonic() - start > wall_timeout:
                                run_meta["timed_out"] = True
                                _write_progress(
                                    f"Wall timeout ({wall_timeout}s) exceeded -- stopping"
                                )
                                _kill_proc_group(proc)
                                break
                    finally:
                        proc.wait()
                        run_meta["return_code"] = proc.returncode
                        # Drain remaining stdout (e.g. result event emitted
                        # just as the hard stop fired).
                        try:
                            for remaining in proc.stdout:
                                decoded = remaining.decode("utf-8", errors="replace")
                                log_file.write(decoded)
                                log_file.flush()
                                t_offset = round(time.monotonic() - start, 3)
                                try:
                                    evt = json.loads(remaining)
                                    trajectory_events.append(
                                        {"t_offset": t_offset, "parsed": evt, "raw_line": None}
                                    )
                                    if evt.get("type") == "result":
                                        cumulative_turns += evt.get("num_turns", 0)
                                        seg_cost = evt.get("total_cost_usd", 0) or 0
                                        if seg_cost > total_cost_usd:
                                            total_cost_usd = seg_cost
                                except (json.JSONDecodeError, ValueError):
                                    stripped = decoded.strip()
                                    if stripped:
                                        trajectory_events.append(
                                            {"t_offset": t_offset, "parsed": None, "raw_line": stripped}
                                        )
                        except OSError:
                            pass
                        _write_progress(
                            f"Process exited (code {proc.returncode}),"
                            f" cumulative turns: {cumulative_turns}"
                        )

            # Run process in a thread; poll solution file for checkpoints.
            run_future = loop.run_in_executor(None, _run_with_turn_limit)
            last_ckpt_content = initial_code
            ckpt_count = 0
            ckpt_interval = self.config.checkpoint_interval

            while not run_future.done():
                if self.shutdown_event.is_set():
                    logger.info("Shutdown requested -- stopping local Claude Code process")
                    p = proc_holder.get("proc")
                    if p:
                        _kill_proc_group(p)
                    break
                await asyncio.sleep(10)
                try:
                    cur = solution_path.read_text()
                except OSError:
                    continue
                if cur == last_ckpt_content or not cur.strip():
                    continue
                last_ckpt_content = cur
                ckpt_count += 1
                iteration = max(cumulative_turns, ckpt_count)
                try:
                    pid = str(uuid.uuid4())
                    er = await self.evaluator.evaluate_program(cur, pid)
                    prog = Program(
                        id=pid,
                        solution=cur,
                        language=self.config.language or "python",
                        metrics=er.metrics,
                        iteration_found=iteration,
                        parent_id=initial.id if initial else None,
                        other_context_ids=[],
                        metadata={"claude_code_checkpoint_turn": cumulative_turns},
                        artifacts=er.artifacts,
                    )
                    self.database.add(prog, iteration=iteration)
                    score = er.metrics.get("combined_score", "?")
                    _write_progress(f"[CHECKPOINT] turn ~{cumulative_turns}, score={score}")
                    if checkpoint_callback and ckpt_count % ckpt_interval == 0:
                        checkpoint_callback(iteration)
                except Exception:
                    logger.debug("Checkpoint eval failed", exc_info=True)

            await run_future

            actual_turns = cumulative_turns if cumulative_turns > 0 else stream_turns

            # Fallback: scan log for result events we might have missed.
            if total_cost_usd == 0.0:
                try:
                    for line in log_path.read_text(errors="replace").splitlines():
                        try:
                            evt = json.loads(line)
                            if evt.get("type") != "result":
                                continue
                            c = evt.get("total_cost_usd", 0) or 0
                            if c > total_cost_usd:
                                total_cost_usd = c
                            if cumulative_turns == 0:
                                actual_turns = max(actual_turns, evt.get("num_turns", 0))
                        except (json.JSONDecodeError, ValueError):
                            continue
                except OSError:
                    pass

            eval_result = await self._final_evaluation(solution_path, initial_code, initial)
            final_iter = max(actual_turns, 1)

            program = Program(
                id=str(uuid.uuid4()),
                solution=eval_result.solution,
                language=self.config.language or "python",
                metrics=eval_result.er.metrics,
                iteration_found=final_iter,
                parent_id=initial.id if initial else None,
                other_context_ids=[],
                metadata={
                    "claude_code_max_turns": max_turns,
                    "actual_turns": actual_turns,
                    "final_score_source": eval_result.source,
                },
                artifacts=eval_result.er.artifacts,
            )
            self.database.add(program, iteration=final_iter)

            if checkpoint_callback:
                checkpoint_callback(final_iter)

            run_elapsed = time.monotonic() - run_start
            if out:
                try:
                    shutil.copy(log_path, out / "claude.log")
                except OSError:
                    pass
                summary = {
                    "model": model,
                    "max_turns": max_turns,
                    "web_search_hint": _web_search_hint_enabled(),
                    "web_search_mode": _web_search_mode(),
                    "web_tools_disabled": _web_tools_disabled(),
                    "keep_improving": _keep_improving_enabled(),
                    "actual_turns": actual_turns,
                    "cost_usd": round(total_cost_usd, 4),
                    "wall_seconds": round(run_elapsed, 1),
                    "baseline_score": (
                        initial.metrics.get("combined_score")
                        if initial and initial.metrics
                        else None
                    ),
                    "final_score": eval_result.er.metrics.get("combined_score"),
                    "final_score_source": eval_result.source,
                }
                (out / "run_summary.json").write_text(
                    json.dumps(summary, indent=2, default=str) + "\n"
                )
                _write_progress(
                    f"Run complete: turns={actual_turns}/{max_turns}, "
                    f"cost=${total_cost_usd:.4f}, "
                    f"time={run_elapsed:.0f}s, "
                    f"score={eval_result.er.metrics.get('combined_score', '?')}"
                    f" (source={eval_result.source})"
                )

        finally:
            p = proc_holder.get("proc")
            if p and p.poll() is None:
                _kill_proc_group(p)

            # Persist the full Claude Code session trajectory (even on crash /
            # timeout). Written to output_dir, so it survives workspace cleanup.
            out_dir = Path(self.output_dir) if self.output_dir else None
            if out_dir and trajectory_events:
                try:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    elapsed_seconds = time.monotonic() - run_start
                    metadata = {
                        "session_id": run_meta["session_id"],
                        "model": model,
                        "effort": effort,
                        "web_search_hint": _web_search_hint_enabled(),
                        "web_search_mode": _web_search_mode(),
                        "web_tools_disabled": _web_tools_disabled(),
                        "keep_improving": _keep_improving_enabled(),
                        "claude_version": _claude_version(claude_bin),
                        "claude_bin": claude_bin,
                        "workspace": str(workspace),
                        "timeout_seconds": run_meta["wall_timeout"],
                        "timed_out": run_meta["timed_out"],
                        "return_code": run_meta["return_code"],
                        "start_utc": start_iso,
                        "end_utc": datetime.now(timezone.utc).isoformat(),
                        "elapsed_seconds": round(elapsed_seconds, 3),
                        "elapsed_human": _format_elapsed(elapsed_seconds),
                        "event_count": len(trajectory_events),
                    }
                    _save_trajectory(
                        out_dir / _TRAJECTORY_FILENAME, metadata, trajectory_events
                    )
                    logger.info(
                        "Saved trajectory (%d events) to %s",
                        len(trajectory_events),
                        out_dir / _TRAJECTORY_FILENAME,
                    )
                except Exception:
                    logger.warning("Failed to write trajectory", exc_info=True)

            shutil.rmtree(workspace, ignore_errors=True)

        return self.database.get_best_program()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _final_evaluation(
        self, solution_path: Path, initial_code: str, initial: Optional[Program]
    ):
        """Evaluate the final solution, falling back to the best checkpoint."""

        class _FinalResult:
            __slots__ = ("solution", "er", "source")

            def __init__(self, solution, er, source):
                self.solution = solution
                self.er = er
                self.source = source

        try:
            final_code = solution_path.read_text()
        except OSError:
            final_code = initial_code
        if not final_code.strip():
            final_code = initial_code

        # Try evaluating the last solution Claude wrote.
        try:
            er = await self.evaluator.evaluate_program(final_code, str(uuid.uuid4()))
            if er.metrics.get("timeout") or er.metrics.get("combined_score") is None:
                raise ValueError("Final eval timed out or returned no score")
            return _FinalResult(final_code, er, "final_eval")
        except Exception as e:
            logger.warning(f"Final eval failed ({e}), re-evaluating best checkpoint code")

        # Fall back to re-evaluating the best checkpoint's code.
        best = self.database.get_best_program()
        if best and best.solution and best.solution.strip():
            try:
                er = await self.evaluator.evaluate_program(best.solution, str(uuid.uuid4()))
                return _FinalResult(best.solution, er, "best_program_reeval")
            except Exception as e2:
                logger.warning(f"Best program re-eval also failed ({e2})")

        return _FinalResult(final_code, _EMPTY_RESULT, "none")
