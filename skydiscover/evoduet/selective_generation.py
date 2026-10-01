"""Iteration-local generation counts for the opt-in EvoDuet selective mode.

The original config retains N for validation, checkpoints and concurrency limits.
Only the controller's view changes: an iteration starts with one generation and
uses N after its own EvoDuet gate chooses retrieve or look-up. The standard controller
and candidate runner therefore keep their existing parsing/evaluation behavior.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps

from skydiscover.evoduet.retrieval_gating import GateDecision


@dataclass
class _IterationState:
    controller: object
    iteration: int
    configured_num_generations: int
    num_generations: int = 1
    gate_decision: str | None = None


_iteration_state: ContextVar[_IterationState | None] = ContextVar(
    "evoduet_selective_generation", default=None
)


class _ConfigView:
    """Forward config access without changing shared state across async iterations."""

    def __init__(self, config, controller):
        object.__setattr__(self, "_config", config)
        object.__setattr__(self, "_controller", controller)

    @property
    def num_generations(self):
        state = _iteration_state.get()
        if state is not None and state.controller is self._controller:
            return state.num_generations
        return self._config.num_generations

    def __getattr__(self, name):
        return getattr(self._config, name)

    def __setattr__(self, name, value):
        setattr(self._config, name, value)


def _configure_controller(controller):
    if not isinstance(controller.config, _ConfigView):
        controller.config = _ConfigView(controller.config, controller)
    return controller.config._config


def record_gate_action(layer, decision, iteration):
    """Publish a parsed action to its owning iteration, even if evidence is empty.

    In ordinary EvoDuet runs there is no active state, so this is a no-op. Binding both
    the layer and iteration prevents another controller's gate or a preflight
    search from changing the solution generation budget.
    """
    state = _iteration_state.get()
    if (
        state is None
        or state.iteration != iteration
        or getattr(state.controller, "evoduet", None) is not layer
    ):
        return
    action = GateDecision(decision)
    state.gate_decision = action.value
    state.num_generations = (
        state.configured_num_generations
        if action in (GateDecision.RETRIEVE, GateDecision.LOOK_UP)
        else 1
    )


def _record_result(result, state):
    child = getattr(result, "child_program_dict", None)
    if child is not None:
        child.setdefault("metadata", {})["selective_generation"] = {
            "gate_decision": state.gate_decision,
            "configured_num_generations": state.configured_num_generations,
            "num_generations": state.num_generations,
        }
    return result


def _iteration_wrapper(method):
    @wraps(method)
    async def run(controller, iteration, *args, **kwargs):
        config = _configure_controller(controller)
        active = _iteration_state.get()
        # _run_iteration delegates to _run_from_scratch_iteration when there is
        # no parent. Reuse that iteration's scope; it has no gate and stays at 1.
        if active is not None and active.controller is controller and active.iteration == iteration:
            return await method(controller, iteration, *args, **kwargs)
        state = _IterationState(controller, iteration, config.num_generations)
        token = _iteration_state.set(state)
        try:
            result = await method(controller, iteration, *args, **kwargs)
            return _record_result(result, state)
        finally:
            _iteration_state.reset(token)

    return run


@contextmanager
def selective_generation_scope():
    """Install routing for this launcher invocation, restoring it on exit.

    Keep the exact controller class: its standard constructor validates support
    for N-generation mode using class identity.
    """
    from skydiscover.search.default_discovery_controller import DiscoveryController

    original_init = DiscoveryController.__init__
    original_iteration = DiscoveryController._run_iteration
    original_scratch = DiscoveryController._run_from_scratch_iteration

    @wraps(original_init)
    def initialize(controller, *args, **kwargs):
        original_init(controller, *args, **kwargs)
        _configure_controller(controller)

    DiscoveryController.__init__ = initialize
    DiscoveryController._run_iteration = _iteration_wrapper(original_iteration)
    DiscoveryController._run_from_scratch_iteration = _iteration_wrapper(original_scratch)
    try:
        yield
    finally:
        DiscoveryController.__init__ = original_init
        DiscoveryController._run_iteration = original_iteration
        DiscoveryController._run_from_scratch_iteration = original_scratch
