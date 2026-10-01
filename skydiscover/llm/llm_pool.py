"""LLM pool -- weighted sampling over one or more LLM backends."""

import asyncio
import logging
import random
from contextvars import ContextVar
from typing import Any, Dict, List, Union

from skydiscover.config import LLMModelConfig
from skydiscover.llm.base import LLMResponse
from skydiscover.llm.openai import OpenAILLM, _validate_sample_count

logger = logging.getLogger("skydiscover.llm")


class LLMPool:
    """Weighted pool of LLM backends. Samples one per generate() call."""

    def __init__(self, models_cfg: List[LLMModelConfig]):
        if not models_cfg:
            raise ValueError("LLMPool requires at least one model config")

        self.models_cfg = models_cfg

        # Validate weights before creating clients to fail fast on bad config.
        self.weights = [m.weight for m in models_cfg]
        if any(w < 0 for w in self.weights):
            raise ValueError("LLMPool model weights must be non-negative")
        total = sum(self.weights)
        if total <= 0:
            raise ValueError("LLMPool model weights must sum to a positive value")
        self.weights = [w / total for w in self.weights]

        self.models = [
            model_cfg.init_client(model_cfg) if model_cfg.init_client else OpenAILLM(model_cfg)
            for model_cfg in models_cfg
        ]
        self.random_state = random.Random()
        # One choice per asynchronous iteration context: retries share it, while
        # concurrent iterations cannot overwrite each other's model selection.
        self._iteration_model: ContextVar[tuple[int, Any, LLMModelConfig] | None] = ContextVar(
            "llm_pool_iteration_model", default=None
        )

        # Logging
        if len(models_cfg) > 1:
            pool_key = tuple((c.name, w) for c, w in zip(models_cfg, self.weights))
            if not hasattr(logger, "_logged_pools"):
                logger._logged_pools = set()
            if pool_key not in logger._logged_pools:
                parts = ", ".join(f"{c.name}={w:.2f}" for c, w in zip(models_cfg, self.weights))
                logger.debug(f"Pool weights: {parts}")
                logger._logged_pools.add(pool_key)

    def _sample_model(self):
        """
        Simple weighted sampling mechanism. Override this to implement a more complex sampling mechanism.
        """
        idx = self.random_state.choices(range(len(self.models)), weights=self.weights, k=1)[0]
        return self.models[idx]

    def reserve_model(self, iteration: int) -> LLMModelConfig:
        """Select an iteration's solution model before its EvoDuet calls.

        Keep a single binding in the caller's context rather than a growing map.
        A later iteration replaces it; child tasks inherit the choice, and
        concurrent iteration tasks maintain independent bindings.
        """
        current = self._iteration_model.get()
        if current is not None and current[0] == iteration:
            return current[2]
        model = self._sample_model()
        config = next(
            (cfg for backend, cfg in zip(self.models, self.models_cfg) if backend is model),
            None,
        )
        if config is None:
            raise ValueError("Cannot reserve a model outside this LLM pool")
        self._iteration_model.set((iteration, model, config))
        return config

    def get_model_for_context(self, llm_context=None):
        """Use a reserved iteration choice, or preserve normal per-call sampling."""
        current = self._iteration_model.get()
        iteration = (llm_context or {}).get("iteration")
        if current is not None and current[0] == iteration:
            return current[1]
        return self._sample_model()

    def release_model(self, iteration: int) -> None:
        """Forget a completed iteration's choice in the caller's context."""
        current = self._iteration_model.get()
        if current is not None and current[0] == iteration:
            self._iteration_model.set(None)

    async def generate(
        self, system_message: str, messages: List[Dict[str, Any]], **kwargs
    ) -> Union[LLMResponse, List[LLMResponse]]:
        """Sample one model; n > 1 returns that model's native response samples."""
        n = _validate_sample_count(kwargs.get("n", 1))
        model = self.get_model_for_context(kwargs.get("llm_context"))
        response = await model.generate(system_message, messages, **kwargs)
        return self._associate_samples(response, model, n)

    @staticmethod
    def _associate_samples(response, model, n):
        if n > 1:
            if (
                not isinstance(response, list)
                or len(response) != n
                or any(not isinstance(sample, LLMResponse) for sample in response)
            ):
                raise ValueError(
                    f"LLM backend must return a list of exactly {n} LLMResponse samples for n={n}; "
                    "use a backend supporting native multi-sampling or set n=1."
                )
            for sample in response:
                sample.generation_model = model
        else:
            if isinstance(response, list):
                raise ValueError("LLM backend must return one LLMResponse for n=1, not a list")
            response.generation_model = model
        return response

    async def check_availability(self, timeout: float = 15.0) -> bool:
        """Probe the first model with a minimal request to check connectivity.

        This is a representative check only: it tests the first configured
        backend, not the entire pool.  A healthy first backend does not
        guarantee all backends are reachable (and vice-versa).

        The max_tokens=1 probe verifies endpoint connectivity and auth, not
        generation correctness.  Reasoning models may return empty content on
        such minimal requests without raising an error.
        """
        try:
            model = self.models[0]
            await asyncio.wait_for(
                model.generate("", [{"role": "user", "content": "ping"}], max_tokens=1),
                timeout=timeout,
            )
            return True
        except Exception:
            return False

    async def generate_all(
        self, system_message: str, messages: List[Dict[str, Any]], **kwargs
    ) -> List[Union[LLMResponse, List[LLMResponse]]]:
        """Generate concurrently; n > 1 keeps each model's sample list together."""
        n = _validate_sample_count(kwargs.get("n", 1))
        responses = await asyncio.gather(
            *(model.generate(system_message, messages, **kwargs) for model in self.models)
        )
        for model, response in zip(self.models, responses):
            self._associate_samples(response, model, n)
        return responses
