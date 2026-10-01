"""Base LLM interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union


@dataclass
class LLMResponse:
    """Response from an LLM generation call.

    text: generated text content.
    image_path: path to generated image file, or None for text-only.
    web_search_results: complete Tavily provenance for tools used by this call.
    llm_reasoning: structured request/response records exposed by the provider.
    llm_reasoning_content: flattened provider-visible reasoning text, when available.
    """

    text: str = ""
    image_path: Optional[str] = None
    web_search_results: Dict[str, Any] = field(default_factory=dict)
    llm_reasoning: Dict[str, Any] = field(default_factory=dict)
    llm_reasoning_content: Optional[str] = None

    @property
    def generation_model(self) -> Optional["LLMInterface"]:
        """The exact generating client, kept outside serialized dataclass fields."""
        return getattr(self, "_generation_model", None)

    @generation_model.setter
    def generation_model(self, model: "LLMInterface") -> None:
        self._generation_model = model

    @property
    def generation_context(self) -> Optional[Dict[str, Any]]:
        """Final generation request, including tool rounds, for self-assessment."""
        return getattr(self, "_generation_context", None)

    @generation_context.setter
    def generation_context(self, context: Dict[str, Any]) -> None:
        self._generation_context = context


class LLMInterface(ABC):
    """Abstract base for LLM backends.

    Subclass this and implement generate() to add a new LLM provider.
    """

    async def score_labels(
        self,
        system_message: str,
        messages: List[Dict[str, Any]],
        *,
        timeout: float = 120.0,
        top_logprobs: int = 20,
        openrouter_provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Score True/False at a fixed answer position, if the backend supports it."""
        return {
            "status": "unsupported",
            "confidence": None,
            "logprob_true": None,
            "logprob_false": None,
            "label_probability_mass": None,
            "model": getattr(self, "model", None),
            "api_base": getattr(self, "api_base", None),
            "error": "This LLM backend does not implement True/False logprob scoring.",
        }

    async def score_verbalized(
        self,
        system_message: str,
        messages: List[Dict[str, Any]],
        *,
        timeout: float = 120.0,
        openrouter_provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Ask for a numeric probability, if the backend supports assessment."""
        timestamp = datetime.now(timezone.utc).isoformat()
        return {
            "status": "unsupported",
            "confidence": None,
            "method": "verbalized_probability",
            "raw_text": None,
            "model": getattr(self, "model", None),
            "api_base": getattr(self, "api_base", None),
            "request": {"system_message": system_message, "messages": messages},
            "responses": [],
            "started_at": timestamp,
            "completed_at": timestamp,
            "duration_ms": 0.0,
            "error": "This LLM backend does not implement verbalized probability scoring.",
        }

    @abstractmethod
    async def generate(
        self, system_message: str, messages: List[Dict[str, Any]], **kwargs
    ) -> Union[LLMResponse, List[LLMResponse]]:
        """Generate a response from the LLM.

        Args:
            system_message: system prompt string.
            messages: conversation history as list of {role, content} dicts.
            **kwargs: backend-specific options (e.g. image_output=True for
                image generation, output_dir, program_id, temperature).

        Returns:
            LLMResponse with text and optional image_path, or a list of responses
            when a supporting backend is explicitly called with n > 1.
        """
        pass
