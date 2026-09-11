"""Model ABC.

Models map (acquired observations, remaining candidates) -> a
ModelPrediction over the candidates. They are the "internal model" of
the inner loop: the thing being optimised by collecting more data.

Models are decoupled from Acquisitions. An Acquisition may consume the
ModelPrediction (greedy, UCB) or ignore it (LLM agent, random).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .types import ModelPrediction, Observation


class Model(ABC):
    """Abstract Model interface."""

    @abstractmethod
    def predict(
        self,
        observations: list[Observation],
        candidates: list[Any],
        task_context: dict[str, Any] | None = None,
    ) -> ModelPrediction:
        """Return a ModelPrediction over `candidates`.

        observations: all (candidate, label) Observations acquired so far
            (cumulative).
        candidates: the set to score. Implementations should produce a
            score for every candidate in `candidates` (missing keys
            default to 0 in downstream consumers).
        task_context: optional task metadata (screen description, etc.).
        """

    # ---- optional ----

    def reset(self) -> None:
        """Clear any per-screen state (e.g. cached classifier weights).
        Default no-op."""

    def name(self) -> str:
        return self.__class__.__name__

    def coverage(self, candidates: list[Any]) -> float | None:
        """Return embedding coverage fraction for ``candidates``, or ``None``
        if this model has no embedding provider."""
        return None
