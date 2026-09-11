"""AcquisitionFunction ABC.

An AcquisitionFunction picks the next `batch_size` candidates given the
inner-loop history, the current candidate pool, and (optionally) the
internal Model's current ModelPrediction.

Implementations that consume the model prediction (greedy, UCB) MUST
gracefully degrade to a uniform / random pick when `model_prediction is
None` (so they can be paired with `NullModel` for ablations).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .types import ModelPrediction, StepRecord


class AcquisitionFunction(ABC):
    """Abstract Acquisition interface."""

    @abstractmethod
    def suggest(
        self,
        history: list[StepRecord],
        candidates: list[Any],
        batch_size: int,
        model_prediction: ModelPrediction | None = None,
        task_context: dict[str, Any] | None = None,
    ) -> list[Any]:
        """Return the next batch of candidates.

        history: inner-loop step records so far.
        candidates: candidate pool (unacquired). The acquisition MUST pick
            elements ONLY from this list. Returned list length == batch_size
            (or fewer if `len(candidates) < batch_size`).
        batch_size: target number of candidates to pick.
        model_prediction: current ModelPrediction (may be None).
        task_context: optional task metadata.
        """

    # ---- optional ----

    def reset(self) -> None:
        """Clear any per-screen state. Default no-op."""

    def name(self) -> str:
        return self.__class__.__name__

    def last_trace(self) -> dict[str, Any]:
        """Optional structured trace from the last suggest() call (agent
        rationale, top-K considered, ...). The inner loop attaches this
        to the StepRecord for the dashboard.

        Default: empty dict.
        """
        return {}
