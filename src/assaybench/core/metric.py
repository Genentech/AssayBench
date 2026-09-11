"""Metric ABC.

Metrics score the inner-loop state. The two main kinds:

- "Curve" metrics (e.g. `hits_auc`) consume the cumulative history of
  acquired observations vs. ground truth. They measure the acquisition's
  data-efficiency.
- "Prediction" metrics (e.g. `andcg_at_k`) consume the current
  ModelPrediction vs. ground truth. They measure the internal model's
  quality at a given step.

Both share the same interface so the inner loop can compute them
uniformly per step.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .types import ModelPrediction, Observation


class Metric(ABC):
    """Abstract Metric interface."""

    @abstractmethod
    def score(
        self,
        observations: list[Observation],
        model_prediction: ModelPrediction | None,
        ground_truth: Any,
        candidates_remaining: list[Any],
        *,
        new_observations: list[Observation] | None = None,
        acquired_batch: list[Any] | None = None,
        n_requested: int | None = None,
    ) -> dict[str, float]:
        """Return one or more scalar scores keyed by name.

        observations: all (candidate, label) Observations acquired so far.
        model_prediction: the internal model's current prediction (may be
            None for fused / no-model setups).
        ground_truth: whatever `task.ground_truth()` returned.
        candidates_remaining: the unacquired candidates as of this step.
        new_observations: Observations from the current batch only (None at
            warm-start / step 0).
        acquired_batch: candidates suggested by the acquisition function in
            the current step (None at warm-start / step 0).
        n_requested: cumulative budget the policy was *asked* to acquire so
            far (sum of per-step batch sizes, capped by the pool) — i.e. how
            many genes it *should* have sampled. Curve metrics use this as the
            x-axis so under-supply is penalised rather than rewarded. ``None``
            falls back to the number actually acquired (legacy behaviour).
        """

    def name(self) -> str:
        return self.__class__.__name__
