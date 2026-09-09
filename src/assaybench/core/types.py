"""Shared dataclasses used across the AssayBench sequential-design core.

The design keeps the Model and the Acquisition separate, so an acquisition
can consume a model's belief (greedy, UCB) or ignore it entirely (random, an
LLM agent). These are the values that pass between them:

- `Observation`  — one labelled candidate after a `task.reveal()` call.
- `ModelPrediction` — the internal model's belief about candidates.
- `StepRecord`   — what one inner-loop step produces: the acquired batch,
                   the new observations, the model's prediction (if any),
                   and per-step metrics. Aggregated into a list, this IS
                   the inner-loop history.
- `RunResult`    — what the inner-loop runner returns: history + final
                   metrics + run metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Observation:
    """One labelled candidate that the task has revealed.

    candidate: a candidate identifier -- a gene symbol when the task is
        "which genes do I assay next within one screen", a screen id when
        it is "which screen do I run next". Must be hashable.
    label: the ground-truth label for this candidate. Free-form so a
        task can return whatever its metrics need: a `bool` hit flag,
        a float relevance score, a dict, the full revealed Screen, ...
    metadata: extra info the task wants to attach (e.g. the original
        gene index, an organism tag).
    """

    candidate: Any
    label: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelPrediction:
    """The internal model's belief over a set of (currently unacquired)
    candidates.

    scores: candidate -> belief / score. Higher = more likely to be a hit.
        The keys are whatever `task.candidates()` yields -- gene symbols
        for a within-screen task, screen ids for a screen-selection one.
    uncertainty: optional candidate -> uncertainty (e.g. tree variance).
        Acquisition functions that need uncertainty (UCB, BALD, ...) use
        this; if `None`, those acquisitions fall back to greedy.
    metadata: free-form extra info (e.g. raw ranking the model produced,
        per-source contributions, ...).
    """

    scores: dict[Any, float]
    uncertainty: Optional[dict[Any, float]] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepRecord:
    """One inner-loop step's record."""

    step: int
    acquired_batch: list[Any]
    new_observations: list[Observation]
    model_prediction: Optional[ModelPrediction] = None
    metrics: dict[str, float] = field(default_factory=dict)
    # Optional structured trace for the acquisition (e.g. agent rationale).
    acquisition_trace: dict[str, Any] = field(default_factory=dict)

    @property
    def n_acquired_so_far(self) -> int | None:  # filled in by the runner
        return self.metrics.get("n_acquired")


@dataclass
class RunResult:
    """The full result of one inner-loop run."""

    history: list[StepRecord]
    final_metrics: dict[str, float]
    run_id: str = ""
    task_id: str = ""
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def last_step(self) -> Optional[StepRecord]:
        return self.history[-1] if self.history else None


# ---------------------------------------------------------------------------
# Backwards-compatible alias for the name this type had in the first
# AssayBench release. New code should use `StepRecord`.
# ---------------------------------------------------------------------------

HistoryEntry = StepRecord

__all__ = [
    "Observation",
    "ModelPrediction",
    "StepRecord",
    "RunResult",
    "HistoryEntry",
]
