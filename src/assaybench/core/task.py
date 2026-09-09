"""Task ABC.

A Task owns:
- the candidate pool (`candidates()`),
- a way to reveal labels for a batch of candidates (`reveal()`), and
- ground truth for end-of-run scoring (`ground_truth()`).

The Task is responsible for whatever bookkeeping is needed to keep the
candidate pool consistent (e.g. a gene-batch task tracks which genes have
been revealed so they don't re-appear in `candidates()`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .types import Observation


class Task(ABC):
    """Abstract task interface for the inner loop.

    Subclasses MUST implement:
        - candidates(): the currently unacquired candidate pool
        - reveal(batch): record labels for `batch` and remove them from
          the candidate pool
        - ground_truth(): the full ground-truth structure used by metrics

    Subclasses MAY override:
        - context(): screen / target metadata that LLM models/acquisitions
          want in their prompt. Returns a dict by default.
        - task_id(): unique identifier (used as run_id prefix).
        - reset(): re-randomise initial state (for repeated runs).
    """

    # ---- required ----

    @abstractmethod
    def candidates(self) -> list[Any]:
        """Return the list of currently UNACQUIRED candidates."""

    @abstractmethod
    def reveal(self, batch: list[Any]) -> list[Observation]:
        """Reveal labels for `batch`, mark them acquired, and return
        Observation objects (in input order)."""

    @abstractmethod
    def ground_truth(self) -> Any:
        """Return the full ground truth used by Metric implementations."""

    # ---- optional ----

    def context(self) -> dict[str, Any]:
        """Free-form metadata about the task (used by LLM prompts)."""
        return {}

    def task_id(self) -> str:
        return self.__class__.__name__

    def reset(self) -> None:
        """Re-randomise / clear acquired state. Optional."""
        raise NotImplementedError(
            f"{type(self).__name__}.reset() not implemented"
        )

    def total_positives(self) -> int | None:
        """Total number of positive labels in the ground truth (e.g. total
        hits in a screen). Returned so the AUC metric can normalise.
        Default: None (metric will infer from ground_truth)."""
        return None

    def initial_observations(self) -> list[Observation]:
        """Observations that were revealed by the Task's constructor (e.g.
        from warm-start sampling) and should be visible to the model,
        acquisition, and metrics before step 1.

        Default: empty. Subclasses that pre-reveal data MUST override this
        to return those observations, otherwise warm-start state will be
        invisible to the inner loop.
        """
        return []
