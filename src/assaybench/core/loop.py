"""The sequential-design loop (`SequentialLoop`).

One run = a series of acquisition rounds within a single task instance --
typically one screen, whose genes are revealed a batch at a time.

Per step the loop does:

    1. observations_so_far = flattened from history
    2. candidates = task.candidates()
    3. pred       = model.predict(observations_so_far, candidates, ctx)
    4. batch      = acquisition.suggest(history, candidates,
                       batch_size, model_prediction=pred, task_context=ctx)
    5. new_obs    = task.reveal(batch)
    6. metrics    = {m.name: m.score(...)  for m in metrics}
    7. appends StepRecord(...) to history

To benchmark across many screens, instantiate one task per screen and
aggregate the `RunResult`s.

The loop deliberately does no I/O: it does not write results, and it knows
nothing about LLM call logging. Pass ``trace_scope`` if you want each run
wrapped in a context manager of your own (that is how AssayLoop attaches
its per-run ``llm_calls.jsonl``).
"""

from __future__ import annotations

import logging
import uuid
from contextlib import nullcontext
from typing import Any, Callable, ContextManager

from .acquisition import AcquisitionFunction
from .metric import Metric
from .model import Model
from .task import Task
from .types import Observation, RunResult, StepRecord

log = logging.getLogger("assaybench.core.loop")


class SequentialLoop:
    """The AssayBench sequential-design loop."""

    def __init__(
        self,
        task: Task,
        model: Model,
        acquisition: AcquisitionFunction,
        metrics: Metric | list[Metric],
        batch_size: int = 100,
        run_id: str | None = None,
        *,
        sweep_id: str | None = None,
        max_shortfall_frac: float | None = 0.5,
        max_shortfall_frac_warmup: int = 2,
        on_step: Callable[[dict[str, Any]], None] | None = None,
        trace_scope: Callable[..., ContextManager[Any]] | None = None,
    ):
        """Initialise the loop.

        Args:
            ...standard args...
            trace_scope: Optional factory called as
                ``trace_scope(run_id, sweep_id=..., task_id=...)`` and used
                as a context manager around the whole run. The loop itself
                logs nothing; this is the hook a caller uses to associate
                whatever its model and acquisition do underneath (LLM calls,
                say) with this run. Default: no scope.
            max_shortfall_frac: If, after
                ``max_shortfall_frac_warmup`` steps, the running
                fraction of acquisition *shortfall* (requested slots
                minus actually-acquired slots) exceeds this, the run is
                aborted with a loud error. Set to ``None`` to disable.
                Default ``0.5`` catches agent / LLM runs that under-
                supply or time out on most steps. Note that the inner
                loop NEVER pads under-supplied batches with random
                genes; it simply records a smaller actual batch and
                lets the run continue (or fails it via this guard).
            max_shortfall_frac_warmup: Number of steps the loop
                accumulates before the guard can fire (so a single
                bad first step doesn't immediately kill the run).
        """
        self.task = task
        self.model = model
        self.acquisition = acquisition
        if isinstance(metrics, Metric):
            metrics = [metrics]
        self.metrics: list[Metric] = list(metrics)
        self.batch_size = batch_size
        self.run_id = run_id or f"run-{uuid.uuid4().hex[:10]}"
        self.sweep_id = sweep_id
        self.max_shortfall_frac = max_shortfall_frac
        self.max_shortfall_frac_warmup = max(0, int(max_shortfall_frac_warmup))
        # Optional per-step progress hook, called after each AL step with a
        # compact dict, for callers that want in-loop progress without the
        # noisy ``verbose`` print path.
        self.on_step = on_step
        self.trace_scope = trace_scope

    def run(self, n_steps: int | None = None, *, verbose: bool = True) -> RunResult:
        """Run up to `n_steps` acquisition rounds. If `n_steps is None`, run
        until the candidate pool is exhausted."""
        if self.trace_scope is None:
            scope: ContextManager[Any] = nullcontext()
        else:
            scope = self.trace_scope(
                self.run_id,
                sweep_id=self.sweep_id,
                task_id=self.task.task_id(),
            )
        with scope:
            return self._run_impl(n_steps=n_steps, verbose=verbose)

    def _run_impl(self, n_steps: int | None, *, verbose: bool) -> RunResult:
        history: list[StepRecord] = []
        ctx = self.task.context()

        # Per-loop reset so the model/acquisition can clear per-screen state.
        try:
            self.model.reset()
        except NotImplementedError:
            pass
        try:
            self.acquisition.reset()
        except NotImplementedError:
            pass

        # Warm-start observations (revealed by the task constructor) become
        # the loop's initial history so the model, acquisition, and metrics
        # all see them. Step 0 is reserved for warm-start.
        all_observations: list[Observation] = list(self.task.initial_observations())
        warm_count = len(all_observations)


        # Emit a synthetic step 0 covering warm-start, so the curves and
        # metrics include the warm-start contribution at t=0.
        if warm_count > 0:
            remaining_after = self.task.candidates()
            step0_metrics: dict[str, float] = {
                "step": 0.0,
                "n_acquired": float(warm_count),
                "n_remaining": float(len(remaining_after)),
                "batch_size": float(warm_count),
                "warm_start": 1.0,
            }
            for m in self.metrics:
                try:
                    out = m.score(
                        observations=list(all_observations),
                        model_prediction=None,
                        ground_truth=self.task.ground_truth(),
                        candidates_remaining=list(remaining_after),
                        new_observations=None,
                        acquired_batch=None,
                        n_requested=warm_count,
                    )
                    step0_metrics.update(out or {})
                except Exception:
                    log.exception("Warm-start metric %s failed", m.name())
                    step0_metrics[f"{m.name()}__error"] = 1.0
            warm_record = StepRecord(
                step=0,
                acquired_batch=[o.candidate for o in all_observations],
                new_observations=list(all_observations),
                model_prediction=None,
                metrics=step0_metrics,
                acquisition_trace={"warm_start": True, "warm_start_size": warm_count},
            )
            history.append(warm_record)

        # Running totals for the shortfall guard.
        # ``cum_requested`` = sum of batch_size we ASKED the acquisition
        # for; ``cum_shortfall`` = requested minus the number of valid
        # genes the acquisition actually returned. We never pad with
        # random fills, so ``shortfall_frac`` directly measures how
        # much of the requested budget the policy under-supplied.
        cum_shortfall = 0
        cum_requested = 0
        cum_acq_timeouts = 0
        run_aborted = False
        abort_reason: str | None = None

        step = 0
        max_steps = n_steps if n_steps is not None else 10**9
        while step < max_steps:
            candidates = self.task.candidates()
            if not candidates:
                if verbose:
                    print(f"[{self.run_id}] candidate pool exhausted at step {step}")
                break

            # Model.predict
            pred = self.model.predict(
                observations=list(all_observations),
                candidates=list(candidates),
                task_context=ctx,
            )

            # Acquisition.suggest
            batch = self.acquisition.suggest(
                history=list(history),
                candidates=list(candidates),
                batch_size=self.batch_size,
                model_prediction=pred,
                task_context=ctx,
            )
            # Defensive: ensure all suggested items are in `candidates`
            # and unique, truncate to batch_size. We DO NOT pad
            # under-supplied batches with random/deterministic fills:
            # honest metrics require the curve to reflect ONLY genes
            # the policy actually chose.
            n_proposed = len(batch)
            seen = set()
            cleaned: list = []
            cand_by_eq = {c: c for c in candidates}
            n_invalid = 0
            for b in batch:
                if b in cand_by_eq and b not in seen:
                    seen.add(b)
                    cleaned.append(b)
                elif b not in cand_by_eq:
                    n_invalid += 1
                if len(cleaned) >= self.batch_size:
                    break
            batch = cleaned

            # Per-step shortfall is requested minus actually-acquired.
            # ``requested_this_step`` is capped by the remaining pool so
            # a small pool at the tail of a run isn't reported as a
            # huge shortfall.
            requested_this_step = min(self.batch_size, len(candidates))
            n_shortfall = max(0, requested_this_step - len(batch))

            # Task.reveal
            new_obs = self.task.reveal(batch)
            all_observations.extend(new_obs)

            # Track shortfall/timeout running totals before computing
            # the current step's metrics so ``shortfall_frac`` is
            # consistent with what callers see in ``acquisition_trace``.
            cum_shortfall += n_shortfall
            cum_requested += requested_this_step
            acq_err = (self.acquisition.last_trace() or {}).get("error")
            if isinstance(acq_err, str) and "timeout" in acq_err.lower():
                cum_acq_timeouts += 1
            shortfall_frac_now = (
                (cum_shortfall / cum_requested) if cum_requested else 0.0
            )

            # Metrics
            remaining_after = self.task.candidates()
            step_metrics: dict[str, float] = {
                "step": float(step + 1),
                "n_acquired": float(len(all_observations)),
                "n_remaining": float(len(remaining_after)),
                "batch_size": float(len(batch)),
                "requested_size": float(requested_this_step),
                # ``shortfall_frac`` is the running fraction of slots
                # the policy was asked to fill but didn't. ``0.0`` =
                # policy supplied full batches; ``1.0`` = policy
                # supplied nothing.
                "shortfall_frac": float(shortfall_frac_now),
                "shortfall_count_step": float(n_shortfall),
                "acq_timeouts": float(cum_acq_timeouts),
            }
            for m in self.metrics:
                try:
                    out = m.score(
                        observations=list(all_observations),
                        model_prediction=pred,
                        ground_truth=self.task.ground_truth(),
                        candidates_remaining=list(remaining_after),
                        new_observations=list(new_obs),
                        acquired_batch=list(batch),
                        n_requested=cum_requested,
                    )
                    step_metrics.update(out or {})
                except Exception:
                    log.exception("Metric %s failed at step %d", m.name(), step + 1)
                    # Record an explicit failure marker so persisted results
                    # don't silently look "successful with missing keys".
                    step_metrics[f"{m.name()}__error"] = 1.0

            # Stash shortfall/invalid-candidate diagnostics on the
            # acquisition trace so partial LLM/agent outputs don't
            # disappear silently.
            trace = dict(self.acquisition.last_trace() or {})
            if n_shortfall > 0 or n_invalid > 0 or n_proposed != len(batch):
                trace.setdefault("shortfall_count", n_shortfall)
                trace.setdefault("invalid_count", n_invalid)
                trace.setdefault("proposed_count", n_proposed)
                trace.setdefault("requested_size", requested_this_step)

            record = StepRecord(
                step=step + 1,
                acquired_batch=list(batch),
                new_observations=list(new_obs),
                model_prediction=pred,
                metrics=step_metrics,
                acquisition_trace=trace,
            )
            history.append(record)


            if verbose:
                metric_str = " ".join(
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in record.metrics.items()
                    if k not in ("step",)
                )
                print(f"[{self.run_id}] step {record.step}: {metric_str}")

            if self.on_step is not None:
                try:
                    self.on_step({
                        "run_id": self.run_id,
                        "task_id": self.task.task_id(),
                        "step": record.step,
                        "n_steps": (max_steps if n_steps is not None else None),
                        "n_batch": len(batch),
                        "n_hits_in_batch": int(sum(
                            1 for o in new_obs
                            if isinstance(o.label, dict) and o.label.get("hit")
                        )),
                        "shortfall": int(n_shortfall),
                        "cum_n_hits": record.metrics.get("n_hits"),
                        "n_hits_vs_random": record.metrics.get("n_hits_vs_random"),
                        "frac_hits": record.metrics.get("frac_hits"),
                    })
                except Exception:  # noqa: BLE001 — progress must never break a run
                    log.debug("on_step progress hook failed", exc_info=True)

            step += 1

            # Shortfall-guard: bail loudly if the acquisition is
            # under-supplying most slots. Without this, an LLM/agent
            # run whose calls all time out will silently emit tiny or
            # empty batches and produce a flat curve that looks like
            # the policy simply didn't help. Failing fast lets the
            # sweep aggregate count this as ``n_failed`` instead of a
            # noisy "ok".
            if (
                self.max_shortfall_frac is not None
                and step >= self.max_shortfall_frac_warmup
                and shortfall_frac_now > self.max_shortfall_frac
            ):
                run_aborted = True
                abort_reason = (
                    f"shortfall_frac={shortfall_frac_now:.2f} exceeds "
                    f"max_shortfall_frac={self.max_shortfall_frac:.2f} "
                    f"after {step} step(s); "
                    f"{cum_acq_timeouts} acquisition timeout(s); "
                    f"acquisition under-supplied {cum_shortfall} of "
                    f"{cum_requested} requested slots. Aborting run."
                )
                log.error("[%s] %s", self.run_id, abort_reason)
                break

        # Compose final result
        final_metrics: dict[str, float] = {}
        if history:
            final_metrics = dict(history[-1].metrics)
        if run_aborted:
            # An ``"error"`` key in final_metrics is the signal to whatever
            # aggregates these runs that this one is a failed screen, whose
            # scores must be excluded from any mean rather than averaged in
            # as a genuinely poor result.
            final_metrics["error"] = abort_reason or "aborted"
            final_metrics["aborted"] = 1.0

        result = RunResult(
            history=history,
            final_metrics=final_metrics,
            run_id=self.run_id,
            task_id=self.task.task_id(),
            config={
                "model": self.model.name(),
                "acquisition": self.acquisition.name(),
                "batch_size": self.batch_size,
                "n_steps": n_steps,
                "max_shortfall_frac": self.max_shortfall_frac,
            },
        )
        return result
