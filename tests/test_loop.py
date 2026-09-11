"""Tests for ``assaybench.core`` -- the four ABCs and SequentialLoop.

These cover the contracts a third-party implementer relies on, and in
particular the two places where a quiet loop bug would show up as a
plausible-looking benchmark number rather than as an error:

- warm-start observations must reach the model, the acquisition and the
  metrics, or a method given a head start is scored as if it had none;
- an under-supplied batch must never be padded, or a policy that returns
  nothing gets credited with random picks.
"""

from contextlib import contextmanager

import pytest

from assaybench.core import (
    AcquisitionFunction,
    Metric,
    Model,
    ModelPrediction,
    Observation,
    SequentialLoop,
    Task,
)

HITS = {f"G{i}" for i in range(0, 100, 10)}


class ToyTask(Task):
    def __init__(self, n=100, warm=0):
        self.pool = [f"G{i}" for i in range(n)]
        self._warm = [
            Observation(self.pool.pop(0), {"hit": f"G{i}" in HITS})
            for i in range(warm)
        ]

    def candidates(self):
        return list(self.pool)

    def reveal(self, batch):
        out = []
        for b in batch:
            self.pool.remove(b)
            out.append(Observation(b, {"hit": b in HITS}))
        return out

    def ground_truth(self):
        return HITS

    def initial_observations(self):
        return list(self._warm)

    def task_id(self):
        return "toy"


class ConstantModel(Model):
    def predict(self, observations, candidates, task_context=None):
        return ModelPrediction(scores={c: 1.0 for c in candidates})


class HeadAcquisition(AcquisitionFunction):
    """Takes the first ``batch_size`` candidates -- or fewer, on demand."""

    def __init__(self, supply=None):
        self.supply = supply

    def suggest(self, history, candidates, batch_size,
                model_prediction=None, task_context=None):
        n = batch_size if self.supply is None else self.supply
        return candidates[:n]


class HitCount(Metric):
    def score(self, observations, model_prediction, ground_truth,
              candidates_remaining, **kwargs):
        return {"n_hits": float(sum(1 for o in observations if o.label["hit"]))}


def _loop(task, acq=None, **kw):
    return SequentialLoop(
        task, ConstantModel(), acq or HeadAcquisition(), HitCount(),
        batch_size=10, **kw,
    )


def test_runs_requested_number_of_steps():
    result = _loop(ToyTask()).run(n_steps=3, verbose=False)
    assert [s.step for s in result.history] == [1, 2, 3]
    assert result.final_metrics["n_acquired"] == 30


def test_stops_when_pool_is_exhausted():
    result = _loop(ToyTask(n=25)).run(n_steps=99, verbose=False)
    assert result.final_metrics["n_remaining"] == 0
    assert result.final_metrics["n_acquired"] == 25


def test_warm_start_observations_are_visible():
    # Five pre-revealed candidates, of which G0 is a hit. If the loop
    # ignored initial_observations() this would score 0 at step 0 and the
    # warm-started run would be indistinguishable from a cold one.
    result = _loop(ToyTask(warm=5)).run(n_steps=2, verbose=False)
    step0 = result.history[0]
    assert step0.step == 0
    assert step0.metrics["warm_start"] == 1.0
    assert step0.metrics["n_acquired"] == 5
    assert step0.metrics["n_hits"] == 1.0
    assert result.final_metrics["n_acquired"] == 25  # 5 warm + 2 x 10


def test_under_supplied_batch_is_recorded_not_padded():
    result = _loop(ToyTask(), acq=HeadAcquisition(supply=4)).run(
        n_steps=2, verbose=False, )
    assert result.final_metrics["n_acquired"] == 8      # not 20
    assert result.final_metrics["batch_size"] == 4
    assert result.final_metrics["requested_size"] == 10
    assert result.final_metrics["shortfall_frac"] == pytest.approx(0.6)


def test_persistent_under_supply_aborts_the_run():
    # A policy supplying 1 of every 10 slots is a broken run, not a bad
    # score: the guard marks it so an aggregate can exclude it.
    result = _loop(
        ToyTask(), acq=HeadAcquisition(supply=1), max_shortfall_frac=0.5,
    ).run(n_steps=10, verbose=False)
    assert result.final_metrics["aborted"] == 1.0
    assert "shortfall_frac" in result.final_metrics["error"]
    assert len(result.history) < 10


def test_no_trace_scope_by_default():
    # The loop must be usable with no I/O hook whatsoever.
    result = _loop(ToyTask()).run(n_steps=1, verbose=False)
    assert result.history


def test_trace_scope_wraps_the_run():
    seen = {}

    @contextmanager
    def scope(run_id, *, sweep_id=None, task_id=None):
        seen.update(run_id=run_id, sweep_id=sweep_id, task_id=task_id)
        seen["depth"] = seen.get("depth", 0) + 1
        yield
        seen["closed"] = True

    _loop(ToyTask(), run_id="r1", sweep_id="s1", trace_scope=scope).run(
        n_steps=2, verbose=False)
    assert seen == {"run_id": "r1", "sweep_id": "s1", "task_id": "toy",
                    "depth": 1, "closed": True}


def test_metric_failure_is_marked_not_swallowed():
    class Broken(Metric):
        def score(self, *a, **k):
            raise RuntimeError("boom")

    loop = SequentialLoop(ToyTask(), ConstantModel(), HeadAcquisition(),
                          Broken(), batch_size=10)
    result = loop.run(n_steps=1, verbose=False)
    assert result.final_metrics["Broken__error"] == 1.0


def test_acquisition_may_not_invent_candidates():
    class Cheater(AcquisitionFunction):
        def suggest(self, history, candidates, batch_size, **kw):
            return ["NOT_A_GENE"] * batch_size

    result = _loop(ToyTask(), acq=Cheater()).run(n_steps=1, verbose=False)
    assert result.final_metrics["n_acquired"] == 0
    assert result.history[0].acquisition_trace["invalid_count"] == 10
