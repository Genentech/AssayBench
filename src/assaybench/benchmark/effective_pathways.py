"""Effective number of pathways covered by sequential gene acquisitions.

``exp(Shannon entropy)`` over the pathway distribution of a bag of genes is an
effective count: a method concentrated on a few programs scores low, while one
spread across distinct biology scores high.

Each annotated gene is assigned to exactly one of its pathways, drawn
uniformly, and the estimate is averaged over draws. Splitting a gene's weight
over every annotation is useful for visualisation, but not for this metric: it
gives a bag of ``m`` genes a support ceiling of ``sum(k)`` rather than ``m`` and
therefore rewards densely annotated genes. One-pathway-per-gene assignment
keeps the ceiling at ``m``.

Three scopes are calculated from the same pick stream:

==========  ======================================  ==========================
Column      Unit                                    Aggregation
==========  ======================================  ==========================
``EP-B``    one acquisition batch                   mean over batches in a
                                                    screen, then over screens
``EP-S``    all picks in one screen                 mean over screens
``EP-D``    all picks pooled over the test set      none -- one number
==========  ======================================  ==========================

The observed effective count depends on the number of annotated genes. Each
scope is therefore rarefied to a fixed annotated-gene count and averaged over
repeated draws. Units that cannot supply the reference count are dropped and
reported in the diagnostics. If fewer than ``RETENTION`` of a scope's eligible
units survive, that scope is returned as ``None``.

This module deliberately contains no pathway dataset or file discovery.
Callers must supply ``membership``, a mapping from gene symbols to one or more
pathway names. This keeps the metric usable with Reactome or another explicitly
chosen, appropriately licensed pathway vocabulary.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

import numpy as np

# Fixed rarefaction counts, in annotated genes per unit, used for the
# AssayLoop paper. Keeping them here makes reproduced results independent of
# which comparison methods happen to be present in a table.
M_BATCH = 30
M_SCREEN = 200
M_DATASET = 6_000

# Minimum fraction of eligible units that must survive rarefaction.
RETENTION = 0.25

# Subsample draws per unit. These make seed-to-seed variation smaller than the
# one-decimal precision used for reported EP values.
R_BATCH = 400
R_SCREEN = 400
R_DATASET = 300

SEED = 0

_SCOPES = ("batch", "screen", "dataset")


def scope_rng(scope: str, seed: int = SEED) -> np.random.Generator:
    """Return the independent RNG stream used for one EP scope.

    Separate streams ensure that, for example, EP-D does not change when the
    number of batches or the batch-level draw count changes.
    """
    if scope not in _SCOPES:
        choices = ", ".join(_SCOPES)
        raise ValueError(f"Unknown scope {scope!r}; expected one of: {choices}")
    return np.random.default_rng([seed, _SCOPES.index(scope)])


def effective_n(weights: Iterable[float]) -> float:
    """Return ``exp(Shannon entropy)`` for a non-negative weight distribution."""
    w = np.asarray(list(weights), float)
    w = w[w > 0]
    if w.size == 0:
        return float("nan")
    w = w / w.sum()
    return float(math.exp(-(w * np.log(w)).sum()))


def pathway_weights(
    genes: Iterable[str],
    membership: Mapping[str, Sequence[str]],
) -> tuple[dict[str, float], int]:
    """Return fractional pathway weights and the annotated-gene count.

    Each annotated gene contributes total weight one, divided evenly across
    its pathways. Unannotated genes are omitted. These weights are useful for
    pathway visualisations and for the raw diagnostic values returned by
    :func:`effective_pathways`; they are not the rarefied EP statistic.
    """
    by_path: dict[str, float] = defaultdict(float)
    n_ann = 0
    for gene in genes:
        pathways = membership.get(gene.upper())
        if not pathways:
            continue
        n_ann += 1
        weight = 1.0 / len(pathways)
        for pathway in pathways:
            by_path[pathway] += weight
    return dict(by_path), n_ann


class _Unit:
    """A bag of annotated genes in CSR form, cheap to subsample repeatedly."""

    __slots__ = ("ptr", "pid", "pw", "n")

    def __init__(self, genes, membership, pid_of):
        ptr = [0]
        pid: list[int] = []
        pw: list[float] = []
        for gene in genes:
            pathways = membership.get(gene.upper())
            if not pathways:
                continue
            weight = 1.0 / len(pathways)
            for pathway in pathways:
                pid.append(pid_of[pathway])
                pw.append(weight)
            ptr.append(len(pid))
        self.ptr = np.asarray(ptr, dtype=np.int64)
        self.pid = np.asarray(pid, dtype=np.int64)
        self.pw = np.asarray(pw, dtype=np.float64)
        self.n = len(ptr) - 1

    def _entropy_of(self, pos: np.ndarray) -> float:
        pid = self.pid[pos]
        pw = self.pw[pos]
        order = np.argsort(pid, kind="stable")
        pid_s, pw_s = pid[order], pw[order]
        starts = np.concatenate(([0], np.flatnonzero(np.diff(pid_s)) + 1))
        sums = np.add.reduceat(pw_s, starts)
        sums = sums[sums > 0]
        if sums.size == 0:
            return float("nan")
        probabilities = sums / sums.sum()
        return float(math.exp(-(probabilities * np.log(probabilities)).sum()))

    def raw(self) -> float:
        """Return the un-rarefied fractional-weight plug-in diagnostic."""
        if self.n == 0:
            return float("nan")
        return self._entropy_of(np.arange(self.pid.size))

    def _one_pathway_each(self, selected, lengths, rng) -> float:
        """Return EP with one pathway drawn uniformly per selected gene."""
        positions = self.ptr[selected] + (
            rng.random(selected.size) * lengths[selected]
        ).astype(np.int64)
        counts = np.bincount(self.pid[positions])
        probabilities = counts[counts > 0] / selected.size
        return float(math.exp(-(probabilities * np.log(probabilities)).sum()))

    def rarefied(self, m: int | None, r: int, rng) -> float | None:
        """Average EP over ``r`` draws of ``m`` annotated genes.

        ``m=None`` uses every annotated gene and averages only over pathway
        assignment. ``None`` is returned when this unit cannot supply ``m``.
        """
        if m is None:
            m = self.n
        if self.n < m or self.n == 0:
            return None
        lengths = np.diff(self.ptr)
        every = np.arange(self.n) if m == self.n else None
        total = 0.0
        for _ in range(r):
            selected = (
                every
                if every is not None
                else rng.choice(self.n, size=m, replace=False)
            )
            total += self._one_pathway_each(selected, lengths, rng)
        return total / r


def effective_pathways(
    screen_batches,
    *,
    membership: Mapping[str, Sequence[str]],
    rarefy: bool = True,
    seed: int = SEED,
) -> dict:
    """Calculate EP at batch, screen, and dataset scope for one method.

    ``screen_batches`` has shape ``list[screen][batch] -> list[gene]``.
    ``membership`` maps uppercase gene symbols to one or more pathway names.

    The result contains rarefied ``ep_batch``, ``ep_screen``, and ``ep_dataset``
    values; fractional-weight ``*_raw`` diagnostics; annotation coverage; and
    rarefaction retention and drop diagnostics.
    """
    pid_of: dict[str, int] = {}
    for pathways in membership.values():
        for pathway in pathways:
            if pathway not in pid_of:
                pid_of[pathway] = len(pid_of)

    all_picks = [
        len(batch)
        for batches in screen_batches
        for batch in batches
        if batch
    ]
    min_full = 0.9 * float(np.median(all_picks)) if all_picks else 0.0

    rng_b, rng_s, rng_d = (scope_rng(scope, seed) for scope in _SCOPES)
    n_picks = 0
    dropped = 0
    batch_eligible = batch_kept = 0
    screen_eligible = screen_kept = 0

    batch_rarefied: list[float] = []
    batch_raw: list[float] = []
    screen_rarefied: list[float] = []
    screen_raw: list[float] = []
    pooled: list[str] = []
    batch_annotated: list[int] = []
    batch_pick_counts: list[int] = []
    screen_annotated: list[int] = []

    for batches in screen_batches:
        per_screen_rarefied: list[float] = []
        per_screen_raw: list[float] = []
        screen_genes: list[str] = []
        for batch in batches:
            if not batch:
                continue
            n_picks += len(batch)
            screen_genes.extend(batch)
            unit = _Unit(batch, membership, pid_of)
            batch_annotated.append(unit.n)
            batch_pick_counts.append(len(batch))

            # Truncated batches still contribute their genes to EP-S and EP-D,
            # but do not count as batch-scope units. Shortfall scores the run
            # artifact separately.
            if len(batch) < min_full or unit.n < 2:
                continue
            batch_eligible += 1
            per_screen_raw.append(unit.raw())
            if rarefy:
                value = unit.rarefied(M_BATCH, R_BATCH, rng_b)
                if value is None:
                    dropped += 1
                else:
                    batch_kept += 1
                    per_screen_rarefied.append(value)
        if per_screen_raw:
            batch_raw.append(float(np.mean(per_screen_raw)))
        if per_screen_rarefied:
            batch_rarefied.append(float(np.mean(per_screen_rarefied)))
        if screen_genes:
            pooled.extend(screen_genes)
            unit = _Unit(screen_genes, membership, pid_of)
            screen_annotated.append(unit.n)
            if unit.n >= 2:
                screen_eligible += 1
                screen_raw.append(unit.raw())
                if rarefy:
                    value = unit.rarefied(M_SCREEN, R_SCREEN, rng_s)
                    if value is None:
                        dropped += 1
                    else:
                        screen_kept += 1
                        screen_rarefied.append(value)

    ep_dataset_raw = ep_dataset = None
    n_annotated = 0
    if pooled:
        unit = _Unit(pooled, membership, pid_of)
        n_annotated = unit.n
        if unit.n >= 2:
            ep_dataset_raw = unit.raw()
            if rarefy:
                ep_dataset = unit.rarefied(M_DATASET, R_DATASET, rng_d)
                if ep_dataset is None:
                    dropped += 1

    def _mean(values):
        return float(np.mean(values)) if values else None

    batch_ann_array = (
        np.asarray(batch_annotated) if batch_annotated else np.zeros(0)
    )
    batch_pick_array = (
        np.asarray(batch_pick_counts) if batch_pick_counts else np.zeros(0)
    )
    screen_ann_array = (
        np.asarray(screen_annotated) if screen_annotated else np.zeros(0)
    )
    full = (
        batch_ann_array[batch_pick_array >= min_full]
        if batch_pick_array.size
        else batch_ann_array
    )

    ep_batch = _mean(batch_rarefied) if rarefy else _mean(batch_raw)
    ep_screen = _mean(screen_rarefied) if rarefy else _mean(screen_raw)
    batch_retention = (
        batch_kept / batch_eligible if batch_eligible else None
    )
    screen_retention = (
        screen_kept / screen_eligible if screen_eligible else None
    )
    if rarefy and batch_retention is not None and batch_retention < RETENTION:
        ep_batch = None
    if rarefy and screen_retention is not None and screen_retention < RETENTION:
        ep_screen = None

    return {
        "ep_batch_retention": batch_retention,
        "ep_screen_retention": screen_retention,
        "ep_batch_ann_min": (
            int(batch_ann_array.min()) if batch_ann_array.size else None
        ),
        "ep_batch_ann_p5": (
            float(np.percentile(batch_ann_array, 5))
            if batch_ann_array.size
            else None
        ),
        "ep_batch_pick_med": (
            float(np.median(batch_pick_array)) if batch_pick_array.size else None
        ),
        "ep_batch_pick_p5": (
            float(np.percentile(batch_pick_array, 5))
            if batch_pick_array.size
            else None
        ),
        "ep_batch_n_full": int(full.size),
        "ep_batch_full_min": int(full.min()) if full.size else None,
        "ep_batch_full_p5": (
            float(np.percentile(full, 5)) if full.size else None
        ),
        "ep_screen_ann_min": (
            int(screen_ann_array.min()) if screen_ann_array.size else None
        ),
        "ep_n_batches": int(batch_ann_array.size),
        "ep_batch": ep_batch,
        "ep_screen": ep_screen,
        "ep_dataset": ep_dataset if rarefy else ep_dataset_raw,
        "ep_batch_raw": _mean(batch_raw),
        "ep_screen_raw": _mean(screen_raw),
        "ep_dataset_raw": ep_dataset_raw,
        "ep_annotated_frac": n_annotated / n_picks if n_picks else None,
        "ep_n_annotated": n_annotated,
        "ep_n_picks": n_picks,
        "ep_n_dropped": dropped,
    }


__all__ = [
    "M_BATCH",
    "M_DATASET",
    "M_SCREEN",
    "RETENTION",
    "R_BATCH",
    "R_DATASET",
    "R_SCREEN",
    "SEED",
    "effective_n",
    "effective_pathways",
    "pathway_weights",
    "scope_rng",
]
