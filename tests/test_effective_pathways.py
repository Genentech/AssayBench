"""Synthetic tests for Effective Pathways (EP-B / EP-S / EP-D).

All tests supply a synthetic membership explicitly, so they require no pathway
dataset or network access.
"""

from __future__ import annotations

import math

import pytest

from assaybench.benchmark.effective_pathways import (
    M_BATCH,
    RETENTION,
    effective_n,
    effective_pathways,
    pathway_weights,
    scope_rng,
)


def _membership(n_genes: int, k: int, n_paths: int) -> dict[str, tuple[str, ...]]:
    """Return genes with ``k`` pathway memberships cycled over ``n_paths``."""
    return {
        f"G{i}": tuple(f"P{(i * k + j) % n_paths}" for j in range(k))
        for i in range(n_genes)
    }


def _batches(genes, n_screens: int, n_batches: int, size: int):
    """Deal genes into ``list[screen][batch] -> list[gene]``."""
    iterator = iter(genes)
    return [
        [[next(iterator) for _ in range(size)] for _ in range(n_batches)]
        for _ in range(n_screens)
    ]


def test_effective_n_uniform_is_the_category_count():
    assert effective_n([1.0] * 7) == pytest.approx(7.0)


def test_effective_n_point_mass_is_one():
    assert effective_n([3.0]) == pytest.approx(1.0)
    assert effective_n([5.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_effective_n_is_scale_invariant():
    assert effective_n([2.0, 6.0]) == pytest.approx(effective_n([1.0, 3.0]))


def test_effective_n_empty_is_nan():
    assert math.isnan(effective_n([]))
    assert math.isnan(effective_n([0.0, 0.0]))


def test_ep_batch_is_capped_by_the_reference_gene_count():
    """Dense annotation must not buy a batch more than M_BATCH pathways."""
    membership = _membership(n_genes=600, k=8, n_paths=4_000)
    screen_batches = _batches(
        list(membership), n_screens=4, n_batches=3, size=50
    )

    result = effective_pathways(screen_batches, membership=membership)

    assert result["ep_batch"] <= M_BATCH + 1e-9
    assert result["ep_batch_raw"] > M_BATCH


def test_unrarefied_batch_scope_is_the_fractional_weight_plugin():
    membership = _membership(n_genes=600, k=8, n_paths=4_000)
    screen_batches = _batches(
        list(membership), n_screens=4, n_batches=3, size=50
    )

    rarefied = effective_pathways(screen_batches, membership=membership)
    plugin = effective_pathways(
        screen_batches, membership=membership, rarefy=False
    )

    assert plugin["ep_batch"] == pytest.approx(rarefied["ep_batch_raw"])
    assert plugin["ep_batch"] != pytest.approx(rarefied["ep_batch"])


def test_broad_picks_score_above_narrow_picks():
    broad = {f"G{i}": (f"P{i}",) for i in range(600)}
    narrow = {f"G{i}": (f"P{i % 3}",) for i in range(600)}
    screen_batches = _batches(
        [f"G{i}" for i in range(600)], n_screens=4, n_batches=3, size=50
    )

    ep_broad = effective_pathways(
        screen_batches, membership=broad
    )["ep_batch"]
    ep_narrow = effective_pathways(
        screen_batches, membership=narrow
    )["ep_batch"]

    assert ep_broad == pytest.approx(M_BATCH, rel=1e-6)
    assert ep_narrow < 3.01
    assert ep_broad > ep_narrow


def test_units_too_small_to_rarefy_are_dropped_and_counted():
    membership = {f"G{i}": (f"P{i}",) for i in range(400)}
    small = M_BATCH - 5
    screen_batches = _batches(
        list(membership), n_screens=4, n_batches=2, size=small
    )

    result = effective_pathways(screen_batches, membership=membership)

    assert result["ep_n_batches"] == 8
    assert result["ep_n_dropped"] >= 8
    assert result["ep_batch_retention"] == 0.0
    assert result["ep_batch"] is None
    assert result["ep_n_picks"] == 8 * small
    assert result["ep_dataset_raw"] is not None


def test_very_low_retention_suppresses_the_mean():
    membership = {f"G{i}": (f"P{i}",) for i in range(400)}
    unannotated = [f"X{i}" for i in range(1_000)]
    screen_batches = []
    for screen in range(4):
        rich = [f"G{screen * 50 + i}" for i in range(50)]
        poor = []
        for batch in range(4):
            offset = (screen * 4 + batch) * 45
            poor.append(
                [f"G{200 + screen * 20 + batch * 5 + i}" for i in range(5)]
                + unannotated[offset : offset + 45]
            )
        screen_batches.append([rich, *poor])

    result = effective_pathways(screen_batches, membership=membership)

    assert result["ep_batch_retention"] == pytest.approx(0.2)
    assert result["ep_batch_retention"] < RETENTION
    assert result["ep_batch"] is None
    assert result["ep_n_dropped"] == 21
    assert result["ep_batch_raw"] is not None


def test_annotated_fraction_tracks_unannotated_picks():
    membership = {f"G{i}": (f"P{i}",) for i in range(100)}
    screen_batches = [
        [[f"G{i}" for i in range(40)] + [f"X{i}" for i in range(10)]]
    ]

    result = effective_pathways(screen_batches, membership=membership)

    assert result["ep_n_picks"] == 50
    assert result["ep_n_annotated"] == 40
    assert result["ep_annotated_frac"] == pytest.approx(0.8)


def test_same_seed_gives_the_same_numbers():
    membership = _membership(n_genes=600, k=4, n_paths=500)
    screen_batches = _batches(
        list(membership), n_screens=3, n_batches=4, size=50
    )

    first = effective_pathways(screen_batches, membership=membership, seed=0)
    second = effective_pathways(screen_batches, membership=membership, seed=0)

    assert first == second


def test_a_different_seed_moves_the_estimate_only_within_noise():
    membership = _membership(n_genes=600, k=4, n_paths=500)
    screen_batches = _batches(
        list(membership), n_screens=3, n_batches=4, size=50
    )

    first = effective_pathways(
        screen_batches, membership=membership, seed=0
    )["ep_batch"]
    second = effective_pathways(
        screen_batches, membership=membership, seed=17
    )["ep_batch"]

    assert abs(first - second) < 0.05


def test_pathway_weights_split_each_gene_across_its_pathways():
    membership = {"A": ("P1", "P2"), "B": ("P2",)}

    weights, n_annotated = pathway_weights(
        ["a", "B", "MISSING"], membership=membership
    )

    assert n_annotated == 2
    assert weights == pytest.approx({"P1": 0.5, "P2": 1.5})
    assert sum(weights.values()) == pytest.approx(n_annotated)


def test_pathway_weights_ignores_unannotated_genes():
    weights, n_annotated = pathway_weights(
        ["NOPE"], membership={"A": ("P1",)}
    )

    assert weights == {}
    assert n_annotated == 0


def test_no_picks_reports_nothing_rather_than_zero():
    result = effective_pathways([], membership={"A": ("P1",)})

    assert result["ep_n_picks"] == 0
    assert result["ep_batch"] is None
    assert result["ep_screen"] is None
    assert result["ep_dataset"] is None
    assert result["ep_annotated_frac"] is None


def test_membership_is_explicitly_required():
    with pytest.raises(TypeError, match="membership"):
        effective_pathways([])


def test_scope_rng_rejects_unknown_scope():
    with pytest.raises(ValueError, match="Unknown scope"):
        scope_rng("experiment")
