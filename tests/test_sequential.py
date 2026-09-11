"""Tests for the ASSAYBENCH-LOOP sequential metrics.

The point of these is that extracting EF, shortfall and %essential out of the
AssayLoop research code changed nothing. Three levels of check:

1. The worked examples printed in AssayLoop Fig. 3C.
2. The behaviours the definitions are *for* -- forgiving out-of-library picks,
   charging hallucinations and unspent budget, hit-rate invariance.
3. A differential test against verbatim transcriptions of the research code's
   own arithmetic, over randomised trajectories.
"""

from __future__ import annotations

import random

import pytest

import assaybench.benchmark.sequential as sequential
from assaybench.benchmark.sequential import (
    adjusted_nauc,
    classify_acquisitions,
    enrichment_factor,
    enrichment_factor_from_value,
    fraction_of_hits,
    load_common_essentials,
    percent_essential,
    shortfall,
)


# ---------------------------------------------------------------------------
# Reference implementations, transcribed verbatim from the AssayLoop research
# code. These are the numbers the paper reports; the extracted functions must
# agree with them exactly. Do not tidy these up -- their value is being
# unedited copies.
# ---------------------------------------------------------------------------


def _reference_adjusted_nvr(
    n_hits, n_in_lib, n_out_lib_in_universe, domain_size, total_hits, budget
):
    """bridgeloop.metrics.hits_auc.adjusted_nvr_value"""
    n3 = budget - n_in_lib - n_out_lib_in_universe
    eff = n_in_lib + n3
    rand_exp = eff * total_hits / domain_size if domain_size > 0 else 0.0
    return float(n_hits / rand_exp) if rand_exp > 0 else 0.0


def _reference_adj_nauc(rounds, screen_genes, universe_set, total_hits, domain_size,
                        hit_set):
    """bridgeloop.scripts.full_genome_table._adj_nauc_from_run.

    The research function reads a persisted ``result.json``; the loop body is
    transcribed unedited, with ``rd["steps"]`` replaced by ``rounds`` and the
    recorded per-gene hit flag ``h`` replaced by the same lookup the runner
    used to record it.
    """
    import numpy as np

    if total_hits <= 0 or domain_size <= 0:
        return 0.0
    eff = 0      # charged picks so far (n1 + out-of-universe)
    cum = 0      # in-library hits so far
    n2 = 0       # forgiven picks
    xs = [0.0]
    ys = [0.0]
    for batch in rounds:
        for g in batch:
            h = g in hit_set
            if g in screen_genes:
                eff += 1
                if h:
                    cum += 1
            elif g in universe_set:
                n2 += 1                       # forgiven: no budget, no hit
            else:
                eff += 1                      # out-of-universe: charged, no hit
        xs.append(eff / domain_size)
        ys.append(cum / total_hits)
    auc = float(np.trapezoid(ys, xs))
    ys_best = [min(x * domain_size / total_hits, 1.0) for x in xs]
    best = float(np.trapezoid(ys_best, xs))
    return float(auc / best) if best > 0 else 0.0


def _reference_oob_frac(acquired, library):
    """bridgeloop.scripts.full_genome_table._oob_frac_from_runs, single run."""
    total_oob, total_picks = 0, 0
    total_picks += len(acquired)
    total_oob += sum(1 for g in acquired if g not in library)
    return total_oob / total_picks if total_picks > 0 else 0.0


def _reference_pct_essential(acquired, library, hit_set, essentials):
    """bridgeloop.scripts.full_genome_table._noness_nvr_from_run, pct_ess term."""
    hits_noness = 0
    for g in acquired:
        if g in essentials:
            continue
        if g in library and g in hit_set:
            hits_noness += 1
    hits_ess = 0
    for g in acquired:
        if g in essentials and g in library and g in hit_set:
            hits_ess += 1
    total_found = hits_noness + hits_ess
    return hits_ess / total_found if total_found > 0 else 0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def screen():
    """A 100-gene library with 10 hits, in a 200-gene universe."""
    library = [f"LIB{i:03d}" for i in range(100)]
    hits = library[:10]
    universe = set(library) | {f"EXT{i:03d}" for i in range(100)}
    return library, hits, universe


# ---------------------------------------------------------------------------
# The paper's worked example (Fig. 3C)
# ---------------------------------------------------------------------------


def test_figure_3c_hit_rich_screen():
    """Screen A: h_rand = 10%, 18 hits in a 100-gene batch -> EF 1.8."""
    assert enrichment_factor_from_value(
        n_hits=18, effective_budget=100, library_size=1000, total_hits=100
    ) == pytest.approx(1.8)


def test_figure_3c_hit_sparse_screen():
    """Screen B: h_rand = 1%, 4 hits in a 100-gene batch -> EF 4.0."""
    assert enrichment_factor_from_value(
        n_hits=4, effective_budget=100, library_size=1000, total_hits=10
    ) == pytest.approx(4.0)


def test_figure_3c_more_raw_hits_can_mean_less_enrichment():
    """The panel's actual claim: 18 hits beats 4 hits on count, loses on EF."""
    hit_rich = enrichment_factor_from_value(18, 100, 1000, 100)
    hit_sparse = enrichment_factor_from_value(4, 100, 1000, 10)
    assert 18 > 4
    assert hit_rich < hit_sparse


# ---------------------------------------------------------------------------
# Enrichment factor
# ---------------------------------------------------------------------------


def test_random_policy_scores_one(screen):
    """Acquiring exactly the library hit rate is EF 1.0 by construction."""
    library, hits, universe = screen
    # 10 picks, of which 1 is a hit, in a 10%-hit-rate screen
    acquired = library[9:10] + library[50:59]
    ef = enrichment_factor(acquired, library, hits, universe, budget=10)
    assert ef == pytest.approx(1.0)


def test_perfect_policy_scores_inverse_hit_rate(screen):
    """All 10 picks are hits in a 10%-hit-rate screen -> EF 10."""
    library, hits, universe = screen
    ef = enrichment_factor(hits, library, hits, universe, budget=10)
    assert ef == pytest.approx(10.0)


def test_out_of_library_real_genes_are_forgiven(screen):
    """A real gene this screen did not measure costs nothing.

    Five hits plus five unmeasured-but-real genes scores the same as five hits
    out of five picks: the forgiven picks leave the effective budget.
    """
    library, hits, universe = screen
    forgiven = enrichment_factor(
        hits[:5] + ["EXT001", "EXT002", "EXT003", "EXT004", "EXT005"],
        library, hits, universe, budget=10,
    )
    clean = enrichment_factor(hits[:5], library, hits, universe, budget=5)
    assert forgiven == pytest.approx(clean)
    assert forgiven == pytest.approx(10.0)


def test_hallucinated_genes_are_charged(screen):
    """A symbol outside the universe consumes budget and cannot be a hit."""
    library, hits, universe = screen
    ef = enrichment_factor(
        hits[:5] + ["NOTAGENE1", "NOTAGENE2", "NOTAGENE3", "NOTAGENE4", "NOTAGENE5"],
        library, hits, universe, budget=10,
    )
    # 5 hits over an effective budget of 10 -> 5 / (10 * 0.1)
    assert ef == pytest.approx(5.0)


def test_unspent_budget_is_charged(screen):
    """Naming three sure things and skipping the rest does not inflate EF."""
    library, hits, universe = screen
    ef = enrichment_factor(hits[:3], library, hits, universe, budget=100)
    assert ef == pytest.approx(3 / (100 * 0.1))

    unbudgeted = enrichment_factor(hits[:3], library, hits, universe)
    assert unbudgeted == pytest.approx(10.0)
    assert ef < unbudgeted


def test_no_universe_forgives_everything(screen):
    """Without a universe there is nothing to call a hallucination."""
    library, hits, _ = screen
    ef = enrichment_factor(
        hits[:5] + ["NOTAGENE1", "NOTAGENE2", "NOTAGENE3", "NOTAGENE4", "NOTAGENE5"],
        library, hits, universe=None,
    )
    assert ef == pytest.approx(10.0)


def test_hits_outside_the_library_are_ignored(screen):
    """A gene cannot be a hit in a screen that never measured it."""
    library, hits, universe = screen
    inflated = list(hits) + ["EXT001", "EXT002"]
    assert enrichment_factor(hits, library, inflated, universe, budget=10) == (
        pytest.approx(enrichment_factor(hits, library, hits, universe, budget=10))
    )


@pytest.mark.parametrize(
    "library_size,total_hits,acquired",
    [(0, 10, 5), (100, 0, 5), (100, 10, 0)],
)
def test_degenerate_inputs_score_zero(library_size, total_hits, acquired):
    assert enrichment_factor_from_value(3, acquired, library_size, total_hits) == 0.0


def test_empty_trajectory_scores_zero(screen):
    library, hits, universe = screen
    assert enrichment_factor([], library, hits, universe) == 0.0


# ---------------------------------------------------------------------------
# Acquisition bookkeeping
# ---------------------------------------------------------------------------


def test_counts_partition_the_trajectory(screen):
    library, hits, universe = screen
    acquired = hits[:4] + library[50:53] + ["EXT001", "EXT002"] + ["NOTAGENE"]
    counts = classify_acquisitions(acquired, library, hits, universe, budget=20)

    assert counts.n_in_library == 7
    assert counts.n_out_library_in_universe == 2
    assert counts.n_out_of_universe == 1
    assert counts.n_hits == 4
    assert counts.n_acquired == len(acquired)
    assert counts.n_unfilled == 20 - len(acquired)
    # n_eff = budget - forgiven
    assert counts.effective_budget == 20 - 2


def test_effective_budget_matches_the_research_code_expression(screen):
    """``N_L + N_Ū`` and ``budget - forgiven`` are the same quantity."""
    library, hits, universe = screen
    acquired = hits[:3] + ["EXT001"] + ["NOTAGENE"] + library[60:64]
    for budget in (5, 9, 40):
        counts = classify_acquisitions(acquired, library, hits, universe, budget)
        research_code = (
            counts.n_in_library
            + budget
            - counts.n_in_library
            - counts.n_out_library_in_universe
        )
        assert counts.effective_budget == research_code


# ---------------------------------------------------------------------------
# Adjusted nAUC
# ---------------------------------------------------------------------------


def test_nauc_of_the_oracle_is_one(screen):
    """Taking every hit first, one per round, is the curve nAUC divides by."""
    library, hits, universe = screen
    rounds = [[gene] for gene in hits]
    assert adjusted_nauc(rounds, library, hits, universe) == pytest.approx(1.0)


def test_nauc_rewards_finding_hits_early(screen):
    """Same picks, same endpoint, different order -> different nAUC."""
    library, hits, universe = screen
    misses = library[50:70]
    early = [hits[:5], misses[:10], hits[5:], misses[10:]]
    late = [misses[:10], hits[:5], misses[10:], hits[5:]]

    assert sorted(g for r in early for g in r) == sorted(g for r in late for g in r)
    assert adjusted_nauc(early, library, hits, universe) > adjusted_nauc(
        late, library, hits, universe
    )


def test_nauc_is_invariant_to_order_within_a_round(screen):
    """Policies propose a batch, not a ranking, so intra-batch order is noise."""
    library, hits, universe = screen
    forward = [hits[:5] + library[50:55], hits[5:] + library[55:60]]
    shuffled = [library[50:55] + hits[:5], library[55:60] + hits[5:]]
    assert adjusted_nauc(forward, library, hits, universe) == pytest.approx(
        adjusted_nauc(shuffled, library, hits, universe)
    )


def test_nauc_forgives_out_of_library_real_genes(screen):
    """A forgiven pick advances neither axis, so the curve is unchanged."""
    library, hits, universe = screen
    clean = [hits[:5], hits[5:]]
    padded = [hits[:5] + ["EXT001", "EXT002"], hits[5:] + ["EXT003"]]
    assert adjusted_nauc(padded, library, hits, universe) == pytest.approx(
        adjusted_nauc(clean, library, hits, universe)
    )


def test_nauc_charges_hallucinations(screen):
    """A hallucinated symbol advances x without advancing y."""
    library, hits, universe = screen
    clean = [hits[:5], hits[5:]]
    padded = [hits[:5] + ["NOTAGENE1", "NOTAGENE2"], hits[5:] + ["NOTAGENE3"]]
    assert adjusted_nauc(padded, library, hits, universe) < adjusted_nauc(
        clean, library, hits, universe
    )


def test_nauc_does_not_charge_unspent_budget(screen):
    """The curve ends at the last gene acquired; EF carries the under-supply.

    A policy asked for 100 genes that names three sure things and stops has a
    perfect *ordering* over what it committed to, so nAUC is 1.0 -- there is no
    budget argument here to say otherwise. EF, on the same trajectory, is
    heavily penalised. The two columns are meant to separate those effects.
    """
    library, hits, universe = screen
    assert adjusted_nauc([[h] for h in hits[:3]], library, hits, universe) == (
        pytest.approx(1.0)
    )
    assert enrichment_factor(
        hits[:3], library, hits, universe, budget=100
    ) == pytest.approx(0.3)


def test_nauc_of_a_wholly_forgiven_trajectory_is_zero(screen):
    """No charged pick, no curve, nothing to score."""
    library, hits, universe = screen
    assert adjusted_nauc([["EXT001", "EXT002"]], library, hits, universe) == 0.0


@pytest.mark.parametrize("rounds", [[], [[]]])
def test_nauc_of_an_empty_trajectory_is_zero(screen, rounds):
    library, hits, universe = screen
    assert adjusted_nauc(rounds, library, hits, universe) == 0.0


def test_nauc_of_a_screen_with_no_hits_is_zero(screen):
    library, _, universe = screen
    assert adjusted_nauc([library[:10]], library, [], universe) == 0.0


# ---------------------------------------------------------------------------
# Fraction of hits
# ---------------------------------------------------------------------------


def test_fraction_of_hits_is_recovery_of_the_hit_set(screen):
    library, hits, _ = screen
    assert fraction_of_hits(hits[:4], library, hits) == pytest.approx(0.4)
    assert fraction_of_hits(hits, library, hits) == 1.0


def test_fraction_of_hits_ignores_unscoreable_picks(screen):
    library, hits, _ = screen
    padded = hits[:4] + ["EXT001", "NOTAGENE", *library[50:60]]
    assert fraction_of_hits(padded, library, hits) == pytest.approx(0.4)


def test_fraction_of_hits_with_no_hits_in_the_screen(screen):
    library, _, _ = screen
    assert fraction_of_hits(library[:10], library, []) == 0.0


def test_fraction_of_hits_is_not_hit_rate_corrected(screen):
    """Its whole difference from EF: a hit-rich screen hands out a high FH."""
    library, _, _ = screen
    hit_rich = library[:50]  # 50% hit rate
    assert fraction_of_hits(library[:25], library, hit_rich) == pytest.approx(0.5)
    assert enrichment_factor(
        library[:25], library, hit_rich, budget=25
    ) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Shortfall
# ---------------------------------------------------------------------------


def test_shortfall_counts_every_unscoreable_pick(screen):
    library, hits, _ = screen
    acquired = library[:6] + ["EXT001", "EXT002", "NOTAGENE1", "NOTAGENE2"]
    assert shortfall(acquired, library) == pytest.approx(0.4)


def test_shortfall_is_zero_for_a_fully_in_library_trajectory(screen):
    library, hits, _ = screen
    assert shortfall(library[:20], library) == 0.0


def test_shortfall_is_one_when_nothing_lands(screen):
    library, _, _ = screen
    assert shortfall(["EXT001", "NOTAGENE"], library) == 1.0


def test_shortfall_of_empty_trajectory_is_zero(screen):
    library, _, _ = screen
    assert shortfall([], library) == 0.0


def test_shortfall_does_not_charge_unspent_budget(screen):
    """Unlike EF, shortfall is a fraction of picks made, not of budget."""
    library, _, _ = screen
    assert shortfall(library[:2] + ["EXT001", "EXT002"], library) == 0.5


# ---------------------------------------------------------------------------
# Percentage essential
# ---------------------------------------------------------------------------


def test_percent_essential_is_a_share_of_hits_found(screen):
    library, hits, _ = screen
    essentials = set(hits[:3])
    acquired = hits[:6] + library[50:55]
    assert percent_essential(acquired, library, hits, essentials) == pytest.approx(0.5)


def test_percent_essential_ignores_essentials_that_were_not_hits(screen):
    library, hits, _ = screen
    essentials = {library[50], library[51]}
    acquired = hits[:4] + library[50:52]
    assert percent_essential(acquired, library, hits, essentials) == 0.0


def test_percent_essential_ignores_essentials_that_were_not_acquired(screen):
    library, hits, _ = screen
    essentials = set(hits)
    acquired = library[50:60]
    assert percent_essential(acquired, library, hits, essentials) == 0.0


def test_percent_essential_with_no_hits_found_is_zero(screen):
    library, hits, _ = screen
    assert percent_essential(library[50:60], library, hits, set(hits)) == 0.0


def test_percent_essential_all_essential(screen):
    library, hits, _ = screen
    assert percent_essential(hits, library, hits, set(hits)) == 1.0


# ---------------------------------------------------------------------------
# User-supplied DepMap essentials
# ---------------------------------------------------------------------------


@pytest.fixture
def supplied_essentials(tmp_path, monkeypatch):
    path = tmp_path / "CRISPRInferredCommonEssentials.csv"
    path.write_text("Essentials\nAAMP (14)\nAARS1 (16)\n")
    calls = []

    def fake_ensure_asset(name):
        calls.append(name)
        return path

    monkeypatch.setattr(sequential, "ensure_asset", fake_ensure_asset)
    monkeypatch.setattr(sequential, "_essentials_cache", None)
    return calls


def test_essentials_are_loaded_on_first_use(supplied_essentials):
    essentials = load_common_essentials()
    assert essentials == frozenset({"AAMP", "AARS1"})
    assert supplied_essentials == ["depmap-common-essentials-26q1"]


def test_supplied_essentials_are_bare_symbols(supplied_essentials):
    """The source file is ``SYMBOL (ENTREZ_ID)``; the id must be stripped."""
    essentials = load_common_essentials()
    assert "AAMP" in essentials
    assert not any("(" in gene or " " in gene for gene in essentials)


def test_supplied_essentials_are_cached(supplied_essentials):
    assert load_common_essentials() is load_common_essentials()
    assert len(supplied_essentials) == 1


def test_percent_essential_defaults_to_the_supplied_set(supplied_essentials):
    library = ["AAMP", "AARS1", "GENEA", "GENEB"]
    hits = ["AAMP", "GENEA"]
    assert percent_essential(hits, library, hits) == pytest.approx(0.5)


def test_load_essentials_from_explicit_path(tmp_path):
    path = tmp_path / "essentials.csv"
    path.write_text("Essentials\nFOO (1)\nBAR (2)\n")
    assert load_common_essentials(path) == frozenset({"FOO", "BAR"})


def test_load_essentials_from_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_common_essentials(tmp_path / "nope.csv")


# ---------------------------------------------------------------------------
# Differential test against the research code
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(25))
def test_matches_research_code_on_random_trajectories(seed):
    """Randomised trajectories, scored both ways, must agree to the bit.

    Covers under-supply, over-supply, hallucination, forgiveness, and screens
    from very hit-rich to very hit-sparse.
    """
    rng = random.Random(seed)

    library_size = rng.randint(20, 400)
    library = [f"LIB{i:04d}" for i in range(library_size)]
    total_hits = rng.randint(1, max(1, library_size // 3))
    hits = library[:total_hits]
    hit_set = set(hits)
    universe = set(library) | {f"EXT{i:04d}" for i in range(200)}
    essentials = set(rng.sample(library, k=rng.randint(1, library_size // 2)))

    budget = rng.randint(5, 120)
    n_picks = rng.randint(1, budget + 10)  # may over- or under-supply
    pool = (
        library
        + [f"EXT{i:04d}" for i in range(200)]
        + [f"FAKE{i:04d}" for i in range(50)]
    )
    acquired = rng.sample(pool, k=min(n_picks, len(pool)))

    counts = classify_acquisitions(acquired, library, hits, universe, budget)

    expected_ef = _reference_adjusted_nvr(
        n_hits=counts.n_hits,
        n_in_lib=counts.n_in_library,
        n_out_lib_in_universe=counts.n_out_library_in_universe,
        domain_size=library_size,
        total_hits=total_hits,
        budget=budget,
    )
    actual_ef = enrichment_factor(acquired, library, hits, universe, budget)
    assert actual_ef == pytest.approx(expected_ef, rel=1e-12, abs=1e-12)

    assert shortfall(acquired, library) == pytest.approx(
        _reference_oob_frac(acquired, set(library)), rel=1e-12, abs=1e-12
    )

    assert percent_essential(acquired, library, hits, essentials) == pytest.approx(
        _reference_pct_essential(acquired, set(library), hit_set, essentials),
        rel=1e-12,
        abs=1e-12,
    )

    # nAUC needs the trajectory split into rounds, since it samples the curve
    # once per round. Chop the same picks into uneven batches.
    rounds, cursor = [], 0
    while cursor < len(acquired):
        step = rng.randint(1, 25)
        rounds.append(acquired[cursor : cursor + step])
        cursor += step

    expected_nauc = _reference_adj_nauc(
        rounds,
        screen_genes=set(library),
        universe_set=universe,
        total_hits=total_hits,
        domain_size=library_size,
        hit_set=hit_set,
    )
    assert adjusted_nauc(rounds, library, hits, universe) == pytest.approx(
        expected_nauc, rel=1e-12, abs=1e-12
    )

    # FH has no research-code counterpart to transcribe: the table reads it off
    # HitsAUC's ``frac_hits``, which is cum_hits / total_hits. Same quantity,
    # recomputed from the trajectory.
    assert fraction_of_hits(acquired, library, hits) == pytest.approx(
        sum(1 for g in acquired if g in hit_set) / total_hits, rel=1e-12, abs=1e-12
    )
