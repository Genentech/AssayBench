"""Tests for the batch-diversity metrics.

Same shape of argument as ``test_sequential``: the point is that moving Vendi
and pathway overlap out of the AssayLoop research code changed no number.

1. The properties the Vendi score is defined by (``1`` for collinear, ``n`` for
   orthogonal, invariance to scale and to row order).
2. The behaviours the metric exists to catch, and the refusals it must make.
3. A differential test against verbatim transcriptions of the research code,
   over randomised batches.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from assaybench.benchmark.diversity import (
    batch_diversity,
    embedding_diversity,
    mean_pairwise_cosine_distance,
    pathway_diversity,
    vendi_score,
)
from assaybench.assets import MissingAsset
from assaybench.data import gene_sets as gene_sets_module
from assaybench.data.gene_embeddings import GeneEmbeddings
from assaybench.data.gene_sets import (
    DEFAULT_GENE_SETS,
    GeneSetMembership,
    load_gene_sets,
)


# ---------------------------------------------------------------------------
# Reference implementations, transcribed verbatim from the AssayLoop research
# code (bridgeloop.metrics.batch_diversity.BatchDiversity). Do not tidy these
# up -- their value is being unedited copies.
# ---------------------------------------------------------------------------


def _reference_unit_centered(embs, mean):
    if mean is not None and mean.shape[0] == embs.shape[1]:
        embs = embs - mean
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return embs / norms


def _reference_mean_pairwise_cos_dist(unit):
    cos_sim = np.clip(unit @ unit.T, -1.0, 1.0)
    i, j = np.triu_indices(unit.shape[0], k=1)
    return float((1.0 - cos_sim[i, j]).mean() / 2.0)


def _reference_vendi(unit):
    n = unit.shape[0]
    kernel = (unit @ unit.T) / n
    w = np.linalg.eigvalsh(kernel)
    w = w[w > 1e-12]
    if w.size == 0:
        return 1.0
    entropy = float(-(w * np.log(w)).sum())
    return float(np.exp(entropy))


def _reference_mean_pairwise_jaccard(sets):
    """bridgeloop.data.gene_sets.GeneSetMembership._mean_pairwise_jaccard"""
    sims = []
    for a in range(len(sets)):
        sa = sets[a]
        for b in range(a + 1, len(sets)):
            sb = sets[b]
            union = len(sa | sb)
            sims.append(len(sa & sb) / union if union else 0.0)
    return sum(sims) / len(sims) if sims else 0.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def space():
    """A 500-gene, 32-dimensional embedding space with a deliberate offset.

    The offset makes the space anisotropic, which is what centering is for.
    """
    rng = np.random.default_rng(0)
    genes = [f"GENE{i:03d}" for i in range(500)]
    matrix = rng.normal(size=(500, 32)) + 3.0
    return GeneEmbeddings(genes, matrix, name="test-space")


@pytest.fixture
def gmt(tmp_path):
    """A small ``.gmt``: two pathway families that do not overlap."""
    lines = [
        "PATH_A\thttp://example\t" + "\t".join(f"GENE{i:03d}" for i in range(0, 20)),
        "PATH_A2\thttp://example\t" + "\t".join(f"GENE{i:03d}" for i in range(0, 15)),
        "PATH_B\thttp://example\t" + "\t".join(f"GENE{i:03d}" for i in range(50, 70)),
        "PATH_C\thttp://example\t" + "\t".join(f"GENE{i:03d}" for i in range(100, 140)),
    ]
    path = tmp_path / "test.symbols.gmt"
    path.write_text("\n".join(lines) + "\n")
    return GeneSetMembership(path)


# ---------------------------------------------------------------------------
# Vendi score: the defining properties
# ---------------------------------------------------------------------------


def test_vendi_of_orthogonal_genes_is_n():
    """Mutually orthogonal genes are n effectively-distinct genes."""
    assert vendi_score(np.eye(8)) == pytest.approx(8.0)


def test_vendi_of_collinear_genes_is_one():
    """A batch of copies of one gene is effectively one gene."""
    vector = np.array([1.0, 2.0, -3.0, 0.5])
    assert vendi_score(np.tile(vector, (6, 1))) == pytest.approx(1.0)


def test_vendi_ignores_magnitude():
    """It is a cosine kernel: only direction can matter."""
    rng = np.random.default_rng(1)
    matrix = rng.normal(size=(10, 16))
    scaled = matrix * rng.uniform(0.1, 10.0, size=(10, 1))
    assert vendi_score(scaled) == pytest.approx(vendi_score(matrix))


def test_vendi_ignores_row_order():
    rng = np.random.default_rng(2)
    matrix = rng.normal(size=(12, 16))
    assert vendi_score(matrix[::-1]) == pytest.approx(vendi_score(matrix))


def test_vendi_is_bounded_by_the_batch_size():
    rng = np.random.default_rng(3)
    for n in (2, 5, 20):
        score = vendi_score(rng.normal(size=(n, 8)))
        assert 1.0 <= score <= n + 1e-9


def test_vendi_of_a_spread_batch_beats_a_clustered_one():
    """The metric's whole purpose, in one assertion."""
    rng = np.random.default_rng(4)
    spread = rng.normal(size=(10, 16))
    clustered = np.tile(rng.normal(size=(1, 16)), (10, 1)) + 0.01 * rng.normal(
        size=(10, 16)
    )
    assert vendi_score(clustered) < vendi_score(spread)


@pytest.mark.parametrize("rows", [0, 1])
def test_vendi_of_a_batch_too_small_to_score(rows):
    assert vendi_score(np.zeros((rows, 4))) == 1.0


def test_vendi_of_zero_vectors_does_not_produce_nan():
    """Zero rows have no direction; they must not poison the eigenvalues."""
    assert np.isfinite(vendi_score(np.zeros((5, 4))))


# ---------------------------------------------------------------------------
# Cosine distance and centering
# ---------------------------------------------------------------------------


def test_cosine_distance_of_identical_genes_is_zero():
    assert mean_pairwise_cosine_distance(np.ones((4, 6))) == pytest.approx(0.0)


def test_cosine_distance_of_opposed_genes_is_one():
    assert mean_pairwise_cosine_distance(
        np.array([[1.0, 0.0], [-1.0, 0.0]])
    ) == pytest.approx(1.0)


def test_centering_widens_the_dynamic_range(space):
    """The reason ``center`` exists: an offset space compresses every batch.

    Two batches of very different spread should be distinguishable. Without
    centering the shared offset dominates every cosine and they are not.
    """
    rng = np.random.default_rng(5)
    tight = np.tile(rng.normal(size=(1, 32)), (10, 1)) + 3.0
    wide = rng.normal(size=(10, 32)) + 3.0
    center = space.mean_embedding()

    raw_gap = abs(
        mean_pairwise_cosine_distance(wide) - mean_pairwise_cosine_distance(tight)
    )
    centered_gap = abs(
        mean_pairwise_cosine_distance(wide, center=center)
        - mean_pairwise_cosine_distance(tight, center=center)
    )
    assert centered_gap > raw_gap


def test_center_of_the_wrong_width_is_ignored():
    """Guards a dimension mismatch rather than raising mid-metric."""
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert mean_pairwise_cosine_distance(
        matrix, center=np.zeros(7)
    ) == pytest.approx(mean_pairwise_cosine_distance(matrix))


@pytest.mark.parametrize("rows", [0, 1])
def test_cosine_distance_of_a_batch_too_small_to_score(rows):
    assert mean_pairwise_cosine_distance(np.zeros((rows, 4))) == 0.0


# ---------------------------------------------------------------------------
# Embedding view
# ---------------------------------------------------------------------------


def test_embedding_diversity_reports_every_key(space):
    out = embedding_diversity([f"GENE{i:03d}" for i in range(10)], space)
    assert set(out) == {
        "batch_diversity",
        "batch_diversity_n",
        "batch_diversity_vs_random",
        "batch_vendi",
        "batch_vendi_ratio",
    }


def test_embedding_diversity_drops_uncovered_genes(space):
    """Coverage is reported, not silently folded into the denominator."""
    out = embedding_diversity(
        ["GENE001", "GENE002", "GENE003", "NOTAGENE1", "NOTAGENE2"], space
    )
    assert out["batch_diversity_n"] == 3.0
    assert out["batch_vendi_ratio"] == pytest.approx(out["batch_vendi"] / 3)


def test_embedding_diversity_of_a_batch_with_one_covered_gene(space):
    assert embedding_diversity(["GENE001", "NOTAGENE"], space) == {}


def test_embedding_diversity_lookup_is_case_insensitive(space):
    assert embedding_diversity(["gene001", "gene002", "gene003"], space) == (
        pytest.approx(embedding_diversity(["GENE001", "GENE002", "GENE003"], space))
    )


def test_random_draw_scores_about_one_versus_random(space):
    """A random batch is, by construction, as spread out as a random draw."""
    rng = random.Random(6)
    genes = rng.sample([f"GENE{i:03d}" for i in range(500)], 50)
    out = embedding_diversity(genes, space)
    assert out["batch_diversity_vs_random"] == pytest.approx(1.0, abs=0.1)


def test_baseline_is_memoised_on_the_space(space):
    embedding_diversity(["GENE001", "GENE002", "GENE003"], space)
    assert (1024, 0) in space._baseline_cache
    first = space._baseline_cache[(1024, 0)]
    embedding_diversity(["GENE010", "GENE011"], space)
    assert space._baseline_cache[(1024, 0)] == first


def test_baseline_is_deterministic():
    """Two identical spaces must produce the same reported number."""
    rng = np.random.default_rng(7)
    genes = [f"GENE{i:03d}" for i in range(200)]
    matrix = rng.normal(size=(200, 16))
    first = embedding_diversity(genes[:20], GeneEmbeddings(genes, matrix))
    second = embedding_diversity(genes[:20], GeneEmbeddings(genes, matrix))
    assert first == pytest.approx(second)


# ---------------------------------------------------------------------------
# Pathway view
# ---------------------------------------------------------------------------


def test_pathway_diversity_reports_every_key(gmt):
    out = pathway_diversity([f"GENE{i:03d}" for i in (0, 1, 55, 105)], gmt)
    assert set(out) == {
        "batch_pathway_diversity",
        "batch_pathway_coverage",
        "batch_pathway_n",
        "batch_pathway_overlap_vs_random",
    }


def test_genes_in_one_pathway_have_zero_diversity(gmt):
    """PATH_B's genes belong to PATH_B and nothing else: Jaccard 1 throughout."""
    out = pathway_diversity(["GENE050", "GENE051", "GENE052"], gmt)
    assert out["batch_pathway_diversity"] == pytest.approx(0.0)
    assert out["batch_pathway_coverage"] == pytest.approx(1 / 3)


def test_genes_in_disjoint_pathways_have_full_diversity(gmt):
    out = pathway_diversity(["GENE050", "GENE105"], gmt)
    assert out["batch_pathway_diversity"] == pytest.approx(1.0)


def test_pathway_diversity_drops_unannotated_genes(gmt):
    out = pathway_diversity(["GENE050", "GENE051", "GENE400", "NOTAGENE"], gmt)
    assert out["batch_pathway_n"] == 2.0


def test_pathway_diversity_of_a_batch_with_one_annotated_gene(gmt):
    assert pathway_diversity(["GENE050", "NOTAGENE"], gmt) == {}


def test_pathway_overlap_vs_random_is_above_one_for_a_clustered_batch(gmt):
    """The direction the column is read in: > 1 = more clustered than chance."""
    out = pathway_diversity(["GENE050", "GENE051", "GENE052"], gmt)
    assert out["batch_pathway_overlap_vs_random"] > 1.0


def test_gene_set_lookup_is_case_insensitive(gmt):
    assert gmt.get("gene050") == gmt.get("GENE050")


def test_gene_set_membership_counts_sets_and_genes(gmt):
    assert gmt.n_sets == 4
    assert len(gmt) == 20 + 20 + 40  # PATH_A2's genes are a subset of PATH_A's


def test_baseline_overlap_is_cached(gmt):
    assert gmt.baseline_overlap() == gmt.baseline_overlap()


# ---------------------------------------------------------------------------
# Loading a collection
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("ASSAYBENCH_CACHE", str(tmp_path))
    monkeypatch.setattr(gene_sets_module, "_CACHE", {})
    return tmp_path


def test_load_gene_sets_without_the_download_raises(empty_cache):
    """The whole no-silent-fallback contract, at the point it matters."""
    with pytest.raises(MissingAsset) as excinfo:
        load_gene_sets()
    message = str(excinfo.value)
    assert f"assaybench download {DEFAULT_GENE_SETS}" in message
    assert "gsea-msigdb" in message


def test_load_gene_sets_rejects_an_unregistered_name(empty_cache):
    with pytest.raises(KeyError, match="msigdb-go-bp"):
        load_gene_sets("msigdb-not-a-collection")


def test_load_gene_sets_is_memoised(empty_cache):
    """A .gmt takes seconds to parse; the same collection is parsed once."""
    path = empty_cache / "c5.go.bp.v2023.2.Hs.symbols.gmt"
    path.write_text("SET1\tdesc\tGENE001\tGENE002\nSET2\tdesc\tGENE002\tGENE003\n")
    assert load_gene_sets() is load_gene_sets()
    assert load_gene_sets().n_sets == 2


# ---------------------------------------------------------------------------
# Both views, and the refusals
# ---------------------------------------------------------------------------


def test_batch_diversity_merges_both_views(space, gmt):
    out = batch_diversity([f"GENE{i:03d}" for i in (0, 1, 55, 105)], space, gmt)
    assert "batch_vendi" in out and "batch_pathway_diversity" in out


def test_batch_diversity_with_only_one_view(space, gmt):
    embedding_only = batch_diversity(["GENE001", "GENE002", "GENE003"], space)
    assert embedding_only and not any(k.startswith("batch_pathway") for k in embedding_only)

    pathway_only = batch_diversity(["GENE050", "GENE051"], gene_sets=gmt)
    assert pathway_only and not any(k.startswith("batch_vendi") for k in pathway_only)


def test_batch_diversity_without_a_data_source_raises():
    """No silent empty dict: that is indistinguishable from a tiny batch."""
    with pytest.raises(ValueError, match="embedding space"):
        batch_diversity(["GENE001", "GENE002"])


def test_batch_diversity_of_a_batch_too_small_to_score(space, gmt):
    assert batch_diversity(["GENE001"], space, gmt) == {}
    assert batch_diversity([], space, gmt) == {}


# ---------------------------------------------------------------------------
# Differential test against the research code
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(20))
def test_matches_research_code_on_random_batches(seed):
    """Randomised spaces and batches, scored both ways, must agree to the bit."""
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    n_genes = int(rng.integers(50, 400))
    dim = int(rng.integers(4, 64))
    genes = [f"GENE{i:04d}" for i in range(n_genes)]
    # Vary the anisotropy: some spaces are centred, some heavily offset.
    matrix = rng.normal(size=(n_genes, dim)) + rng.normal() * 5.0
    space = GeneEmbeddings(genes, matrix)

    batch = py_rng.sample(genes, k=py_rng.randint(2, min(60, n_genes)))
    embedded, mask = space.embed_batch(batch)
    covered = np.asarray(embedded[mask], dtype=np.float64)

    unit = _reference_unit_centered(covered, space.mean_embedding())
    expected_diversity = _reference_mean_pairwise_cos_dist(unit)
    expected_vendi = _reference_vendi(unit)

    out = embedding_diversity(batch, space)
    assert out["batch_diversity"] == pytest.approx(
        expected_diversity, rel=1e-12, abs=1e-12
    )
    assert out["batch_vendi"] == pytest.approx(expected_vendi, rel=1e-12, abs=1e-12)
    assert out["batch_vendi_ratio"] == pytest.approx(
        expected_vendi / len(batch), rel=1e-12, abs=1e-12
    )

    # The research code's baseline: sample_matrix(1024, seed=0), same centering.
    sample = space.sample_matrix(1024, seed=0)
    expected_baseline = _reference_mean_pairwise_cos_dist(
        _reference_unit_centered(
            np.asarray(sample, dtype=np.float64), space.mean_embedding()
        )
    )
    assert out["batch_diversity_vs_random"] == pytest.approx(
        expected_diversity / expected_baseline, rel=1e-12, abs=1e-12
    )


@pytest.mark.parametrize("seed", range(20))
def test_pathway_stats_match_research_code_on_random_batches(seed, tmp_path):
    py_rng = random.Random(seed)

    genes = [f"GENE{i:04d}" for i in range(300)]
    lines = []
    for set_index in range(py_rng.randint(3, 30)):
        members = py_rng.sample(genes, k=py_rng.randint(2, 80))
        lines.append(f"SET{set_index}\tdesc\t" + "\t".join(members))
    path = tmp_path / "random.gmt"
    path.write_text("\n".join(lines) + "\n")
    membership = GeneSetMembership(path)

    batch = py_rng.sample(genes, k=py_rng.randint(2, 50))
    annotated = [s for s in (membership.get(g) for g in batch) if s]
    if len(annotated) < 2:
        assert pathway_diversity(batch, membership) == {}
        return

    expected = _reference_mean_pairwise_jaccard(annotated)
    out = pathway_diversity(batch, membership)
    assert out["batch_pathway_diversity"] == pytest.approx(
        1.0 - expected, rel=1e-12, abs=1e-12
    )
    assert out["batch_pathway_n"] == float(len(annotated))
