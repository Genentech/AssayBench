"""Batch-diversity metrics for ASSAYBENCH-LOOP.

EF and nAUC ask whether a batch found hits. These ask whether it *explored*.
A policy that proposes twelve subunits of the same complex can score well on
enrichment while telling the experimentalist almost nothing new, and the
failure is invisible in a hit count.

Two complementary views, because they disagree in informative ways:

**Vendi score** -- the effective number of distinct genes in the batch, from
the embedding geometry. Formally the exponentiated Shannon entropy of the
eigenvalues of the batch's normalised cosine-similarity matrix: ``1`` when
every gene is collinear with every other, ``n`` when all ``n`` are mutually
orthogonal. It is reported as a ratio to ``n`` so batch sizes are comparable.
Alongside it, the mean pairwise cosine distance, calibrated against a random
draw from the same space.

**Pathway overlap** -- embedding-free. The mean pairwise Jaccard of the genes'
MSigDB pathway memberships, divided by the same statistic for a random draw.
Above 1 means the batch clusters into shared pathways more than chance.

The two can disagree, and that is the point of reporting both: text-derived
embeddings place genes near each other when the literature discusses them
together, which is not the same thing as sharing a pathway.

Neither view has a default data source. :func:`batch_diversity` requires the
caller to supply the embedding space, the gene sets, or both -- there is no
stand-in. One-hot vectors in particular would yield a Vendi score of exactly
``n`` for every batch: a meaningless number, and indistinguishable in a results
table from a real one.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..data.gene_embeddings import GeneEmbeddings
from ..data.gene_sets import GeneSetMembership

__all__ = [
    "vendi_score",
    "mean_pairwise_cosine_distance",
    "embedding_diversity",
    "pathway_diversity",
    "batch_diversity",
]


# ---------------------------------------------------------------------------
# Embedding geometry
# ---------------------------------------------------------------------------


def _unit_centered(embeddings: np.ndarray, center: np.ndarray | None) -> np.ndarray:
    """Centre on ``center`` (when its shape fits) and L2-normalise the rows.

    A zero row would divide by zero; its norm is treated as 1, leaving it zero,
    which contributes a cosine of 0 rather than a NaN.
    """
    if center is not None and np.asarray(center).shape[0] == embeddings.shape[1]:
        embeddings = embeddings - center
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return embeddings / norms


def mean_pairwise_cosine_distance(
    embeddings: np.ndarray, *, center: np.ndarray | None = None
) -> float:
    """Mean pairwise cosine distance over a batch, in ``[0, 1]``.

    ``(1 - cos) / 2``, averaged over the distinct pairs.

    Args:
        embeddings: An ``(n, dim)`` array. Rows must already be filtered to the
            genes the space covers.
        center: Subtracted before normalising -- pass
            :meth:`GeneEmbeddings.mean_embedding`. Anisotropic spaces need it;
            see that method for why.

    Returns:
        The mean distance. ``0.0`` for fewer than two rows.
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        return 0.0
    unit = _unit_centered(embeddings, center)
    similarity = np.clip(unit @ unit.T, -1.0, 1.0)
    rows, cols = np.triu_indices(unit.shape[0], k=1)
    return float((1.0 - similarity[rows, cols]).mean() / 2.0)


def vendi_score(embeddings: np.ndarray, *, center: np.ndarray | None = None) -> float:
    """Effective number of distinct genes in a batch.

    ``exp(H(λ))`` where ``λ`` are the eigenvalues of ``K/n``, and ``K`` is the
    batch's cosine-similarity Gram matrix. Because ``K/n`` is positive
    semi-definite with unit trace, its eigenvalues are a probability
    distribution, and the exponentiated entropy of that distribution counts how
    many genes the batch is *effectively* made of.

    Args:
        embeddings: An ``(n, dim)`` array, already filtered to covered genes.
        center: Subtracted before normalising. See
            :meth:`GeneEmbeddings.mean_embedding`.

    Returns:
        A value in ``[1, n]``. ``1.0`` for fewer than two rows, or for a batch
        whose genes are all collinear -- both mean "effectively one gene".
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        return 1.0
    unit = _unit_centered(embeddings, center)
    n = unit.shape[0]
    eigenvalues = np.linalg.eigvalsh((unit @ unit.T) / n)
    # Numerical noise puts near-zero eigenvalues on either side of 0; they
    # contribute nothing to the entropy but would produce NaN through log.
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    if eigenvalues.size == 0:
        return 1.0
    entropy = float(-(eigenvalues * np.log(eigenvalues)).sum())
    return float(np.exp(entropy))


def embedding_diversity(
    genes: Sequence[str],
    embeddings: GeneEmbeddings,
    *,
    baseline_sample: int = 1024,
    baseline_seed: int = 0,
) -> dict[str, float]:
    """The embedding view of one batch's diversity.

    Args:
        genes: The batch. Genes the space does not cover are dropped, and
            reported via ``batch_diversity_n``.
        embeddings: The space to score in.
        baseline_sample: Genes drawn at random to calibrate
            ``batch_diversity_vs_random``.
        baseline_seed: Seed for that draw.

    Returns:
        ``batch_diversity``
            Mean pairwise centred cosine distance.
        ``batch_diversity_vs_random``
            That, over a random draw from the same space. ``~1`` is as spread
            out as chance, ``< 1`` is focused. Omitted if the space is too
            small to draw a baseline from.
        ``batch_vendi``
            Effective number of distinct genes.
        ``batch_vendi_ratio``
            ``batch_vendi / n``, in ``(0, 1]``. The batch-size-free form, and
            the one the paper's table reports.
        ``batch_diversity_n``
            Covered genes actually used.

        An empty dict when fewer than two genes are covered -- a pairwise
        statistic needs a pair. This is the only reason a key is ever missing;
        absent reference data raises instead.
    """
    matrix, mask = embeddings.embed_batch([str(g) for g in genes])
    covered = np.asarray(matrix[mask], dtype=np.float64)
    n = int(mask.sum())
    if n < 2:
        return {}

    center = embeddings.mean_embedding()
    diversity = mean_pairwise_cosine_distance(covered, center=center)
    vendi = vendi_score(covered, center=center)
    out = {
        "batch_diversity": diversity,
        "batch_diversity_n": float(n),
        "batch_vendi": vendi,
        "batch_vendi_ratio": vendi / n,
    }

    baseline = _random_draw_baseline(embeddings, baseline_sample, baseline_seed)
    if baseline:
        out["batch_diversity_vs_random"] = diversity / baseline
    return out


def _random_draw_baseline(
    embeddings: GeneEmbeddings, sample: int, seed: int
) -> float:
    """Mean pairwise distance of a random draw, memoised on the space.

    The draw is deterministic in ``seed`` and independent of the batch, so
    every batch scored against a given space shares one baseline -- computing
    it per batch would be the same number at a thousand times the cost.
    """
    key = (sample, seed)
    cached = embeddings._baseline_cache.get(key)
    if cached is not None:
        return cached
    matrix = embeddings.sample_matrix(sample, seed=seed)
    value = (
        0.0
        if matrix.shape[0] < 2
        else mean_pairwise_cosine_distance(
            np.asarray(matrix, dtype=np.float64),
            center=embeddings.mean_embedding(),
        )
    )
    embeddings._baseline_cache[key] = value
    return value


# ---------------------------------------------------------------------------
# Pathway coverage
# ---------------------------------------------------------------------------


def pathway_diversity(
    genes: Sequence[str], gene_sets: GeneSetMembership
) -> dict[str, float]:
    """The pathway view of one batch's diversity.

    Args:
        genes: The batch. Unannotated genes are dropped, and reported via
            ``batch_pathway_n``.
        gene_sets: The membership map, from
            :func:`~assaybench.data.gene_sets.load_gene_sets`.

    Returns:
        ``batch_pathway_diversity``
            ``1 - mean pairwise Jaccard`` of pathway membership.
        ``batch_pathway_coverage``
            Distinct pathways spanned, per annotated gene.
        ``batch_pathway_overlap_vs_random``
            Overlap over a random draw's overlap. ``> 1`` means the batch
            clusters into shared pathways more than chance would.
        ``batch_pathway_n``
            Annotated genes used.

        An empty dict when fewer than two genes are annotated.
    """
    stats = gene_sets.batch_stats([str(g) for g in genes])
    if stats is None:
        return {}
    out = {
        "batch_pathway_diversity": stats["diversity"],
        "batch_pathway_coverage": stats["coverage"],
        "batch_pathway_n": stats["n"],
    }
    baseline = gene_sets.baseline_overlap()
    if baseline:
        out["batch_pathway_overlap_vs_random"] = stats["mean_jaccard"] / baseline
    return out


# ---------------------------------------------------------------------------
# Both views
# ---------------------------------------------------------------------------


def batch_diversity(
    genes: Sequence[str],
    embeddings: GeneEmbeddings | None = None,
    gene_sets: GeneSetMembership | None = None,
    *,
    baseline_sample: int = 1024,
    baseline_seed: int = 0,
) -> dict[str, float]:
    """Both diversity views for one acquired batch.

    Args:
        genes: The batch, as acquired.
        embeddings: Space for the Vendi and cosine views. Omit to skip them.
        gene_sets: Membership map for the pathway view. Omit to skip it.
        baseline_sample: Genes drawn to calibrate the embedding baseline.
        baseline_seed: Seed for that draw.

    Returns:
        The union of :func:`embedding_diversity` and
        :func:`pathway_diversity`, restricted to the views requested. Empty if
        the batch has fewer than two genes.

    Raises:
        ValueError: If neither ``embeddings`` nor ``gene_sets`` is given.
            Returning an empty dict instead would look identical, downstream,
            to a batch too small to score.
    """
    if embeddings is None and gene_sets is None:
        raise ValueError(
            "batch_diversity needs an embedding space, a gene-set membership, "
            "or both. Pass embeddings= for the Vendi view (see "
            "assaybench.data.gene_embeddings.GeneEmbeddings) and gene_sets= "
            "for the pathway view (assaybench.data.gene_sets.load_gene_sets)."
        )
    if len(genes) < 2:
        return {}

    out: dict[str, float] = {}
    if embeddings is not None:
        out.update(
            embedding_diversity(
                genes,
                embeddings,
                baseline_sample=baseline_sample,
                baseline_seed=baseline_seed,
            )
        )
    if gene_sets is not None:
        out.update(pathway_diversity(genes, gene_sets))
    return out
