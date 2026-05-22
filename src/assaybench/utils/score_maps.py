"""Relevance score transformations for ranking metrics."""

import numpy as np
from typing import List
from scipy.stats import rankdata


def hit_scale_relevance_scores(relevance_scores: List[float]) -> List[float]:
    """
    Rescale relevance scores to within-hit percentile ranks.

    For positive scores (hits), replaces each score with its percentile rank
    among positive scores only (1 = best hit, 1/n_hits = worst hit).
    For negative scores, does the same within negative scores
    (−1 = most negative, −1/n_neg_hits = least negative).
    Zero scores are left unchanged.

    This makes the metric comparable across screens with different hit rates:
    the best hit always scores 1, regardless of how many hits exist.
    """
    scores = np.array(relevance_scores, dtype=float)

    pos_mask = scores > 0
    n_pos = int(pos_mask.sum())
    if n_pos > 0:
        ranks = rankdata(-scores[pos_mask], method="average")  # rank 1 = highest
        scores[pos_mask] = 1 - (ranks - 1) / n_pos

    neg_mask = scores < 0
    n_neg = int(neg_mask.sum())
    if n_neg > 0:
        ranks = rankdata(scores[neg_mask], method="average")   # rank 1 = most negative
        scores[neg_mask] = -(1 - (ranks - 1) / n_neg)

    return scores.tolist()
