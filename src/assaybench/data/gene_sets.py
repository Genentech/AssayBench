"""MSigDB gene-set membership, for pathway-based batch diversity.

A ``.gmt`` file is one gene set per line::

    <set_name>\\t<description>\\t<gene1>\\t<gene2>\\t...

This module exposes the raw gene -> set-of-pathways mapping, so diversity can
be measured as literal pathway overlap rather than as geometry in an embedding
space. The two disagree often enough to be worth reporting separately: genes
can be neighbours in a text embedding because they are discussed in the same
literature while belonging to entirely different pathways.

The ``.gmt`` files are not bundled -- see :mod:`assaybench.assets` for why, and
for the ``assaybench download`` command that fetches them. A missing download
raises :class:`~assaybench.assets.MissingAsset`; nothing here falls back to a
partial or substitute collection.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from pathlib import Path

from ..assets import asset_path

__all__ = [
    "GeneSetMembership",
    "DEFAULT_GENE_SETS",
    "load_gene_sets",
]

#: The collection the published pathway-overlap numbers use. GO biological
#: process has the broadest gene coverage of the three registered collections,
#: so the fewest picks go unscored.
DEFAULT_GENE_SETS = "msigdb-go-bp"


class GeneSetMembership:
    """Gene -> frozenset of pathway names, read from a MSigDB ``.gmt``.

    Args:
        path: The ``.gmt`` to read.
        source: Label for :meth:`name`. Defaults to the file's stem.

    Attributes:
        n_sets: How many gene sets the file defined.
    """

    def __init__(self, path: str | Path, source: str | None = None):
        self.path = Path(path)
        self.source = source or self.path.stem
        self._gene_to_sets: dict[str, frozenset[str]] = {}
        self.n_sets = 0
        self._load()
        self._baseline_overlap: float | None = None

    def _load(self) -> None:
        gene_to_sets: dict[str, set[str]] = {}
        n_sets = 0
        with open(self.path) as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                name = parts[0]
                n_sets += 1
                for gene in parts[2:]:
                    if gene:
                        gene_to_sets.setdefault(gene.upper(), set()).add(name)
        self._gene_to_sets = {g: frozenset(s) for g, s in gene_to_sets.items()}
        self.n_sets = n_sets

    def name(self) -> str:
        return f"geneset:{self.source}"

    def __len__(self) -> int:
        """How many genes carry at least one annotation."""
        return len(self._gene_to_sets)

    def get(self, gene: str) -> frozenset[str] | None:
        """Pathways containing ``gene``, or ``None`` if it is unannotated.

        Lookup is case-insensitive: MSigDB symbols are upper-case, but callers
        are not required to be.
        """
        if not gene:
            return None
        return self._gene_to_sets.get(gene.upper())

    @staticmethod
    def _mean_pairwise_jaccard(sets: list[frozenset[str]]) -> float:
        similarities: list[float] = []
        for a in range(len(sets)):
            first = sets[a]
            for b in range(a + 1, len(sets)):
                second = sets[b]
                union = len(first | second)
                similarities.append(len(first & second) / union if union else 0.0)
        return sum(similarities) / len(similarities) if similarities else 0.0

    def batch_stats(self, genes: Iterable[str]) -> dict[str, float] | None:
        """Pathway-overlap statistics for one batch.

        Args:
            genes: The batch. Unannotated genes are dropped.

        Returns:
            ``None`` when fewer than two genes are annotated, since a pairwise
            statistic needs a pair. Otherwise:

            ``mean_jaccard``
                Mean pairwise Jaccard of the genes' pathway memberships.
            ``diversity``
                ``1 - mean_jaccard``. High = the batch spreads across pathways.
            ``coverage``
                Distinct pathways spanned, per annotated gene.
            ``n``
                Annotated genes used.
        """
        sets = [s for s in (self.get(g) for g in genes) if s]
        if len(sets) < 2:
            return None
        mean_jaccard = self._mean_pairwise_jaccard(sets)
        union: set[str] = set()
        for entry in sets:
            union |= entry
        return {
            "mean_jaccard": mean_jaccard,
            "diversity": 1.0 - mean_jaccard,
            "coverage": len(union) / len(sets),
            "n": float(len(sets)),
        }

    def baseline_overlap(self, *, sample: int = 512, seed: int = 0) -> float:
        """Mean pairwise Jaccard of a random draw of annotated genes.

        A batch's raw overlap is not interpretable on its own -- a dense
        collection gives every batch a high Jaccard. Dividing by this baseline
        cancels the collection's density, so ``> 1`` means "more clustered than
        a random draw from the same annotation set" regardless of which
        collection is in use.

        Cached after the first call. Args are only read on that first call.
        """
        if self._baseline_overlap is not None:
            return self._baseline_overlap
        genes = list(self._gene_to_sets)
        if len(genes) < 2:
            self._baseline_overlap = 0.0
            return 0.0
        rng = random.Random(seed)
        picked = rng.sample(genes, min(sample, len(genes)))
        self._baseline_overlap = self._mean_pairwise_jaccard(
            [self._gene_to_sets[g] for g in picked]
        )
        return self._baseline_overlap


_CACHE: dict[str, GeneSetMembership] = {}


def load_gene_sets(name: str = DEFAULT_GENE_SETS) -> GeneSetMembership:
    """Load a downloaded MSigDB collection, memoised by asset name.

    Args:
        name: An :mod:`assaybench.assets` registry key --
            ``"msigdb-go-bp"``, ``"msigdb-hallmark"``, or
            ``"msigdb-canonical-pathways"``.

    Returns:
        The membership map. Parsing a ``.gmt`` takes a second or two, so the
        result is cached for the process.

    Raises:
        KeyError: If ``name`` is not a registered asset.
        MissingAsset: If it has not been downloaded. Callers for whom pathway
            diversity is genuinely optional must catch this and report the
            omission -- dropping the column silently would leave a results
            table that looks complete and is not.
    """
    if name not in _CACHE:
        _CACHE[name] = GeneSetMembership(asset_path(name), source=name)
    return _CACHE[name]
