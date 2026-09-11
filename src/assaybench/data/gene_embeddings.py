"""A gene -> vector table, and the population statistics diversity needs.

:mod:`assaybench.benchmark.diversity` scores a batch against an embedding
space. It needs three things from that space: the vectors for the batch, the
mean vector over the whole space, and a random draw from it. This module is
that interface, and nothing else -- it does not fetch, compute, or cache
embeddings, so importing it costs nothing and depends on nothing.

Where the vectors come from is the caller's choice. The published AssayLoop
numbers use GenePT ada-002 embeddings via the `PRESAGE
<https://github.com/Genentech/PRESAGE>`_ cache, which is 3.4 GB and therefore
neither a dependency of this package nor a download it performs::

    import pickle
    with open(presage_cache / "other_embeddings/GenePT_ada.embeddings.pkl", "rb") as f:
        frame = pickle.load(f)
    embeddings = GeneEmbeddings.from_frame(frame, name="presage:GenePT_ada")

Any other source works the same way, as long as it can be reduced to symbols
and rows.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np

__all__ = ["GeneEmbeddings"]


class GeneEmbeddings:
    """Gene symbols and their vectors, in a fixed row order.

    Args:
        genes: The symbols, one per row of ``matrix``. Duplicate symbols keep
            their first row. Lookup is case-insensitive, preferring an exact
            match.
        matrix: An ``(n_genes, dim)`` array.
        name: Label for :meth:`__repr__` and for recording which space a
            reported number was computed in. Say where the vectors came from.

    Raises:
        ValueError: If ``genes`` and ``matrix`` disagree on length, or the
            matrix is not two-dimensional.
    """

    def __init__(
        self,
        genes: Sequence[str],
        matrix: np.ndarray,
        *,
        name: str = "gene-embeddings",
    ):
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(
                f"Expected a 2-D (n_genes, dim) matrix, got shape {matrix.shape}."
            )
        if len(genes) != matrix.shape[0]:
            raise ValueError(
                f"{len(genes)} genes but {matrix.shape[0]} rows in the matrix."
            )

        self.name = name
        self._genes = [str(g) for g in genes]
        self._matrix = matrix
        self._row: dict[str, int] = {}
        self._row_upper: dict[str, int] = {}
        for index, gene in enumerate(self._genes):
            self._row.setdefault(gene, index)
            self._row_upper.setdefault(gene.upper(), index)
        self._mean: np.ndarray | None = None
        # Memo for :mod:`assaybench.benchmark.diversity`, keyed by
        # ``(sample, seed)``. The random-draw baseline it holds is a property
        # of this space, not of any batch, so it is computed once and reused
        # across every batch scored against it.
        self._baseline_cache: dict[tuple[int, int], float] = {}

    # -- construction --------------------------------------------------------

    @classmethod
    def from_mapping(
        cls, vectors: Mapping[str, Iterable[float]], *, name: str = "gene-embeddings"
    ) -> GeneEmbeddings:
        """Build from a ``{symbol: vector}`` mapping, in iteration order."""
        genes = list(vectors)
        if not genes:
            raise ValueError("Cannot build GeneEmbeddings from an empty mapping.")
        matrix = np.stack([np.asarray(vectors[g], dtype=np.float32) for g in genes])
        return cls(genes, matrix, name=name)

    @classmethod
    def from_frame(cls, frame, *, name: str = "gene-embeddings") -> GeneEmbeddings:
        """Build from a symbol-indexed table -- a pandas DataFrame, or anything
        else exposing ``.index`` and ``.values``.

        Duck-typed on purpose: pandas is not a dependency of this package.
        """
        return cls(list(frame.index), np.asarray(frame.values), name=name)

    # -- accessors -----------------------------------------------------------

    def __len__(self) -> int:
        return len(self._genes)

    def __contains__(self, gene: object) -> bool:
        return isinstance(gene, str) and self.index(gene) is not None

    def __repr__(self) -> str:
        return (
            f"GeneEmbeddings({self.name!r}, {len(self)} genes, dim={self.dim})"
        )

    @property
    def dim(self) -> int:
        return int(self._matrix.shape[1])

    def index(self, gene: str) -> int | None:
        """Row of ``gene``, or ``None`` if the space does not cover it."""
        if not gene:
            return None
        row = self._row.get(gene)
        return self._row_upper.get(gene.upper()) if row is None else row

    def embed(self, gene: str) -> np.ndarray | None:
        """Vector for ``gene``, or ``None`` if it is not covered."""
        row = self.index(gene)
        return None if row is None else self._matrix[row]

    def embed_batch(self, genes: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """Vectors for a batch, with a coverage mask.

        Returns:
            ``(matrix, mask)``. ``matrix`` is ``(len(genes), dim)`` with zero
            rows where the gene is not covered; ``mask`` is a boolean array
            marking the rows that are real. Callers must apply the mask --
            the zero rows are padding, not embeddings.
        """
        matrix = np.zeros((len(genes), self.dim), dtype=np.float32)
        mask = np.zeros(len(genes), dtype=bool)
        for position, gene in enumerate(genes):
            row = self.index(str(gene))
            if row is not None:
                matrix[position] = self._matrix[row]
                mask[position] = True
        return matrix, mask

    def coverage(self, genes: Sequence[str]) -> float:
        """Fraction of ``genes`` the space covers. ``0.0`` for an empty list."""
        if not genes:
            return 0.0
        return float(np.mean([self.index(str(g)) is not None for g in genes]))

    # -- population statistics -----------------------------------------------

    def mean_embedding(self) -> np.ndarray:
        """Mean vector over every gene in the space, cached.

        Subtracted before cosine distances are taken. Text-derived gene
        embeddings are strongly anisotropic -- random GenePT/ada gene pairs sit
        at cosine ~0.82 -- so raw ``(1 - cos)/2`` compresses every batch into a
        narrow band and is near-constant. Centering restores the range.
        """
        if self._mean is None:
            self._mean = self._matrix.mean(axis=0)
        return self._mean

    def sample_matrix(self, n: int, *, seed: int = 0) -> np.ndarray:
        """A random ``(min(n, len(self)), dim)`` draw, without replacement.

        Used to calibrate a batch's diversity against a random draw from the
        same space. Deterministic in ``seed``.
        """
        total = self._matrix.shape[0]
        rng = np.random.default_rng(seed)
        chosen = rng.choice(total, size=min(n, total), replace=False)
        return self._matrix[chosen]
