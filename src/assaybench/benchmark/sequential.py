"""Sequential-acquisition metrics for ASSAYBENCH-LOOP.

:mod:`assaybench.benchmark.metrics` scores a *single* ranked gene list. This
module scores a *trajectory*: the genes a policy acquires over N rounds of an
adaptive screen, where each round's labels are revealed before the next round
is proposed.

The metrics implemented here are the ones defined in AssayLoop §3.3:

============  =========================================  ====================
Metric        Definition                                  Function
============  =========================================  ====================
EF            ``|G ∩ H| / ((N_L + N_Ū) · h_rand)``        :func:`enrichment_factor`
adjusted nAUC ``AUC / AUC_oracle`` on the effective axis  :func:`adjusted_nauc`
FH            ``|G ∩ H| / |H|``                            :func:`fraction_of_hits`
Shortfall     ``|G \\ L| / |G|``                           :func:`shortfall`
%essential    ``|G ∩ H ∩ E| / |G ∩ H|``                    :func:`percent_essential`
============  =========================================  ====================

with

``U``
    the candidate universe -- all gene symbols the policy is allowed to name
    (in practice, approved HGNC symbols).
``L``
    the screen library: genes actually measured in this screen, so the only
    genes for which a label exists. ``L ⊆ U``.
``H``
    the hits within ``L``.
``G``
    the genes the policy acquired over the whole trajectory.
``E``
    DepMap common-essential genes (see :func:`load_common_essentials`).

Two conventions deserve emphasis, because they are what makes EF comparable
across screens with very different hit rates and gene libraries:

*Out-of-library but real genes are forgiven.* A policy that names a real gene
which this particular screen happens not to have measured is not penalised: the
pick is dropped from the effective budget rather than counted as a miss. No
label exists for it, so scoring it either way would be an artefact of library
composition rather than of the policy.

*Hallucinated and unspent picks are charged.* A pick that is not a real gene,
and a budget slot the policy simply failed to fill, both consume effective
budget without any chance of a hit. This is what stops a policy from gaming EF
by proposing three genes it is sure about and leaving the other 97 slots empty.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..assets import ensure_asset

__all__ = [
    "AcquisitionCounts",
    "classify_acquisitions",
    "enrichment_factor",
    "enrichment_factor_from_value",
    "adjusted_nauc",
    "fraction_of_hits",
    "shortfall",
    "percent_essential",
    "load_common_essentials",
    "DEPMAP_ESSENTIALS_CITATION",
]

# ``np.trapz`` was renamed to ``np.trapezoid`` in NumPy 2.0; assaybench supports
# both.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


# ---------------------------------------------------------------------------
# Acquisition bookkeeping
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AcquisitionCounts:
    """How a trajectory's picks break down against the library and universe.

    Attributes:
        n_in_library: ``N_L = |G ∩ L|``. Picks that were measured in this
            screen, and so are the only picks that can be hits.
        n_out_library_in_universe: Picks that are real genes (in ``U``) but
            outside this screen's library. Forgiven -- they consume no
            effective budget.
        n_out_of_universe: ``|G \\ U|``. Hallucinated or unrecognised symbols.
            Charged.
        n_unfilled: Budget slots the policy did not use, i.e.
            ``budget - len(acquired)``. Charged. Zero when no budget is given.
            May be negative if a policy over-supplies; see
            :meth:`effective_budget`.
        n_hits: ``|G ∩ H|``. In-library picks that were hits.
    """

    n_in_library: int
    n_out_library_in_universe: int
    n_out_of_universe: int
    n_unfilled: int
    n_hits: int

    @property
    def n_acquired(self) -> int:
        """``|G|`` -- every pick the policy actually made."""
        return (
            self.n_in_library
            + self.n_out_library_in_universe
            + self.n_out_of_universe
        )

    @property
    def effective_budget(self) -> int:
        """``n_eff = N_L + N_Ū`` -- the denominator's budget term.

        Forgiven picks are excluded; hallucinations and unspent slots are not.
        Equivalently ``budget - n_out_library_in_universe`` when a budget is
        given, which is how the quantity is written in AssayLoop Fig. 3B.
        """
        return self.n_in_library + self.n_out_of_universe + self.n_unfilled


def classify_acquisitions(
    acquired: Sequence[str],
    library: Iterable[str],
    hits: Iterable[str],
    universe: Iterable[str] | None = None,
    budget: int | None = None,
) -> AcquisitionCounts:
    """Break a trajectory's picks down into the terms EF needs.

    Args:
        acquired: Every gene the policy acquired, in order, concatenated across
            rounds. Order is irrelevant to the metrics in this module but is
            preserved so the same list can feed a trajectory metric such as
            nAUC. Picks are counted once each as they appear; a policy that can
            re-acquire the same gene should de-duplicate before calling.
        library: ``L``, the genes measured in this screen.
        hits: ``H``, the hits. Members outside ``library`` are ignored, since a
            gene with no label in this screen cannot be scored as a hit.
        universe: ``U``, the valid gene symbols. If ``None``, every pick outside
            the library is treated as a real-but-unmeasured gene and therefore
            forgiven -- i.e. hallucination is not penalised. Pass the approved
            HGNC symbol set to get the paper's behaviour.
        budget: The number of genes the policy was asked for over the whole
            trajectory. If ``None``, unspent budget is not charged.

    Returns:
        The populated :class:`AcquisitionCounts`.
    """
    library = set(library)
    hit_set = set(hits) & library
    universe_set = None if universe is None else set(universe)

    n_in_library = n_out_lib_in_universe = n_out_of_universe = n_hits = 0
    for gene in acquired:
        if gene in library:
            n_in_library += 1
            if gene in hit_set:
                n_hits += 1
        elif universe_set is None or gene in universe_set:
            n_out_lib_in_universe += 1
        else:
            n_out_of_universe += 1

    n_unfilled = 0 if budget is None else budget - len(acquired)
    return AcquisitionCounts(
        n_in_library=n_in_library,
        n_out_library_in_universe=n_out_lib_in_universe,
        n_out_of_universe=n_out_of_universe,
        n_unfilled=n_unfilled,
        n_hits=n_hits,
    )


# ---------------------------------------------------------------------------
# Enrichment factor
# ---------------------------------------------------------------------------


def enrichment_factor_from_value(
    n_hits: float,
    effective_budget: float,
    library_size: int,
    total_hits: int,
) -> float:
    """EF from pre-computed terms -- the arithmetic, with nothing else in it.

    ``EF = n_hits / (effective_budget · total_hits / library_size)``

    Args:
        n_hits: ``|G ∩ H|``.
        effective_budget: ``n_eff = N_L + N_Ū``.
        library_size: ``|L|``.
        total_hits: ``|H|``.

    Returns:
        The ratio of hits found to hits a uniformly-random policy would expect
        over the same effective budget. ``1.0`` is random, ``2.0`` is twice as
        good as random. ``0.0`` when the random expectation is zero or negative
        (an empty library, a screen with no hits, or a wholly forgiven
        trajectory), which is the only sensible value: no comparison is
        possible.
    """
    if library_size <= 0 or total_hits <= 0:
        return 0.0
    random_expected = effective_budget * total_hits / library_size
    if random_expected <= 0:
        return 0.0
    return float(n_hits / random_expected)


def enrichment_factor(
    acquired: Sequence[str],
    library: Iterable[str],
    hits: Iterable[str],
    universe: Iterable[str] | None = None,
    budget: int | None = None,
) -> float:
    """Adjusted hit enrichment factor (AssayLoop Eq. 3).

    The headline ASSAYBENCH-LOOP metric: how many more hits the policy found
    than uniformly-random acquisition would expect over the same effective
    budget.

    EF is comparable across screens in a way that raw hit counts are not. A
    policy finding 18 hits in a 10%-hit-rate screen (EF 1.8) is doing less well
    than one finding 4 hits in a 1%-hit-rate screen (EF 4.0), even though the
    first found four times as many.

    Args:
        acquired: Every gene acquired over the trajectory, in order.
        library: ``L``, the genes measured in this screen.
        hits: ``H``, the hits in the library.
        universe: ``U``, the valid gene symbols. ``None`` forgives every
            out-of-library pick, including hallucinations.
        budget: Genes the policy was asked for. ``None`` does not charge for
            unspent budget.

    Returns:
        EF as a ratio (not a percentage, and not bounded above by 1).
    """
    library = set(library)
    counts = classify_acquisitions(acquired, library, hits, universe, budget)
    return enrichment_factor_from_value(
        n_hits=counts.n_hits,
        effective_budget=counts.effective_budget,
        library_size=len(library),
        total_hits=len(set(hits) & library),
    )


# ---------------------------------------------------------------------------
# Adjusted normalised AUC
# ---------------------------------------------------------------------------


def adjusted_nauc(
    rounds: Sequence[Sequence[str]],
    library: Iterable[str],
    hits: Iterable[str],
    universe: Iterable[str] | None = None,
) -> float:
    """Adjusted normalised area under the hit-recovery curve (Fig. 3B).

    EF summarises the *endpoint* of a trajectory. nAUC summarises its *shape*:
    a policy that finds all its hits in round one scores higher than one that
    finds the same hits in the last round, because the point of an adaptive
    screen is to reach the hits early.

    The curve is built on the same effective axis EF's denominator uses::

        x_t = Σ_{i≤t} (N_L,i + N_Ū,i) / |L|      y_t = Σ_{i≤t} N_H,i / |H|

    so forgiven picks advance neither axis, and hallucinated picks advance
    ``x`` without advancing ``y``. The result is divided by the area under the
    oracle curve -- the policy that acquires every hit first -- integrated on
    the same x-grid.

    Two conventions differ from :func:`enrichment_factor`, deliberately:

    *One point per round, not per gene.* Policies are asked for a batch, not
    for a ranking within that batch, so the curve is sampled once per round.
    nAUC is therefore invariant to the order of genes inside a round, and two
    policies are only ever compared on orderings they actually chose.

    *Unspent budget is not charged.* The curve ends at the last gene acquired.
    Under-supply is already penalised by EF (through its effective budget) and
    reported directly by :func:`shortfall`; charging it a third time here would
    move all three columns together on a single deficiency.

    Args:
        rounds: The genes acquired in each round, in acquisition order --
            ``[["TP53", "MYC"], ["KRAS"], ...]``. Pass ``[acquired]`` for a
            single-shot policy; the result is then the area under a single
            straight segment.
        library: ``L``, the genes measured in this screen.
        hits: ``H``, the hits in the library.
        universe: ``U``, the valid gene symbols. ``None`` forgives every
            out-of-library pick, so hallucinations cost nothing.

    Returns:
        ``AUC / AUC_oracle``, in ``[0, 1]``. ``0.0`` for an empty library, a
        screen with no hits, or a trajectory that charged no budget at all
        (every pick forgiven), where no curve exists to score.
    """
    library = set(library)
    hit_set = set(hits) & library
    universe_set = None if universe is None else set(universe)

    library_size = len(library)
    total_hits = len(hit_set)
    if library_size <= 0 or total_hits <= 0:
        return 0.0

    effective = 0  # charged picks so far: in-library + hallucinated
    cumulative_hits = 0
    xs = [0.0]
    ys = [0.0]
    for batch in rounds:
        for gene in batch:
            if gene in library:
                effective += 1
                if gene in hit_set:
                    cumulative_hits += 1
            elif universe_set is not None and gene not in universe_set:
                effective += 1  # hallucinated: charged, and never a hit
            # else: real gene, unmeasured here -- forgiven, advances neither axis
        xs.append(effective / library_size)
        ys.append(cumulative_hits / total_hits)

    area = float(_trapezoid(ys, xs))
    # The oracle takes every hit first, so its curve climbs at slope
    # |L|/|H| until it saturates at 1. Integrating it on the *same* x-grid
    # keeps numerator and denominator at one granularity, rather than
    # comparing a per-round numerator against a continuous closed form.
    ys_oracle = [min(x * library_size / total_hits, 1.0) for x in xs]
    oracle_area = float(_trapezoid(ys_oracle, xs))
    return float(area / oracle_area) if oracle_area > 0 else 0.0


# ---------------------------------------------------------------------------
# Fraction of hits
# ---------------------------------------------------------------------------


def fraction_of_hits(
    acquired: Sequence[str],
    library: Iterable[str],
    hits: Iterable[str],
) -> float:
    """Share of the screen's hits the policy recovered.

    ``FH = |G ∩ H| / |H|``

    The plainest possible read on a trajectory: of everything there was to
    find, how much was found. Unlike EF it is not corrected for hit rate, so it
    is not comparable across screens -- a screen where 30% of the library are
    hits hands out a high FH to any policy. It is reported alongside EF because
    it is the number an experimentalist actually cares about once the screen is
    fixed.

    Args:
        acquired: Every gene acquired over the trajectory.
        library: ``L``, the genes measured in this screen.
        hits: ``H``, the hits in the library.

    Returns:
        A fraction in ``[0, 1]``. ``0.0`` for a screen with no hits.
    """
    library = set(library)
    hit_set = set(hits) & library
    if not hit_set:
        return 0.0
    found = {gene for gene in acquired if gene in hit_set}
    return len(found) / len(hit_set)


# ---------------------------------------------------------------------------
# Shortfall
# ---------------------------------------------------------------------------


def shortfall(acquired: Sequence[str], library: Iterable[str]) -> float:
    """Fraction of picks that fall outside the screen library (Eq. 6).

    ``SF = |G \\ L| / |G|``

    This counts both real-but-unmeasured genes and hallucinated symbols: it is
    a diagnostic for how much of the policy's budget produced no label at all,
    not a correctness measure. EF forgives the former and charges the latter;
    shortfall reports them together so the two effects can be read separately.

    Args:
        acquired: Every gene acquired over the trajectory.
        library: ``L``, the genes measured in this screen.

    Returns:
        A fraction in ``[0, 1]``. ``0.0`` for an empty trajectory.
    """
    library = set(library)
    if not acquired:
        return 0.0
    return sum(1 for gene in acquired if gene not in library) / len(acquired)


# ---------------------------------------------------------------------------
# Percentage of essential genes
# ---------------------------------------------------------------------------


def percent_essential(
    acquired: Sequence[str],
    library: Iterable[str],
    hits: Iterable[str],
    essentials: Iterable[str] | None = None,
) -> float:
    """Share of the hits found that are common-essential genes (Eq. 7).

    ``%ess = |G ∩ H ∩ E| / |G ∩ H|``

    Common essentials kill most cell lines, so they are hits in a great many
    screens regardless of the biology being interrogated. They are genuine
    hits, but a policy that scores well by naming them has learned which genes
    are always hits rather than which genes matter *here*. A low value means
    the hits found are specific to the screen.

    Args:
        acquired: Every gene acquired over the trajectory.
        library: ``L``, the genes measured in this screen.
        hits: ``H``, the hits in the library.
        essentials: ``E``. Defaults to :func:`load_common_essentials`.

    Returns:
        A fraction in ``[0, 1]``. ``0.0`` when no hits were found, since there
        is nothing to take a share of.
    """
    library = set(library)
    hit_set = set(hits) & library
    essential_set = (
        load_common_essentials() if essentials is None else set(essentials)
    )

    found = [gene for gene in acquired if gene in hit_set]
    if not found:
        return 0.0
    return sum(1 for gene in found if gene in essential_set) / len(found)


DEPMAP_ESSENTIALS_CITATION = (
    "Common-essential genes from Cancer Dependency Map (DepMap) Public 26Q1, "
    "Broad Institute. Use is governed by the DepMap portal terms. See "
    "assaybench/data/depmap/PROVENANCE.md."
)

_ESSENTIALS_ASSET = "depmap-common-essentials-26q1"
_essentials_cache: frozenset[str] | None = None


def load_common_essentials(path: str | Path | None = None) -> frozenset[str]:
    """Load the DepMap common-essential gene set.

    The file is a single column of ``SYMBOL (ENTREZ_ID)`` entries under an
    ``Essentials`` header; the Entrez id is stripped. By default AssayBench
    looks for the pinned DepMap Public 26Q1 file at
    ``$ASSAYBENCH_DEPMAP_PATH`` or in its asset cache and verifies its checksum.
    If it is absent, the exception explains how to download it manually from
    DepMap. AssayBench never downloads or hosts this file.

    Args:
        path: Read this file instead of the downloaded one. Use to pin a
            different DepMap release.

    Returns:
        The gene symbols, as a frozenset. The default set is cached after the
        first call; an explicit ``path`` is never cached.

    Raises:
        FileNotFoundError: If ``path`` is given and does not exist.
    """
    global _essentials_cache

    if path is None:
        if _essentials_cache is not None:
            return _essentials_cache
        text = ensure_asset(_ESSENTIALS_ASSET).read_text()
    else:
        text = Path(path).read_text()

    symbols = set()
    for row in csv.reader(text.splitlines()):
        if not row:
            continue
        cell = row[0].strip()
        if not cell or cell == "Essentials":
            continue
        symbols.add(cell.split(" (")[0].strip())

    result = frozenset(symbols)
    if path is None:
        _essentials_cache = result
    return result
