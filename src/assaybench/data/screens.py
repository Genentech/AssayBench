"""Loading benchmark screens, and the candidate pool they define.

A screen set answers "which screens was this number averaged over"; the
manifests in :mod:`assaybench.data.screen_sets` record that. This module is
the other half: turning a manifest into loaded :class:`ScreenRecord` objects,
and turning a set of those into the gene pool a policy is allowed to acquire
from.

Both belong here rather than in a method repository. ``enrichment_factor``
takes a ``universe`` argument and its docstring says to pass the approved
symbol set to get the paper's behaviour -- a benchmark that asks for a pool it
gives you no way to build is incomplete. :func:`gene_universe` builds it.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from assaybench.data.screen_sets import (
    ScreenSetManifest,
    available_manifests,
    load_manifest,
    load_manifest_file,
)
from assaybench.dataset.dataset import AssayBenchDataset

log = logging.getLogger("assaybench.data.screens")


# ---------------------------------------------------------------------------
# ScreenRecord -- one AssayBench row, normalised into the shape the loop wants.
# ---------------------------------------------------------------------------


@dataclass
class ScreenRecord:
    dataset_name: str
    split: str
    organism: str
    gene_symbol_convention: str  # e.g. "HGNC", "MGI"
    phenotype: str
    cell_line: str
    cell_type: str
    library_type: str
    library_methodology: str
    direction_str: str
    condition_clause: str
    num_genes: int
    genes: list[str]
    relevance_scores: list[float]
    hits: list[bool]
    question: str = ""             # pre-rendered ranking prompt
    description: str = ""
    contrast_label: str = ""
    reverse: bool = False
    cleaned_phenotype: str = ""
    # Continuous relevance score for EVERY gene (not just hits): non-hits,
    # which carry a masked ``relevance_score`` of 0, are filled with their
    # ``combined_scores`` value. This is the regression target a model that
    # predicts a score per gene trains against. Empty when the source row
    # lacked ``combined_scores`` (falls back to ``relevance_scores`` in that
    # case -- see ``screen_from_example``).
    unmasked_relevance_scores: list[float] = field(default_factory=list)

    @property
    def total_hits(self) -> int:
        return int(sum(1 for h in self.hits if h))

    @property
    def gene_to_index(self) -> dict[str, int]:
        return {g: i for i, g in enumerate(self.genes)}

    def context(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "split": self.split,
            "organism": self.organism,
            "gene_symbol_convention": self.gene_symbol_convention,
            "phenotype": self.phenotype,
            "cell_line": self.cell_line,
            "cell_type": self.cell_type,
            "library_type": self.library_type,
            "library_methodology": self.library_methodology,
            "direction_str": self.direction_str,
            "condition_clause": self.condition_clause,
            "num_genes": self.num_genes,
            "total_hits": self.total_hits,
            "description": self.description,
            "contrast_label": self.contrast_label,
            "question": self.question,
            "cleaned_phenotype": self.cleaned_phenotype,
        }


def screen_from_example(example: dict[str, Any]) -> ScreenRecord:
    """Normalise one AssayBench example dict into a ScreenRecord."""
    genes = list(example.get("relevance_genes") or [])
    rs = list(example.get("relevance_scores") or [])
    hits = list(example.get("hit") or [])
    # Pad/truncate parallel arrays defensively (some legacy rows have
    # shorter hit arrays than relevance arrays).
    n = len(genes)
    rs = rs + [0.0] * (n - len(rs)) if len(rs) < n else rs[:n]
    hits = hits + [False] * (n - len(hits)) if len(hits) < n else hits[:n]
    hits = [bool(h) for h in hits]

    org = example.get("organism") or "Homo sapiens"
    conv = example.get("gene_symbol_convention") or (
        "HGNC" if "sapiens" in str(org).lower() or not org else "MGI"
    )

    # Unmasked relevance scores: fill the masked (==0) non-hit entries with
    # the gene's ``combined_scores`` value so every gene has a continuous
    # target (mirrors dataset.py). Falls back to the masked
    # ``relevance_scores`` when the source row has no ``combined_scores``.
    combined = example.get("combined_scores")
    if combined is not None:
        combined = list(combined)
        combined = (
            combined + [0.0] * (n - len(combined))
            if len(combined) < n else combined[:n]
        )
        unmasked = [
            float(combined[i]) if float(rs[i]) == 0.0 else float(rs[i])
            for i in range(n)
        ]
    else:
        unmasked = [float(x) for x in rs]

    return ScreenRecord(
        dataset_name=example.get("dataset_name") or example.get("screen_name", "unknown"),
        split=example.get("split", ""),
        organism=org,
        gene_symbol_convention=conv,
        phenotype=example.get("phenotype", "") or "",
        cell_line=example.get("cell_line", "") or "",
        cell_type=example.get("cell_type", "") or "",
        library_type=example.get("library_type", "") or "",
        library_methodology=example.get("library_methodology", "") or "",
        direction_str=example.get("direction_str", "") or "",
        condition_clause=example.get("condition_clause", "") or "",
        num_genes=int(example.get("num_genes", n) or n),
        genes=genes,
        relevance_scores=[float(x) for x in rs],
        hits=hits,
        unmasked_relevance_scores=unmasked,
        question=example.get("question", "") or "",
        description=example.get("description", "") or "",
        contrast_label=example.get("contrast_label", "") or "",
        reverse=bool(example.get("reverse", False)),
        cleaned_phenotype=example.get("cleaned_phenotype", "") or "",
    )


# ---------------------------------------------------------------------------
# Rendering the ranking prompt for rows that don't carry one
# ---------------------------------------------------------------------------


class _DefaultFormatDict(dict):
    """dict that returns ``"Not specified"`` for missing format keys.

    Lets ``str.format_map`` render a template even when a row is missing an
    optional placeholder field, instead of raising ``KeyError``.
    """

    def __missing__(self, key: str) -> str:  # noqa: D401
        return "Not specified"


def _render_question(item: dict[str, Any]) -> str:
    """Render the ``biogrid_ranking_prompt`` for a public row.

    Mirrors :mod:`assaybench.dataset.dataset`: load the
    ``biogrid_ranking_prompt`` template and ``.format(**item)`` it against the
    screen's structured fields (``cell_line``, ``library_type``,
    ``experimental_setup``, ``duration``, ``condition_clause``, ``phenotype``,
    ``significance_criteria``, ``ranking_rationale``, ``notes``, ...), which
    the public Genentech/assaybench rows carry but do NOT pre-render into a
    ``question`` field. Falls back to the bare template on any error.
    """
    from assaybench.utils.prompt_loaders import load_objective_prompt

    template = load_objective_prompt("biogrid_ranking_prompt")
    merged = dict(item)
    # A trailing period is stripped from phenotype for cleaner prose (the
    # template appends its own punctuation around it).
    phen = str(merged.get("phenotype") or "")
    if phen.endswith("."):
        merged["phenotype"] = phen[:-1]
    try:
        return template.format_map(_DefaultFormatDict(merged))
    except Exception:
        log.warning(
            "Failed to render biogrid_ranking_prompt for screen %r; "
            "using bare template.",
            item.get("dataset_name") or item.get("screen_name"),
        )
        return template


# ---------------------------------------------------------------------------
# Loading screens
# ---------------------------------------------------------------------------


def _source_is_public(source: dict[str, Any]) -> bool:
    """A public manifest's ``source`` block names the public Hub dataset.

    A manifest without one is an internal file, which this package cannot
    resolve.
    """
    return ((source or {}).get("dataset") or "").lower().endswith("/assaybench")


def _load_from_hub(
    *,
    wanted: set[str] | None,
    split_field: str = "yearfold0",
    split_value: str | list[str] = "test",
    strict: bool,
) -> list[ScreenRecord]:
    """Load screens from the public ``Genentech/assaybench`` dataset.

    The public dataset packs every benchmark entry into one ``train`` split
    with per-fold split labels in side columns (``yearfold0``,
    ``randomfold0``, ...). We filter on the requested ``split_field ==
    split_value`` and then by ``dataset_name``.

    ``split_value`` may be a list to match multiple folds (e.g.
    ``["test", "validation"]`` for combined splits).
    """
    ds = AssayBenchDataset(dataset_group="Genentech/assaybench", dataset_name="biogrid")
    ds.load()
    # The public dataset has a single 'train' arrow split; the actual fold
    # labels live in side columns.
    rows = ds.dataset["train"]

    accepted_values = {split_value} if isinstance(split_value, str) else set(split_value)

    out: list[ScreenRecord] = []
    seen: set[str] = set()
    for ex in rows:
        if ex.get(split_field) not in accepted_values:
            continue
        name = ex.get("dataset_name") or ex.get("screen_name")
        if not name or name in seen:
            continue
        if wanted is not None and name not in wanted:
            continue
        seen.add(name)
        ex2 = dict(ex)
        # Provenance label: which fold column and value this screen came in
        # on. Set unconditionally -- if the raw row happens to carry its own
        # ``split`` column it is about the arrow split, not the fold we asked
        # for, and would be misleading here.
        ex2["split"] = f"public_{split_field}_{split_value}"
        # Public rows don't carry a pre-rendered prompt; render the official
        # ``biogrid_ranking_prompt`` so an LLM policy uses the paper prompt
        # rather than a generic fallback block.
        if not ex2.get("question"):
            ex2["question"] = _render_question(ex2)
        out.append(screen_from_example(ex2))

    if wanted is not None:
        missing = wanted - seen
        if missing:
            msg = (
                f"{len(missing)} public screens not found "
                f"(first few: {sorted(missing)[:5]})"
            )
            if strict:
                raise ValueError(f"load_screens: {msg}")
            log.warning("load_screens: %s", msg)
    return out


def load_screens(
    screen_set: str | None = None,
    *,
    dataset_names: Iterable[str] | None = None,
    yaml_path: Path | str | None = None,
    split_field: str = "yearfold0",
    split_value: str | list[str] = "test",
    strict: bool | None = None,
) -> list[ScreenRecord]:
    """Load benchmark screens from the public ``Genentech/assaybench`` dataset.

    Four ways to say which screens, checked in this order:

    - ``dataset_names`` -- exactly these screens, resolved on the fold given
      by ``split_field``/``split_value``. Strict by default: a name that does
      not resolve raises, because a typo should not silently shrink the
      benchmark.
    - ``yaml_path`` -- a manifest file on disk.
    - ``screen_set`` -- the name of a manifest shipped with this package (see
      :func:`assaybench.data.screen_sets.available_manifests`, e.g.
      ``"assayloop-test"`` or ``"lopo-drug-test"``), or a path to one.
    - none of the above -- every screen on the fold given by
      ``split_field``/``split_value``.

    A manifest's own ``source`` block wins over the ``split_field`` and
    ``split_value`` arguments: which fold a curated set was drawn from is part
    of the set's definition.

    Args:
        screen_set: Shipped manifest name, or a path to a manifest YAML.
        dataset_names: Explicit screen names, bypassing manifests.
        yaml_path: Manifest file to read.
        split_field: Fold column to filter on. The public dataset carries
            ``yearfold0`` (the paper's temporal split) and ``randomfold0``.
        split_value: Fold value, or a list of them, to keep.
        strict: Whether an unresolved requested screen raises. Defaults to
            ``True`` for ``dataset_names`` and ``False`` for manifests, which
            are allowed to drift.

    Returns:
        The loaded screens, in dataset order.

    Raises:
        ValueError: If ``screen_set`` is neither a shipped manifest nor a
            readable file, or if a manifest names a non-public corpus.
    """
    manifest: ScreenSetManifest | None = None
    source: dict[str, Any] | None = None
    wanted: set[str] | None = None

    if dataset_names is not None:
        wanted = set(dataset_names)
        default_strict = True
    elif yaml_path is not None:
        manifest = load_manifest_file(yaml_path)
        default_strict = False
    elif screen_set is not None:
        if screen_set in available_manifests():
            manifest = load_manifest(screen_set)
        elif Path(screen_set).is_file():
            manifest = load_manifest_file(screen_set)
        else:
            raise ValueError(
                f"screen_set={screen_set!r} is neither a manifest shipped with "
                "assaybench nor a readable manifest path. Shipped manifests: "
                f"{', '.join(available_manifests())}."
            )
        default_strict = False
    else:
        # Whole fold, no curated subset.
        default_strict = False

    if manifest is not None:
        # Log which set is being loaded. Two runs of the same model can differ
        # only in the screen subset behind them, and that is invisible in the
        # resulting numbers.
        log.info("load_screens: %s", manifest.summary())
        source = manifest.source
        wanted = set(manifest.dataset_names)
        if not _source_is_public(source):
            raise ValueError(
                f"{yaml_path or screen_set} is not a public screen set: its "
                "'source' block does not name the Genentech/assaybench "
                "dataset. Internal screen manifests cannot be resolved."
            )
        split_field = source.get("split_field", split_field)
        split_value = source.get("split_value", split_value)

    strict = default_strict if strict is None else bool(strict)
    return _load_from_hub(
        wanted=wanted,
        split_field=split_field,
        split_value=split_value,
        strict=strict,
    )


# ---------------------------------------------------------------------------
# The candidate pool
# ---------------------------------------------------------------------------


def gene_universe(
    screens: Iterable[ScreenRecord], min_screen_freq: int = 2
) -> list[str]:
    """The candidate pool a full-genome run acquires from.

    A single screen's library is the wrong pool to compare an open-vocabulary
    policy against: an LLM names real genes that this particular library did
    not happen to measure, and charging those as misses measures the library,
    not the biology. So full-genome runs acquire from the union of the screens'
    libraries instead, and EF forgives the picks that land outside the one
    screen being scored.

    The union alone is too permissive. A gene measured in exactly one of twenty
    libraries is usually a pseudogene or an assembly artefact of that library,
    and leaving those in inflates the pool with candidates no method could
    reasonably be expected to know about. ``min_screen_freq`` drops them.

    ``min_screen_freq=2`` over the public 20-screen set gives the **f2
    universe**, 21,147 genes -- the pool every number in the AssayLoop paper is
    scored against. ``0`` disables the filter and returns the plain union,
    22,174.

    Args:
        screens: The screens whose libraries form the pool. For the paper's
            setting this is ``load_screens("assayloop-test")``.
        min_screen_freq: Keep a gene only if it appears in at least this many
            of ``screens``. ``0`` or ``1`` keeps everything.

    Returns:
        The pool, sorted, suitable for ``universe=`` on
        :func:`assaybench.enrichment_factor` or as a task's candidate pool.
    """
    freq: Counter[str] = Counter()
    for s in screens:
        # set() per screen: a library with a duplicated symbol must not count
        # twice toward that gene's screen frequency.
        for g in set(s.genes):
            freq[g] += 1
    return sorted(g for g, n in freq.items() if n >= min_screen_freq)


__all__ = [
    "ScreenRecord",
    "gene_universe",
    "load_screens",
    "screen_from_example",
]
