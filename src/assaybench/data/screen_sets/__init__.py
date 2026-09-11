"""Named screen-set manifests: which screens a reported number was computed on.

A benchmark result is only comparable if the set of screens behind it is the
same, and "the AssayBench test split" is not specific enough to pin that down --
papers subset it. These manifests are the subsets themselves, committed by name
and shipped with the package, so a number can be traced to a screen list instead
of to a sentence in a methods section.

What is here:

``assayloop-test`` / ``assayloop-validation``
    The 20-screen test and validation sets the AssayLoop paper reports on,
    drawn from the ``Genentech/assaybench`` biogrid ``yearfold0`` test and
    validation folds. **These are subsets, not the full folds** -- see each
    manifest's ``description`` and ``selection_criteria``.

``lopo-{drug,fitness,infection,molecular,trafficking}-{train,test}``
    Leave-one-phenotype-out folds: the ``test`` half is every screen with the
    named phenotype, the ``train`` half is everything else. Used to measure how
    a model transfers to a phenotype class it never saw.

A manifest stores screen *names*, not screen data. Resolving those names against
the Hub dataset is the caller's job -- this module does no network I/O:

    from assaybench.data.screen_sets import load_manifest

    m = load_manifest("assayloop-test")
    print(m.summary())          # says exactly which split is loaded
    names = m.dataset_names     # -> pass to your own loader

Asking for a name that does not exist raises :class:`UnknownScreenSet` and lists
what does. Nothing here falls back to a different set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

_PACKAGE = "assaybench.data.screen_sets"
_SUFFIX = ".yaml"


class UnknownScreenSet(KeyError):
    """Raised for a manifest name that is not shipped with the package."""


@dataclass(frozen=True)
class ScreenSetManifest:
    """One named screen set: its provenance, its criteria, and its screens.

    Attributes:
        name: The manifest name, e.g. ``"assayloop-test"``.
        description: Prose from the manifest saying what the set is and, where
            it is a subset of a larger split, that it is one.
        source: The corpus the names resolve against -- ``dataset``, ``config``,
            ``split_field``, and ``split_value`` (a string, or a list when the
            set spans folds).
        selection_criteria: How the set was chosen. Free-form and manifest
            specific; empty for sets defined by a single rule stated in
            ``description`` (the LOPO folds).
        screens: The screen entries verbatim, each carrying at least
            ``dataset_name``. Order is the manifest's order.
        path: Where the manifest was read from.
    """

    name: str
    description: str
    source: dict[str, Any]
    selection_criteria: dict[str, Any] = field(default_factory=dict)
    screens: list[dict[str, Any]] = field(default_factory=list)
    path: Path | None = None

    def __len__(self) -> int:
        return len(self.screens)

    @property
    def dataset_names(self) -> list[str]:
        """The ``dataset_name`` of every screen, in manifest order."""
        return [s["dataset_name"] for s in self.screens]

    @property
    def split_values(self) -> list[str]:
        """The fold value(s) this set was drawn from, always as a list."""
        value = self.source.get("split_value")
        if value is None:
            return []
        return [value] if isinstance(value, str) else list(value)

    def summary(self) -> str:
        """One line naming the set, its size, and the fold it came from.

        Worth logging at the top of an evaluation run. Two AssayBench results
        can differ because they used different subsets of the same split, and
        that is invisible in the numbers themselves.
        """
        src = self.source
        folds = "+".join(self.split_values) or "?"
        return (
            f"{self.name}: {len(self.screens)} screens from "
            f"{src.get('dataset', '?')} [{src.get('config', '?')}] "
            f"{src.get('split_field', '?')}={folds}"
        )


def available_manifests() -> list[str]:
    """Every manifest name shipped with the package, sorted."""
    root = files(_PACKAGE)
    return sorted(
        entry.name[: -len(_SUFFIX)]
        for entry in root.iterdir()
        if entry.name.endswith(_SUFFIX)
    )


def manifest_path(name: str) -> Path:
    """Filesystem path of a shipped manifest.

    Raises:
        UnknownScreenSet: If ``name`` is not shipped. The message lists the
            names that are; there is no nearest-match fallback, because
            silently evaluating on a neighbouring screen set produces a number
            that looks fine and is not comparable to anything.
    """
    root = files(_PACKAGE)
    candidate = root / f"{name}{_SUFFIX}"
    if not candidate.is_file():
        raise UnknownScreenSet(
            f"No screen set named {name!r}. Available: "
            f"{', '.join(available_manifests())}."
        )
    return Path(str(candidate))


def load_manifest(name: str) -> ScreenSetManifest:
    """Read a shipped manifest by name.

    Args:
        name: A name from :func:`available_manifests`, e.g.
            ``"assayloop-test"`` or ``"lopo-drug-train"``.

    Returns:
        The parsed :class:`ScreenSetManifest`, including its provenance and
        selection criteria -- not just the screen names -- so a caller can
        report which set it ran on.

    Raises:
        UnknownScreenSet: If ``name`` is not shipped.
    """
    path = manifest_path(name)
    doc = yaml.safe_load(path.read_text()) or {}
    return ScreenSetManifest(
        name=name,
        description=str(doc.get("description") or ""),
        source=dict(doc.get("source") or {}),
        selection_criteria=dict(doc.get("selection_criteria") or {}),
        screens=list(doc.get("screens") or []),
        path=path,
    )


def load_manifest_file(path: Path | str) -> ScreenSetManifest:
    """Read a manifest from an arbitrary path, for sets you curate yourself.

    The file must have the same shape as the shipped ones: ``description``,
    ``source``, optional ``selection_criteria``, and ``screens``. The manifest's
    :attr:`~ScreenSetManifest.name` is taken from the filename stem.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    doc = yaml.safe_load(path.read_text()) or {}
    return ScreenSetManifest(
        name=path.stem,
        description=str(doc.get("description") or ""),
        source=dict(doc.get("source") or {}),
        selection_criteria=dict(doc.get("selection_criteria") or {}),
        screens=list(doc.get("screens") or []),
        path=path,
    )


__all__ = [
    "ScreenSetManifest",
    "UnknownScreenSet",
    "available_manifests",
    "load_manifest",
    "load_manifest_file",
    "manifest_path",
]
