"""Tests for the shipped screen-set manifests.

Two things matter here and neither is about code. First, the manifests have to
actually be in the wheel -- they are data files, so a packaging mistake removes
them without breaking any import. Second, the sets have to stay what the paper
says they are: the AssayLoop numbers are averages over exactly these 20 screens,
so a manifest that quietly gains or loses one makes published numbers
irreproducible in a way no test of the metric code would catch.
"""

from __future__ import annotations

import pytest

from assaybench.data.screen_sets import (
    UnknownScreenSet,
    available_manifests,
    load_manifest,
    load_manifest_file,
    manifest_path,
)

# Every manifest that ships, and the screen count it must have.
EXPECTED_SIZES = {
    "assayloop-test": 20,
    "assayloop-validation": 20,
    "lopo-drug-train": 1322,
    "lopo-drug-test": 579,
    "lopo-fitness-train": 874,
    "lopo-fitness-test": 1027,
    "lopo-infection-train": 1741,
    "lopo-infection-test": 160,
    "lopo-molecular-train": 1810,
    "lopo-molecular-test": 91,
    "lopo-trafficking-train": 1857,
    "lopo-trafficking-test": 44,
}


def test_every_expected_manifest_ships():
    assert sorted(available_manifests()) == sorted(EXPECTED_SIZES)


@pytest.mark.parametrize("name", sorted(EXPECTED_SIZES))
def test_manifest_size_is_pinned(name):
    """A set that silently changes size invalidates every number reported on it."""
    assert len(load_manifest(name)) == EXPECTED_SIZES[name]


@pytest.mark.parametrize("name", sorted(EXPECTED_SIZES))
def test_manifest_is_self_describing(name):
    m = load_manifest(name)
    assert m.description, "a set with no description cannot be identified later"
    assert m.source["dataset"] == "Genentech/assaybench"
    assert m.source["config"] == "biogrid"
    assert m.source["split_field"] == "yearfold0"
    assert m.split_values, "must say which fold(s) it was drawn from"
    assert name in m.summary()
    assert str(len(m)) in m.summary()


@pytest.mark.parametrize("name", sorted(EXPECTED_SIZES))
def test_screen_names_are_unique(name):
    names = load_manifest(name).dataset_names
    assert len(set(names)) == len(names)


@pytest.mark.parametrize(
    "phenotype", ["drug", "fitness", "infection", "molecular", "trafficking"]
)
def test_lopo_folds_partition_the_pool(phenotype):
    """Leave-one-phenotype-out means exactly that: disjoint, and the same total.

    If a screen leaked from test into train, a fold model would be evaluated on
    a phenotype it had in fact seen -- the one thing the experiment is designed
    to rule out.
    """
    train = set(load_manifest(f"lopo-{phenotype}-train").dataset_names)
    test = set(load_manifest(f"lopo-{phenotype}-test").dataset_names)
    assert not (train & test)
    assert len(train | test) == 1901


def test_curated_sets_are_disjoint():
    """The paper selects on validation and reports on test; overlap would leak."""
    test = set(load_manifest("assayloop-test").dataset_names)
    val = set(load_manifest("assayloop-validation").dataset_names)
    assert not (test & val)


def test_unknown_name_raises_and_lists_alternatives():
    """No nearest-match fallback: a wrong set scores fine and means nothing."""
    with pytest.raises(UnknownScreenSet) as exc:
        load_manifest("public_screens")
    assert "assayloop-test" in str(exc.value)


def test_load_manifest_file_round_trips():
    from_name = load_manifest("assayloop-test")
    from_path = load_manifest_file(manifest_path("assayloop-test"))
    assert from_path.name == from_name.name
    assert from_path.dataset_names == from_name.dataset_names
