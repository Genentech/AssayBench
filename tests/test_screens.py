"""The candidate pool has one definition, and it lives with the metrics.

``enrichment_factor`` takes a ``universe`` argument, so the package has to
offer a way to build one. Before ``gene_universe`` moved here it was rebuilt
inline in four places in the AssayLoop repo, and one of those copies silently
omitted the frequency filter -- handing a ranker 22,174 candidates while every
published number was scored against 21,147. These tests pin the semantics.
"""

from __future__ import annotations

from dataclasses import dataclass

from assaybench import ScreenRecord, gene_universe, screen_from_example


@dataclass
class _Screen:
    """Just enough of a ScreenRecord for the helper: it only reads ``.genes``."""

    genes: list[str]


def test_keeps_genes_in_at_least_two_screens_by_default():
    screens = [_Screen(["A", "B", "C"]), _Screen(["B", "C", "D"]), _Screen(["C"])]
    # A and D appear once each; B twice; C three times.
    assert gene_universe(screens) == ["B", "C"]


def test_min_screen_freq_zero_is_the_plain_union():
    screens = [_Screen(["A", "B"]), _Screen(["B", "C"])]
    assert gene_universe(screens, min_screen_freq=0) == ["A", "B", "C"]


def test_result_is_sorted_and_deduplicated():
    screens = [_Screen(["Z", "A"]), _Screen(["A", "Z"])]
    assert gene_universe(screens) == ["A", "Z"]


def test_duplicate_symbol_within_one_library_counts_once():
    """A library that lists a symbol twice must not satisfy freq >= 2 alone.

    Duplicated symbols do occur, and counting both occurrences would admit a
    gene that only one screen ever measured.
    """
    screens = [_Screen(["A", "A", "B"]), _Screen(["B"])]
    assert gene_universe(screens) == ["B"]


def test_higher_cutoff_is_a_subset_of_a_lower_one():
    screens = [_Screen(["A", "B", "C"]), _Screen(["B", "C"]), _Screen(["C"])]
    f1 = gene_universe(screens, min_screen_freq=1)
    f2 = gene_universe(screens, min_screen_freq=2)
    f3 = gene_universe(screens, min_screen_freq=3)
    assert set(f3) <= set(f2) <= set(f1)
    assert (f1, f2, f3) == (["A", "B", "C"], ["B", "C"], ["C"])


def test_empty_input_is_an_empty_pool():
    assert gene_universe([]) == []


def test_gene_universe_accepts_real_screen_records():
    """The stub above is a convenience; the real record must work too."""
    screens = [
        screen_from_example(
            {"dataset_name": "s1", "relevance_genes": ["A", "B"],
             "relevance_scores": [1.0, 0.0], "hit": [True, False]}
        ),
        screen_from_example(
            {"dataset_name": "s2", "relevance_genes": ["B", "C"],
             "relevance_scores": [0.0, 1.0], "hit": [False, True]}
        ),
    ]
    assert all(isinstance(s, ScreenRecord) for s in screens)
    assert gene_universe(screens) == ["B"]
    assert gene_universe(screens, min_screen_freq=0) == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# screen_from_example
# ---------------------------------------------------------------------------


def test_screen_from_example_pads_short_hit_arrays():
    """Some rows carry fewer hit flags than genes; the arrays must stay parallel."""
    s = screen_from_example(
        {"dataset_name": "s", "relevance_genes": ["A", "B", "C"],
         "relevance_scores": [1.0], "hit": [True]}
    )
    assert len(s.genes) == len(s.relevance_scores) == len(s.hits) == 3
    assert s.hits == [True, False, False]
    assert s.total_hits == 1


def test_unmasked_scores_fill_non_hits_from_combined_scores():
    s = screen_from_example(
        {"dataset_name": "s", "relevance_genes": ["A", "B"],
         "relevance_scores": [2.0, 0.0], "hit": [True, False],
         "combined_scores": [9.0, 0.5]}
    )
    # The hit keeps its own score; the masked non-hit takes combined_scores.
    assert s.unmasked_relevance_scores == [2.0, 0.5]


def test_unmasked_scores_fall_back_when_no_combined_scores():
    s = screen_from_example(
        {"dataset_name": "s", "relevance_genes": ["A", "B"],
         "relevance_scores": [2.0, 0.0], "hit": [True, False]}
    )
    assert s.unmasked_relevance_scores == [2.0, 0.0]
