"""Tests for the prompt-side plumbing in :mod:`assaybench.llm`.

These two functions decide how a method's output is read and how its history is
written, so a change in either shifts scores without touching a model. That is
the reason they are in the package at all, and the reason their behaviour is
pinned here rather than left to each implementation.
"""

from __future__ import annotations

import pytest

from assaybench.core.types import Observation, StepRecord
from assaybench.llm import (
    extract_gene_list,
    format_history_by_round,
    organism_suffix,
    parse_json_genes,
)


def _step(step, genes, hits, warm=False):
    return StepRecord(
        step=step,
        acquired_batch=list(genes),
        new_observations=[
            Observation(candidate=g, label={"hit": bool(h)})
            for g, h in zip(genes, hits)
        ],
        acquisition_trace={"warm_start": True} if warm else {},
    )


# --- extract_gene_list -----------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "TP53, BRCA1, EGFR",
        "TP53\nBRCA1\nEGFR",
        "1. TP53\n2. BRCA1\n3. EGFR",
        "- TP53\n- BRCA1\n- EGFR",
        "Here are my picks: TP53; BRCA1; EGFR",
        "<Final Answer>TP53, BRCA1, EGFR</Final Answer>",
        '{"genes": ["TP53", "BRCA1", "EGFR"]}',
        '```json\n{"predictions": ["tp53", "brca1", "egfr"]}\n```',
    ],
)
def test_every_common_shape_reads_the_same(text):
    """The point of the module: format must not change the score."""
    assert extract_gene_list(text) == ["TP53", "BRCA1", "EGFR"]


def test_order_is_preserved_because_order_is_the_ranking():
    assert extract_gene_list("EGFR, TP53, BRCA1") == ["EGFR", "TP53", "BRCA1"]


def test_duplicates_collapse_to_first_position():
    assert extract_gene_list("TP53, BRCA1, TP53") == ["TP53", "BRCA1"]


def test_prose_is_dropped_not_ranked():
    out = extract_gene_list("I would pick these genes, TP53, BRCA1")
    assert out == ["TP53", "BRCA1"]


def test_parse_json_genes_returns_none_when_not_json():
    assert parse_json_genes("TP53, BRCA1") is None


def test_json_wins_over_a_trailing_final_answer_block():
    text = '{"genes": ["TP53"]}\n<Final Answer>BRCA1</Final Answer>'
    assert extract_gene_list(text) == ["TP53"]


# --- organism_suffix -------------------------------------------------------


def test_organism_suffix_defaults_to_human():
    assert "HGNC" in organism_suffix({})


def test_organism_suffix_infers_mgi_for_mouse():
    suffix = organism_suffix({"organism": "Mus musculus"})
    assert "MGI" in suffix and "Mus musculus" in suffix


def test_explicit_convention_overrides_the_guess():
    assert "RGD" in organism_suffix(
        {"organism": "Rattus norvegicus", "gene_symbol_convention": "RGD"}
    )


# --- format_history_by_round -----------------------------------------------


def test_empty_history_renders_nothing():
    assert format_history_by_round([]) == ""


def test_rounds_stay_separate_and_carry_their_hit_rate():
    out = format_history_by_round([
        _step(1, ["A", "B"], [1, 0]),
        _step(2, ["C", "D"], [0, 0]),
    ])
    assert "### Round 1 (2 genes, hit rate 1/2 = 50.0%)" in out
    assert "### Round 2 (2 genes, hit rate 0/2 = 0.0%)" in out
    assert "Cumulative: 1/4 hits" in out


def test_warm_start_rounds_are_labelled_as_random():
    out = format_history_by_round([_step(1, ["A", "B"], [1, 0], warm=True)])
    assert "Warm-start (random sample" in out
    assert "### Round 1" not in out


def test_nothing_is_truncated_by_default():
    """Truncating silently breaks the "do not re-suggest" contract in the intro."""
    genes = [f"G{i}" for i in range(200)]
    out = format_history_by_round([_step(1, genes, [0] * 200)])
    assert "G199" in out
    assert "more)" not in out


def test_truncation_is_opt_in_and_says_how_much_it_hid():
    genes = [f"G{i}" for i in range(10)]
    out = format_history_by_round([_step(1, genes, [0] * 10)], max_each_round=3)
    assert "(+7 more)" in out


def test_blinded_mode_hides_labels_and_the_cumulative_tail():
    out = format_history_by_round(
        [_step(1, ["A", "B"], [1, 0])], include_labels=False
    )
    assert "Sampled: A, B" in out
    assert "hit rate" not in out
    assert "Cumulative" not in out


def test_do_not_resuggest_can_be_turned_off():
    history = [_step(1, ["A"], [1])]
    assert "Do NOT re-suggest" in format_history_by_round(history)
    assert "Do NOT re-suggest" not in format_history_by_round(
        history, do_not_resuggest=False
    )
