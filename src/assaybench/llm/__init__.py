"""Prompt-side plumbing for LLM policies: history in, gene list out.

Two pieces of unglamorous code that a benchmark cannot leave to each
implementer, because doing them differently changes the score without
changing the method. :func:`format_history_by_round` writes the observations
so far into a prompt; :func:`extract_gene_list` reads the model's answer back
into a ranking.

Neither talks to a model or to the network -- bring your own client. See
:mod:`assaybench.core` for the loop these plug into.
"""

from .history_format import format_history_by_round
from .parse_genes import extract_gene_list, organism_suffix, parse_json_genes

__all__ = [
    "format_history_by_round",
    "extract_gene_list",
    "parse_json_genes",
    "organism_suffix",
]
