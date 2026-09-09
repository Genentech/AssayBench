"""How an LLM policy talks to the sequential loop: history in, gene list out.

Two functions decide this, and both live in :mod:`assaybench.llm` rather than in
each method, because a method that writes its history differently or reads its
own output more leniently is not being scored on the same task as the others.

Runs offline against a made-up two-round history -- no dataset, no API key:

    python examples/sequential_llm_prompt.py
"""

from assaybench.core.types import Observation, StepRecord
from assaybench.llm import extract_gene_list, format_history_by_round, organism_suffix


def round_record(step, picks, hits, warm_start=False):
    """One loop step: what was acquired, and what came back."""
    return StepRecord(
        step=step,
        acquired_batch=list(picks),
        new_observations=[
            Observation(candidate=g, label={"hit": g in hits}) for g in picks
        ],
        acquisition_trace={"warm_start": True} if warm_start else {},
    )


history = [
    round_record(1, ["ATM", "CHEK2", "MDM2", "RPL5"], {"MDM2"}, warm_start=True),
    round_record(2, ["TP53BP1", "USP7", "PPM1D", "CDKN1A"], {"USP7", "PPM1D"}),
]

# --- 1. Render what the model already knows -------------------------------
# Grouped by round, in acquisition order, with nothing truncated: the block
# tells the model not to re-suggest these, and hiding some of them breaks that.
prompt = (
    "You are selecting genes for a genome-wide CRISPR screen for regulators "
    "of p53 stability.\n"
    + organism_suffix({"organism": "Homo sapiens"})
    + format_history_by_round(history)
    + "\nPropose 4 more genes as a comma-separated list."
)
print(prompt)

# The blinded variant drops the labels but keeps the "already measured" list --
# it separates what the model learns from the results from what it gains just
# by not repeating itself.
print("\n--- blinded ---")
print(format_history_by_round(history, include_labels=False))

# --- 2. Read the reply ----------------------------------------------------
# All of these are answers real models give to the prompt above, and all of
# them have to parse to the same ranking.
replies = [
    "BRCA1, USP28, TRIM24, HUWE1",
    "1. BRCA1\n2. USP28\n3. TRIM24\n4. HUWE1",
    "- BRCA1\n- USP28\n- TRIM24\n- HUWE1",
    '```json\n{"genes": ["BRCA1", "USP28", "TRIM24", "HUWE1"]}\n```',
    "<Final Answer>BRCA1, USP28, TRIM24, HUWE1</Final Answer>",
]
print("\n--- parsed ---")
for reply in replies:
    print(f"{reply!r:>62}  ->  {extract_gene_list(reply)}")
