"""Reading a gene list back out of free-text LLM output.

Prompts ask for a comma-separated list of gene symbols; models return one
inside prose, inside a code fence, as a numbered list, or wrapped in JSON,
more or less at random. Parsing that back is not interesting work, but doing
it inconsistently is a real source of benchmark noise -- a method that loses
three of its hundred picks to a stray bullet character scores worse for a
reason that has nothing to do with its biology. These helpers exist so every
method is read the same way.

Accepted shapes: a JSON object with a gene-list key,
``<Final Answer>...</Final Answer>`` tags, numbered lists, mixed commas and
newlines, and the plain raw list. Symbols are upper-cased and de-duplicated
with order preserved, since order is the ranking.
"""

from __future__ import annotations

import json
import re

_FINAL_ANSWER_RE = re.compile(
    r"<Final Answer>(.*?)</Final Answer>", re.DOTALL | re.IGNORECASE
)
_NUM_PREFIX_RE = re.compile(r"^\s*\d+[.)\s]+")
_CODE_FENCE_RE = re.compile(r"```[a-zA-Z]*\n?|\n?```")


def parse_json_genes(text: str) -> list[str] | None:
    """If ``text`` is wrapped in a JSON object with a 'genes' key, return that.
    Returns ``None`` on parse failure.
    """
    if "{" not in text or "}" not in text:
        return None
    try:
        # Strip code fences if present.
        cleaned = _CODE_FENCE_RE.sub("", text)
        first_brace = cleaned.index("{")
        last_brace = cleaned.rindex("}")
        obj = json.loads(cleaned[first_brace : last_brace + 1])
        for key in ("genes", "predictions", "ranking", "top_genes", "selected_genes"):
            if key in obj and isinstance(obj[key], list):
                return [str(x).strip().upper() for x in obj[key]]
    except (ValueError, json.JSONDecodeError):
        return None
    return None


def extract_gene_list(text: str) -> list[str]:
    """Best-effort extraction of an ordered gene list from LLM output.

    Resolution order:
      1. JSON object with a 'genes' / 'predictions' / 'ranking' key.
      2. ``<Final Answer>`` block.
      3. The whole text.

    Then split on commas / newlines, strip numbering and quotes,
    upper-case, dedupe (preserving order), drop empties.
    """
    json_out = parse_json_genes(text)
    if json_out is not None:
        return _dedupe_clean(json_out)

    m = _FINAL_ANSWER_RE.search(text)
    body = m.group(1) if m else text

    # Replace newlines with commas to flatten lists.
    flat = body.replace("\n", ",").replace(";", ",")
    candidates = [g.strip() for g in flat.split(",")]
    return _dedupe_clean(candidates)


def _dedupe_clean(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for g in items:
        g = _NUM_PREFIX_RE.sub("", g)
        # Strip again after the punctuation strip: a "- TP53" bullet leaves a
        # leading space behind, and " TP53" matches no gene universe, so the
        # whole bulleted answer scores as out-of-library picks.
        g = g.strip().strip(",.;:'\"`*-").strip()
        if not g:
            continue
        # Drop anything that looks like a sentence rather than a symbol -- but
        # a lead-in ("Here are my picks: TP53") carries the first-ranked gene
        # in the same chunk, and dropping the chunk drops the model's top pick.
        if " " in g and len(g.split()) > 1:
            if ":" in g:
                g = g.rsplit(":", 1)[-1].strip().strip(",.;'\"`*-").strip()
            if not g or len(g.split()) > 1:
                continue
        g = g.upper()
        if g in seen:
            continue
        seen.add(g)
        out.append(g)
    return out


def organism_suffix(task_context: dict) -> str:
    """A sentence pinning which gene-symbol namespace the model should use.

    Append it to the prompt. Without it, models asked about a mouse screen
    return human symbols often enough to matter, and every one of those is
    scored as an out-of-library pick -- a parsing artefact that looks in the
    results table exactly like a bad prediction.
    """
    org = (task_context or {}).get("organism", "Homo sapiens") or "Homo sapiens"
    conv = (task_context or {}).get("gene_symbol_convention") or (
        "HGNC" if "sapiens" in str(org).lower() else "MGI"
    )
    return (
        f"All gene symbols must use the {conv} convention for {org} "
        f"(e.g. 'TP53' for HGNC, 'Trp53' for MGI). "
        f"Do not return symbols from another organism."
    )


__all__ = ["extract_gene_list", "parse_json_genes", "organism_suffix"]
