"""Rendering a :class:`~assaybench.core.StepRecord` history into a prompt.

An LLM policy in a sequential loop has to be told what it already measured.
How that block is written turns out to matter as much as the model: two
methods given the same observations in different formats are not being
compared on the same task. This is the one format, so they are.

The design choices, and why each one is load-bearing:

* **Group by round** (one ``StepRecord`` == one round). The LLM gets
  to see which hypothesis classes paid off per batch and how hit rate
  evolved, instead of an alphabetically-merged flat list.

* **Preserve within-batch acquisition order.** When the LLM did the ranking
  itself, this is its own ordering handed back to it, so it can see how
  well-calibrated its prior was. For other acquisitions the order is
  still meaningful (model score, agent rationale, etc.).

* **No truncation by default.** The block carries a "do NOT re-suggest"
  instruction, and truncating silently breaks that contract -- the model
  re-proposes a gene it cannot see it already measured, and the pick is
  wasted. ``max_each_round=None`` shows every gene; pass an integer only
  under a real prompt-budget constraint, accepting that cost.

* **Mark warm-start rounds.** Warm-start observations are random, not
  the result of any policy, so they should be flagged so the LLM
  doesn't mis-attribute their hit/non-hit signal to a previous decision.
"""

from __future__ import annotations

from ..core.types import Observation, StepRecord


def _is_hit(o: Observation) -> bool:
    """Treat the canonical ``label = {"hit": bool}`` schema as hit/miss."""
    if isinstance(o.label, dict):
        return bool(o.label.get("hit"))
    # Be defensive: a raw bool label also works.
    if isinstance(o.label, bool):
        return o.label
    return False


def format_history_by_round(
    history: list[StepRecord],
    *,
    intro: str = (
        "You have already validated {n} gene(s) in this screen, shown "
        "below grouped by round in acquisition order."
    ),
    do_not_resuggest: bool = True,
    max_each_round: int | None = None,
    include_cumulative_tail: bool = True,
    skip_empty_rounds: bool = True,
    include_labels: bool = True,
) -> str:
    """Render ``history`` as a round-by-round prompt block.

    Returns the empty string if there are no observations (so a caller
    can ``f"{prefix}{format_history_by_round(...)}"`` without worrying
    about extra whitespace on cold-start prompts).

    Args:
        history: the full AL history (every step the inner loop has
            recorded so far, in chronological order).
        intro: leading sentence; ``{n}`` is substituted with the total
            number of observations.
        do_not_resuggest: append "Do NOT re-suggest any of them." to
            the intro. Set ``False`` for acquisitions that filter via a
            candidate list and don't need the LLM to enforce de-dup.
        max_each_round: if not ``None``, truncate each round's hits and
            non-hits to this many genes and append "(+N more)". The
            default ``None`` shows everything — strongly preferred.
        include_cumulative_tail: append a final "Cumulative: X/Y hits"
            summary line for the LLM to read off easily. Implicitly
            disabled when ``include_labels=False``.
        skip_empty_rounds: drop rounds whose ``new_observations`` is
            empty (e.g. if a step recorded zero new genes for some
            reason). Default ``True``.
        include_labels: when ``True`` (default) split each round into
            hits vs non-hits with hit-rate header. When ``False``, list
            only the sampled genes per round without revealing whether
            they were hits. That is the blinded ablation: it separates
            what the model learns from the *labels* from what it gains
            merely by being told not to repeat itself.
    """
    if not history:
        return ""

    rounds_text: list[str] = []
    total_hits = 0
    total_obs = 0
    for r in history:
        obs = list(r.new_observations)
        if not obs and skip_empty_rounds:
            continue
        hits = [str(o.candidate) for o in obs if _is_hit(o)]
        misses = [str(o.candidate) for o in obs if not _is_hit(o)]
        total_hits += len(hits)
        total_obs += len(obs)

        is_warm = bool(
            (r.acquisition_trace or {}).get("warm_start")
        )

        if not include_labels:
            # Blinded mode: show the sampled-gene list only, no hit/miss
            # split, no hit rate. Order is preserved from the round so
            # the LLM still sees acquisition-order signal.
            symbols = [str(o.candidate) for o in obs]
            if max_each_round is not None and len(symbols) > max_each_round:
                shown = symbols[:max_each_round]
                more = f" (+{len(symbols) - max_each_round} more)"
            else:
                shown = symbols
                more = ""
            if is_warm:
                header = (
                    f"### Warm-start (random sample, {len(obs)} genes sampled)"
                )
            else:
                header = f"### Round {r.step} ({len(obs)} genes sampled)"
            rounds_text.append(
                f"{header}\n"
                f"  Sampled: {', '.join(shown) or '(none)'}{more}"
            )
            continue

        if is_warm:
            header = (
                f"### Warm-start (random sample, {len(obs)} genes)"
            )
        else:
            hit_rate = (len(hits) / len(obs)) * 100 if obs else 0.0
            header = (
                f"### Round {r.step} ({len(obs)} genes, "
                f"hit rate {len(hits)}/{len(obs)} = {hit_rate:.1f}%)"
            )

        if max_each_round is not None:
            h_show = hits[:max_each_round]
            m_show = misses[:max_each_round]
            h_more = (
                f" (+{len(hits) - len(h_show)} more)"
                if len(hits) > len(h_show) else ""
            )
            m_more = (
                f" (+{len(misses) - len(m_show)} more)"
                if len(misses) > len(m_show) else ""
            )
        else:
            h_show, m_show = hits, misses
            h_more = m_more = ""

        rounds_text.append(
            f"{header}\n"
            f"  Hits ({len(hits)}): "
            f"{', '.join(h_show) or '(none)'}{h_more}\n"
            f"  Non-hits ({len(misses)}): "
            f"{', '.join(m_show) or '(none)'}{m_more}"
        )

    if not rounds_text:
        return ""

    intro_line = intro.format(n=total_obs)
    if not include_labels:
        intro_line = (
            f"You have already sampled {total_obs} gene(s) in this "
            f"screen, shown below grouped by round in acquisition "
            f"order. Hit/non-hit labels are NOT shown."
        )
    if do_not_resuggest:
        intro_line = f"{intro_line} Do NOT re-suggest any of them."

    tail = ""
    if include_cumulative_tail and total_obs and include_labels:
        hit_pct = (total_hits / total_obs) * 100
        tail = (
            f"\n\n  Cumulative: {total_hits}/{total_obs} hits "
            f"({hit_pct:.1f}%)"
        )

    return (
        f"\n\n## Active-Learning History\n\n"
        f"{intro_line}\n\n"
        + "\n\n".join(rounds_text)
        + tail
        + "\n"
    )


__all__ = ["format_history_by_round"]
