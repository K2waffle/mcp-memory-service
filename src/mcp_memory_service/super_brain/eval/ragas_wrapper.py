"""Local, dependency-free RAGAS-style scorers.

RAGAS proper needs an LLM-as-judge. We ship cheap, local approximations
that capture the *shape* of the same metrics so the harness runs offline
and dashboards show real trends. Swap to the real ragas package by
writing a drop-in scorer that keeps the same metric names.

Metrics implemented:
  * faithfulness_lex  — token-overlap fraction between answer and contexts.
  * answer_relevance  — token-overlap fraction between answer and question.
  * context_precision — how many contexts have at least one answer token.
  * context_recall    — how many answer tokens appear in any context.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set:
    return set(_TOKEN_RE.findall((text or "").lower()))


def faithfulness_lex(answer: str, contexts: Sequence[str]) -> float:
    """Fraction of answer tokens that appear in any retrieved context."""
    a = _tokens(answer)
    if not a:
        return 0.0
    ctx = set()
    for c in contexts:
        ctx |= _tokens(c)
    return len(a & ctx) / max(1, len(a))


def answer_relevance(answer: str, question: str) -> float:
    """Fraction of question tokens the answer mentions."""
    q = _tokens(question)
    a = _tokens(answer)
    if not q:
        return 0.0
    return len(q & a) / max(1, len(q))


def context_precision(answer: str, contexts: Sequence[str]) -> float:
    """Fraction of contexts that contain at least one answer token."""
    if not contexts:
        return 0.0
    a = _tokens(answer)
    if not a:
        return 0.0
    hits = 0
    for c in contexts:
        if _tokens(c) & a:
            hits += 1
    return hits / len(contexts)


def context_recall(answer: str, contexts: Sequence[str]) -> float:
    """Fraction of answer tokens that appear across all contexts."""
    return faithfulness_lex(answer, contexts)  # same operational definition here


def score_case(
    question: str,
    answer: Optional[str],
    contexts: Sequence[str],
) -> Dict[str, float]:
    """Compute all RAGAS-ish metrics for one (q, a, contexts) tuple."""
    if not answer:
        return {
            "faithfulness_lex": 0.0,
            "answer_relevance": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
        }
    return {
        "faithfulness_lex": faithfulness_lex(answer, contexts),
        "answer_relevance": answer_relevance(answer, question),
        "context_precision": context_precision(answer, contexts),
        "context_recall": context_recall(answer, contexts),
    }


def make_scorer(contexts_key: str = "contexts"):
    """Factory: returns a harness-compatible scorer pulling contexts from case.meta."""
    def _scorer(case: Any, answer: Optional[str]) -> Dict[str, float]:
        meta = getattr(case, "meta", {}) or {}
        contexts = meta.get(contexts_key) or []
        q = getattr(case, "prompt", "") or ""
        return score_case(q, answer, contexts)
    return _scorer
