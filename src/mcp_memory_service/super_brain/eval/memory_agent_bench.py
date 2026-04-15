"""MemoryAgentBench / LongMemEval-shaped runner.

Both benchmarks share the same end-to-end pattern: given an agent with
memory, pump it a long stream of facts / events / instructions, then ask
questions that require remembering earlier turns. We score by exact-match
or token-F1 against the gold answer.

This module stays benchmark-agnostic — concrete loaders (JSONL paths,
HuggingFace datasets) live at the caller site. We accept an iterable of
:class:`EvalCase`, a ``prime`` phase (store everything), then a ``query``
phase (retrieve + answer).
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Optional

from .harness import CaseResult, EvalCase, Runner, run_harness, summarize

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


def exact_match(case: EvalCase, answer: Optional[str]) -> Dict[str, float]:
    if case.gold is None or answer is None:
        return {}
    gold = case.gold.strip().lower()
    ans = answer.strip().lower()
    return {"em": 1.0 if gold == ans else 0.0}


def token_f1(case: EvalCase, answer: Optional[str]) -> Dict[str, float]:
    if case.gold is None or answer is None:
        return {}
    g_toks = set(_tokens(case.gold))
    a_toks = set(_tokens(answer))
    if not g_toks or not a_toks:
        return {"f1": 0.0}
    tp = len(g_toks & a_toks)
    if tp == 0:
        return {"f1": 0.0}
    precision = tp / len(a_toks)
    recall = tp / len(g_toks)
    return {"f1": 2 * precision * recall / (precision + recall)}


async def prime(server: Any, facts: Iterable[Dict[str, Any]]) -> int:
    """Store all priming facts as episodic memories. Returns the count written."""
    from ...models.memory import Memory
    import hashlib
    import time

    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    count = 0
    for f in facts:
        content = f["content"]
        ch = hashlib.sha256(content.encode("utf-8")).hexdigest()
        mem = Memory(
            content=content,
            content_hash=ch,
            tags=list(f.get("tags") or []) + ["sb:eval_fact"],
            memory_type="episodic",
            metadata={
                "tldr": (f.get("tldr") or content[:140]),
                "eval_case_id": f.get("case_id"),
                "created_at_iso": now_iso,
                "schema_version": "1",
                "source_type": "benchmark",
            },
        )
        try:
            ok, _ = await server.storage.store(mem)
            if ok:
                count += 1
        except Exception as exc:
            logger.debug("prime: store failed: %s", exc)
    return count


def simple_retrieval_runner(n_results: int = 5) -> Runner:
    """Default runner: retrieves top-k and concatenates their tldrs as the answer.

    Use this to sanity-check that the memory layer returns *something* relevant
    before wiring a downstream LLM answerer.
    """
    async def _runner(server: Any, case: EvalCase) -> Dict[str, Any]:
        storage = getattr(server, "storage", None)
        if storage is None:
            return {"answer": None, "metrics": {"retrieved": 0.0}}
        try:
            res = await storage.retrieve(case.prompt, n_results=n_results,
                                         tags=None, min_confidence=0.0)
        except Exception as exc:
            logger.debug("retrieval runner failed: %s", exc)
            return {"answer": None, "metrics": {"retrieved": 0.0}}
        tldrs = []
        for r in res:
            md = getattr(r.memory, "metadata", {}) or {}
            tldrs.append(md.get("tldr") or (getattr(r.memory, "content", "") or "")[:140])
        return {"answer": " | ".join(tldrs), "metrics": {"retrieved": float(len(res))}}
    return _runner


async def run(
    server: Any,
    cases: Iterable[EvalCase],
    runner: Optional[Runner] = None,
    concurrency: int = 4,
) -> Dict[str, Any]:
    """Run the query phase and return a summary dict + per-case results."""
    runner = runner or simple_retrieval_runner()
    scorers = [exact_match, token_f1]
    results: List[CaseResult] = []
    async for res in run_harness(server, cases, runner, scorers=scorers,
                                 concurrency=concurrency):
        results.append(res)
    return {"summary": summarize(results), "results": results}
