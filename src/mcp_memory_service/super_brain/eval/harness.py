"""Shared evaluation harness.

Benchmarks like MemoryAgentBench, LongMemEval, and RAGAS all look the
same from our point of view: run a set of cases against the server,
collect per-case answers + metrics, aggregate, and report. This module
is the skeleton that each concrete benchmark plugs into.

Intentional shape:
  * ``EvalCase`` — one input + optional gold answer + optional tags.
  * ``CaseResult`` — what came back, and any per-case metrics.
  * ``run_harness`` — async iterator-driven runner; never loads the full
    benchmark into memory.
  * ``summarize`` — aggregate, stable key set so dashboards don't drift.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EvalCase:
    id: str
    prompt: str
    gold: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CaseResult:
    case_id: str
    answer: Optional[str]
    latency_ms: float
    ok: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None


# Runner signature: given (server, case), return (answer, metrics).
Runner = Callable[[Any, EvalCase], Awaitable[Dict[str, Any]]]
# Scorer signature: given (case, answer), return a dict of metric name → float.
Scorer = Callable[[EvalCase, Optional[str]], Dict[str, float]]


async def run_harness(
    server: Any,
    cases: Iterable[EvalCase],
    runner: Runner,
    scorers: Optional[List[Scorer]] = None,
    concurrency: int = 4,
) -> AsyncIterator[CaseResult]:
    """Run each case through ``runner`` and yield :class:`CaseResult`.

    ``concurrency`` controls the bounded worker pool so we don't hammer
    the server when a benchmark has hundreds of cases.
    """
    sem = asyncio.Semaphore(max(1, concurrency))
    scorers = scorers or []

    async def _one(case: EvalCase) -> CaseResult:
        async with sem:
            t0 = time.perf_counter()
            try:
                out = await runner(server, case)
                answer = out.get("answer") if isinstance(out, dict) else None
                metrics = dict(out.get("metrics", {})) if isinstance(out, dict) else {}
                for scorer in scorers:
                    metrics.update(scorer(case, answer) or {})
                return CaseResult(
                    case_id=case.id,
                    answer=answer,
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                    ok=True,
                    metrics=metrics,
                )
            except Exception as exc:
                logger.warning("eval.harness: case %s failed: %s", case.id, exc)
                return CaseResult(
                    case_id=case.id,
                    answer=None,
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                    ok=False,
                    error=str(exc),
                )

    tasks = [asyncio.create_task(_one(c)) for c in cases]
    for fut in asyncio.as_completed(tasks):
        yield await fut


def summarize(results: List[CaseResult]) -> Dict[str, Any]:
    """Aggregate a list of case results into a stable summary dict."""
    n = len(results)
    if n == 0:
        return {"n": 0, "ok": 0, "failed": 0, "metrics": {}}
    ok = sum(1 for r in results if r.ok)
    metric_names = set()
    for r in results:
        metric_names.update(r.metrics.keys())

    metric_summary: Dict[str, Dict[str, float]] = {}
    for name in sorted(metric_names):
        vals = [r.metrics[name] for r in results if name in r.metrics]
        if not vals:
            continue
        vals_sorted = sorted(vals)
        mean = sum(vals) / len(vals)
        p50 = vals_sorted[len(vals_sorted) // 2]
        p95 = vals_sorted[int(0.95 * (len(vals_sorted) - 1))]
        metric_summary[name] = {
            "mean": mean,
            "p50": p50,
            "p95": p95,
            "min": min(vals),
            "max": max(vals),
            "n": float(len(vals)),
        }

    latencies = [r.latency_ms for r in results]
    return {
        "n": n,
        "ok": ok,
        "failed": n - ok,
        "latency_ms": {
            "mean": sum(latencies) / n,
            "p50": sorted(latencies)[n // 2],
            "p95": sorted(latencies)[int(0.95 * (n - 1))],
        },
        "metrics": metric_summary,
    }


def results_to_dicts(results: List[CaseResult]) -> List[Dict[str, Any]]:
    """Serialize results for JSON / JSONL logging."""
    return [asdict(r) for r in results]
