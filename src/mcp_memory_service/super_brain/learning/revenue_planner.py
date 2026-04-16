"""Economic Monte Carlo Tree Search (MCTS) revenue planning.

Applies the chess-AI planning loop to revenue decisions:

  1. Current state  = your assets (live infra, built tools, content, budget)
  2. Legal moves    = revenue actions available from current state
  3. Rollout        = simulate N-step downstream effects via value formula
  4. Backprop       = credit through parent chain with confidence decay
  5. Selection      = highest expected_value / time_to_first_dollar

The move library is grounded in actual super-brain infrastructure so the
planner recommends actions the operator can immediately execute.

Key types
---------
RevenueMove  — atomic revenue action with cost, prerequisites, and estimates
MCTSNode     — tree node wrapping a move; tracks visits + cumulative value
               and exposes the UCB1 score for balanced exploration/exploitation

Public API
----------
plan_revenue_path(server, current_assets, target_monthly, ...) -> dict
next_best_move(server) -> dict
"""
from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------

@dataclass
class RevenueMove:
    """Atomic revenue action with cost, prerequisites, and projections."""
    action: str
    time_cost_hours: float
    prerequisites: List[str]          # action names that must precede this
    expected_revenue_30d: float       # USD
    automation_possible: bool
    confidence: float                 # 0–1


@dataclass
class MCTSNode:
    """A node in the MCTS tree representing one revenue move."""
    move: RevenueMove
    parent: Optional["MCTSNode"]
    children: List["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    total_value: float = 0.0

    @property
    def ucb1_score(self) -> float:
        """Upper Confidence Bound 1 — balances exploitation and exploration.

        Returns inf for unvisited nodes so they are always expanded first.
        """
        if self.visits == 0:
            return float("inf")
        if self.parent is None or self.parent.visits == 0:
            return self.total_value / self.visits
        exploit = self.total_value / self.visits
        explore = math.sqrt(2.0 * math.log(self.parent.visits) / self.visits)
        return exploit + explore


# ---------------------------------------------------------------------------
# Known move library — grounded in actual super-brain infrastructure
# ---------------------------------------------------------------------------

KNOWN_MOVES: List[RevenueMove] = [
    RevenueMove("post_launch_tweet",        0.1,  [],                             200,    True,  0.70),
    RevenueMove("post_reddit_locallama",    0.2,  ["post_launch_tweet"],          800,    True,  0.65),
    RevenueMove("deploy_api_metering",      4.0,  [],                             2000,   True,  0.50),
    RevenueMove("launch_affiliate_program", 2.0,  ["post_launch_tweet"],          500,    True,  0.40),
    RevenueMove("automated_weekly_digest",  3.0,  ["deploy_api_metering"],        5000,   True,  0.45),
    RevenueMove("domain_research_product",  40.0, ["deploy_api_metering"],        20000,  True,  0.30),
    RevenueMove("developer_docs_page",      6.0,  [],                             3000,   True,  0.55),
    RevenueMove("mcp_marketplace_listing",  1.0,  ["developer_docs_page"],        1500,   True,  0.60),
    RevenueMove("verify_intelligence_domain", 20.0, ["domain_research_product"],  50000,  True,  0.25),
]

# Index by action name for fast prerequisite checking.
_MOVE_INDEX: Dict[str, RevenueMove] = {m.action: m for m in KNOWN_MOVES}


# ---------------------------------------------------------------------------
# Value formula
# ---------------------------------------------------------------------------

_AUTOMATION_BONUS = 1.5


def _move_value(move: RevenueMove) -> float:
    """Rollout value estimate for a single move.

    value = expected_revenue_30d * confidence * automation_bonus / (time_cost_hours + 1)

    The +1 prevents division-by-zero for instant moves and keeps units in
    revenue-per-effective-hour.
    """
    bonus = _AUTOMATION_BONUS if move.automation_possible else 1.0
    return (move.expected_revenue_30d * move.confidence * bonus) / (move.time_cost_hours + 1.0)


# ---------------------------------------------------------------------------
# Prerequisite graph helpers
# ---------------------------------------------------------------------------

def _prereqs_met(action: str, completed: set) -> bool:
    """Return True if all prerequisites for *action* are in *completed*."""
    move = _MOVE_INDEX.get(action)
    if move is None:
        return False
    return all(p in completed for p in move.prerequisites)


def _legal_moves(completed: set) -> List[RevenueMove]:
    """Return all moves whose prerequisites are fully satisfied."""
    return [m for m in KNOWN_MOVES if m.action not in completed and _prereqs_met(m.action, completed)]


# ---------------------------------------------------------------------------
# MCTS core
# ---------------------------------------------------------------------------

_CONFIDENCE_DECAY = 0.9   # per tree level during backpropagation


def _rollout(node: MCTSNode, depth: int = 5) -> float:
    """Simulate a random playout from *node* and return cumulative value.

    At each step we pick one of the legal successor moves at random
    (uniform), add its value (decayed by depth), and advance state.
    """
    completed: set = set()
    # Walk up to root to reconstruct completed set at this node.
    current = node
    while current is not None:
        completed.add(current.move.action)
        current = current.parent

    total = _move_value(node.move)
    decay = _CONFIDENCE_DECAY

    for _ in range(depth - 1):
        candidates = _legal_moves(completed)
        if not candidates:
            break
        chosen = random.choice(candidates)
        total += _move_value(chosen) * decay
        decay *= _CONFIDENCE_DECAY
        completed.add(chosen.action)

    return total


def _backpropagate(node: MCTSNode, value: float) -> None:
    """Walk up the tree adding *value* (decayed by level) to each ancestor."""
    current: Optional[MCTSNode] = node
    depth = 0
    while current is not None:
        current.visits += 1
        current.total_value += value * (_CONFIDENCE_DECAY ** depth)
        current = current.parent
        depth += 1


def _expand(node: MCTSNode, completed: set) -> List[MCTSNode]:
    """Attach child nodes for every legal move reachable from *node*."""
    child_completed = completed | {node.move.action}
    children = []
    for move in _legal_moves(child_completed):
        child = MCTSNode(move=move, parent=node)
        node.children.append(child)
        children.append(child)
    return children


def _select(root: MCTSNode, completed: set) -> MCTSNode:
    """Descend the tree following the highest UCB1 score until a leaf."""
    current = root
    visited: set = set(completed)
    visited.add(root.move.action)

    while current.children:
        # Pick the child with the highest UCB1 score.
        current = max(current.children, key=lambda n: n.ucb1_score)
        visited.add(current.move.action)

    return current


def _best_sequence(roots: List[MCTSNode], max_steps: int = 5) -> List[RevenueMove]:
    """Greedily reconstruct the best move sequence from root nodes.

    At each step, pick the root node with the highest average value, then
    descend its subtree following highest average value until *max_steps*
    or no more children exist.
    """
    if not roots:
        return []
    # Start from the highest-value root.
    current = max(roots, key=lambda n: (n.total_value / n.visits if n.visits else 0.0))
    sequence: List[RevenueMove] = [current.move]

    for _ in range(max_steps - 1):
        if not current.children:
            break
        current = max(
            current.children,
            key=lambda n: (n.total_value / n.visits if n.visits else 0.0),
        )
        sequence.append(current.move)

    return sequence


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def plan_revenue_path(
    server: Any,
    current_assets: Dict[str, Any],
    target_monthly: float = 5000.0,
    horizon_days: int = 30,
    n_simulations: int = 50,
) -> Dict[str, Any]:
    """Run MCTS to find the highest-expected-value revenue path.

    Parameters
    ----------
    server:
        MCP server instance (used for optional memory retrieval in future
        extensions; not strictly required for the pure MCTS pass).
    current_assets:
        Description of current state, e.g. {"deployed": ["api_metering"]}.
        The "deployed" list is used to pre-mark prerequisites as completed.
    target_monthly:
        Monthly revenue target in USD.  Used to annotate output and compute
        how many months the projected path covers.
    horizon_days:
        Planning horizon (informational; currently 30 is assumed in estimates).
    n_simulations:
        Number of MCTS rollout iterations.  50 is enough for this move
        library size; increase for richer exploration.

    Returns
    -------
    dict with keys:
        recommended_sequence  — ordered list of RevenueMove dicts
        expected_30d_revenue  — float USD
        confidence            — float 0–1
        reasoning             — human-readable explanation
        automated_moves       — subset of action names that need no human input
    """
    t_start = time.monotonic()

    # --- Build initial completed set from current_assets ---
    completed: set = set()
    for item in current_assets.get("deployed", []):
        # Accept either action names or partial matches.
        for move in KNOWN_MOVES:
            if item in move.action or move.action in item:
                completed.add(move.action)
    logger.debug("plan_revenue_path: pre-completed=%s", completed)

    # --- Create root nodes for every immediately legal move ---
    root_moves = _legal_moves(completed)
    if not root_moves:
        return {
            "recommended_sequence": [],
            "expected_30d_revenue": 0.0,
            "confidence": 0.0,
            "reasoning": "No legal moves available given current assets.",
            "automated_moves": [],
        }

    roots: List[MCTSNode] = [MCTSNode(move=m, parent=None) for m in root_moves]

    # --- MCTS iterations ---
    for iteration in range(n_simulations):
        # Pick a starting root (round-robin to ensure all roots get visits).
        root = roots[iteration % len(roots)]

        # Selection
        leaf = _select(root, completed | {root.move.action})

        # Expansion (expand on first visit)
        if leaf.visits == 0:
            leaf_completed = completed | {root.move.action}
            _expand(leaf, leaf_completed)

        # Rollout
        value = _rollout(leaf)

        # Backpropagation
        _backpropagate(leaf, value)

    # --- Extract best sequence ---
    sequence = _best_sequence(roots)

    # --- Compute aggregate stats ---
    if sequence:
        total_rev = sum(m.expected_revenue_30d * m.confidence for m in sequence)
        avg_conf = sum(m.confidence for m in sequence) / len(sequence)
        automated = [m.action for m in sequence if m.automation_possible]
        total_hours = sum(m.time_cost_hours for m in sequence)
    else:
        total_rev = 0.0
        avg_conf = 0.0
        automated = []
        total_hours = 0.0

    elapsed = time.monotonic() - t_start

    # --- Build reasoning narrative ---
    steps_text = " → ".join(m.action for m in sequence) if sequence else "(none)"
    gap = max(0.0, target_monthly - total_rev)
    reasoning_parts = [
        f"MCTS ran {n_simulations} simulations across {len(roots)} root moves in "
        f"{elapsed:.2f}s.",
        f"Best sequence ({len(sequence)} steps): {steps_text}.",
        f"Projected 30-day revenue: ${total_rev:,.0f} "
        f"({'above' if gap == 0.0 else 'below'} ${target_monthly:,.0f} target"
        + (f" by ${gap:,.0f}" if gap > 0 else "") + ").",
        f"Estimated time cost: {total_hours:.1f} hours.",
        f"{len(automated)} of {len(sequence)} moves are fully automatable.",
    ]
    reasoning = "  ".join(reasoning_parts)

    return {
        "recommended_sequence": [
            {
                "action": m.action,
                "time_cost_hours": m.time_cost_hours,
                "expected_revenue_30d": m.expected_revenue_30d,
                "confidence": m.confidence,
                "automation_possible": m.automation_possible,
                "prerequisites": m.prerequisites,
            }
            for m in sequence
        ],
        "expected_30d_revenue": round(total_rev, 2),
        "confidence": round(avg_conf, 4),
        "reasoning": reasoning,
        "automated_moves": automated,
    }


async def next_best_move(server: Any) -> Dict[str, Any]:
    """Return the single highest-gradient action to take RIGHT NOW.

    Pulls stored opportunity_rank memories (tagged entity:opportunities),
    extracts any action names that match the known move library, marks those
    as context for the MCTS evaluation, then returns the top unblocked move
    ranked by raw move value (expected_revenue_30d * confidence *
    automation_bonus / (time_cost_hours + 1)).

    Falls back to pure KNOWN_MOVES ranking when no relevant memories exist.
    """
    # --- Attempt to hydrate completed moves from stored opportunities ---
    completed: set = set()
    opportunity_context: List[str] = []

    try:
        storage = getattr(server, "storage", None)
        if storage is not None:
            results = await storage.retrieve(
                "revenue opportunity deployment",
                n_results=20,
                tags=["entity:opportunities"],
                min_confidence=0.0,
            )
            for r in results or []:
                mem = getattr(r, "memory", r)
                content = getattr(mem, "content", "") or ""
                tldr = (getattr(mem, "metadata", {}) or {}).get("tldr", "") or ""
                combined = (content + " " + tldr).lower()
                # Check if the memory references any known action (marks it done).
                for move in KNOWN_MOVES:
                    # A memory mentioning an action as "deployed" or "live" or
                    # "completed" implies the move is already executed.
                    action_slug = move.action.replace("_", " ")
                    if action_slug in combined:
                        for signal in ("deployed", "live", "completed", "done", "launched"):
                            if signal in combined:
                                completed.add(move.action)
                                break
                opportunity_context.append(tldr or content[:80])
    except Exception as exc:
        logger.debug("next_best_move: memory retrieval failed (non-fatal): %s", exc)

    # --- Rank every unblocked move by immediate value ---
    candidates = _legal_moves(completed)
    if not candidates:
        # All moves blocked or already done — nothing left to recommend.
        return {
            "action": None,
            "reason": "All known moves are either completed or blocked.",
            "value_score": 0.0,
            "opportunity_context_used": len(opportunity_context),
        }

    ranked = sorted(candidates, key=_move_value, reverse=True)
    best = ranked[0]

    return {
        "action": best.action,
        "time_cost_hours": best.time_cost_hours,
        "expected_revenue_30d": best.expected_revenue_30d,
        "confidence": best.confidence,
        "automation_possible": best.automation_possible,
        "prerequisites": best.prerequisites,
        "value_score": round(_move_value(best), 4),
        "reason": (
            f"Highest immediate value among {len(candidates)} unblocked moves.  "
            f"Est. ${best.expected_revenue_30d:,.0f}/30d at {best.confidence:.0%} confidence "
            f"({'automatable' if best.automation_possible else 'manual'}, "
            f"{best.time_cost_hours}h cost)."
        ),
        "opportunity_context_used": len(opportunity_context),
        "all_unblocked": [
            {
                "action": m.action,
                "value_score": round(_move_value(m), 4),
                "expected_revenue_30d": m.expected_revenue_30d,
                "confidence": m.confidence,
                "time_cost_hours": m.time_cost_hours,
            }
            for m in ranked
        ],
    }
