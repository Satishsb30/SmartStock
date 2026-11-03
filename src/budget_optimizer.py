"""Budget-constrained purchasing.

Given the order recommendations from ``inventory.compute_inventory_plan`` and a
budget, decide how many units of each line to buy.

Method (a transparent greedy heuristic, not an exact optimum):

1. Each line's need is split into *critical* units (bringing the inventory
   position back up to the reorder point) and *optional* units (the rest of the
   way up to max stock).
2. Lines are ranked by priority = urgency x margin per unit of cost, where
   urgency = 1 / (1 + days of cover), so empty shelves and high-margin items
   come first.
3. Critical units are funded for all lines first, in priority order; any budget
   left then funds optional units, again in priority order.

Within a stage, a line takes as many units as the remaining budget allows, so
leftover money flows on to the next line instead of being stranded.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

ALLOCATION_COLUMNS: tuple[str, ...] = (
    "store_id",
    "item_id",
    "name",
    "unit_cost",
    "days_of_cover",
    "recommended_qty",
    "critical_qty",
    "approved_qty",
    "approved_cost",
    "shortfall_qty",
    "critical_shortfall_qty",
    "potential_margin",
    "priority",
)


def _priority(plan: pd.DataFrame) -> pd.Series:
    cost = plan["unit_cost"].astype(float)
    margin = (plan["unit_price"] - plan["unit_cost"]).clip(lower=0)
    margin_ratio = np.where(cost > 0, margin / cost.where(cost > 0, 1.0), margin)
    urgency = 1.0 / (1.0 + plan["days_of_cover"].clip(lower=0))
    return pd.Series(urgency * margin_ratio, index=plan.index)


def optimize_budget(plan: pd.DataFrame, budget: float) -> pd.DataFrame:
    """Allocate ``budget`` across the recommended orders in ``plan``.

    Returns one row per line that needs an order (see ``ALLOCATION_COLUMNS``),
    highest priority first. Total ``approved_cost`` never exceeds ``budget``.
    """
    budget = float(budget)
    if budget < 0 or not math.isfinite(budget):
        raise ValueError("budget must be a non-negative number")

    lines = plan[plan["recommended_qty"] > 0].copy()
    if lines.empty:
        return pd.DataFrame(columns=list(ALLOCATION_COLUMNS))

    lines["priority"] = _priority(lines)
    lines["critical_qty"] = (lines["reorder_point"] - lines["inventory_position"]).clip(lower=0)
    lines["critical_qty"] = np.minimum(np.ceil(lines["critical_qty"]), lines["recommended_qty"])
    # Stable, deterministic ordering: priority, then cheaper items, then ids.
    lines = lines.sort_values(
        ["priority", "unit_cost", "store_id", "item_id"],
        ascending=[False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    remaining = budget
    approved = np.zeros(len(lines))
    wanted = lines["recommended_qty"].to_numpy(dtype=float)
    critical = lines["critical_qty"].to_numpy(dtype=float)
    cost = lines["unit_cost"].to_numpy(dtype=float)

    def take(idx: int, units: float) -> float:
        nonlocal remaining
        if units <= 0:
            return 0.0
        if cost[idx] <= 0:
            return units
        affordable = math.floor(round(remaining / cost[idx], 9))
        got = float(min(units, max(affordable, 0)))
        remaining -= got * cost[idx]
        return got

    for i in range(len(lines)):  # stage 1: critical units
        approved[i] += take(i, critical[i])
    for i in range(len(lines)):  # stage 2: optional units
        approved[i] += take(i, wanted[i] - approved[i])

    lines["approved_qty"] = approved
    lines["approved_cost"] = approved * cost
    lines["shortfall_qty"] = wanted - approved
    lines["critical_shortfall_qty"] = np.clip(critical - approved, 0, None)
    lines["potential_margin"] = approved * (lines["unit_price"] - lines["unit_cost"])
    return lines[list(ALLOCATION_COLUMNS)]


def budget_summary(allocation: pd.DataFrame, budget: float) -> dict[str, float]:
    """Headline numbers for an allocation."""
    if allocation.empty:
        return {
            "budget": float(budget),
            "spent": 0.0,
            "remaining": float(budget),
            "lines": 0,
            "lines_fully_funded": 0,
            "units_short": 0.0,
            "critical_units_short": 0.0,
            "potential_margin": 0.0,
        }
    spent = float(allocation["approved_cost"].sum())
    return {
        "budget": float(budget),
        "spent": spent,
        "remaining": float(budget) - spent,
        "lines": int(len(allocation)),
        "lines_fully_funded": int((allocation["shortfall_qty"] <= 0).sum()),
        "units_short": float(allocation["shortfall_qty"].sum()),
        "critical_units_short": float(allocation["critical_shortfall_qty"].sum()),
        "potential_margin": float(allocation["potential_margin"].sum()),
    }


def budget_scenarios(plan: pd.DataFrame, budgets: list[float]) -> pd.DataFrame:
    """Compare several budget levels side by side."""
    rows = []
    for b in budgets:
        summary = budget_summary(optimize_budget(plan, b), b)
        rows.append(
            {
                "budget": summary["budget"],
                "spent": summary["spent"],
                "lines_fully_funded": summary["lines_fully_funded"],
                "lines": summary["lines"],
                "critical_units_short": summary["critical_units_short"],
                "units_short": summary["units_short"],
                "potential_margin": summary["potential_margin"],
            }
        )
    return pd.DataFrame(rows)
