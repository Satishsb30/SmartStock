"""Budget optimizer: hand-computed cases and property checks."""

import math

import numpy as np
import pandas as pd
import pytest

from src import budget_optimizer as bo
from src import inventory as inv


def make_plan(lines):
    """lines: (unit_cost, unit_price, days_of_cover, critical_qty, recommended_qty)."""
    rows = []
    for n, (cost, price, cover, critical, rec) in enumerate(lines, start=1):
        rows.append(
            {"store_id": 1, "item_id": n, "name": f"Item {n}", "unit_cost": cost, "unit_price": price,
             "days_of_cover": cover, "recommended_qty": rec, "inventory_position": 0.0, "reorder_point": critical,
             "order_cost": rec * cost}
        )
    return pd.DataFrame(rows)


# A: urgent and high margin.  B: plenty of cover and thin margin.
TWO_LINES = make_plan([(10.0, 20.0, 0.0, 5, 10), (5.0, 6.0, 20.0, 2, 4)])


def approved(plan, budget):
    return bo.optimize_budget(plan, budget).set_index("item_id")["approved_qty"].to_dict()


def test_budget_just_covers_critical_units():
    # critical: A 5 x 10 = 50, B 2 x 5 = 10
    assert approved(TWO_LINES, 60) == {1: 5.0, 2: 2.0}


def test_leftover_budget_funds_top_up_units_in_priority_order():
    # after critical (60) there is 15 left: A takes one more unit (10), then B one (5)
    assert approved(TWO_LINES, 75) == {1: 6.0, 2: 3.0}


def test_full_budget_funds_everything_and_extra_is_not_spent():
    alloc = bo.optimize_budget(TWO_LINES, 10_000)
    assert alloc["shortfall_qty"].sum() == 0 and alloc["approved_cost"].sum() == 10 * 10 + 4 * 5


def test_tiny_budget_goes_to_the_highest_priority_line():
    assert approved(TWO_LINES, 20) == {1: 2.0, 2: 0.0}


def test_zero_budget_approves_nothing():
    alloc = bo.optimize_budget(TWO_LINES, 0)
    assert alloc["approved_qty"].sum() == 0 and alloc["critical_shortfall_qty"].sum() == 7


def test_priority_order_and_margin_numbers():
    alloc = bo.optimize_budget(TWO_LINES, 60)
    assert alloc["item_id"].tolist() == [1, 2]
    assert alloc["potential_margin"].tolist() == [5 * 10.0, 2 * 1.0]
    summary = bo.budget_summary(alloc, 60)
    assert summary["spent"] == 60 and summary["remaining"] == 0 and summary["lines_fully_funded"] == 0
    assert summary["critical_units_short"] == 0 and summary["units_short"] == 7  # 5 + 2 top-up units


def test_free_items_are_fully_funded_without_budget():
    free = make_plan([(0.0, 5.0, 1.0, 3, 8)])
    alloc = bo.optimize_budget(free, 0)
    assert alloc["approved_qty"].iloc[0] == 8 and alloc["approved_cost"].iloc[0] == 0


def test_lines_without_a_recommendation_are_excluded():
    plan = make_plan([(10.0, 20.0, 0.0, 5, 10), (5.0, 6.0, 20.0, 0, 0)])
    assert bo.optimize_budget(plan, 1000)["item_id"].tolist() == [1]
    assert bo.optimize_budget(plan.iloc[:0], 100).empty
    summary = bo.budget_summary(bo.optimize_budget(plan.iloc[:0], 100), 100)
    assert summary["spent"] == 0 and summary["remaining"] == 100


@pytest.mark.parametrize("budget", [-1, float("nan"), float("inf")])
def test_invalid_budgets(budget):
    with pytest.raises(ValueError):
        bo.optimize_budget(TWO_LINES, budget)


def test_deterministic():
    assert bo.optimize_budget(TWO_LINES, 33).equals(bo.optimize_budget(TWO_LINES, 33))


def test_properties_on_random_plans():
    rng = np.random.default_rng(0)
    for _ in range(300):
        n = int(rng.integers(1, 12))
        lines = []
        for _ in range(n):
            cost = float(rng.integers(1, 40))
            rec = int(rng.integers(1, 60))
            lines.append((cost, cost * float(rng.uniform(0.8, 2.0)), float(rng.uniform(0, 30)), int(rng.integers(0, rec + 1)), rec))
        plan = make_plan(lines)
        budget = float(rng.uniform(0, plan["order_cost"].sum() * 1.3))
        alloc = bo.optimize_budget(plan, budget)
        spent = alloc["approved_cost"].sum()
        assert spent <= budget + 1e-6
        assert (alloc["approved_qty"] >= 0).all() and (alloc["approved_qty"] <= alloc["recommended_qty"]).all()
        assert (alloc["approved_qty"] == np.floor(alloc["approved_qty"])).all()
        # no stranded money: any line left short could not afford even one more unit
        short = alloc[alloc["shortfall_qty"] > 0]
        assert (budget - spent < short["unit_cost"] + 1e-6).all()
        assert (alloc["shortfall_qty"] == alloc["recommended_qty"] - alloc["approved_qty"]).all()
        assert (alloc["critical_shortfall_qty"] <= alloc["critical_qty"]).all()


def test_scenarios_and_realistic_plan(db_path, history):
    from database import database as db

    needy = db.get_inventory(db_path)
    needy["on_hand"] = (needy["on_hand"] * 0.1).round()
    plan = inv.compute_inventory_plan(needy, history)
    total = plan["order_cost"].sum()
    assert total > 0 and (plan["recommended_qty"] > 0).sum() >= 5
    scenarios = bo.budget_scenarios(plan, [0, total * 0.25, total * 0.5, total, total * 3])
    assert scenarios["spent"].is_monotonic_increasing
    assert scenarios["critical_units_short"].is_monotonic_decreasing
    assert scenarios["spent"].iloc[-1] == pytest.approx(total) and scenarios["units_short"].iloc[-1] == 0
    assert scenarios["spent"].iloc[0] == 0
    assert (scenarios["spent"] <= scenarios["budget"] + 1e-6).all()
    # a budget equal to the critical cost leaves no critical units unfunded
    crit_cost = sum(
        min(math.ceil(max(r.reorder_point - r.inventory_position, 0)), r.recommended_qty) * r.unit_cost
        for r in plan[plan["recommended_qty"] > 0].itertuples()
    )
    assert bo.budget_summary(bo.optimize_budget(plan, crit_cost), crit_cost)["critical_units_short"] == 0
