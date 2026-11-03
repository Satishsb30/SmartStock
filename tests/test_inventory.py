"""Inventory policy, using small hand-computable cases plus the bundled data."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from database import database as db
from src import inventory as inv
from src import preprocessing as pp
from helpers import make_case
from src import settings


def test_safety_stock_formula_and_validation():
    assert inv.safety_stock(10, 9, 0.95) == pytest.approx(norm.ppf(0.95) * 30)
    assert inv.safety_stock(0, 7, 0.9) == 0
    assert inv.safety_stock(10, 4, 0.99) > inv.safety_stock(10, 4, 0.90)
    for args in [(-1, 1, 0.9), (1, -1, 0.9), (1, 1, 1.0), (1, 1, 0.3)]:
        with pytest.raises(ValueError):
            inv.safety_stock(*args)


def test_eoq():
    assert inv.economic_order_quantity(1000, 50, 2.0, 0.25) == pytest.approx(math.sqrt(2 * 1000 * 50 / 0.5))
    assert inv.economic_order_quantity(0) == 0
    assert inv.economic_order_quantity(100, unit_cost=0) == 0


def test_policy_values_and_statuses_on_a_constant_demand_case():
    # demand 10/day, lead 5, review 7, no variance -> SS 0, ROP 50, max stock 120
    history, inventory = make_case([(0, 0), (30, 0), (60, 0), (1000, 0), (20, 100)])
    plan = inv.compute_inventory_plan(inventory, history).set_index("item_id")
    assert (plan["safety_stock"] == 0).all()
    assert (plan["reorder_point"] == 50).all() and (plan["max_stock"] == 120).all()
    assert plan["status"].to_dict() == {1: "stockout", 2: "reorder", 3: "ok", 4: "overstock", 5: "ok"}
    assert plan["recommended_qty"].to_dict() == {1: 120.0, 2: 90.0, 3: 0.0, 4: 0.0, 5: 0.0}
    assert plan.loc[2, "order_cost"] == 180.0  # 90 units at 2.0
    assert plan.loc[3, "days_of_cover"] == pytest.approx(6.0)
    assert plan.loc[5, "inventory_position"] == 120.0  # on-order stock counts toward the position


def test_forecast_demand_replaces_history_when_available():
    history, inventory = make_case([(60, 0)])
    forecast = pd.DataFrame({"store_id": 1, "item_id": 1, "date": pd.date_range("2024-05-01", periods=30), "predicted": 20.0})
    plan = inv.compute_inventory_plan(inventory, history, forecast).iloc[0]
    assert plan["forecast_used"] and plan["avg_daily_demand"] == 20.0 and plan["hist_avg_daily"] == 10.0
    assert plan["reorder_point"] == 100 and plan["max_stock"] == 240  # 5 and 12 days at 20/day
    assert plan["status"] == "reorder" and plan["recommended_qty"] == 180


def test_short_forecast_is_extrapolated_by_its_mean():
    history, inventory = make_case([(0, 0)])
    forecast = pd.DataFrame({"store_id": 1, "item_id": 1, "date": pd.date_range("2024-05-01", periods=3), "predicted": 20.0})
    plan = inv.compute_inventory_plan(inventory, history, forecast).iloc[0]
    assert plan["reorder_point"] == 100 and plan["max_stock"] == 240


def test_item_without_demand_history_is_overstock_not_a_crash():
    history, inventory = make_case([(0, 0), (50, 0)])
    history = history[history["item_id"] == 1]  # item 2 has no sales at all
    plan = inv.compute_inventory_plan(inventory, history).set_index("item_id")
    assert plan.loc[2, "avg_daily_demand"] == 0 and plan.loc[2, "status"] == "overstock"
    assert plan.loc[2, "days_of_cover"] == inv.MAX_DAYS_COVER and plan.loc[2, "recommended_qty"] == 0
    assert np.isfinite(plan.select_dtypes("number").to_numpy()).all()


def test_zero_lead_time_uses_the_default():
    history, inventory = make_case([(0, 0)], lead=0)
    plan = inv.compute_inventory_plan(inventory, history).iloc[0]
    assert plan["lead_time_days"] == settings.DEFAULT_LEAD_TIME_DAYS


def test_empty_inventory_gives_empty_plan():
    history, inventory = make_case([(0, 0)])
    plan = inv.compute_inventory_plan(inventory.iloc[:0], history)
    assert plan.empty and list(plan.columns) == list(inv.PLAN_COLUMNS)


@pytest.mark.parametrize("kwargs", [{"service_level": 1.0}, {"service_level": 0.2}, {"review_days": -1}])
def test_invalid_policy_arguments(kwargs):
    history, inventory = make_case([(0, 0)])
    with pytest.raises(ValueError):
        inv.compute_inventory_plan(inventory, history, **kwargs)


def test_plan_on_real_data_matches_hand_calculation_and_invariants(db_path, history):
    plan = inv.compute_inventory_plan(db.get_inventory(db_path), history)
    assert len(plan) == 30 and list(plan.columns) == list(inv.PLAN_COLUMNS)
    assert (plan["max_stock"] >= plan["reorder_point"]).all() and (plan["reorder_point"] >= plan["safety_stock"]).all()
    ordering = plan["recommended_qty"] > 0
    assert (plan.loc[ordering, "inventory_position"] <= plan.loc[ordering, "reorder_point"]).all()
    assert (plan.loc[~ordering, "inventory_position"] > plan.loc[~ordering, "reorder_point"]).all()
    assert np.allclose(plan.loc[ordering, "inventory_position"] + plan.loc[ordering, "recommended_qty"], plan.loc[ordering, "max_stock"])

    row = plan.iloc[0]
    stats = pp.demand_statistics(history).query("store_id == @row.store_id and item_id == @row.item_id").iloc[0]
    ss = norm.ppf(settings.SERVICE_LEVEL) * stats["std_daily"] * math.sqrt(row["lead_time_days"])
    assert row["safety_stock"] == math.ceil(ss)
    assert row["reorder_point"] == math.ceil(stats["avg_daily"] * row["lead_time_days"] + ss)


def test_higher_service_level_means_more_safety_stock(db_path, history):
    inventory = db.get_inventory(db_path)
    low = inv.compute_inventory_plan(inventory, history, service_level=0.90)
    high = inv.compute_inventory_plan(inventory, history, service_level=0.99)
    assert (high["safety_stock"] >= low["safety_stock"]).all() and high["safety_stock"].sum() > low["safety_stock"].sum()


def test_save_policies_keeps_stock_levels(db_path, history):
    before = db.get_inventory(db_path)
    plan = inv.compute_inventory_plan(before, history)
    assert inv.save_policies(plan, db_path) == 30
    after = db.get_inventory(db_path)
    assert (after["on_hand"].to_numpy() == before["on_hand"].to_numpy()).all()
    assert (after["reorder_point"].to_numpy() == plan["reorder_point"].to_numpy()).all()
    assert inv.save_policies(plan.iloc[:0], db_path) == 0


def test_place_order_adds_to_on_order(db_path):
    assert inv.place_order(1, 1, 25, db_path) == 25
    assert inv.place_order(1, 1, 5, db_path) == 30
    with pytest.raises(ValueError):
        inv.place_order(1, 1, 0, db_path)


def test_plan_summary(db_path, history):
    plan = inv.compute_inventory_plan(db.get_inventory(db_path), history)
    summary = inv.plan_summary(plan)
    assert summary["positions"] == 30
    assert summary["stockouts"] + summary["reorders"] + summary["overstocks"] <= 30
    assert summary["inventory_value"] == pytest.approx((plan["on_hand"] * plan["unit_cost"]).sum())
    assert inv.plan_summary(plan.iloc[:0])["positions"] == 0
