"""Alert rules."""

import pandas as pd
import pytest

from database import database as db
from src import alerts as al
from src import inventory as inv
from helpers import make_case

STOCK_TYPES = {"stockout", "stockout_risk", "reorder", "low_stock", "overstock"}


@pytest.fixture
def plan():
    # item: 1 stockout, 2 reorder with 3 days of cover (< 5-day lead), 3 ok but 6 days cover,
    #       4 overstock, 5 ok with 2 days of cover (stock is on order)
    history, inventory = make_case([(0, 0), (30, 0), (60, 0), (1000, 0), (20, 100)])
    return inv.compute_inventory_plan(inventory, history)


def types_by_item(alerts):
    return alerts.groupby("item_id")["alert_type"].apply(list).to_dict()


def test_each_status_maps_to_the_expected_alert(plan):
    alerts = al.generate_alerts(plan)
    assert types_by_item(alerts) == {
        1: ["stockout"], 2: ["stockout_risk"], 3: ["low_stock"], 4: ["overstock"], 5: ["low_stock"],
    }
    severity = alerts.set_index("item_id")["severity"].to_dict()
    assert severity == {1: "critical", 2: "critical", 3: "warning", 4: "info", 5: "warning"}


def test_alerts_are_sorted_by_severity_and_well_formed(plan):
    alerts = al.generate_alerts(plan)
    assert list(alerts.columns) == list(al.ALERT_COLUMNS)
    assert alerts["severity"].map({"critical": 0, "warning": 1, "info": 2}).is_monotonic_increasing
    assert alerts["message"].str.len().gt(10).all() and alerts["message"].str.contains("Item").all()


def test_reorder_with_enough_cover_is_a_warning_not_critical():
    # demand 10/day, lead 4 days -> reorder point 40; 40 on hand is exactly at it, with 4.0 days of cover
    history, inventory = make_case([(40, 0)], lead=4)
    plan = inv.compute_inventory_plan(inventory, history)
    assert plan.iloc[0]["status"] == "reorder"
    alerts = al.generate_alerts(plan)
    assert alerts["alert_type"].tolist() == ["reorder"] and alerts["severity"].tolist() == ["warning"]


def test_at_most_one_stock_alert_per_position(plan):
    alerts = al.generate_alerts(plan)
    stock = alerts[alerts["alert_type"].isin(STOCK_TYPES)]
    assert not stock.duplicated(["store_id", "item_id"]).any()


def test_zero_demand_empty_stock_is_not_an_alert():
    history, inventory = make_case([(0, 0)])
    plan = inv.compute_inventory_plan(inventory, history.iloc[:0])
    assert plan.iloc[0]["status"] == "stockout"
    assert al.generate_alerts(plan).empty


def test_demand_surge_and_drop_need_a_forecast(plan):
    changed = plan.copy()
    changed["forecast_used"] = True
    changed["hist_avg_daily"] = 10.0
    changed["avg_daily_demand"] = [15.0, 10.0, 6.0, 10.0, 12.9]
    demand = al.generate_alerts(changed)
    demand = demand[demand["alert_type"].isin(["demand_surge", "demand_drop"])].set_index("item_id")["alert_type"].to_dict()
    assert demand == {1: "demand_surge", 3: "demand_drop"}  # 12.9/10 is below the 1.3 threshold
    no_forecast = changed.assign(forecast_used=False)
    assert not al.generate_alerts(no_forecast)["alert_type"].isin(["demand_surge", "demand_drop"]).any()


def test_empty_plan_gives_no_alerts(plan):
    assert al.generate_alerts(plan.iloc[:0]).empty


def test_refresh_alerts_replaces_open_alerts_in_the_database(plan, empty_db):
    first = al.refresh_alerts(plan, empty_db)
    assert len(db.get_alerts(db_path=empty_db)) == len(first) == 5
    al.refresh_alerts(plan, empty_db)
    assert len(db.get_alerts(db_path=empty_db)) == 5  # replaced, not appended
    al.refresh_alerts(plan.iloc[:0], empty_db)
    assert db.get_alerts(db_path=empty_db).empty
