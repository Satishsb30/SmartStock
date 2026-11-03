"""SQLite layer."""

import sqlite3

import pandas as pd
import pytest

from database import database as db


def _inv(on_hand=10.0, on_order=20.0):
    return pd.DataFrame({"store_id": [1], "item_id": [1], "on_hand": [on_hand], "on_order": [on_order]})


def test_init_db_is_idempotent_and_creates_all_tables(empty_db):
    db.init_db(empty_db)
    for table in ("items", "stores", "sales_history", "inventory", "stock_movements", "forecasts", "alerts"):
        assert db.table_row_count(table, empty_db) == 0
    assert not db.is_populated(empty_db)


def test_unknown_table_name_is_rejected(empty_db):
    with pytest.raises(ValueError):
        db.table_row_count("items; DROP TABLE items", empty_db)


def test_items_get_defaults_and_upsert_updates(empty_db):
    db.upsert_items(pd.DataFrame({"item_id": [1, 2]}), empty_db)
    items = db.get_items(empty_db)
    assert items["name"].tolist() == ["Item 1", "Item 2"]
    db.upsert_items(pd.DataFrame({"item_id": [1], "name": ["Widget"], "unit_cost": [9.5]}), empty_db)
    items = db.get_items(empty_db).set_index("item_id")
    assert len(items) == 2 and items.loc[1, "name"] == "Widget" and items.loc[1, "unit_cost"] == 9.5


def test_sales_upsert_does_not_duplicate_and_parses_dates(empty_db):
    frame = pd.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=5), "store_id": 1, "item_id": 1, "sales": [1, 2, 3, 4, 5]}
    )
    db.upsert_sales_history(frame, empty_db)
    frame["sales"] = [9, 9, 9, 9, 9]
    db.upsert_sales_history(frame, empty_db)
    got = db.get_sales_history(1, 1, empty_db)
    assert len(got) == 5 and got["sales"].tolist() == [9.0] * 5
    assert pd.api.types.is_datetime64_any_dtype(got["date"])
    assert db.get_sales_history(2, 1, empty_db).empty


def test_sales_missing_columns_and_negative_values_are_rejected(empty_db):
    with pytest.raises(ValueError):
        db.upsert_sales_history(pd.DataFrame({"date": ["2024-01-01"], "sales": [1]}), empty_db)
    bad = pd.DataFrame({"date": ["2024-01-01"], "store_id": [1], "item_id": [1], "sales": [-1]})
    with pytest.raises(sqlite3.IntegrityError):
        db.upsert_sales_history(bad, empty_db)


def test_stock_movements_update_stock_and_log(empty_db):
    db.upsert_inventory(_inv(10, 20), empty_db)
    assert db.record_stock_movement(1, 1, "sale", 4, db_path=empty_db) == 6
    assert db.record_stock_movement(1, 1, "receipt", 15, db_path=empty_db) == 21
    row = db.read_sql("SELECT * FROM inventory", db_path=empty_db).iloc[0]
    assert row["on_hand"] == 21 and row["on_order"] == 5  # receipt reduces what is on order
    assert db.record_stock_movement(1, 1, "receipt", 50, db_path=empty_db) == 71
    assert db.read_sql("SELECT on_order FROM inventory", db_path=empty_db).iloc[0, 0] == 0  # never negative
    assert db.record_stock_movement(1, 1, "adjustment", -1, db_path=empty_db) == 70
    assert db.record_stock_movement(1, 1, "write_off", 10, db_path=empty_db) == 60
    log = db.get_stock_movements(10, empty_db)
    assert len(log) == 5
    assert log.iloc[0]["movement_type"] == "write_off" and log.iloc[0]["quantity"] == -10  # newest first, signed


@pytest.mark.parametrize(
    "args",
    [("sale", 100), ("sale", 0), ("sale", -1), ("receipt", -5), ("bogus", 1), ("adjustment", 0), ("adjustment", -999)],
)
def test_invalid_movements_are_rejected_and_leave_no_trace(empty_db, args):
    db.upsert_inventory(_inv(10, 0), empty_db)
    with pytest.raises(ValueError):
        db.record_stock_movement(1, 1, args[0], args[1], db_path=empty_db)
    assert db.read_sql("SELECT on_hand FROM inventory", db_path=empty_db).iloc[0, 0] == 10
    assert db.table_row_count("stock_movements", empty_db) == 0


def test_movement_for_unknown_position_is_rejected(empty_db):
    with pytest.raises(ValueError):
        db.record_stock_movement(9, 9, "sale", 1, db_path=empty_db)


def test_add_on_order(empty_db):
    db.upsert_inventory(_inv(10, 5), empty_db)
    assert db.add_on_order(1, 1, 7, empty_db) == 12
    for bad in [(1, 1, 0), (1, 1, -2), (9, 9, 3)]:
        with pytest.raises(ValueError):
            db.add_on_order(*bad, db_path=empty_db)


def test_forecasts_upsert_and_roundtrip(empty_db):
    fc = pd.DataFrame(
        {"store_id": 1, "item_id": 1, "date": pd.date_range("2024-02-01", periods=3), "predicted": [1.0, 2.0, 3.0],
         "lower": 0.0, "upper": 5.0, "model": "naive"}
    )
    db.save_forecasts(fc, empty_db)
    fc["predicted"] = [4.0, 5.0, 6.0]
    db.save_forecasts(fc, empty_db)
    got = db.get_forecasts(1, 1, empty_db)
    assert got["predicted"].tolist() == [4.0, 5.0, 6.0] and len(got) == 3
    with pytest.raises(ValueError):
        db.save_forecasts(fc.drop(columns="model"), empty_db)
    db.clear_forecasts(empty_db)
    assert db.get_forecasts(db_path=empty_db).empty


def test_alert_lifecycle(empty_db):
    alerts = pd.DataFrame(
        {"store_id": [1, 1, 1], "item_id": [1, 2, 3], "alert_type": ["a", "b", "c"],
         "severity": ["info", "critical", "warning"], "message": ["x", "y", "z"]}
    )
    assert db.replace_open_alerts(alerts, empty_db) == 3
    open_alerts = db.get_alerts(db_path=empty_db)
    assert open_alerts["severity"].tolist() == ["critical", "warning", "info"]
    assert db.resolve_alert(int(open_alerts.iloc[0]["alert_id"]), empty_db)
    assert not db.resolve_alert(99999, empty_db)
    assert len(db.get_alerts(db_path=empty_db)) == 2
    db.replace_open_alerts(alerts.iloc[:0], empty_db)  # replaces open ones, keeps resolved history
    assert db.get_alerts(db_path=empty_db).empty
    assert len(db.get_alerts(include_resolved=True, db_path=empty_db)) == 1
    with pytest.raises(ValueError):
        db.replace_open_alerts(alerts.drop(columns="message"), empty_db)


def test_alert_severity_constraint(empty_db):
    bad = pd.DataFrame({"store_id": [1], "item_id": [1], "alert_type": ["a"], "severity": ["urgent"], "message": ["m"]})
    with pytest.raises(sqlite3.IntegrityError):
        db.replace_open_alerts(bad, empty_db)


def test_connection_scope_rolls_back_on_error(empty_db):
    with pytest.raises(RuntimeError):
        with db.connection_scope(empty_db) as conn:
            conn.execute("INSERT INTO stores (store_id, name) VALUES (1, 'x')")
            raise RuntimeError("boom")
    assert db.table_row_count("stores", empty_db) == 0


def test_clear_all_keeps_schema(db_path):
    assert db.is_populated(db_path)
    db.clear_all(db_path)
    assert not db.is_populated(db_path)
    assert db.table_row_count("items", db_path) == 0
