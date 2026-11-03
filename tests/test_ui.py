"""Streamlit pages, driven headlessly with Streamlit's AppTest."""

import pytest
from streamlit.testing.v1 import AppTest

from conftest import ROOT
from database import database as db

PAGES = ["1_Dashboard", "2_Forecasting", "3_Inventory", "4_Budget_Optimizer", "5_Alerts", "6_Data"]


def page(name: str) -> AppTest:
    return AppTest.from_file(str(ROOT / "pages" / f"{name}.py"), default_timeout=120)


def click(at: AppTest, label: str) -> AppTest:
    return next(b for b in at.button if b.label == label).click().run()


def clean(at: AppTest) -> AppTest:
    assert not at.exception, [e.value for e in at.exception]
    return at


@pytest.mark.parametrize("name", PAGES)
def test_page_renders(ui_db, name):
    clean(page(name).run())


def test_entry_point_renders(ui_db):
    clean(AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run())


@pytest.mark.parametrize("name", [p for p in PAGES if p != "6_Data"])
def test_pages_show_an_empty_state_without_data(ui_db, name):
    db.clear_all(ui_db)
    at = clean(page(name).run())
    assert any("No data loaded" in w.value for w in at.warning)


def test_data_page_loads_the_sample_after_a_reset(ui_db):
    db.clear_all(ui_db)
    at = clean(page("6_Data").run())
    assert any("empty" in w.value for w in at.warning)
    clean(click(at, "Load small sample (sample_data.csv)"))
    assert db.table_row_count("sales_history", ui_db) == 1840
    assert len(db.get_inventory(ui_db)) == 10


def test_data_page_delete_requires_confirmation(ui_db):
    at = clean(page("6_Data").run())
    assert next(b for b in at.button if b.label == "Delete all data").disabled
    at.checkbox[0].check().run()
    clean(click(at, "Delete all data"))
    assert not db.is_populated(ui_db)


def test_stock_scaling_demo_tool(ui_db):
    before = db.get_inventory(ui_db)["on_hand"].sum()
    clean(click(clean(page("6_Data").run()), "Apply stock scaling"))
    after = db.get_inventory(ui_db)["on_hand"].sum()
    assert after < before * 0.35  # default factor is 25%, rounding aside


def test_forecast_run_save_and_forecast_all(ui_db):
    at = clean(page("2_Forecasting").run())
    at = clean(click(at, "Run forecast"))
    assert {m.label for m in at.metric} >= {"Model used", "Expected demand (horizon)"}
    clean(click(at, "Save this forecast"))
    assert len(db.get_forecasts(1, 1, ui_db)) == 30
    clean(click(clean(page("2_Forecasting").run()), "Forecast all and save"))
    assert len(db.get_forecasts(db_path=ui_db)) == 30 * 30
    clean(page("1_Dashboard").run())  # dashboard with forecast overlay


def test_forecasting_reports_a_short_series_as_an_error(ui_db):
    db.execute("DELETE FROM sales_history WHERE NOT (store_id = 1 AND item_id = 1 AND date <= '2021-01-05')", db_path=ui_db)
    at = clean(page("2_Forecasting").run())
    clean(click(at, "Run forecast"))
    assert any("at least 14 days" in e.value for e in at.error)


def test_inventory_movement_and_order_forms(ui_db):
    at = clean(page("3_Inventory").run())
    logged = len(db.get_stock_movements(500, ui_db))
    at.number_input[0].set_value(2.0)
    clean(click(at, "Record movement"))
    assert len(db.get_stock_movements(500, ui_db)) == logged + 1

    at = clean(page("3_Inventory").run())
    at.number_input[1].set_value(40.0)
    clean(click(at, "Place order"))
    assert db.get_inventory(ui_db)["on_order"].sum() == 40.0


def test_inventory_rejects_overdrawing_stock(ui_db):
    at = clean(page("3_Inventory").run())
    movement_type = next(s for s in at.selectbox if s.label == "Type")
    movement_type.select("sale")
    at.number_input[0].set_value(10_000_000.0)
    at = clean(click(at, "Record movement"))
    assert any("Insufficient stock" in e.value for e in at.error)
    assert db.table_row_count("stock_movements", ui_db) == 0


def test_save_policies_button(ui_db):
    clean(click(clean(page("3_Inventory").run()), "Save policies to database"))
    assert (db.get_inventory(ui_db)["max_stock"] > 0).all()


def test_budget_optimizer_places_orders(ui_db):
    clean(click(clean(page("6_Data").run()), "Apply stock scaling"))
    at = clean(page("4_Budget_Optimizer").run())
    assert any(m.label == "Lines fully funded" for m in at.metric)
    clean(click(at, "Place all approved orders"))
    assert db.get_inventory(ui_db)["on_order"].sum() > 0


def test_budget_page_says_so_when_nothing_needs_ordering(ui_db):
    db.execute("UPDATE inventory SET on_hand = 1000000", db_path=ui_db)
    at = clean(page("4_Budget_Optimizer").run())
    assert any("nothing to allocate" in s.value for s in at.success)


def test_alerts_page_lists_and_resolves(ui_db):
    clean(click(clean(page("6_Data").run()), "Apply stock scaling"))
    at = clean(page("5_Alerts").run())
    open_before = len(db.get_alerts(db_path=ui_db))
    assert open_before > 0
    resolve = [b for b in at.button if b.label == "Resolve"]
    assert resolve
    clean(resolve[0].click().run())
    assert len(db.get_alerts(db_path=ui_db)) == open_before - 1
