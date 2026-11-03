"""Data loading, validation and seeding."""

import io

import pandas as pd
import pytest

from database import database as db
from src import data_loader as dl
from src import settings


def test_bundled_datasets_are_valid(sales):
    assert list(sales.columns) == ["date", "store", "item", "sales"]
    assert len(sales) == 32850
    assert sales["store"].nunique() == 3 and sales["item"].nunique() == 10
    assert (sales["sales"] >= 0).all()
    assert not sales.duplicated(["date", "store", "item"]).any()
    sample = dl.load_sales_csv(settings.SAMPLE_CSV_PATH)
    assert len(sample) == 2 * 5 * 184


def test_synthetic_generation_is_deterministic_and_shaped():
    a = dl.generate_synthetic_sales(2, 3, "2024-01-01", "2024-03-31", seed=5)
    b = dl.generate_synthetic_sales(2, 3, "2024-01-01", "2024-03-31", seed=5)
    c = dl.generate_synthetic_sales(2, 3, "2024-01-01", "2024-03-31", seed=6)
    assert a.equals(b) and not a.equals(c)
    assert len(a) == 2 * 3 * 91 and (a["sales"] >= 0).all()


def test_synthetic_data_has_weekly_seasonality():
    data = dl.generate_synthetic_sales(1, 5, "2022-01-01", "2023-12-31")
    by_dow = data.groupby(data["date"].dt.dayofweek)["sales"].mean()
    assert by_dow[5] > by_dow[0] * 1.3  # Saturday clearly above Monday


def test_column_aliases_duplicates_and_negatives():
    raw = io.StringIO("Date,Store_ID,Item_ID,Sales\n2024-01-01,1,1,5\n2024-01-01,1,1,3\n2024-01-02,1,1,-4\n")
    out = dl.load_sales_csv(raw)
    assert out["sales"].tolist() == [8.0, 0.0]
    assert list(out.columns) == ["date", "store", "item", "sales"]


def test_rows_with_missing_values_are_dropped():
    raw = io.StringIO("date,store,item,sales\n2024-01-01,1,1,5\n2024-01-02,1,1,\n2024-01-03,1,1,7\n")
    assert len(dl.load_sales_csv(raw)) == 2


@pytest.mark.parametrize(
    "text",
    [
        "a,b\n1,2\n",
        "date,store,item,sales\nx,y,z,w\n",
        "date,store,item,sales\n2024-01-01,1,1,5\nbad,1,1,x\nbad,1,1,x\nbad,1,1,x\n",
        "",
    ],
)
def test_invalid_files_raise_validation_error(text):
    with pytest.raises(dl.DataValidationError):
        dl.load_sales_csv(io.StringIO(text))


def test_missing_file_raises_validation_error(tmp_path):
    with pytest.raises(dl.DataValidationError):
        dl.load_sales_csv(tmp_path / "nope.csv")


def test_item_catalog_is_deterministic_and_sensible(sales):
    a, b = dl.build_item_catalog(sales), dl.build_item_catalog(sales)
    assert a.equals(b) and len(a) == 10
    assert (a["unit_price"] > a["unit_cost"]).all() and a["lead_time_days"].between(3, 14).all()


def test_ensure_database_seeds_once(tmp_path):
    path = tmp_path / "seed.db"
    assert dl.ensure_database(path) is True
    assert dl.ensure_database(path) is False
    assert db.table_row_count("sales_history", path) == 32850
    assert len(db.get_items(path)) == 10 and len(db.get_stores(path)) == 3
    inv = db.get_inventory(path)
    assert len(inv) == 30 and (inv["on_hand"] > 0).all() and (inv["on_order"] == 0).all()


def test_reload_without_reset_preserves_inventory(db_path, sales):
    before = db.get_inventory(db_path).set_index(["store_id", "item_id"])
    db.record_stock_movement(1, 1, "sale", 3, db_path=db_path)
    dl.load_into_database(sales, db_path)
    after = db.get_inventory(db_path).set_index(["store_id", "item_id"])
    assert after.loc[(1, 1), "on_hand"] == before.loc[(1, 1), "on_hand"] - 3
    assert db.table_row_count("sales_history", db_path) == 32850  # upsert, no duplicates


def test_reload_with_reset_replaces_everything(db_path):
    small = dl.generate_synthetic_sales(1, 2, "2024-01-01", "2024-02-29")
    result = dl.load_into_database(small, db_path, reset=True)
    assert result["stores"] == 1 and result["items"] == 2
    assert db.table_row_count("sales_history", db_path) == len(small)
    assert len(db.get_inventory(db_path)) == 2


def test_adding_new_items_keeps_existing_ones(db_path):
    extra = dl.generate_synthetic_sales(1, 1, "2024-01-01", "2024-02-29", seed=3)
    extra["item"] = 99
    dl.load_into_database(extra, db_path)
    assert 99 in set(db.get_items(db_path)["item_id"]) and len(db.get_items(db_path)) == 11
    assert len(db.get_inventory(db_path)) == 31
