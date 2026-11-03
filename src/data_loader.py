"""Loading, validating and seeding demand data for SmartStock.

The expected CSV layout matches the Kaggle "Store Item Demand Forecasting"
dataset: ``date, store, item, sales``. ``store_id`` / ``item_id`` are accepted
as column aliases.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import IO, Union

import numpy as np
import pandas as pd

from database import database as db
from src import settings

PathOrBuffer = Union[str, Path, IO]

_COLUMN_ALIASES = {
    "store_id": "store",
    "item_id": "item",
    "storeid": "store",
    "itemid": "item",
    "demand": "sales",
    "quantity": "sales",
}

_CATEGORIES = ("Beverages", "Snacks", "Dairy", "Bakery", "Household", "Personal Care")


class DataValidationError(ValueError):
    """Raised when an uploaded or loaded file cannot be used."""


# --------------------------------------------------------------------------- #
# Synthetic data
# --------------------------------------------------------------------------- #

def generate_synthetic_sales(
    n_stores: int = 3,
    n_items: int = 10,
    start: str = "2021-01-01",
    end: str = "2023-12-31",
    seed: int | None = None,
) -> pd.DataFrame:
    """Create realistic daily demand with trend, weekly and yearly seasonality."""
    rng = np.random.default_rng(settings.RANDOM_SEED if seed is None else seed)
    dates = pd.date_range(start, end, freq="D")
    n_days = len(dates)
    t = np.arange(n_days)
    dow_profile = np.array([0.85, 0.85, 0.9, 0.95, 1.1, 1.35, 1.2])
    day_of_year = dates.dayofyear.to_numpy()

    frames = []
    for store in range(1, n_stores + 1):
        store_mult = rng.uniform(0.7, 1.4)
        for item in range(1, n_items + 1):
            base = rng.uniform(8, 60)
            trend = rng.uniform(-0.05, 0.25) / 365.0
            amp = rng.uniform(0.05, 0.35)
            phase = rng.uniform(0, 2 * np.pi)
            seasonal = 1 + amp * np.sin(2 * np.pi * day_of_year / 365.25 + phase)
            weekly = dow_profile[dates.dayofweek.to_numpy()]
            level = base * store_mult * (1 + trend * t) * seasonal * weekly
            sales = rng.poisson(np.clip(level, 0.1, None))
            frames.append(
                pd.DataFrame({"date": dates, "store": store, "item": item, "sales": sales})
            )
    return pd.concat(frames, ignore_index=True)


def write_default_datasets(data_dir: str | Path | None = None) -> tuple[Path, Path]:
    """Write ``train.csv`` (full history) and ``sample_data.csv`` (small subset)."""
    out = Path(data_dir) if data_dir else settings.DATA_DIR
    out.mkdir(parents=True, exist_ok=True)
    train = generate_synthetic_sales()
    train_path = out / "train.csv"
    train.to_csv(train_path, index=False, date_format="%Y-%m-%d")

    sample = generate_synthetic_sales(
        n_stores=2, n_items=5, start="2023-07-01", end="2023-12-31", seed=7
    )
    sample_path = out / "sample_data.csv"
    sample.to_csv(sample_path, index=False, date_format="%Y-%m-%d")
    return train_path, sample_path


# --------------------------------------------------------------------------- #
# Loading and validation
# --------------------------------------------------------------------------- #

def validate_sales_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names and types; raise DataValidationError on bad data.

    Returns a DataFrame with columns ``date, store, item, sales`` where duplicate
    (date, store, item) rows are summed, negative sales are clipped to zero and
    rows with missing values are dropped.
    """
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    out = out.rename(columns={k: v for k, v in _COLUMN_ALIASES.items() if k in out.columns})

    missing = [c for c in settings.REQUIRED_SALES_COLUMNS if c not in out.columns]
    if missing:
        raise DataValidationError(
            f"Missing required column(s): {', '.join(missing)}. "
            f"Expected: {', '.join(settings.REQUIRED_SALES_COLUMNS)}."
        )
    out = out[list(settings.REQUIRED_SALES_COLUMNS)]

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for col in ("store", "item", "sales"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    before = len(out)
    out = out.dropna()
    if out.empty:
        raise DataValidationError("No valid rows found after parsing dates and numbers.")
    if len(out) < 0.5 * before:
        raise DataValidationError(
            f"More than half of the rows ({before - len(out)} of {before}) could not be parsed."
        )

    out["store"] = out["store"].astype(int)
    out["item"] = out["item"].astype(int)
    out["sales"] = out["sales"].clip(lower=0).astype(float)
    out["date"] = out["date"].dt.normalize()
    out = out.groupby(["date", "store", "item"], as_index=False)["sales"].sum()
    return out.sort_values(["store", "item", "date"]).reset_index(drop=True)


def load_sales_csv(source: PathOrBuffer) -> pd.DataFrame:
    """Read and validate a demand-history CSV (path or file-like object)."""
    try:
        raw = pd.read_csv(source)
    except FileNotFoundError as exc:
        raise DataValidationError(f"File not found: {source}") from exc
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise DataValidationError(f"Could not read CSV: {exc}") from exc
    return validate_sales_frame(raw)


# --------------------------------------------------------------------------- #
# Building master data from sales
# --------------------------------------------------------------------------- #

def build_item_catalog(sales: pd.DataFrame, seed: int | None = None) -> pd.DataFrame:
    """Create item master data (category, cost, price, lead time) deterministically."""
    rng = np.random.default_rng(settings.RANDOM_SEED if seed is None else seed)
    item_ids = sorted(int(i) for i in sales["item"].unique())
    rows = []
    for item_id in item_ids:
        cost = round(float(rng.uniform(1.5, 25.0)), 2)
        margin = float(rng.uniform(1.25, 1.8))
        rows.append(
            {
                "item_id": item_id,
                "name": f"Item {item_id}",
                "category": _CATEGORIES[(item_id - 1) % len(_CATEGORIES)],
                "unit_cost": cost,
                "unit_price": round(cost * margin, 2),
                "lead_time_days": int(rng.integers(3, 15)),
            }
        )
    return pd.DataFrame(rows)


def build_initial_inventory(
    sales: pd.DataFrame, seed: int | None = None
) -> pd.DataFrame:
    """Starting stock positions: a random 5-40 days of recent average demand."""
    rng = np.random.default_rng(settings.RANDOM_SEED if seed is None else seed)
    last_date = sales["date"].max()
    recent = sales[sales["date"] > last_date - pd.Timedelta(days=28)]
    avg = recent.groupby(["store", "item"])["sales"].mean().reset_index()
    avg["on_hand"] = (avg["sales"] * rng.uniform(5, 40, size=len(avg))).round()
    avg["on_order"] = 0.0
    return avg.rename(columns={"store": "store_id", "item": "item_id"})[
        ["store_id", "item_id", "on_hand", "on_order"]
    ]


# --------------------------------------------------------------------------- #
# Database population
# --------------------------------------------------------------------------- #

def load_into_database(
    sales: pd.DataFrame,
    db_path: str | Path | None = None,
    reset: bool = False,
) -> dict[str, int]:
    """Validate ``sales`` and store it with generated master data and stock levels.

    With ``reset=True`` every existing table is cleared first. Otherwise sales
    are upserted and items/stores/inventory that already exist are preserved.
    """
    clean = validate_sales_frame(sales)
    db.init_db(db_path)
    if reset:
        db.clear_all(db_path)

    existing_items = set(db.get_items(db_path)["item_id"].astype(int))
    existing_inv = db.get_inventory(db_path)
    existing_pairs = set(zip(existing_inv["store_id"].astype(int), existing_inv["item_id"].astype(int)))

    catalog = build_item_catalog(clean)
    new_items = catalog[~catalog["item_id"].isin(existing_items)]
    db.upsert_items(new_items, db_path)

    stores = pd.DataFrame({"store_id": sorted(int(s) for s in clean["store"].unique())})
    existing_stores = set(db.get_stores(db_path)["store_id"].astype(int))
    db.upsert_stores(stores[~stores["store_id"].isin(existing_stores)], db_path)

    n_sales = db.upsert_sales_history(
        clean.rename(columns={"store": "store_id", "item": "item_id"}), db_path
    )

    inv = build_initial_inventory(clean)
    mask = [
        (int(s), int(i)) not in existing_pairs for s, i in zip(inv["store_id"], inv["item_id"])
    ]
    db.upsert_inventory(inv[mask], db_path)

    return {
        "sales_rows": len(clean),
        "items": int(clean["item"].nunique()),
        "stores": int(clean["store"].nunique()),
        "written": int(n_sales),
    }


def ensure_database(db_path: str | Path | None = None) -> bool:
    """Seed an empty database from ``train.csv`` (generating it if missing).

    Returns True if data was loaded, False if the database already had data.
    """
    db.init_db(db_path)
    if db.is_populated(db_path):
        return False
    if not settings.TRAIN_CSV_PATH.is_file():
        write_default_datasets()
    load_into_database(load_sales_csv(settings.TRAIN_CSV_PATH), db_path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="SmartStock data utilities")
    parser.add_argument("--generate", action="store_true", help="write data/train.csv and data/sample_data.csv")
    args = parser.parse_args()
    if args.generate:
        train_path, sample_path = write_default_datasets()
        print(f"Wrote {train_path}\nWrote {sample_path}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
