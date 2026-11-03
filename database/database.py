"""SQLite persistence layer for SmartStock.

All SQL used by the application lives here so the rest of the code works with
pandas DataFrames and plain Python values.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import pandas as pd

from src import settings

MOVEMENT_TYPES: tuple[str, ...] = ("receipt", "sale", "adjustment", "write_off")

_TABLES: tuple[str, ...] = (
    "items",
    "stores",
    "sales_history",
    "inventory",
    "stock_movements",
    "forecasts",
    "alerts",
)


def _resolve(db_path: str | Path | None) -> Path:
    return Path(db_path) if db_path is not None else settings.DB_PATH


def get_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Open a connection with foreign keys on and Row access by column name."""
    path = _resolve(db_path)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def connection_scope(db_path: str | Path | None = None) -> Iterator[sqlite3.Connection]:
    """Commit on success, roll back on error, always close."""
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: str | Path | None = None) -> None:
    """Create all tables and indexes (safe to call repeatedly)."""
    schema = settings.SCHEMA_PATH.read_text(encoding="utf-8")
    with connection_scope(db_path) as conn:
        conn.executescript(schema)


def execute(query: str, params: Sequence[Any] = (), db_path: str | Path | None = None) -> int:
    """Run one write statement and return the number of affected rows."""
    with connection_scope(db_path) as conn:
        cur = conn.execute(query, params)
        return cur.rowcount


def executemany(
    query: str, rows: Iterable[Sequence[Any]], db_path: str | Path | None = None
) -> int:
    with connection_scope(db_path) as conn:
        cur = conn.executemany(query, list(rows))
        return cur.rowcount


def read_sql(
    query: str, params: Sequence[Any] = (), db_path: str | Path | None = None
) -> pd.DataFrame:
    """Run a SELECT and return a DataFrame."""
    conn = get_connection(db_path)
    try:
        return pd.read_sql_query(query, conn, params=list(params))
    finally:
        conn.close()


def table_row_count(table: str, db_path: str | Path | None = None) -> int:
    if table not in _TABLES:
        raise ValueError(f"Unknown table: {table}")
    df = read_sql(f"SELECT COUNT(*) AS n FROM {table}", db_path=db_path)
    return int(df.loc[0, "n"])


def is_populated(db_path: str | Path | None = None) -> bool:
    """True when the database already holds items and sales history."""
    try:
        return (
            table_row_count("items", db_path) > 0
            and table_row_count("sales_history", db_path) > 0
        )
    except sqlite3.OperationalError:
        return False


def clear_all(db_path: str | Path | None = None) -> None:
    """Delete every row from every table (schema is kept)."""
    with connection_scope(db_path) as conn:
        for table in _TABLES:
            conn.execute(f"DELETE FROM {table}")


# --------------------------------------------------------------------------- #
# Items and stores
# --------------------------------------------------------------------------- #

def upsert_items(items: pd.DataFrame, db_path: str | Path | None = None) -> int:
    """Insert or update items. Required column: item_id."""
    if items.empty:
        return 0
    df = items.copy()
    defaults = {
        "name": None,
        "category": "General",
        "unit_cost": 1.0,
        "unit_price": 2.0,
        "lead_time_days": settings.DEFAULT_LEAD_TIME_DAYS,
    }
    if "item_id" not in df.columns:
        raise ValueError("items DataFrame needs an 'item_id' column")
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
    df["item_id"] = df["item_id"].astype(int)
    df["name"] = df["name"].where(df["name"].notna(), "Item " + df["item_id"].astype(str))
    rows = [
        (
            int(r.item_id),
            str(r.name),
            str(r.category),
            float(r.unit_cost),
            float(r.unit_price),
            int(r.lead_time_days),
        )
        for r in df.itertuples(index=False)
    ]
    sql = """
        INSERT INTO items (item_id, name, category, unit_cost, unit_price, lead_time_days)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(item_id) DO UPDATE SET
            name = excluded.name,
            category = excluded.category,
            unit_cost = excluded.unit_cost,
            unit_price = excluded.unit_price,
            lead_time_days = excluded.lead_time_days
    """
    return executemany(sql, rows, db_path)


def upsert_stores(stores: pd.DataFrame, db_path: str | Path | None = None) -> int:
    if stores.empty:
        return 0
    df = stores.copy()
    if "store_id" not in df.columns:
        raise ValueError("stores DataFrame needs a 'store_id' column")
    df["store_id"] = df["store_id"].astype(int)
    if "name" not in df.columns:
        df["name"] = "Store " + df["store_id"].astype(str)
    rows = [(int(r.store_id), str(r.name)) for r in df.itertuples(index=False)]
    sql = """
        INSERT INTO stores (store_id, name) VALUES (?, ?)
        ON CONFLICT(store_id) DO UPDATE SET name = excluded.name
    """
    return executemany(sql, rows, db_path)


def get_items(db_path: str | Path | None = None) -> pd.DataFrame:
    return read_sql("SELECT * FROM items ORDER BY item_id", db_path=db_path)


def get_stores(db_path: str | Path | None = None) -> pd.DataFrame:
    return read_sql("SELECT * FROM stores ORDER BY store_id", db_path=db_path)


# --------------------------------------------------------------------------- #
# Sales history
# --------------------------------------------------------------------------- #

def upsert_sales_history(sales: pd.DataFrame, db_path: str | Path | None = None) -> int:
    """Insert or replace daily sales rows. Columns: date, store_id, item_id, sales."""
    needed = {"date", "store_id", "item_id", "sales"}
    missing = needed - set(sales.columns)
    if missing:
        raise ValueError(f"sales DataFrame missing columns: {sorted(missing)}")
    if sales.empty:
        return 0
    df = sales[["date", "store_id", "item_id", "sales"]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    rows = list(
        zip(
            df["date"].tolist(),
            df["store_id"].astype(int).tolist(),
            df["item_id"].astype(int).tolist(),
            df["sales"].astype(float).tolist(),
        )
    )
    sql = """
        INSERT INTO sales_history (date, store_id, item_id, sales)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(date, store_id, item_id) DO UPDATE SET sales = excluded.sales
    """
    return executemany(sql, rows, db_path)


def get_sales_history(
    store_id: int | None = None,
    item_id: int | None = None,
    db_path: str | Path | None = None,
) -> pd.DataFrame:
    """Return sales history with a parsed ``date`` column, oldest first."""
    clauses: list[str] = []
    params: list[Any] = []
    if store_id is not None:
        clauses.append("store_id = ?")
        params.append(int(store_id))
    if item_id is not None:
        clauses.append("item_id = ?")
        params.append(int(item_id))
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    df = read_sql(
        f"SELECT date, store_id, item_id, sales FROM sales_history {where} "
        "ORDER BY store_id, item_id, date",
        params,
        db_path,
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


# --------------------------------------------------------------------------- #
# Inventory and stock movements
# --------------------------------------------------------------------------- #

def upsert_inventory(inventory: pd.DataFrame, db_path: str | Path | None = None) -> int:
    """Insert or update inventory positions. Required: store_id, item_id."""
    if inventory.empty:
        return 0
    df = inventory.copy()
    for col in ("store_id", "item_id"):
        if col not in df.columns:
            raise ValueError(f"inventory DataFrame needs a '{col}' column")
    for col in ("on_hand", "on_order", "reorder_point", "safety_stock", "max_stock"):
        if col not in df.columns:
            df[col] = 0.0
    rows = [
        (
            int(r.store_id),
            int(r.item_id),
            float(r.on_hand),
            float(r.on_order),
            float(r.reorder_point),
            float(r.safety_stock),
            float(r.max_stock),
        )
        for r in df.itertuples(index=False)
    ]
    sql = """
        INSERT INTO inventory
            (store_id, item_id, on_hand, on_order, reorder_point, safety_stock, max_stock, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT(store_id, item_id) DO UPDATE SET
            on_hand = excluded.on_hand,
            on_order = excluded.on_order,
            reorder_point = excluded.reorder_point,
            safety_stock = excluded.safety_stock,
            max_stock = excluded.max_stock,
            updated_at = datetime('now')
    """
    return executemany(sql, rows, db_path)


def get_inventory(db_path: str | Path | None = None) -> pd.DataFrame:
    """Inventory joined with item details for display and calculations."""
    return read_sql(
        """
        SELECT inv.store_id, inv.item_id, i.name, i.category, i.unit_cost, i.unit_price,
               i.lead_time_days, inv.on_hand, inv.on_order, inv.reorder_point,
               inv.safety_stock, inv.max_stock, inv.updated_at
        FROM inventory inv
        JOIN items i ON i.item_id = inv.item_id
        ORDER BY inv.store_id, inv.item_id
        """,
        db_path=db_path,
    )


def record_stock_movement(
    store_id: int,
    item_id: int,
    movement_type: str,
    quantity: float,
    note: str = "",
    db_path: str | Path | None = None,
) -> float:
    """Log a movement and update on-hand stock atomically.

    Quantities are positive numbers. ``receipt`` adds stock; ``sale`` and
    ``write_off`` remove it; ``adjustment`` sets the signed change directly
    (positive or negative). Returns the new on-hand quantity. Raises
    ValueError if the change would drive stock below zero or the position is
    unknown.
    """
    if movement_type not in MOVEMENT_TYPES:
        raise ValueError(f"movement_type must be one of {MOVEMENT_TYPES}")
    qty = float(quantity)
    if movement_type != "adjustment" and qty <= 0:
        raise ValueError("quantity must be greater than zero")
    if movement_type == "adjustment" and qty == 0:
        raise ValueError("adjustment quantity must not be zero")

    delta = qty if movement_type in ("receipt", "adjustment") else -qty

    with connection_scope(db_path) as conn:
        row = conn.execute(
            "SELECT on_hand, on_order FROM inventory WHERE store_id = ? AND item_id = ?",
            (int(store_id), int(item_id)),
        ).fetchone()
        if row is None:
            raise ValueError(f"No inventory record for store {store_id}, item {item_id}")
        new_on_hand = float(row["on_hand"]) + delta
        if new_on_hand < 0:
            raise ValueError(
                f"Insufficient stock: on hand {row['on_hand']:.0f}, change {delta:+.0f}"
            )
        new_on_order = float(row["on_order"])
        if movement_type == "receipt":
            new_on_order = max(0.0, new_on_order - qty)
        conn.execute(
            """
            UPDATE inventory
               SET on_hand = ?, on_order = ?, updated_at = datetime('now')
             WHERE store_id = ? AND item_id = ?
            """,
            (new_on_hand, new_on_order, int(store_id), int(item_id)),
        )
        conn.execute(
            """
            INSERT INTO stock_movements (store_id, item_id, movement_type, quantity, note)
            VALUES (?, ?, ?, ?, ?)
            """,
            (int(store_id), int(item_id), movement_type, delta, note),
        )
    return new_on_hand


def add_on_order(
    store_id: int, item_id: int, quantity: float, db_path: str | Path | None = None
) -> float:
    """Increase the on-order quantity for a position; returns the new on-order total."""
    qty = float(quantity)
    if qty <= 0:
        raise ValueError("quantity must be greater than zero")
    with connection_scope(db_path) as conn:
        cur = conn.execute(
            """
            UPDATE inventory
               SET on_order = on_order + ?, updated_at = datetime('now')
             WHERE store_id = ? AND item_id = ?
            """,
            (qty, int(store_id), int(item_id)),
        )
        if cur.rowcount == 0:
            raise ValueError(f"No inventory record for store {store_id}, item {item_id}")
        row = conn.execute(
            "SELECT on_order FROM inventory WHERE store_id = ? AND item_id = ?",
            (int(store_id), int(item_id)),
        ).fetchone()
    return float(row["on_order"])


def get_stock_movements(
    limit: int = 200, db_path: str | Path | None = None
) -> pd.DataFrame:
    return read_sql(
        """
        SELECT movement_id, store_id, item_id, movement_type, quantity, note, created_at
        FROM stock_movements
        ORDER BY movement_id DESC
        LIMIT ?
        """,
        [int(limit)],
        db_path,
    )


# --------------------------------------------------------------------------- #
# Forecasts
# --------------------------------------------------------------------------- #

def save_forecasts(forecasts: pd.DataFrame, db_path: str | Path | None = None) -> int:
    """Store forecasts. Columns: store_id, item_id, date, predicted, lower, upper, model."""
    needed = {"store_id", "item_id", "date", "predicted", "lower", "upper", "model"}
    missing = needed - set(forecasts.columns)
    if missing:
        raise ValueError(f"forecasts DataFrame missing columns: {sorted(missing)}")
    if forecasts.empty:
        return 0
    df = forecasts.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    rows = [
        (
            int(r.store_id),
            int(r.item_id),
            str(r.date),
            float(r.predicted),
            float(r.lower),
            float(r.upper),
            str(r.model),
        )
        for r in df.itertuples(index=False)
    ]
    sql = """
        INSERT INTO forecasts
            (store_id, item_id, forecast_date, predicted, lower, upper, model, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT(store_id, item_id, forecast_date) DO UPDATE SET
            predicted = excluded.predicted,
            lower = excluded.lower,
            upper = excluded.upper,
            model = excluded.model,
            created_at = datetime('now')
    """
    return executemany(sql, rows, db_path)


def get_forecasts(
    store_id: int | None = None,
    item_id: int | None = None,
    db_path: str | Path | None = None,
) -> pd.DataFrame:
    clauses: list[str] = []
    params: list[Any] = []
    if store_id is not None:
        clauses.append("store_id = ?")
        params.append(int(store_id))
    if item_id is not None:
        clauses.append("item_id = ?")
        params.append(int(item_id))
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    df = read_sql(
        f"""
        SELECT store_id, item_id, forecast_date AS date, predicted, lower, upper, model, created_at
        FROM forecasts {where}
        ORDER BY store_id, item_id, forecast_date
        """,
        params,
        db_path,
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


def clear_forecasts(db_path: str | Path | None = None) -> None:
    execute("DELETE FROM forecasts", db_path=db_path)


# --------------------------------------------------------------------------- #
# Alerts
# --------------------------------------------------------------------------- #

def replace_open_alerts(alerts: pd.DataFrame, db_path: str | Path | None = None) -> int:
    """Replace all unresolved alerts with a fresh set.

    Columns: store_id, item_id, alert_type, severity, message. Resolved alerts
    are kept as history.
    """
    needed = {"store_id", "item_id", "alert_type", "severity", "message"}
    missing = needed - set(alerts.columns)
    if missing:
        raise ValueError(f"alerts DataFrame missing columns: {sorted(missing)}")
    rows = [
        (
            int(r.store_id),
            int(r.item_id),
            str(r.alert_type),
            str(r.severity),
            str(r.message),
        )
        for r in alerts.itertuples(index=False)
    ]
    with connection_scope(db_path) as conn:
        conn.execute("DELETE FROM alerts WHERE resolved = 0")
        if rows:
            conn.executemany(
                """
                INSERT INTO alerts (store_id, item_id, alert_type, severity, message)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
    return len(rows)


def get_alerts(
    include_resolved: bool = False, db_path: str | Path | None = None
) -> pd.DataFrame:
    where = "" if include_resolved else "WHERE resolved = 0"
    return read_sql(
        f"""
        SELECT alert_id, store_id, item_id, alert_type, severity, message, created_at, resolved
        FROM alerts {where}
        ORDER BY
            CASE severity WHEN 'critical' THEN 0 WHEN 'warning' THEN 1 ELSE 2 END,
            alert_id DESC
        """,
        db_path=db_path,
    )


def resolve_alert(alert_id: int, db_path: str | Path | None = None) -> bool:
    return execute("UPDATE alerts SET resolved = 1 WHERE alert_id = ?", [int(alert_id)], db_path) > 0
