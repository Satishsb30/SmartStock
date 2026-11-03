"""Inventory policy: safety stock, reorder points, order-up-to levels and status.

For each store/item the module combines current stock, supplier lead time and
demand (forecast when available, otherwise recent history):

* safety stock  = z(service level) * daily demand std * sqrt(lead time)
* reorder point = expected lead-time demand + safety stock
* max stock     = expected demand over (lead time + review period) + safety stock
* order         = when inventory position (on hand + on order) is at or below the
                  reorder point, order up to max stock

Daily demand std comes from the last 90 days of history, so it includes weekly
and seasonal swings and is therefore a conservative estimate.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from database import database as db
from src import preprocessing, settings

STATUS_STOCKOUT = "stockout"
STATUS_REORDER = "reorder"
STATUS_OVERSTOCK = "overstock"
STATUS_OK = "ok"

MAX_DAYS_COVER = 9999.0

PLAN_COLUMNS: tuple[str, ...] = (
    "store_id",
    "item_id",
    "name",
    "category",
    "unit_cost",
    "unit_price",
    "lead_time_days",
    "on_hand",
    "on_order",
    "inventory_position",
    "hist_avg_daily",
    "avg_daily_demand",
    "std_daily",
    "forecast_used",
    "safety_stock",
    "reorder_point",
    "max_stock",
    "days_of_cover",
    "recommended_qty",
    "order_cost",
    "eoq",
    "status",
)


def safety_stock(std_daily: float, lead_time_days: float, service_level: float) -> float:
    """z * sigma_daily * sqrt(lead time)."""
    if std_daily < 0 or lead_time_days < 0:
        raise ValueError("std_daily and lead_time_days must be non-negative")
    if not 0.5 <= service_level < 1.0:
        raise ValueError("service_level must be between 0.5 and 1 (exclusive)")
    return float(norm.ppf(service_level) * std_daily * math.sqrt(lead_time_days))


def economic_order_quantity(
    annual_demand: float,
    order_cost: float = settings.ORDER_COST,
    unit_cost: float = 1.0,
    holding_rate: float = settings.HOLDING_COST_RATE,
) -> float:
    """Classic EOQ = sqrt(2 * D * S / H); 0 when demand or holding cost is zero."""
    holding = unit_cost * holding_rate
    if annual_demand <= 0 or holding <= 0 or order_cost <= 0:
        return 0.0
    return float(math.sqrt(2.0 * annual_demand * order_cost / holding))


def _forecast_lookup(forecasts: pd.DataFrame | None) -> dict[tuple[int, int], np.ndarray]:
    if forecasts is None or forecasts.empty:
        return {}
    ordered = forecasts.sort_values(["store_id", "item_id", "date"])
    return {
        (int(s), int(i)): grp["predicted"].to_numpy(dtype=float)
        for (s, i), grp in ordered.groupby(["store_id", "item_id"])
    }


def _expected_demand(preds: np.ndarray | None, days: int, fallback_daily: float) -> float:
    """Expected demand over ``days`` days from forecast values, else history average."""
    if preds is None or len(preds) == 0:
        return fallback_daily * days
    if len(preds) >= days:
        return float(preds[:days].sum())
    return float(preds.mean()) * days


def compute_inventory_plan(
    inventory: pd.DataFrame,
    sales: pd.DataFrame,
    forecasts: pd.DataFrame | None = None,
    service_level: float | None = None,
    review_days: int | None = None,
) -> pd.DataFrame:
    """Policy and order recommendation for every row of ``inventory``.

    ``inventory`` is the output of ``database.get_inventory``; ``sales`` is the
    history from ``database.get_sales_history``; ``forecasts`` (optional) is from
    ``database.get_forecasts``.
    """
    level = settings.SERVICE_LEVEL if service_level is None else float(service_level)
    review = settings.REVIEW_PERIOD_DAYS if review_days is None else int(review_days)
    if review < 0:
        raise ValueError("review_days must be non-negative")
    if not 0.5 <= level < 1.0:
        raise ValueError("service_level must be between 0.5 and 1 (exclusive)")
    if inventory.empty:
        return pd.DataFrame(columns=list(PLAN_COLUMNS))

    stats = preprocessing.demand_statistics(sales) if not sales.empty else pd.DataFrame(
        columns=["store_id", "item_id", "avg_daily", "std_daily"]
    )
    stats = stats[["store_id", "item_id", "avg_daily", "std_daily"]]
    lookup = _forecast_lookup(forecasts)

    merged = inventory.merge(stats, on=["store_id", "item_id"], how="left")
    merged["avg_daily"] = merged["avg_daily"].fillna(0.0)
    merged["std_daily"] = merged["std_daily"].fillna(0.0)

    rows = []
    for r in merged.itertuples(index=False):
        key = (int(r.store_id), int(r.item_id))
        preds = lookup.get(key)
        lead = int(r.lead_time_days) if int(r.lead_time_days) > 0 else settings.DEFAULT_LEAD_TIME_DAYS
        hist_avg = float(r.avg_daily)
        daily = float(preds.mean()) if preds is not None and len(preds) else hist_avg

        ss = safety_stock(float(r.std_daily), lead, level)
        lead_demand = _expected_demand(preds, lead, hist_avg)
        period_demand = _expected_demand(preds, lead + review, hist_avg)
        rop = math.ceil(lead_demand + ss)
        max_stock = max(math.ceil(period_demand + ss), rop)

        on_hand = float(r.on_hand)
        on_order = float(r.on_order)
        position = on_hand + on_order
        qty = max(0, math.ceil(max_stock - position)) if position <= rop else 0

        if daily > 0:
            cover = min(on_hand / daily, MAX_DAYS_COVER)
        else:
            cover = MAX_DAYS_COVER if on_hand > 0 else 0.0

        if on_hand <= 0:
            status = STATUS_STOCKOUT
        elif position <= rop:
            status = STATUS_REORDER
        elif cover > settings.OVERSTOCK_DAYS_COVER:
            status = STATUS_OVERSTOCK
        else:
            status = STATUS_OK

        rows.append(
            {
                "store_id": key[0],
                "item_id": key[1],
                "name": r.name,
                "category": r.category,
                "unit_cost": float(r.unit_cost),
                "unit_price": float(r.unit_price),
                "lead_time_days": lead,
                "on_hand": on_hand,
                "on_order": on_order,
                "inventory_position": position,
                "hist_avg_daily": hist_avg,
                "avg_daily_demand": daily,
                "std_daily": float(r.std_daily),
                "forecast_used": preds is not None and len(preds) > 0,
                "safety_stock": float(math.ceil(ss)),
                "reorder_point": float(rop),
                "max_stock": float(max_stock),
                "days_of_cover": float(cover),
                "recommended_qty": float(qty),
                "order_cost": float(qty * float(r.unit_cost)),
                "eoq": economic_order_quantity(daily * 365, unit_cost=float(r.unit_cost)),
                "status": status,
            }
        )
    return pd.DataFrame(rows, columns=list(PLAN_COLUMNS))


def save_policies(plan: pd.DataFrame, db_path: str | Path | None = None) -> int:
    """Store safety stock, reorder point and max stock on the inventory rows."""
    if plan.empty:
        return 0
    current = db.get_inventory(db_path)[["store_id", "item_id", "on_hand", "on_order"]]
    update = plan[["store_id", "item_id", "safety_stock", "reorder_point", "max_stock"]].merge(
        current, on=["store_id", "item_id"], how="inner"
    )
    return db.upsert_inventory(update, db_path)


def place_order(
    store_id: int, item_id: int, quantity: float, db_path: str | Path | None = None
) -> float:
    """Record a purchase order (adds to on-order). Returns the new on-order total."""
    return db.add_on_order(store_id, item_id, quantity, db_path)


def plan_summary(plan: pd.DataFrame) -> dict[str, float]:
    """Headline numbers for the dashboard."""
    if plan.empty:
        return {
            "positions": 0,
            "stockouts": 0,
            "reorders": 0,
            "overstocks": 0,
            "inventory_value": 0.0,
            "recommended_spend": 0.0,
        }
    return {
        "positions": int(len(plan)),
        "stockouts": int((plan["status"] == STATUS_STOCKOUT).sum()),
        "reorders": int((plan["status"] == STATUS_REORDER).sum()),
        "overstocks": int((plan["status"] == STATUS_OVERSTOCK).sum()),
        "inventory_value": float((plan["on_hand"] * plan["unit_cost"]).sum()),
        "recommended_spend": float(plan["order_cost"].sum()),
    }
