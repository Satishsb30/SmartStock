"""Alert generation from the inventory plan.

At most one stock-level alert is raised per store/item (the most severe), plus
an optional demand-change alert when forecasts are available.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from database import database as db
from src import settings
from src.inventory import (
    MAX_DAYS_COVER,
    STATUS_OVERSTOCK,
    STATUS_REORDER,
    STATUS_STOCKOUT,
)

ALERT_COLUMNS: tuple[str, ...] = ("store_id", "item_id", "alert_type", "severity", "message")
_SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}


def _label(row: pd.Series) -> str:
    return f"{row['name']} (store {int(row['store_id'])})"


def generate_alerts(plan: pd.DataFrame) -> pd.DataFrame:
    """Alerts for every row of the plan, most severe first."""
    alerts: list[dict] = []
    for _, r in plan.iterrows():
        label = _label(r)
        base = {"store_id": int(r["store_id"]), "item_id": int(r["item_id"])}
        daily = float(r["avg_daily_demand"])
        cover = float(r["days_of_cover"])
        lead = float(r["lead_time_days"])

        if r["status"] == STATUS_STOCKOUT and daily > 0:
            alerts.append(
                {**base, "alert_type": "stockout", "severity": "critical",
                 "message": f"{label} is out of stock; demand is about {daily:.1f} units/day."}
            )
        elif r["status"] == STATUS_REORDER and cover < lead:
            alerts.append(
                {**base, "alert_type": "stockout_risk", "severity": "critical",
                 "message": (
                     f"{label} has {cover:.1f} days of cover but a {lead:.0f}-day lead time; "
                     f"it will run out before a new order arrives. Order {r['recommended_qty']:.0f} units."
                 )}
            )
        elif r["status"] == STATUS_REORDER:
            alerts.append(
                {**base, "alert_type": "reorder", "severity": "warning",
                 "message": (
                     f"{label} is at or below its reorder point "
                     f"({r['inventory_position']:.0f} <= {r['reorder_point']:.0f}). "
                     f"Order {r['recommended_qty']:.0f} units."
                 )}
            )
        elif r["status"] == STATUS_OVERSTOCK:
            cover_txt = "no recent demand" if cover >= MAX_DAYS_COVER else f"{cover:.0f} days of cover"
            tied = float(r["on_hand"]) * float(r["unit_cost"])
            alerts.append(
                {**base, "alert_type": "overstock", "severity": "info",
                 "message": f"{label} has {cover_txt}, tying up {tied:,.0f} in stock."}
            )
        elif cover < settings.LOW_STOCK_DAYS_COVER and daily > 0:
            alerts.append(
                {**base, "alert_type": "low_stock", "severity": "warning",
                 "message": f"{label} has only {cover:.1f} days of cover left (on order: {r['on_order']:.0f})."}
            )

        hist = float(r["hist_avg_daily"])
        if bool(r["forecast_used"]) and hist > 0:
            ratio = daily / hist
            if ratio >= settings.DEMAND_SURGE_RATIO:
                alerts.append(
                    {**base, "alert_type": "demand_surge", "severity": "warning",
                     "message": f"Forecast demand for {label} is {ratio:.0%} of its recent average."}
                )
            elif ratio <= settings.DEMAND_DROP_RATIO:
                alerts.append(
                    {**base, "alert_type": "demand_drop", "severity": "info",
                     "message": f"Forecast demand for {label} is {ratio:.0%} of its recent average."}
                )

    if not alerts:
        return pd.DataFrame(columns=list(ALERT_COLUMNS))
    out = pd.DataFrame(alerts, columns=list(ALERT_COLUMNS))
    out["_rank"] = out["severity"].map(_SEVERITY_ORDER)
    return (
        out.sort_values(["_rank", "store_id", "item_id"], kind="mergesort")
        .drop(columns="_rank")
        .reset_index(drop=True)
    )


def refresh_alerts(plan: pd.DataFrame, db_path: str | Path | None = None) -> pd.DataFrame:
    """Regenerate alerts from ``plan`` and replace the open alerts in the database."""
    alerts = generate_alerts(plan)
    db.replace_open_alerts(alerts, db_path)
    return alerts
