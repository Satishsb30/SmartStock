"""Test data builders shared across test modules."""

import pandas as pd


def make_case(positions, daily=10.0, days=120, lead=5):
    """Constant demand (zero variance) for items 1..n in store 1, plus matching inventory."""
    dates = pd.date_range("2024-01-01", periods=days)
    history = pd.concat(
        [pd.DataFrame({"date": dates, "store_id": 1, "item_id": i, "sales": daily}) for i in range(1, len(positions) + 1)],
        ignore_index=True,
    )
    inventory = pd.DataFrame(
        [
            {"store_id": 1, "item_id": i, "name": f"Item {i}", "category": "General", "unit_cost": 2.0, "unit_price": 3.0,
             "lead_time_days": lead, "on_hand": oh, "on_order": oo, "reorder_point": 0, "safety_stock": 0, "max_stock": 0,
             "updated_at": ""}
            for i, (oh, oo) in enumerate(positions, start=1)
        ]
    )
    return history, inventory
