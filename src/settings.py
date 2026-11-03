"""Central configuration for SmartStock.

Values come from environment variables (optionally loaded from a ``.env`` file in
the project root) with safe defaults, so the app runs without any setup.
"""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
SCHEMA_PATH: Path = BASE_DIR / "database" / "schema.sql"
TRAIN_CSV_PATH: Path = DATA_DIR / "train.csv"
SAMPLE_CSV_PATH: Path = DATA_DIR / "sample_data.csv"


def _load_env_file(path: Path) -> None:
    """Load KEY=VALUE pairs from a .env file without overriding real env vars."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file(BASE_DIR / ".env")


def _env_int(name: str, default: int, minimum: int | None = None) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    if minimum is not None:
        value = max(minimum, value)
    return value


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    return min(max(value, minimum), maximum)


def _resolve_db_path() -> Path:
    raw = os.environ.get("SMARTSTOCK_DB_PATH", "data/smartstock.db")
    path = Path(raw)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


DB_PATH: Path = _resolve_db_path()

FORECAST_HORIZON_DAYS: int = _env_int("SMARTSTOCK_FORECAST_HORIZON", 30, minimum=1)
SERVICE_LEVEL: float = _env_float("SMARTSTOCK_SERVICE_LEVEL", 0.95, 0.50, 0.999)
DEFAULT_LEAD_TIME_DAYS: int = _env_int("SMARTSTOCK_DEFAULT_LEAD_TIME", 7, minimum=1)
DEFAULT_BUDGET: float = _env_float("SMARTSTOCK_DEFAULT_BUDGET", 50000.0, 0.0, 1e12)
RANDOM_SEED: int = _env_int("SMARTSTOCK_RANDOM_SEED", 42)

# Expected columns of the demand history file (Kaggle "Store Item Demand" layout).
REQUIRED_SALES_COLUMNS: tuple[str, ...] = ("date", "store", "item", "sales")

# Alert thresholds.
LOW_STOCK_DAYS_COVER: float = 7.0
OVERSTOCK_DAYS_COVER: float = 90.0

# Model selection.
MIN_HISTORY_DAYS_FOR_ML: int = 120
MIN_HISTORY_DAYS_FOR_SEASONAL: int = 28
BACKTEST_DAYS: int = 30

# Inventory policy.
REVIEW_PERIOD_DAYS: int = 7        # how often stock is reviewed / orders are placed
ORDER_COST: float = 50.0           # fixed cost per purchase order (used for EOQ)
HOLDING_COST_RATE: float = 0.25    # annual holding cost as a fraction of unit cost
DEMAND_SURGE_RATIO: float = 1.3    # forecast vs recent demand that triggers a surge alert
DEMAND_DROP_RATIO: float = 0.7
