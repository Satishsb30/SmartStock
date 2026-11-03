"""Time-series preparation and feature engineering for SmartStock."""

from __future__ import annotations

import numpy as np
import pandas as pd

LAGS: tuple[int, ...] = (1, 7, 14, 28)
ROLL_WINDOWS: tuple[int, ...] = (7, 28)
MAX_LOOKBACK: int = max(max(LAGS), max(ROLL_WINDOWS))

FEATURE_COLUMNS: tuple[str, ...] = (
    "dow",
    "day_of_month",
    "month",
    "week_of_year",
    "is_weekend",
    "doy_sin",
    "doy_cos",
    *(f"lag_{k}" for k in LAGS),
    *(f"roll_mean_{w}" for w in ROLL_WINDOWS),
    *(f"roll_std_{w}" for w in ROLL_WINDOWS),
)


def extract_series(sales: pd.DataFrame, store_id: int, item_id: int) -> pd.Series:
    """Daily demand for one store/item as a Series with a complete daily index.

    Accepts a frame with either ``store_id/item_id`` or ``store/item`` columns.
    Missing days are filled with zero (no recorded sales).
    """
    s_col = "store_id" if "store_id" in sales.columns else "store"
    i_col = "item_id" if "item_id" in sales.columns else "item"
    sub = sales[(sales[s_col] == store_id) & (sales[i_col] == item_id)]
    if sub.empty:
        return pd.Series(dtype=float, name="sales")
    series = sub.groupby("date")["sales"].sum().sort_index()
    full = pd.date_range(series.index.min(), series.index.max(), freq="D")
    series = series.reindex(full, fill_value=0.0).astype(float)
    series.index.name = "date"
    series.name = "sales"
    return series


def cap_outliers(series: pd.Series, iqr_multiplier: float = 3.0) -> pd.Series:
    """Cap extreme spikes at Q3 + k*IQR (lower values are left untouched)."""
    if len(series) < 8:
        return series.copy()
    q1, q3 = series.quantile([0.25, 0.75])
    upper = q3 + iqr_multiplier * (q3 - q1)
    if upper <= 0:
        return series.copy()
    return series.clip(upper=upper)


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar features for each date in ``index``."""
    doy = index.dayofyear.to_numpy()
    return pd.DataFrame(
        {
            "dow": index.dayofweek.to_numpy(),
            "day_of_month": index.day.to_numpy(),
            "month": index.month.to_numpy(),
            "week_of_year": index.isocalendar().week.to_numpy().astype(int),
            "is_weekend": (index.dayofweek.to_numpy() >= 5).astype(int),
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        },
        index=index,
    )


def build_training_frame(series: pd.Series) -> pd.DataFrame:
    """Supervised frame (features + ``target``) for gradient boosting.

    Lags and rolling statistics only use values strictly before each date, so
    there is no leakage. The first ``MAX_LOOKBACK`` days are dropped.
    """
    frame = calendar_features(series.index)
    for k in LAGS:
        frame[f"lag_{k}"] = series.shift(k)
    shifted = series.shift(1)
    for w in ROLL_WINDOWS:
        frame[f"roll_mean_{w}"] = shifted.rolling(w).mean()
        frame[f"roll_std_{w}"] = shifted.rolling(w).std()
    frame["target"] = series
    return frame.dropna().loc[:, [*FEATURE_COLUMNS, "target"]]


def next_day_features(history: np.ndarray, date: pd.Timestamp) -> pd.DataFrame:
    """One feature row for ``date`` given all earlier daily values (oldest first).

    Matches ``build_training_frame`` exactly so recursive multi-step forecasts
    stay consistent with training.
    """
    if len(history) < MAX_LOOKBACK:
        raise ValueError(f"Need at least {MAX_LOOKBACK} days of history, got {len(history)}")
    row = calendar_features(pd.DatetimeIndex([date])).iloc[0].to_dict()
    for k in LAGS:
        row[f"lag_{k}"] = float(history[-k])
    for w in ROLL_WINDOWS:
        window = history[-w:]
        row[f"roll_mean_{w}"] = float(window.mean())
        row[f"roll_std_{w}"] = float(window.std(ddof=1))
    return pd.DataFrame([row], columns=list(FEATURE_COLUMNS))


def demand_statistics(sales: pd.DataFrame, window_days: int = 90) -> pd.DataFrame:
    """Per store/item demand statistics over the most recent ``window_days``.

    Columns: store_id, item_id, avg_daily, std_daily, cv, total_units, days_observed.
    """
    s_col = "store_id" if "store_id" in sales.columns else "store"
    i_col = "item_id" if "item_id" in sales.columns else "item"
    if sales.empty:
        return pd.DataFrame(
            columns=["store_id", "item_id", "avg_daily", "std_daily", "cv", "total_units", "days_observed"]
        )
    cutoff = sales["date"].max() - pd.Timedelta(days=window_days - 1)
    recent = sales[sales["date"] >= cutoff]
    grouped = recent.groupby([s_col, i_col])["sales"]
    stats = grouped.agg(avg_daily="mean", std_daily="std", total_units="sum", days_observed="count")
    stats["std_daily"] = stats["std_daily"].fillna(0.0)
    stats["cv"] = np.where(stats["avg_daily"] > 0, stats["std_daily"] / stats["avg_daily"], 0.0)
    stats = stats.reset_index().rename(columns={s_col: "store_id", i_col: "item_id"})
    return stats[["store_id", "item_id", "avg_daily", "std_daily", "cv", "total_units", "days_observed"]]
