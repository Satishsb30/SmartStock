"""Forecast accuracy metrics and backtesting helpers."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

Forecaster = Callable[[pd.Series, int], np.ndarray]


def _as_arrays(actual, predicted) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if a.shape != p.shape:
        raise ValueError(f"Shape mismatch: actual {a.shape} vs predicted {p.shape}")
    if a.size == 0:
        raise ValueError("Cannot compute metrics on empty arrays")
    return a, p


def mae(actual, predicted) -> float:
    a, p = _as_arrays(actual, predicted)
    return float(np.mean(np.abs(a - p)))


def rmse(actual, predicted) -> float:
    a, p = _as_arrays(actual, predicted)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mape(actual, predicted) -> float:
    """Mean absolute percentage error in percent, ignoring days with zero actuals."""
    a, p = _as_arrays(actual, predicted)
    mask = a != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100)


def smape(actual, predicted) -> float:
    """Symmetric MAPE in percent (days where both values are zero count as perfect)."""
    a, p = _as_arrays(actual, predicted)
    denom = np.abs(a) + np.abs(p)
    ratio = np.divide(2 * np.abs(a - p), denom, out=np.zeros_like(denom), where=denom != 0)
    return float(np.mean(ratio) * 100)


def wape(actual, predicted) -> float:
    """Weighted absolute percentage error: total absolute error / total actual, in percent."""
    a, p = _as_arrays(actual, predicted)
    total = np.abs(a).sum()
    if total == 0:
        return float("nan")
    return float(np.abs(a - p).sum() / total * 100)


def bias(actual, predicted) -> float:
    """Mean signed error (positive = over-forecasting)."""
    a, p = _as_arrays(actual, predicted)
    return float(np.mean(p - a))


def evaluate_forecast(actual, predicted) -> dict[str, float]:
    """All metrics in one dictionary."""
    return {
        "MAE": mae(actual, predicted),
        "RMSE": rmse(actual, predicted),
        "MAPE": mape(actual, predicted),
        "sMAPE": smape(actual, predicted),
        "WAPE": wape(actual, predicted),
        "Bias": bias(actual, predicted),
    }


def holdout_split(series: pd.Series, days: int) -> tuple[pd.Series, pd.Series]:
    """Split a series into (train, last ``days`` days as test)."""
    if days < 1 or days >= len(series):
        raise ValueError("days must be between 1 and len(series) - 1")
    return series.iloc[:-days], series.iloc[-days:]


def backtest(series: pd.Series, forecaster: Forecaster, days: int) -> dict[str, float | np.ndarray]:
    """Fit on all but the last ``days`` days, forecast them and score the result.

    Returns the metrics plus ``predicted`` and ``actual`` arrays.
    """
    train, test = holdout_split(series, days)
    predicted = np.clip(np.asarray(forecaster(train, days), dtype=float), 0, None)
    result: dict[str, float | np.ndarray] = dict(evaluate_forecast(test.to_numpy(), predicted))
    result["predicted"] = predicted
    result["actual"] = test.to_numpy()
    return result


def compare_models(
    series: pd.Series, forecasters: dict[str, Forecaster], days: int
) -> pd.DataFrame:
    """Backtest several forecasters on the same holdout; best (lowest MAE) first.

    A model that raises an error is skipped.
    """
    rows = []
    for name, fn in forecasters.items():
        try:
            res = backtest(series, fn, days)
        except Exception:  # noqa: BLE001 - one failing model must not stop the comparison
            continue
        rows.append({"model": name, **{k: v for k, v in res.items() if isinstance(v, float)}})
    if not rows:
        return pd.DataFrame(columns=["model", "MAE", "RMSE", "MAPE", "sMAPE", "WAPE", "Bias"])
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
