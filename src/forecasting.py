"""Demand forecasting models and automatic model selection.

Three models are available:

* ``seasonal_naive``  - average of the same weekday over recent weeks (baseline)
* ``holt_winters``    - damped-trend exponential smoothing with weekly seasonality
* ``lightgbm``        - gradient boosting on lag/calendar features, forecast recursively

``select_and_forecast`` backtests every eligible model on the most recent
days, picks the one with the lowest MAE, then refits it on the full history.
Prediction intervals use the RMSE measured in that backtest (about 90% coverage
if errors are roughly normal and stable over time).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from src import evaluation, preprocessing, settings

SEASONAL_NAIVE = "seasonal_naive"
HOLT_WINTERS = "holt_winters"
LIGHTGBM = "lightgbm"
AUTO = "auto"

MODEL_NAMES: tuple[str, ...] = (SEASONAL_NAIVE, HOLT_WINTERS, LIGHTGBM)
INTERVAL_Z = 1.645  # two-sided 90% interval
MIN_SERIES_DAYS = 14


@dataclass
class ForecastResult:
    """Forecast for one series."""

    forecast: pd.DataFrame  # columns: date, predicted, lower, upper
    model: str
    backtest_days: int = 0
    comparison: pd.DataFrame = field(default_factory=pd.DataFrame)


# --------------------------------------------------------------------------- #
# Point-forecast models: (history, horizon) -> np.ndarray of length horizon
# --------------------------------------------------------------------------- #

def _future_index(series: pd.Series, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")


def seasonal_naive_forecast(series: pd.Series, horizon: int, weeks: int = 4) -> np.ndarray:
    """Mean of the same weekday over the last ``weeks`` weeks."""
    idx = _future_index(series, horizon)
    tail = series.iloc[-7 * weeks:]
    by_dow = tail.groupby(tail.index.dayofweek).mean()
    overall = float(tail.mean())
    return np.array([float(by_dow.get(d, overall)) for d in idx.dayofweek])


def holt_winters_forecast(series: pd.Series, horizon: int) -> np.ndarray:
    """Additive damped-trend Holt-Winters with a 7-day season (last 2 years of data)."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    if len(series) < settings.MIN_HISTORY_DAYS_FOR_SEASONAL:
        return seasonal_naive_forecast(series, horizon)
    data = series.iloc[-730:].astype(float).reset_index(drop=True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                data,
                trend="add",
                damped_trend=True,
                seasonal="add",
                seasonal_periods=7,
                initialization_method="estimated",
            ).fit()
            pred = np.asarray(model.forecast(horizon), dtype=float)
        if not np.all(np.isfinite(pred)):
            raise ValueError("non-finite forecast")
        return pred
    except Exception:  # noqa: BLE001 - fall back to the baseline if fitting fails
        return seasonal_naive_forecast(series, horizon)


def lightgbm_forecast(series: pd.Series, horizon: int) -> np.ndarray:
    """LightGBM on lag/rolling/calendar features, forecasting recursively day by day."""
    import lightgbm as lgb

    frame = preprocessing.build_training_frame(series.iloc[-1100:])
    if len(frame) < 60:
        return holt_winters_forecast(series, horizon)
    x = frame[list(preprocessing.FEATURE_COLUMNS)]
    y = frame["target"]
    model = lgb.LGBMRegressor(
        n_estimators=250,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=10,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.9,
        random_state=settings.RANDOM_SEED,
        deterministic=True,
        force_row_wise=True,
        n_jobs=1,
        verbose=-1,
    )
    model.fit(x, y)

    history = series.to_numpy(dtype=float)
    idx = _future_index(series, horizon)
    preds = np.empty(horizon)
    for step, date in enumerate(idx):
        feats = preprocessing.next_day_features(history, date)
        value = max(0.0, float(model.predict(feats)[0]))
        preds[step] = value
        history = np.append(history, value)
    return preds


MODEL_FUNCTIONS: dict[str, Callable[[pd.Series, int], np.ndarray]] = {
    SEASONAL_NAIVE: seasonal_naive_forecast,
    HOLT_WINTERS: holt_winters_forecast,
    LIGHTGBM: lightgbm_forecast,
}


# --------------------------------------------------------------------------- #
# Selection and interval construction
# --------------------------------------------------------------------------- #

def _min_train_days(model: str) -> int:
    if model == LIGHTGBM:
        return settings.MIN_HISTORY_DAYS_FOR_ML
    if model == HOLT_WINTERS:
        return settings.MIN_HISTORY_DAYS_FOR_SEASONAL
    return MIN_SERIES_DAYS


def _build_result(
    series: pd.Series,
    point: np.ndarray,
    sigma: float,
    model: str,
    backtest_days: int,
    comparison: pd.DataFrame,
) -> ForecastResult:
    point = np.clip(point, 0, None)
    half_width = INTERVAL_Z * sigma
    frame = pd.DataFrame(
        {
            "date": _future_index(series, len(point)),
            "predicted": point,
            "lower": np.clip(point - half_width, 0, None),
            "upper": point + half_width,
        }
    )
    return ForecastResult(frame, model, backtest_days, comparison)


def select_and_forecast(
    series: pd.Series,
    horizon: int | None = None,
    model: str = AUTO,
    backtest_days: int | None = None,
) -> ForecastResult:
    """Forecast one daily series.

    ``model`` is ``"auto"`` (pick the best by backtest MAE) or one of
    ``MODEL_NAMES``. Raises ValueError for series shorter than 14 days or an
    unknown model name.
    """
    horizon = settings.FORECAST_HORIZON_DAYS if horizon is None else int(horizon)
    if horizon < 1:
        raise ValueError("horizon must be at least 1 day")
    if model != AUTO and model not in MODEL_FUNCTIONS:
        raise ValueError(f"Unknown model '{model}'. Choose from {AUTO}, {', '.join(MODEL_NAMES)}")
    if len(series) < MIN_SERIES_DAYS:
        raise ValueError(f"Need at least {MIN_SERIES_DAYS} days of history, got {len(series)}")

    series = preprocessing.cap_outliers(series.astype(float))
    days = min(backtest_days or settings.BACKTEST_DAYS, len(series) // 4)

    # Too little data to hold anything out: use the baseline and the series spread.
    if days < 7:
        point = seasonal_naive_forecast(series, horizon)
        sigma = float(series.std(ddof=1)) if len(series) > 1 else 0.0
        return _build_result(series, point, sigma, SEASONAL_NAIVE, 0, pd.DataFrame())

    candidates = list(MODEL_NAMES) if model == AUTO else [model]
    eligible = [m for m in candidates if len(series) - days >= _min_train_days(m)]
    if not eligible:
        eligible = [SEASONAL_NAIVE]

    comparison = evaluation.compare_models(
        series, {m: MODEL_FUNCTIONS[m] for m in eligible}, days
    )
    if comparison.empty:
        eligible = [SEASONAL_NAIVE]
        comparison = evaluation.compare_models(
            series, {SEASONAL_NAIVE: seasonal_naive_forecast}, days
        )
    best = str(comparison.loc[0, "model"])
    sigma = float(comparison.loc[0, "RMSE"])

    point = MODEL_FUNCTIONS[best](series, horizon)
    return _build_result(series, point, sigma, best, days, comparison)


def forecast_all(
    sales: pd.DataFrame,
    horizon: int | None = None,
    model: str = AUTO,
    pairs: list[tuple[int, int]] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Forecast every store/item pair (or just ``pairs``).

    Returns rows ready for ``database.save_forecasts``:
    store_id, item_id, date, predicted, lower, upper, model. Series that are
    too short to forecast are skipped.
    """
    s_col = "store_id" if "store_id" in sales.columns else "store"
    i_col = "item_id" if "item_id" in sales.columns else "item"
    if pairs is None:
        pairs = sorted(
            (int(s), int(i)) for s, i in sales[[s_col, i_col]].drop_duplicates().itertuples(index=False)
        )
    frames = []
    for n, (store, item) in enumerate(pairs, start=1):
        series = preprocessing.extract_series(sales, store, item)
        if len(series) >= MIN_SERIES_DAYS:
            res = select_and_forecast(series, horizon, model)
            out = res.forecast.copy()
            out.insert(0, "item_id", item)
            out.insert(0, "store_id", store)
            out["model"] = res.model
            frames.append(out)
        if progress:
            progress(n, len(pairs))
    if not frames:
        return pd.DataFrame(
            columns=["store_id", "item_id", "date", "predicted", "lower", "upper", "model"]
        )
    return pd.concat(frames, ignore_index=True)
