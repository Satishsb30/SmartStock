"""Forecasting models and automatic selection."""

import numpy as np
import pandas as pd
import pytest

from src import evaluation as ev
from src import forecasting as fc
from src import preprocessing as pp


@pytest.fixture(scope="module")
def series(sales):
    return pp.extract_series(sales, 1, 1)


def _weekly(n_weeks=20, pattern=(5, 6, 7, 8, 12, 20, 15)):
    values = np.tile(np.array(pattern, dtype=float), n_weeks)
    return pd.Series(values, index=pd.date_range("2024-01-01", periods=len(values)))  # starts on a Monday


@pytest.mark.parametrize("name", fc.MODEL_NAMES)
def test_every_model_returns_finite_non_negative_forecast(series, name):
    out = fc.MODEL_FUNCTIONS[name](series, 30)
    assert out.shape == (30,) and np.isfinite(out).all() and (out >= 0).all()


def test_seasonal_naive_reproduces_a_perfect_weekly_pattern():
    s = _weekly()
    out = fc.seasonal_naive_forecast(s, 14)
    expected = np.tile(np.array([5, 6, 7, 8, 12, 20, 15], dtype=float), 2)  # next day is a Monday again
    assert np.allclose(out, expected)


def test_holt_winters_and_lightgbm_learn_a_weekly_pattern():
    s = _weekly(n_weeks=40)
    expected = np.tile(np.array([5, 6, 7, 8, 12, 20, 15], dtype=float), 2)
    for model in (fc.holt_winters_forecast, fc.lightgbm_forecast):
        assert ev.mae(expected, model(s, 14)) < 1.5, model.__name__


def test_models_beat_a_flat_average_on_seasonal_data(series):
    res = ev.compare_models(series, fc.MODEL_FUNCTIONS, 30).set_index("model")
    flat = ev.backtest(series, lambda tr, h: np.full(h, tr.iloc[-60:].mean()), 30)["MAE"]
    assert res.loc["holt_winters", "MAE"] < flat
    assert res.loc["lightgbm", "MAE"] < flat


def test_auto_selection_picks_the_best_backtest_model_and_is_deterministic(series):
    first = fc.select_and_forecast(series, 30)
    second = fc.select_and_forecast(series, 30)
    assert first.model == first.comparison.iloc[0]["model"]
    assert first.comparison["MAE"].is_monotonic_increasing
    assert first.model == second.model and np.allclose(first.forecast["predicted"], second.forecast["predicted"])
    assert first.backtest_days == 30


def test_forecast_frame_shape_and_interval_ordering(series):
    res = fc.select_and_forecast(series, 21)
    f = res.forecast
    assert list(f.columns) == ["date", "predicted", "lower", "upper"] and len(f) == 21
    assert f["date"].iloc[0] == series.index[-1] + pd.Timedelta(days=1)
    assert (f["date"].diff().dropna() == pd.Timedelta(days=1)).all()
    assert (f["lower"] >= 0).all() and (f["lower"] <= f["predicted"]).all() and (f["predicted"] <= f["upper"]).all()
    assert (f["upper"] > f["lower"]).any()


def test_interval_coverage_is_reasonable_on_a_true_holdout(sales):
    hits = total = 0
    for item in range(1, 6):
        s = pp.extract_series(sales, 2, item)
        f = fc.select_and_forecast(s.iloc[:-30], 30).forecast
        actual = s.iloc[-30:].to_numpy()
        hits += int(((actual >= f["lower"].to_numpy()) & (actual <= f["upper"].to_numpy())).sum())
        total += 30
    assert hits / total >= 0.70  # nominal is 90%; allow slack so the test is stable, not flattering


def test_forcing_a_model(series):
    assert fc.select_and_forecast(series, 7, model="holt_winters").model == "holt_winters"
    assert fc.select_and_forecast(series, 7, model="seasonal_naive").model == "seasonal_naive"


@pytest.mark.parametrize(
    "kwargs",
    [{"model": "nope"}, {"horizon": 0}, {"horizon": -3}],
)
def test_invalid_arguments(series, kwargs):
    with pytest.raises(ValueError):
        fc.select_and_forecast(series, **{"horizon": 7, **kwargs})


def test_too_short_series_is_rejected(series):
    with pytest.raises(ValueError):
        fc.select_and_forecast(series.iloc[:13], 7)


@pytest.mark.parametrize("days,expected", [(14, "seasonal_naive"), (40, "seasonal_naive")])
def test_short_series_fall_back_to_the_baseline(series, days, expected):
    res = fc.select_and_forecast(series.iloc[:days], 14)
    assert res.model == expected
    assert len(res.forecast) == 14 and np.isfinite(res.forecast[["predicted", "lower", "upper"]].to_numpy()).all()


def test_lightgbm_needs_enough_history_to_be_considered(series):
    assert "lightgbm" not in fc.select_and_forecast(series.iloc[:100], 14).comparison["model"].tolist()
    assert "lightgbm" in fc.select_and_forecast(series.iloc[:200], 14).comparison["model"].tolist()


def test_all_zero_series_forecasts_zero():
    z = pd.Series(0.0, index=pd.date_range("2024-01-01", periods=200))
    out = fc.select_and_forecast(z, 10).forecast
    assert (out[["predicted", "lower", "upper"]] == 0).all().all()


def test_huge_spike_does_not_distort_the_forecast(series):
    spiky = series.copy()
    spiky.iloc[-5] = 100_000
    clean = fc.select_and_forecast(series, 14).forecast["predicted"].mean()
    dirty = fc.select_and_forecast(spiky, 14).forecast["predicted"].mean()
    assert dirty < clean * 1.5


def test_forecast_all(sales):
    out = fc.forecast_all(sales, 14, pairs=[(1, 1), (1, 2), (3, 10)])
    assert set(out.columns) == {"store_id", "item_id", "date", "predicted", "lower", "upper", "model"}
    assert len(out) == 42 and out.groupby(["store_id", "item_id"]).size().eq(14).all()
    seen = []
    fc.forecast_all(sales, 7, model="seasonal_naive", pairs=[(1, 1), (2, 2)], progress=lambda d, t: seen.append((d, t)))
    assert seen == [(1, 2), (2, 2)]


def test_forecast_all_skips_series_that_are_too_short():
    tiny = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=5), "store_id": 1, "item_id": 1, "sales": 3.0})
    assert fc.forecast_all(tiny, 7).empty
