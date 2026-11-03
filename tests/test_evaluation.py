"""Accuracy metrics and backtesting."""

import numpy as np
import pandas as pd
import pytest

from src import evaluation as ev


def test_metrics_hand_computed():
    a, p = np.array([10.0, 0.0, 20.0]), np.array([12.0, 1.0, 18.0])
    assert ev.mae(a, p) == pytest.approx(5 / 3)
    assert ev.rmse(a, p) == pytest.approx(np.sqrt(3))
    assert ev.mape(a, p) == pytest.approx(15.0)  # the zero-actual day is ignored
    assert ev.wape(a, p) == pytest.approx(5 / 30 * 100)
    assert ev.bias(a, p) == pytest.approx(1 / 3)
    assert ev.smape([0, 0], [0, 0]) == 0
    assert ev.smape([10], [0]) == pytest.approx(200.0)


def test_undefined_percentages_are_nan():
    assert np.isnan(ev.mape([0, 0], [1, 1]))
    assert np.isnan(ev.wape([0, 0], [1, 1]))


def test_perfect_forecast_scores_zero():
    result = ev.evaluate_forecast([1, 2, 3], [1, 2, 3])
    assert all(v == 0 for v in result.values())
    assert set(result) == {"MAE", "RMSE", "MAPE", "sMAPE", "WAPE", "Bias"}


@pytest.mark.parametrize("a,p", [([1], [1, 2]), ([], [])])
def test_invalid_metric_inputs(a, p):
    with pytest.raises(ValueError):
        ev.mae(a, p)


def _series(n=60):
    return pd.Series(np.arange(n, dtype=float), index=pd.date_range("2024-01-01", periods=n))


def test_holdout_split():
    train, test = ev.holdout_split(_series(), 10)
    assert len(train) == 50 and len(test) == 10 and train.index[-1] < test.index[0]
    for bad in (0, -1, 60, 100):
        with pytest.raises(ValueError):
            ev.holdout_split(_series(), bad)


def test_backtest_scores_a_known_forecaster():
    last_value = lambda train, h: np.full(h, train.iloc[-1])  # noqa: E731
    res = ev.backtest(_series(), last_value, 5)
    assert res["actual"].tolist() == [55, 56, 57, 58, 59]
    assert res["predicted"].tolist() == [54.0] * 5
    assert res["MAE"] == pytest.approx(3.0) and res["Bias"] == pytest.approx(-3.0)


def test_backtest_clips_negative_predictions():
    res = ev.backtest(_series(), lambda train, h: np.full(h, -5.0), 5)
    assert (res["predicted"] == 0).all()


def test_compare_models_sorts_and_skips_failures():
    def broken(train, h):
        raise RuntimeError("fails")

    table = ev.compare_models(
        _series(),
        {"last": lambda t, h: np.full(h, t.iloc[-1]), "zero": lambda t, h: np.zeros(h), "broken": broken},
        5,
    )
    assert table["model"].tolist() == ["last", "zero"]
    assert table["MAE"].is_monotonic_increasing
    assert ev.compare_models(_series(), {"broken": broken}, 5).empty
