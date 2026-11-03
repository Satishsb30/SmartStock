"""Series preparation and feature engineering."""

import numpy as np
import pandas as pd
import pytest

from src import preprocessing as pp


@pytest.fixture
def series(history):
    return pp.extract_series(history, 1, 1)


def test_extract_series_is_complete_daily(series):
    assert len(series) == 1095
    assert (series.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all()
    assert series.name == "sales"


def test_extract_series_fills_gaps_with_zero_and_handles_unknown(history):
    one = history[(history["store_id"] == 1) & (history["item_id"] == 1)]
    gapped = one.drop(index=one.index[10:15])
    out = pp.extract_series(gapped, 1, 1)
    assert len(out) == 1095
    assert (out.iloc[10:15] == 0).all()
    assert pp.extract_series(history, 99, 99).empty


def test_extract_series_accepts_loader_column_names(sales):
    assert len(pp.extract_series(sales, 2, 3)) == 1095


def test_cap_outliers_limits_spikes_only():
    s = pd.Series(np.full(100, 10.0))
    s.iloc[50] = 10_000
    capped = pp.cap_outliers(s)
    assert capped.iloc[50] == 10.0  # IQR is 0, so the fence is the median itself
    assert (capped.drop(50) == 10.0).all()
    assert pp.cap_outliers(pd.Series([1.0, 2.0])).tolist() == [1.0, 2.0]  # too short to judge
    assert pp.cap_outliers(pd.Series(np.zeros(50))).sum() == 0


def test_calendar_features():
    idx = pd.date_range("2024-03-02", periods=3)  # Sat, Sun, Mon
    feats = pp.calendar_features(idx)
    assert feats["dow"].tolist() == [5, 6, 0]
    assert feats["is_weekend"].tolist() == [1, 1, 0]
    assert feats["month"].tolist() == [3, 3, 3]
    assert np.allclose(feats["doy_sin"] ** 2 + feats["doy_cos"] ** 2, 1)


def test_training_frame_has_no_leakage_or_gaps(series):
    frame = pp.build_training_frame(series)
    assert list(frame.columns) == [*pp.FEATURE_COLUMNS, "target"]
    assert not frame.isna().any().any()
    assert len(frame) == len(series) - pp.MAX_LOOKBACK
    date = frame.index[100]
    pos = series.index.get_loc(date)
    assert frame.loc[date, "lag_1"] == series.iloc[pos - 1]
    assert frame.loc[date, "lag_7"] == series.iloc[pos - 7]
    assert frame.loc[date, "roll_mean_7"] == pytest.approx(series.iloc[pos - 7 : pos].mean())
    assert frame.loc[date, "target"] == series.iloc[pos]  # target is today, features only use the past


@pytest.mark.parametrize("pos", [200, 700, 1094])
def test_recursive_features_match_training_features(series, pos):
    frame = pp.build_training_frame(series)
    date = series.index[pos]
    row = pp.next_day_features(series.to_numpy()[:pos], date)
    assert list(row.columns) == list(pp.FEATURE_COLUMNS)
    assert np.allclose(row.iloc[0].to_numpy(float), frame.loc[date, list(pp.FEATURE_COLUMNS)].to_numpy(float))


def test_next_day_features_needs_enough_history(series):
    with pytest.raises(ValueError):
        pp.next_day_features(series.to_numpy()[:5], series.index[5])


def test_demand_statistics(history):
    stats = pp.demand_statistics(history)
    assert len(stats) == 30 and (stats["avg_daily"] > 0).all() and (stats["days_observed"] == 90).all()
    row = stats[(stats["store_id"] == 1) & (stats["item_id"] == 1)].iloc[0]
    last90 = pp.extract_series(history, 1, 1).iloc[-90:]
    assert row["avg_daily"] == pytest.approx(last90.mean())
    assert row["std_daily"] == pytest.approx(last90.std())
    assert row["cv"] == pytest.approx(last90.std() / last90.mean())
    assert pp.demand_statistics(history.iloc[:0]).empty
