"""Forecasting: run, compare and save demand forecasts."""

import streamlit as st

from database import database as db
from src import forecasting, preprocessing, settings, ui

ui.init_page("Forecasting", "📈")
ui.require_data()
state = ui.load_state()

items = db.get_items()
stores = db.get_stores()
item_names = ui.item_labels(items)
store_names = ui.store_labels(stores)

c1, c2, c3, c4 = st.columns([1, 2, 1, 1])
store_id = c1.selectbox("Store", list(store_names), format_func=store_names.get)
item_id = c2.selectbox("Item", list(item_names), format_func=item_names.get)
horizon = int(c3.number_input("Horizon (days)", 7, 120, settings.FORECAST_HORIZON_DAYS, step=1))
model = c4.selectbox(
    "Model",
    [forecasting.AUTO, *forecasting.MODEL_NAMES],
    help="'auto' backtests every model on the last 30 days and keeps the most accurate one.",
)

series = preprocessing.extract_series(state.history, store_id, item_id)
key = (store_id, item_id, horizon, model)
title = f"{item_names[item_id]} at store {store_id}"

if st.button("Run forecast", type="primary"):
    with st.spinner("Fitting models..."):
        try:
            result = forecasting.select_and_forecast(series, horizon, model)
            st.session_state["fc_result"] = {"key": key, "result": result}
        except ValueError as exc:
            st.session_state.pop("fc_result", None)
            st.error(str(exc))

entry = st.session_state.get("fc_result")
if entry and entry["key"] == key:
    result = entry["result"]
    forecast = result.forecast
    m1, m2, m3 = st.columns(3)
    m1.metric("Model used", result.model.replace("_", " "))
    m2.metric("Expected demand (horizon)", f"{forecast['predicted'].sum():,.0f} units")
    if not result.comparison.empty:
        best = result.comparison.iloc[0]
        m3.metric("Backtest error (WAPE)", f"{best['WAPE']:.1f}%")
    st.plotly_chart(ui.forecast_figure(series, forecast, title), use_container_width=True)

    if not result.comparison.empty:
        st.subheader("Model comparison")
        st.caption(
            f"Each model was fitted on the history before the last {result.backtest_days} days and scored "
            "on those days. Lower is better; the first row was used."
        )
        st.dataframe(result.comparison.round(2), hide_index=True, use_container_width=True)
    else:
        st.caption("History is too short for a backtest, so a simple weekday-average baseline was used.")

    with st.expander("Forecast values"):
        st.dataframe(forecast.round(1), hide_index=True, use_container_width=True)

    if st.button("Save this forecast"):
        out = forecast.copy()
        out["store_id"], out["item_id"], out["model"] = store_id, item_id, result.model
        db.save_forecasts(out)
        ui.flash(f"Saved {len(out)} forecast days for {title}. Inventory plans now use it.")
        st.rerun()
else:
    stored = db.get_forecasts(store_id, item_id)
    if stored.empty:
        st.info("Choose a store, item and model, then press **Run forecast**.")
    else:
        st.caption(
            f"Showing the saved forecast ({stored['model'].iloc[0].replace('_', ' ')}, "
            f"{len(stored)} days). Press **Run forecast** to recalculate."
        )
        st.plotly_chart(ui.forecast_figure(series, stored, title), use_container_width=True)

st.divider()
st.subheader("Forecast every item")
st.write(
    f"Runs the selected model for all {len(state.plan)} store/item combinations and saves the results, "
    "so inventory plans and alerts use forecast demand instead of recent averages."
)
if st.button("Forecast all and save"):
    progress = st.progress(0.0, text="Forecasting...")
    all_forecasts = forecasting.forecast_all(
        state.history,
        horizon,
        model,
        progress=lambda done, total: progress.progress(done / total, text=f"Forecasting {done}/{total}"),
    )
    db.save_forecasts(all_forecasts)
    progress.empty()
    n_series = all_forecasts[["store_id", "item_id"]].drop_duplicates().shape[0]
    ui.flash(f"Saved forecasts for {n_series} store/item combinations.")
    st.rerun()
