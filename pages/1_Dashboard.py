"""Dashboard: headline numbers, stock status, demand outlook and critical alerts."""

import pandas as pd
import plotly.express as px
import streamlit as st

from src import alerts, inventory, ui

ui.init_page("Dashboard", "🏠")
ui.require_data()
state = ui.load_state()
plan = state.plan

summary = inventory.plan_summary(plan)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Stock positions", f"{summary['positions']:,}")
c2.metric("Stockouts", f"{summary['stockouts']:,}")
c3.metric("Need reorder", f"{summary['reorders']:,}")
c4.metric("Overstocked", f"{summary['overstocks']:,}")
c5.metric("Inventory value (at cost)", f"{summary['inventory_value']:,.0f}")
st.caption(
    f"Recommended purchase spend right now: **{summary['recommended_spend']:,.0f}** "
    f"(service level {state.service_level:.1%}, review period {state.review_days} days)."
)

left, right = st.columns(2)

with left:
    counts = (
        plan["status"].value_counts().reindex(list(ui.STATUS_COLORS), fill_value=0).rename_axis("status")
        .reset_index(name="positions")
    )
    fig = px.bar(counts, x="status", y="positions", color="status", color_discrete_map=ui.STATUS_COLORS,
                 title="Stock status")
    fig.update_layout(showlegend=False, xaxis_title=None, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

with right:
    plan_value = plan.assign(value=plan["on_hand"] * plan["unit_cost"])
    by_cat = plan_value.groupby("category", as_index=False)["value"].sum().sort_values("value")
    fig = px.bar(by_cat, x="value", y="category", orientation="h", title="Inventory value by category")
    fig.update_traces(marker_color=ui.COLOR_HISTORY)
    fig.update_layout(xaxis_title="Value at cost", yaxis_title=None, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Demand outlook")
fc = state.forecasts
if fc.empty:
    st.info("No forecasts saved yet. Open **Forecasting** and run a forecast (or forecast all items).")
else:
    pairs = fc[["store_id", "item_id"]].drop_duplicates()
    hist = state.history.merge(pairs, on=["store_id", "item_id"])
    daily = hist.groupby("date")["sales"].sum().iloc[-90:]
    future = fc.groupby("date")[["predicted", "lower", "upper"]].sum().reset_index()
    st.plotly_chart(
        ui.forecast_figure(daily, future, f"Total daily demand for the {len(pairs)} forecasted item(s)",
                           history_days=90),
        use_container_width=True,
    )

st.subheader("Critical alerts")
live = alerts.generate_alerts(plan)
critical = live[live["severity"] == "critical"]
if critical.empty:
    st.success("No critical alerts.")
else:
    st.write(f"{len(critical)} critical alert(s); showing the first 10. See **Alerts** for everything.")
    shown = critical.head(10).copy()
    shown["alert"] = shown["alert_type"].str.replace("_", " ")
    st.dataframe(shown[["store_id", "item_id", "alert", "message"]], hide_index=True, use_container_width=True)
