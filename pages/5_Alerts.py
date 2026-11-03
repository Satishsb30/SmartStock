"""Alerts: stock and demand warnings with acknowledge-and-resolve."""

import streamlit as st

from database import database as db
from src import alerts, ui

ui.init_page("Alerts", "🔔")
ui.require_data()
ui.policy_controls()
state = ui.load_state()

if "alerts_synced" not in st.session_state:
    alerts.refresh_alerts(state.plan)
    st.session_state["alerts_synced"] = True

top1, top2 = st.columns([1, 3])
if top1.button("Refresh alerts", type="primary"):
    count = len(alerts.refresh_alerts(state.plan))
    ui.flash(f"Alerts refreshed: {count} open.")
    st.rerun()
top2.caption(
    "Alerts are rebuilt from the current inventory plan. Resolving an alert hides it until the next refresh, "
    "which raises it again if the condition still holds."
)

open_alerts = db.get_alerts()
severity_filter = st.multiselect("Severity", ["critical", "warning", "info"], default=["critical", "warning", "info"])
open_alerts = open_alerts[open_alerts["severity"].isin(severity_filter)]

counts = db.get_alerts()["severity"].value_counts()
c1, c2, c3 = st.columns(3)
c1.metric("Critical", int(counts.get("critical", 0)))
c2.metric("Warning", int(counts.get("warning", 0)))
c3.metric("Info", int(counts.get("info", 0)))

if open_alerts.empty:
    st.success("No open alerts for the selected severities.")
else:
    shown = open_alerts.head(50)
    if len(open_alerts) > len(shown):
        st.caption(f"Showing the first {len(shown)} of {len(open_alerts)} alerts.")
    for row in shown.itertuples(index=False):
        text, button = st.columns([6, 1])
        text.markdown(
            f"{ui.SEVERITY_ICONS.get(row.severity, '')} **{row.alert_type.replace('_', ' ').title()}** - {row.message}"
        )
        if button.button("Resolve", key=f"resolve_{row.alert_id}"):
            db.resolve_alert(int(row.alert_id))
            st.rerun()

with st.expander("Resolved alert history"):
    history = db.get_alerts(include_resolved=True)
    history = history[history["resolved"] == 1]
    if history.empty:
        st.write("Nothing resolved yet.")
    else:
        st.dataframe(history.drop(columns="resolved"), hide_index=True, use_container_width=True)
