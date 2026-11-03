"""SmartStock - entry point. Run with:  streamlit run app.py"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="SmartStock", page_icon="📦", layout="wide")

PAGES = [
    st.Page("pages/1_Dashboard.py", title="Dashboard", icon="🏠", default=True),
    st.Page("pages/2_Forecasting.py", title="Forecasting", icon="📈"),
    st.Page("pages/3_Inventory.py", title="Inventory", icon="📦"),
    st.Page("pages/4_Budget_Optimizer.py", title="Budget Optimizer", icon="💰"),
    st.Page("pages/5_Alerts.py", title="Alerts", icon="🔔"),
    st.Page("pages/6_Data.py", title="Data", icon="🗂️"),
]

st.sidebar.markdown("### 📦 SmartStock")
st.sidebar.caption("AI-based inventory management with demand forecasting")

st.navigation(PAGES).run()
