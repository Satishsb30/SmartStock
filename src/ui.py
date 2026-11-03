"""Shared Streamlit helpers: page setup, data loading and charts."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from database import database as db
from src import data_loader, inventory, settings

APP_NAME = "SmartStock"

STATUS_COLORS = {
    inventory.STATUS_STOCKOUT: "#c0392b",
    inventory.STATUS_REORDER: "#e69f00",
    inventory.STATUS_OK: "#2e8b57",
    inventory.STATUS_OVERSTOCK: "#5b7fa6",
}
SEVERITY_ICONS = {"critical": "🔴", "warning": "🟠", "info": "🔵"}
COLOR_HISTORY = "#2f6fba"
COLOR_FORECAST = "#e07b39"


@dataclass
class AppState:
    """Everything a page needs, loaded fresh from the database on each run."""

    history: pd.DataFrame
    inventory: pd.DataFrame
    forecasts: pd.DataFrame
    plan: pd.DataFrame
    service_level: float
    review_days: int


@st.cache_resource(show_spinner="Preparing database...")
def _seed_database(db_path: str) -> bool:
    """Create the schema and load the bundled data into an empty database (once per process)."""
    return data_loader.ensure_database(db_path)


def init_page(title: str, icon: str = "📦") -> None:
    """Page config (ignored if the entry point already set it), database seeding, title."""
    try:
        st.set_page_config(page_title=f"{title} - {APP_NAME}", page_icon=icon, layout="wide")
    except Exception:  # noqa: BLE001 - already configured by app.py
        pass
    _seed_database(str(settings.DB_PATH))
    st.title(f"{icon} {title}")
    show_flash()


def flash(message: str, kind: str = "success") -> None:
    """Queue a message to show after the next rerun."""
    st.session_state["_flash"] = (kind, message)


def show_flash() -> None:
    item = st.session_state.pop("_flash", None)
    if item:
        kind, message = item
        getattr(st, kind, st.info)(message)


def require_data() -> None:
    """Stop the page with a hint when the database has no sales history."""
    if not db.is_populated():
        st.warning("No data loaded yet. Open the **Data** page to load the sample data or upload a CSV.")
        st.stop()


def load_state() -> AppState:
    level = float(st.session_state.get("service_level", settings.SERVICE_LEVEL))
    review = int(st.session_state.get("review_days", settings.REVIEW_PERIOD_DAYS))
    history = db.get_sales_history()
    inv = db.get_inventory()
    forecasts = db.get_forecasts()
    plan = inventory.compute_inventory_plan(inv, history, forecasts, level, review)
    return AppState(history, inv, forecasts, plan, level, review)


def policy_controls() -> tuple[float, int]:
    """Sidebar controls for the inventory policy; values persist across pages."""
    st.sidebar.subheader("Inventory policy")
    level = st.sidebar.slider(
        "Service level",
        min_value=0.80,
        max_value=0.999,
        value=float(st.session_state.get("service_level", settings.SERVICE_LEVEL)),
        step=0.005,
        format="%.3f",
        key="w_service_level",
        help="Probability of not stocking out during the supplier lead time.",
    )
    review = st.sidebar.number_input(
        "Review period (days)",
        min_value=0,
        max_value=60,
        value=int(st.session_state.get("review_days", settings.REVIEW_PERIOD_DAYS)),
        step=1,
        key="w_review_days",
        help="How often orders are placed; orders top stock up to cover lead time plus this period.",
    )
    st.session_state["service_level"] = float(level)
    st.session_state["review_days"] = int(review)
    return float(level), int(review)


def item_labels(items: pd.DataFrame) -> dict[int, str]:
    return {int(r.item_id): f"{r.item_id} - {r.name}" for r in items.itertuples(index=False)}


def store_labels(stores: pd.DataFrame) -> dict[int, str]:
    return {int(r.store_id): f"{r.store_id} - {r.name}" for r in stores.itertuples(index=False)}


def forecast_figure(
    history: pd.Series, forecast: pd.DataFrame, title: str, history_days: int = 120
) -> go.Figure:
    """History, forecast line and prediction interval on one chart."""
    fig = go.Figure()
    tail = history.iloc[-history_days:]
    fig.add_trace(
        go.Scatter(x=tail.index, y=tail.values, name="History", line=dict(color=COLOR_HISTORY, width=1.6))
    )
    if not forecast.empty:
        fig.add_trace(
            go.Scatter(x=forecast["date"], y=forecast["upper"], line=dict(width=0), showlegend=False,
                       hoverinfo="skip")
        )
        fig.add_trace(
            go.Scatter(x=forecast["date"], y=forecast["lower"], fill="tonexty",
                       fillcolor="rgba(224,123,57,0.18)", line=dict(width=0), name="90% interval",
                       hoverinfo="skip")
        )
        fig.add_trace(
            go.Scatter(x=forecast["date"], y=forecast["predicted"], name="Forecast",
                       line=dict(color=COLOR_FORECAST, width=2, dash="dash"))
        )
    fig.update_layout(
        title=title, xaxis_title=None, yaxis_title="Units per day", hovermode="x unified",
        margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.15),
    )
    return fig
