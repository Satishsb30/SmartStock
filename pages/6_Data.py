"""Data: load, upload, reset and demo controls."""

import streamlit as st

from database import database as db
from src import data_loader, preprocessing, settings, ui

ui.init_page("Data", "🗂️")

populated = db.is_populated()
if populated:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stores", db.table_row_count("stores"))
    c2.metric("Items", db.table_row_count("items"))
    c3.metric("Sales rows", f"{db.table_row_count('sales_history'):,}")
    hist = db.get_sales_history()
    c4.metric("History", f"{hist['date'].min():%Y-%m-%d} to {hist['date'].max():%Y-%m-%d}")
else:
    st.warning("The database is empty. Load the bundled data or upload your own CSV below.")


def _after_load(summary: dict, label: str) -> None:
    st.session_state.pop("fc_result", None)
    st.session_state.pop("alerts_synced", None)
    ui.flash(f"{label}: {summary['sales_rows']:,} rows, {summary['stores']} stores, {summary['items']} items.")
    st.rerun()


st.subheader("Upload demand history")
st.caption("CSV with columns `date`, `store`, `item`, `sales` (one row per day, store and item).")
upload = st.file_uploader("CSV file", type="csv")
if upload is not None:
    try:
        parsed = data_loader.load_sales_csv(upload)
    except data_loader.DataValidationError as exc:
        st.error(str(exc))
    else:
        st.success(
            f"Valid file: {len(parsed):,} rows, {parsed['store'].nunique()} stores, {parsed['item'].nunique()} items, "
            f"{parsed['date'].min():%Y-%m-%d} to {parsed['date'].max():%Y-%m-%d}."
        )
        st.dataframe(parsed.head(10), hide_index=True, use_container_width=True)
        short = [
            (s, i) for (s, i), n in parsed.groupby(["store", "item"]).size().items() if n < 28
        ]
        if short:
            st.warning(f"{len(short)} store/item series have under 28 days of history; forecasts will be basic.")
        mode = st.radio(
            "How should it be loaded?",
            ["Replace all existing data", "Add to existing data"],
            help="Replacing also clears inventory, forecasts and alerts. Adding keeps existing stock levels.",
        )
        if st.button("Load file", type="primary"):
            summary = data_loader.load_into_database(parsed, reset=mode.startswith("Replace"))
            _after_load(summary, "Loaded your file")

st.divider()
st.subheader("Bundled datasets")
b1, b2 = st.columns(2)
if b1.button("Load full demo data (train.csv)", help="Replaces all existing data."):
    if not settings.TRAIN_CSV_PATH.is_file():
        data_loader.write_default_datasets()
    summary = data_loader.load_into_database(data_loader.load_sales_csv(settings.TRAIN_CSV_PATH), reset=True)
    _after_load(summary, "Loaded train.csv")
if b2.button("Load small sample (sample_data.csv)", help="Replaces all existing data."):
    if not settings.SAMPLE_CSV_PATH.is_file():
        data_loader.write_default_datasets()
    summary = data_loader.load_into_database(data_loader.load_sales_csv(settings.SAMPLE_CSV_PATH), reset=True)
    _after_load(summary, "Loaded sample_data.csv")

if populated:
    st.divider()
    st.subheader("Demo tools")
    st.caption("Scale every on-hand quantity to see how reorder recommendations, alerts and budgets react.")
    factor = st.slider("On-hand stock as a share of current", 0.05, 1.0, 0.25, step=0.05)
    if st.button("Apply stock scaling"):
        inv = db.get_inventory()
        inv["on_hand"] = (inv["on_hand"] * factor).round()
        db.upsert_inventory(inv[["store_id", "item_id", "on_hand", "on_order", "reorder_point", "safety_stock", "max_stock"]])
        ui.flash(f"On-hand stock scaled to {factor:.0%} for {len(inv)} positions.")
        st.rerun()

    st.divider()
    st.subheader("Reset")
    confirm = st.checkbox("I understand this deletes all data in the database")
    if st.button("Delete all data", disabled=not confirm):
        db.clear_all()
        st.session_state.pop("fc_result", None)
        st.session_state.pop("alerts_synced", None)
        ui.flash("All data deleted.", "warning")
        st.rerun()

    st.divider()
    with st.expander("Browse sales history"):
        items = db.get_items()
        stores = db.get_stores()
        store_names, item_names = ui.store_labels(stores), ui.item_labels(items)
        s = st.selectbox("Store", list(store_names), format_func=store_names.get, key="browse_store")
        i = st.selectbox("Item", list(item_names), format_func=item_names.get, key="browse_item")
        series = preprocessing.extract_series(hist, s, i)
        if series.empty:
            st.write("No history for this combination.")
        else:
            st.line_chart(series.rename("units sold"))
