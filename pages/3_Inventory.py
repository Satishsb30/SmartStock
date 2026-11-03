"""Inventory: policy per item, stock movements and purchase orders."""

import streamlit as st

from database import database as db
from src import inventory, ui

STATUS_RANK = {s: n for n, s in enumerate(["stockout", "reorder", "ok", "overstock"])}

ui.init_page("Inventory", "📦")
ui.require_data()
ui.policy_controls()
state = ui.load_state()
plan = state.plan

# ---- Filters --------------------------------------------------------------
f1, f2, f3 = st.columns(3)
store_filter = f1.multiselect("Store", sorted(plan["store_id"].unique()))
category_filter = f2.multiselect("Category", sorted(plan["category"].unique()))
status_filter = f3.multiselect("Status", list(ui.STATUS_COLORS))

view = plan
if store_filter:
    view = view[view["store_id"].isin(store_filter)]
if category_filter:
    view = view[view["category"].isin(category_filter)]
if status_filter:
    view = view[view["status"].isin(status_filter)]

st.caption(
    f"{len(view)} of {len(plan)} positions. Policy: {state.service_level:.1%} service level, "
    f"{state.review_days}-day review period. Demand source: forecast where saved, otherwise the last 90 days."
)

columns = [
    "store_id", "item_id", "name", "category", "status", "on_hand", "on_order", "days_of_cover",
    "avg_daily_demand", "safety_stock", "reorder_point", "max_stock", "recommended_qty", "order_cost",
    "lead_time_days", "forecast_used",
]
st.dataframe(
    view[columns]
    .assign(_rank=view["status"].map(STATUS_RANK))
    .sort_values(["_rank", "days_of_cover"])
    .drop(columns="_rank"),
    hide_index=True,
    use_container_width=True,
    column_config={
        "days_of_cover": st.column_config.NumberColumn("Days of cover", format="%.1f"),
        "avg_daily_demand": st.column_config.NumberColumn("Avg daily demand", format="%.1f"),
        "order_cost": st.column_config.NumberColumn("Order cost", format="%.2f"),
        "forecast_used": st.column_config.CheckboxColumn("Forecast"),
    },
)

b1, b2 = st.columns(2)
b1.download_button(
    "Download plan as CSV", view.to_csv(index=False).encode("utf-8"), "inventory_plan.csv", "text/csv"
)
if b2.button("Save policies to database", help="Stores safety stock, reorder point and max stock per position."):
    n = inventory.save_policies(plan)
    ui.flash(f"Saved policies for {n} positions.")
    st.rerun()

# ---- Actions --------------------------------------------------------------
st.divider()
act1, act2 = st.columns(2)
names = {(int(r.store_id), int(r.item_id)): f"Store {int(r.store_id)} - {r.name}" for r in plan.itertuples()}
positions = sorted(names)

with act1:
    st.subheader("Record stock movement")
    with st.form("movement_form", clear_on_submit=True):
        pos = st.selectbox("Position", positions, format_func=names.get)
        mtype = st.selectbox("Type", list(db.MOVEMENT_TYPES), help="Receipts also reduce the on-order quantity.")
        qty = st.number_input(
            "Quantity", value=1.0, step=1.0,
            help="Positive for receipts, sales and write-offs. For adjustments use a signed change.",
        )
        note = st.text_input("Note", "")
        if st.form_submit_button("Record movement"):
            try:
                new_qty = db.record_stock_movement(pos[0], pos[1], mtype, qty, note)
                ui.flash(f"Recorded {mtype}. {names[pos]} now has {new_qty:,.0f} on hand.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

with act2:
    st.subheader("Place purchase order")
    needing = plan[plan["recommended_qty"] > 0]
    st.caption(f"{len(needing)} position(s) currently below their reorder point.")
    with st.form("order_form", clear_on_submit=True):
        pos = st.selectbox("Position ", positions, format_func=names.get)
        suggested = plan[(plan["store_id"] == pos[0]) & (plan["item_id"] == pos[1])]["recommended_qty"].iloc[0]
        qty = st.number_input("Order quantity", min_value=0.0, value=float(suggested), step=1.0,
                              help="Pre-filled with the recommendation for the selected position at the time of loading.")
        if st.form_submit_button("Place order"):
            try:
                total = inventory.place_order(pos[0], pos[1], qty)
                ui.flash(f"Ordered {qty:,.0f} units. {names[pos]} now has {total:,.0f} on order.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

with st.expander("Recent stock movements"):
    movements = db.get_stock_movements(100)
    if movements.empty:
        st.write("No movements recorded yet.")
    else:
        st.dataframe(movements, hide_index=True, use_container_width=True)
