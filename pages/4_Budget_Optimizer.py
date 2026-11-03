"""Budget optimizer: decide what to buy when money is limited."""

import plotly.graph_objects as go
import streamlit as st

from src import budget_optimizer, inventory, settings, ui

ui.init_page("Budget Optimizer", "💰")
ui.require_data()
ui.policy_controls()
state = ui.load_state()
plan = state.plan

need = plan[plan["recommended_qty"] > 0]
if need.empty:
    st.success("No positions need ordering right now, so there is nothing to allocate.")
    st.stop()

full_cost = float(need["order_cost"].sum())
budget = st.number_input(
    "Purchasing budget",
    min_value=0.0,
    value=float(settings.DEFAULT_BUDGET),
    step=1000.0,
    help="Default comes from SMARTSTOCK_DEFAULT_BUDGET.",
)

allocation = budget_optimizer.optimize_budget(plan, budget)
summary = budget_optimizer.budget_summary(allocation, budget)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Full recommended spend", f"{full_cost:,.0f}")
m2.metric("Spend under budget", f"{summary['spent']:,.0f}", help=f"{summary['remaining']:,.0f} of the budget is left over.")
m3.metric("Lines fully funded", f"{summary['lines_fully_funded']} / {summary['lines']}")
m4.metric("Critical units unfunded", f"{summary['critical_units_short']:,.0f}")

if summary["critical_units_short"] > 0:
    st.warning(
        "The budget does not cover all critical units (stock needed to get back to the reorder point). "
        "Expect stockouts on the lines marked below."
    )
elif summary["units_short"] > 0:
    st.info("All critical units are funded; some top-up units (up to max stock) are not.")
else:
    st.success("The budget covers every recommended order.")

st.caption(
    "Critical units are funded first, then top-up units. Within each stage, lines are ranked by urgency "
    "(fewer days of cover) times margin per unit of cost. This is a greedy heuristic, not a guaranteed optimum."
)

table = allocation.copy()
table["funding"] = table.apply(
    lambda r: "full" if r["shortfall_qty"] <= 0 else ("critical gap" if r["critical_shortfall_qty"] > 0 else "partial"),
    axis=1,
)
st.dataframe(
    table[["store_id", "item_id", "name", "days_of_cover", "recommended_qty", "approved_qty", "approved_cost",
           "critical_shortfall_qty", "funding"]],
    hide_index=True,
    use_container_width=True,
    column_config={
        "days_of_cover": st.column_config.NumberColumn("Days of cover", format="%.1f"),
        "approved_cost": st.column_config.NumberColumn("Approved cost", format="%.2f"),
    },
)

c1, c2 = st.columns(2)
c1.download_button(
    "Download allocation as CSV", allocation.to_csv(index=False).encode("utf-8"), "budget_allocation.csv", "text/csv"
)
approved = allocation[allocation["approved_qty"] > 0]
if c2.button("Place all approved orders", type="primary", disabled=approved.empty):
    for r in approved.itertuples(index=False):
        inventory.place_order(int(r.store_id), int(r.item_id), float(r.approved_qty))
    ui.flash(f"Placed {len(approved)} orders worth {summary['spent']:,.0f}.")
    st.rerun()

st.subheader("What if the budget were different?")
fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
scenarios = budget_optimizer.budget_scenarios(plan, [full_cost * f for f in fractions])
fig = go.Figure()
fig.add_trace(go.Bar(x=[f"{f:.0%}" for f in fractions], y=scenarios["critical_units_short"],
                     name="Critical units unfunded", marker_color="#c0392b"))
fig.add_trace(go.Bar(x=[f"{f:.0%}" for f in fractions], y=scenarios["units_short"] - scenarios["critical_units_short"],
                     name="Top-up units unfunded", marker_color="#e69f00"))
fig.update_layout(barmode="stack", xaxis_title="Budget as % of full recommended spend", yaxis_title="Units unfunded",
                  margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25))
st.plotly_chart(fig, use_container_width=True)
