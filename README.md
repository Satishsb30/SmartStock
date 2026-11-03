# SmartStock: AI-Based Inventory Management with Demand Forecasting

SmartStock is a Streamlit application that forecasts daily demand for every store/item combination, turns the
forecasts into inventory policies (safety stock, reorder points, order quantities), allocates a limited purchasing
budget, and raises alerts when stock is at risk. Data is stored in a local SQLite database.

## Features

| Page | What it does |
|---|---|
| **Dashboard** | Headline stock numbers, status chart, inventory value by category, demand outlook, critical alerts |
| **Forecasting** | Forecast one item or all items with Seasonal-Naive, Holt-Winters or LightGBM. `auto` backtests each model on the last 30 days and keeps the most accurate. 90% prediction intervals |
| **Inventory** | Safety stock, reorder point, max stock and recommended order per position. Record receipts, sales, adjustments and write-offs; place purchase orders; download the plan as CSV |
| **Budget Optimizer** | Spends a budget on the most urgent, highest-margin orders first, and shows what different budgets would buy |
| **Alerts** | Stockout, stockout risk, reorder, low stock, overstock, demand surge and demand drop alerts, with resolve and history |
| **Data** | Upload your own CSV, load the bundled data, scale stock levels for demos, reset the database |

## Requirements

- Windows 11 (the code is plain Python and should also run on macOS and Linux)
- Python 3.12
- VS Code (optional)

## Setup on Windows 11 (PowerShell)

```powershell
cd path\to\SmartStock
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell refuses to run the activation script, run this once and try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

In VS Code, press `Ctrl+Shift+P`, choose **Python: Select Interpreter**, and pick the `.venv` interpreter.

Optional: copy `.env.example` to `.env` to change the defaults (see Configuration below).

## Run the app

```powershell
streamlit run app.py
```

The browser opens at http://localhost:8501. On first start the database (`data/smartstock.db`) is created and filled
from `data/train.csv`. Streamlit may log "st.navigation was called in an app with a pages/ directory"; this is a
harmless warning in the terminal.

## Run the tests

```powershell
python -m pytest
```

The suite has about 140 tests covering the database layer, data loading, feature engineering, forecasting, inventory
policy, budget optimizer, alerts, settings and every Streamlit page (driven headlessly). It takes under a minute.

## A suggested first session

1. **Data**: click **Apply stock scaling** (25%) so some items are short. Without this, the generous starting stock
   means almost nothing needs ordering.
2. **Forecasting**: pick an item, press **Run forecast**, look at the model comparison, then press
   **Forecast all and save**.
3. **Inventory**: change the service level in the sidebar and watch safety stock and reorder points move.
4. **Budget Optimizer**: lower the budget and see which lines are left unfunded. Press **Place all approved orders**.
5. **Alerts**: review what needs attention and resolve items.

## Your own data

Upload a CSV on the **Data** page with these columns (names are case-insensitive; `store_id`/`item_id` also work):

```
date,store,item,sales
2024-01-01,1,1,23
2024-01-02,1,1,31
```

- One row per day, store and item. Duplicate rows are summed; negative sales are set to 0; unparseable rows are dropped.
  A file where more than half the rows are unusable is rejected.
- Missing days inside a series are treated as zero sales.
- At least 14 days per series are needed to forecast; about 150 days or more lets LightGBM take part; 2 years or more
  captures yearly seasonality best.
- The file contains demand only. Item names, categories, costs, prices, lead times and starting stock are **generated
  placeholders** (items are named "Item N"). Replace them with real values in the database (`items` and `inventory`
  tables) before relying on cost-based outputs such as order cost and the budget optimizer.

`data/train.csv` and `data/sample_data.csv` are **synthetic**: trend, weekly and yearly seasonality plus random noise.
Forecast accuracy measured on them says nothing about accuracy on real demand.

## Configuration

All settings are optional environment variables (or entries in `.env`):

| Variable | Default | Meaning |
|---|---|---|
| `SMARTSTOCK_DB_PATH` | `data/smartstock.db` | SQLite file; relative paths resolve from the project root |
| `SMARTSTOCK_FORECAST_HORIZON` | `30` | Default forecast length in days |
| `SMARTSTOCK_SERVICE_LEVEL` | `0.95` | Target probability of not stocking out during lead time (0.50 to 0.999) |
| `SMARTSTOCK_DEFAULT_LEAD_TIME` | `7` | Lead time used when an item has none |
| `SMARTSTOCK_DEFAULT_BUDGET` | `50000` | Starting budget on the Budget Optimizer page |
| `SMARTSTOCK_RANDOM_SEED` | `42` | Seed for sample data and models |

Invalid values fall back to the defaults; out-of-range values are clamped. Other tunables (alert thresholds, review
period, order cost, holding cost rate) are constants in `src/settings.py`.

## How it works

**Forecasting** (`src/forecasting.py`)
- *Seasonal-Naive*: average of the same weekday over the last 4 weeks. The baseline.
- *Holt-Winters*: damped additive trend with a 7-day season (statsmodels).
- *LightGBM*: lag (1, 7, 14, 28), rolling mean/std (7, 28) and calendar features, forecasting recursively one day at a
  time. Features are built identically for training and prediction.
- Extreme spikes are capped (Q3 + 3 x IQR) before fitting.
- In `auto` mode each eligible model is fitted on everything except the last 30 days and scored on those days; the
  lowest MAE wins and is refitted on the full history. Prediction intervals use that backtest's RMSE (90%, assuming
  roughly normal errors), so they are an approximation.

**Inventory policy** (`src/inventory.py`)
- Safety stock = z(service level) x daily demand std x sqrt(lead time). The std comes from the last 90 days, so it
  includes weekly swings and is on the conservative side.
- Reorder point = expected lead-time demand + safety stock.
- Max stock = expected demand over (lead time + review period) + safety stock.
- When on hand + on order is at or below the reorder point, the recommended order tops the position up to max stock.
- Demand comes from the saved forecast when one exists, otherwise from the last 90 days.

**Budget optimizer** (`src/budget_optimizer.py`)
- Critical units (back to the reorder point) are funded for all lines first, then top-up units (to max stock).
- Within a stage, lines are ranked by urgency (1 / (1 + days of cover)) x margin per unit of cost.
- This is a transparent greedy heuristic, not a guaranteed optimum. It never exceeds the budget, and no money is left
  unspent while a line could still afford another unit.

## Project structure

```
SmartStock/
├── app.py                  entry point and navigation
├── pages/                  Dashboard, Forecasting, Inventory, Budget Optimizer, Alerts, Data
├── src/
│   ├── settings.py         configuration
│   ├── data_loader.py      CSV loading, validation, synthetic data, database seeding
│   ├── preprocessing.py    series preparation and features
│   ├── forecasting.py      models and automatic selection
│   ├── evaluation.py       metrics and backtesting
│   ├── inventory.py        safety stock, reorder points, order recommendations
│   ├── budget_optimizer.py budget allocation
│   ├── alerts.py           alert rules
│   └── ui.py               shared Streamlit helpers
├── database/               schema.sql and the SQLite access layer
├── data/                   train.csv, sample_data.csv (synthetic)
└── tests/                  pytest suite
```

## Limitations

- Single-user, local SQLite database; no authentication.
- Items are treated independently: no promotions, prices, holidays, stock-outs in the history (censored demand) or
  substitution effects.
- Lead times are fixed per item; supplier variability is not modelled.
- Forecasts are not refreshed automatically: after adding data, run the forecast again.
- Resolving an alert hides it until the next refresh, which raises it again if the condition still holds.

## Troubleshooting

- **`ModuleNotFoundError: src`**: start the app from the project folder (`streamlit run app.py`).
- **Empty pages saying "No data loaded"**: open the **Data** page and load the sample data.
- **Start over**: delete `data/smartstock.db`, or use **Data > Delete all data**.
- **Port in use**: `streamlit run app.py --server.port 8502`.
