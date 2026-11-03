PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS items (
    item_id        INTEGER PRIMARY KEY,
    name           TEXT    NOT NULL,
    category       TEXT    NOT NULL DEFAULT 'General',
    unit_cost      REAL    NOT NULL DEFAULT 1.0 CHECK (unit_cost >= 0),
    unit_price     REAL    NOT NULL DEFAULT 2.0 CHECK (unit_price >= 0),
    lead_time_days INTEGER NOT NULL DEFAULT 7   CHECK (lead_time_days >= 0)
);

CREATE TABLE IF NOT EXISTS stores (
    store_id INTEGER PRIMARY KEY,
    name     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sales_history (
    date     TEXT    NOT NULL,
    store_id INTEGER NOT NULL,
    item_id  INTEGER NOT NULL,
    sales    REAL    NOT NULL CHECK (sales >= 0),
    PRIMARY KEY (date, store_id, item_id)
);

CREATE INDEX IF NOT EXISTS idx_sales_store_item
    ON sales_history (store_id, item_id, date);

CREATE TABLE IF NOT EXISTS inventory (
    store_id       INTEGER NOT NULL,
    item_id        INTEGER NOT NULL,
    on_hand        REAL    NOT NULL DEFAULT 0 CHECK (on_hand >= 0),
    on_order       REAL    NOT NULL DEFAULT 0 CHECK (on_order >= 0),
    reorder_point  REAL    NOT NULL DEFAULT 0,
    safety_stock   REAL    NOT NULL DEFAULT 0,
    max_stock      REAL    NOT NULL DEFAULT 0,
    updated_at     TEXT    NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (store_id, item_id)
);

CREATE TABLE IF NOT EXISTS stock_movements (
    movement_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    store_id      INTEGER NOT NULL,
    item_id       INTEGER NOT NULL,
    movement_type TEXT    NOT NULL CHECK (movement_type IN ('receipt', 'sale', 'adjustment', 'write_off')),
    quantity      REAL    NOT NULL,
    note          TEXT    NOT NULL DEFAULT '',
    created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS forecasts (
    store_id      INTEGER NOT NULL,
    item_id       INTEGER NOT NULL,
    forecast_date TEXT    NOT NULL,
    predicted     REAL    NOT NULL,
    lower         REAL    NOT NULL,
    upper         REAL    NOT NULL,
    model         TEXT    NOT NULL,
    created_at    TEXT    NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (store_id, item_id, forecast_date)
);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    store_id   INTEGER NOT NULL,
    item_id    INTEGER NOT NULL,
    alert_type TEXT    NOT NULL,
    severity   TEXT    NOT NULL CHECK (severity IN ('critical', 'warning', 'info')),
    message    TEXT    NOT NULL,
    created_at TEXT    NOT NULL DEFAULT (datetime('now')),
    resolved   INTEGER NOT NULL DEFAULT 0 CHECK (resolved IN (0, 1))
);

CREATE INDEX IF NOT EXISTS idx_alerts_open ON alerts (resolved, severity);
