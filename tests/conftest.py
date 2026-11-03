"""Shared pytest fixtures.

The Streamlit UI tests read the database path from the environment at import
time, so it is pointed at a throwaway directory before anything else is imported.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_UI_DIR = tempfile.mkdtemp(prefix="smartstock_tests_")
os.environ["SMARTSTOCK_DB_PATH"] = os.path.join(_UI_DIR, "ui.db")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from database import database as db  # noqa: E402
from src import data_loader, settings  # noqa: E402


@pytest.fixture(scope="session")
def sales() -> pd.DataFrame:
    """The bundled demand history in loader format (date, store, item, sales)."""
    return data_loader.load_sales_csv(settings.TRAIN_CSV_PATH)


@pytest.fixture(scope="session")
def seeded_template(tmp_path_factory) -> Path:
    """A database seeded from train.csv, built once per test session."""
    path = tmp_path_factory.mktemp("template") / "template.db"
    data_loader.ensure_database(path)
    return path


@pytest.fixture
def db_path(seeded_template, tmp_path) -> Path:
    """A private, seeded copy of the database for one test."""
    target = tmp_path / "test.db"
    shutil.copy(seeded_template, target)
    return target


@pytest.fixture
def empty_db(tmp_path) -> Path:
    path = tmp_path / "empty.db"
    db.init_db(path)
    return path


@pytest.fixture
def history(db_path) -> pd.DataFrame:
    return db.get_sales_history(db_path=db_path)


@pytest.fixture
def ui_db(seeded_template) -> Path:
    """Reset the database used by the Streamlit pages to the seeded state."""
    target = Path(settings.DB_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(seeded_template, target)
    return target
