"""Settings parsing (run in subprocesses so module state stays untouched)."""

import json
import os
import subprocess
import sys

from conftest import ROOT


def _settings_with(env: dict[str, str]) -> dict:
    code = (
        "import json; from src import settings as s; "
        "print(json.dumps({'h': s.FORECAST_HORIZON_DAYS, 'sl': s.SERVICE_LEVEL, "
        "'lt': s.DEFAULT_LEAD_TIME_DAYS, 'b': s.DEFAULT_BUDGET, 'seed': s.RANDOM_SEED, 'db': str(s.DB_PATH)}))"
    )
    full_env = {k: v for k, v in os.environ.items() if not k.startswith("SMARTSTOCK_")}
    full_env.update(env)
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=full_env, capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_defaults_match_env_example():
    s = _settings_with({})
    assert (s["h"], s["sl"], s["lt"], s["b"], s["seed"]) == (30, 0.95, 7, 50000.0, 42)
    assert s["db"].endswith(os.path.join("data", "smartstock.db"))


def test_environment_overrides_are_applied():
    s = _settings_with(
        {"SMARTSTOCK_FORECAST_HORIZON": "14", "SMARTSTOCK_SERVICE_LEVEL": "0.9", "SMARTSTOCK_DEFAULT_BUDGET": "1234"}
    )
    assert (s["h"], s["sl"], s["b"]) == (14, 0.9, 1234.0)


def test_invalid_values_fall_back_and_out_of_range_values_are_clamped():
    s = _settings_with(
        {"SMARTSTOCK_FORECAST_HORIZON": "abc", "SMARTSTOCK_SERVICE_LEVEL": "5", "SMARTSTOCK_DEFAULT_LEAD_TIME": "-3"}
    )
    assert s["h"] == 30
    assert s["sl"] == 0.999
    assert s["lt"] == 1


def test_absolute_db_path_is_respected(tmp_path):
    target = str(tmp_path / "custom.db")
    assert _settings_with({"SMARTSTOCK_DB_PATH": target})["db"] == target
