"""Shared configuration; service addresses remain environment specific."""
import os
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[2]
with (ROOT / "config" / "project.toml").open("rb") as source:
    CONFIG = tomllib.load(source)
MODEL_NAME = CONFIG["model"]["name"]
MODEL_ALIAS = CONFIG["model"]["alias"]
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
DB_PATH = Path(os.getenv("PREDICTIONS_DB_PATH", str(ROOT / "data" / "predictions.db")))
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", str(ROOT / "reports")))
