"""Persistent service traces and feedback aggregates, independent of MLflow."""
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path


def connect(path: Path):
    return sqlite3.connect(path, timeout=15)


def init_database(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with connect(path) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("""CREATE TABLE IF NOT EXISTS predictions (
            prediction_id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
            model_version TEXT NOT NULL, input_json TEXT NOT NULL,
            predicted_canceled INTEGER NOT NULL, cancellation_probability REAL NOT NULL,
            duration_ms REAL NOT NULL, actual_canceled INTEGER, feedback_received_at TEXT)""")
        connection.execute("""CREATE TABLE IF NOT EXISTS request_errors (
            request_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, route TEXT NOT NULL,
            status_code INTEGER NOT NULL, model_version TEXT NOT NULL,
            duration_ms REAL NOT NULL, error_type TEXT NOT NULL)""")
        connection.execute("CREATE INDEX IF NOT EXISTS feedback_window ON predictions(feedback_received_at, model_version)")


def record_error(path, request_id, route, status, version, duration, error_type):
    with connect(path) as connection:
        connection.execute("""INSERT INTO request_errors
            (request_id, created_at, route, status_code, model_version, duration_ms, error_type)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
                           (request_id, datetime.now(timezone.utc).isoformat(), route,
                            status, version, duration, error_type))


def feedback_metrics(path: Path, minimum: int, window_days: int) -> list[dict]:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()
    with connect(path) as connection:
        rows = connection.execute("""SELECT model_version, COUNT(*),
            SUM(actual_canceled=1 AND predicted_canceled=1),
            SUM(actual_canceled=0 AND predicted_canceled=1),
            SUM(actual_canceled=1 AND predicted_canceled=0),
            SUM(actual_canceled=0 AND predicted_canceled=0)
            FROM predictions WHERE feedback_received_at >= ? AND actual_canceled IS NOT NULL
            GROUP BY model_version""", (cutoff,)).fetchall()
    results = []
    for version, count, tp, fp, fn, tn in rows:
        ready = count >= minimum and tp + fn > 0 and tn + fp > 0
        results.append({"model_version": version, "feedback_count": count,
                        "window_days": window_days, "minimum_feedback": minimum,
                        "status": "available" if ready else "insufficient_feedback",
                        "precision": tp / (tp + fp) if ready and tp + fp else None,
                        "recall": tp / (tp + fn) if ready else None,
                        "f1": 2 * tp / (2 * tp + fp + fn) if ready else None,
                        "true_positives": tp, "false_positives": fp,
                        "false_negatives": fn, "true_negatives": tn})
    return results


def flow_reports(directory: Path):
    """One atomically written file per execution avoids concurrent append corruption."""
    reports = []
    for path in directory.glob("flow-*.json"):
        try:
            reports.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return sorted(reports, key=lambda report: report["finished_at"])
