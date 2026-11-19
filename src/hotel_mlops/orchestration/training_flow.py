import os
import subprocess
import sys
import json
import time
import urllib.error
import urllib.request
from uuid import uuid4
from pathlib import Path

from prefect import flow, get_run_logger, task

ROOT = Path(__file__).resolve().parents[3]

from hotel_mlops.train import load_split, sha256_file  # noqa: E402
from hotel_mlops.data_quality import validate_values  # noqa: E402
from hotel_mlops.train import read_csv  # noqa: E402
from hotel_mlops.settings import REPORTS_DIR  # noqa: E402


def send_telegram_failure(message: str) -> None:
    """Send a best-effort failure alert when Telegram credentials are configured."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return

    payload = json.dumps({"chat_id": chat_id, "text": message}).encode("utf-8")
    request = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            result = json.loads(response.read().decode("utf-8"))
        if not result.get("ok"):
            raise RuntimeError("Telegram API rejected the notification")
    except (urllib.error.URLError, TimeoutError, RuntimeError, ValueError) as exc:
        # Never print the request URL: it contains the bot token.
        raise RuntimeError(type(exc).__name__) from None


@task(name="validate-training-data", retries=0)
def validate_training_data(train_path: Path, test_path: Path) -> dict[str, object]:
    """Block training when files, labels, rows, or feature schemas are invalid."""
    logger = get_run_logger()

    for path in (train_path, test_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required dataset does not exist: {path}")
        validate_values(read_csv(path), require_schema=True)

    X_train, y_train = load_split(train_path)
    X_test, y_test = load_split(test_path)

    if X_train.empty or X_test.empty:
        raise ValueError("Training and test datasets must both contain rows.")
    if set(X_train.columns) != set(X_test.columns):
        raise ValueError("Train and test datasets must have identical feature columns.")
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError("Train and test targets must each contain both classes 0 and 1.")

    result = {
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "feature_count": len(X_train.columns),
        "train_sha256": sha256_file(train_path),
        "test_sha256": sha256_file(test_path),
    }
    logger.info(
        "Data validation passed: train_rows=%s, test_rows=%s, features=%s",
        result["train_rows"],
        result["test_rows"],
        result["feature_count"],
    )
    return result


@task(name="train-log-and-promote", retries=0)
def train_log_and_promote(train_path: Path, test_path: Path) -> None:
    """Run the existing training entry point and surface failures to Prefect."""
    logger = get_run_logger()
    child_env = os.environ.copy()
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + child_env.get("PYTHONPATH", "")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "hotel_mlops.train",
            "--train",
            str(train_path),
            "--test",
            str(test_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=child_env,
        check=False,
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = REPORTS_DIR / f"training-{uuid4()}.log"
    log_path.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    logger.info("Training output saved to %s", log_path)
    if result.returncode != 0:
        details = (result.stderr or result.stdout)[-4000:]
        # Keep error reporting safe on Windows consoles using legacy encodings.
        details = details.encode("ascii", "backslashreplace").decode("ascii")
        raise RuntimeError(
            "Training failed. "
            f"Exit code: {result.returncode}. "
            f"Details: {details}"
        )
    logger.info("Training subprocess completed successfully.")


@flow(name="hotel-cancellation-training")
def hotel_cancellation_training(
    train_path: Path = ROOT / "data" / "prepared" / "train.csv",
    test_path: Path = ROOT / "data" / "prepared" / "test.csv",
    source_path: str | None = None,
) -> None:
    """Validate data, then train, track in MLflow, and apply promotion gates."""
    started = time.time()
    execution_id = str(uuid4())
    report = {"execution_id": execution_id, "started_at": started, "status": "failed"}
    try:
        if source_path:
            output = ROOT / "data" / execution_id
            ingest_training_data(Path(source_path), output)
            train_path, test_path = output / "train.csv", output / "test.csv"
        report["data"] = validate_training_data(train_path, test_path)
        train_log_and_promote(train_path, test_path)
        report["status"] = "completed"
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        report["message"] = str(exc)[-2000:]
        get_run_logger().error("LOCAL FAILURE ALERT: %s", report["message"])
        alert_text = (
            "ALERT: Hotel Cancellation training failed\n"
            f"Flow run: {execution_id}\n"
            f"Error: {type(exc).__name__}: {str(exc)[-1200:]}"
        )
        try:
            send_telegram_failure(alert_text)
        except RuntimeError as notification_error:
            get_run_logger().warning(
                "Telegram notification could not be delivered (%s).",
                notification_error,
            )
        raise
    finally:
        report["finished_at"] = time.time()
        report["duration_seconds"] = report["finished_at"] - started
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORTS_DIR / f"flow-{execution_id}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(path)


@task(name="ingest-historical-data", retries=0)
def ingest_training_data(source: Path, output: Path):
    from hotel_mlops.ingest import prepare_dataset
    return prepare_dataset(source, output)


if __name__ == "__main__":
    hotel_cancellation_training()
