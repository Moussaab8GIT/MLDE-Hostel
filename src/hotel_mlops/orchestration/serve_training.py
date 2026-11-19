"""Register and serve the local weekly training deployment in Prefect."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

from hotel_mlops.orchestration.training_flow import hotel_cancellation_training  # noqa: E402


if __name__ == "__main__":
    hotel_cancellation_training.serve(
        name="weekly-hotel-cancellation-training",
        interval=7 * 24 * 60 * 60,
        paused=False,
        parameters={
            "train_path": str(ROOT / "data" / "prepared" / "train.csv"),
            "test_path": str(ROOT / "data" / "prepared" / "test.csv"),
        },
        tags=["hotel-cancellation", "scheduled"],
        description=(
            "Weekly demonstration retraining from the fixed historical train/test "
            "splits. Replace these inputs with newly ingested batches for a real "
            "production retraining schedule."
        ),
    )
