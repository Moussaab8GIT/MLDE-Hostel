"""A controlled failed run; never edits the original datasets."""
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hotel_mlops.train import read_csv
from hotel_mlops.orchestration.training_flow import hotel_cancellation_training

if __name__ == "__main__":
    with TemporaryDirectory() as directory:
        frame = read_csv(ROOT / "data" / "prepared" / "train.csv")
        frame.loc[0, "IsCanceled"] = 2
        bad = Path(directory) / "invalid.csv"
        frame.to_csv(bad, index=False)
        # Nonzero exit and a failed Prefect run are the expected evidence.
        hotel_cancellation_training(train_path=bad, test_path=ROOT / "data" / "prepared" / "test.csv")
