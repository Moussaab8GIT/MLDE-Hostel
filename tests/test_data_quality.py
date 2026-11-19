from pathlib import Path

import pandas as pd
import pytest

from hotel_mlops.train import TARGET, load_split


def write_dataset(path: Path, frame: pd.DataFrame) -> Path:
    frame.to_csv(path, index=False)
    return path


def test_load_split_accepts_binary_target_and_returns_features(tmp_path: Path) -> None:
    dataset_path = write_dataset(
        tmp_path / "valid.csv",
        pd.DataFrame(
            {
                TARGET: [0, 1, 0],
                "LeadTime": [12, 30, 7],
                "Country": ["France", "Spain", "France"],
            }
        ),
    )

    features, target = load_split(dataset_path)

    assert list(features.columns) == ["LeadTime", "Country"]
    assert target.tolist() == [0, 1, 0]


def test_load_split_rejects_missing_target(tmp_path: Path) -> None:
    dataset_path = write_dataset(
        tmp_path / "missing-target.csv", pd.DataFrame({"LeadTime": [12, 30]})
    )

    with pytest.raises(ValueError, match="must contain the target column"):
        load_split(dataset_path)


def test_load_split_rejects_non_binary_target(tmp_path: Path) -> None:
    dataset_path = write_dataset(
        tmp_path / "invalid-target.csv",
        pd.DataFrame({TARGET: [0, 2], "LeadTime": [12, 30]}),
    )

    with pytest.raises(ValueError, match="only 0 and 1"):
        load_split(dataset_path)


def test_load_split_rejects_non_numeric_target(tmp_path: Path) -> None:
    dataset_path = write_dataset(
        tmp_path / "text-target.csv",
        pd.DataFrame({TARGET: [0, "cancelled"], "LeadTime": [12, 30]}),
    )

    with pytest.raises(ValueError):
        load_split(dataset_path)


@pytest.mark.parametrize("label", [0.5, 1.9, None])
def test_fractional_or_missing_target_is_not_silently_truncated(tmp_path, label):
    dataset_path = write_dataset(tmp_path / "bad.csv", pd.DataFrame({TARGET: [0, label], "LeadTime": [3, 5]}))
    with pytest.raises(ValueError, match="only 0 and 1"):
        load_split(dataset_path)
