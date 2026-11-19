import pandas as pd
import pytest

from hotel_mlops.ingest import prepare_dataset
from hotel_mlops.schemas import BookingRequest


def test_ingestion_hashes_and_does_not_split_a_date(tmp_path):
    rows = []
    for day in range(1, 11):
        for label in (0, 1):
            row = dict.fromkeys(BookingRequest.model_fields, None)
            row.update(LeadTime=5, ArrivalDateYear=2016, ArrivalDateMonth="July",
                       ArrivalDateDayOfMonth=day, IsCanceled=label)
            rows.append(row)
    source = tmp_path / "source.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    first = prepare_dataset(source, tmp_path / "first")
    second = prepare_dataset(source, tmp_path / "second")
    assert first == second
    train = pd.read_csv(tmp_path / "first" / "train.csv")
    test = pd.read_csv(tmp_path / "first" / "test.csv")
    assert train.ArrivalDateDayOfMonth.max() < test.ArrivalDateDayOfMonth.min()
    with pytest.raises(FileExistsError):
        prepare_dataset(source, tmp_path / "first")
