import sqlite3
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from hotel_mlops.schemas import BookingRequest, FeedbackRequest
from hotel_mlops.storage import feedback_metrics, init_database, record_error


@pytest.mark.parametrize("booking", [{}, {"LeadTime": -1}, {"LeadTime": "100"},
                                     {"IsRepeatedGuest": 2}, {"typo": 3},
                                     {"ArrivalDateYear": 2026, "ArrivalDateMonth": "February",
                                      "ArrivalDateDayOfMonth": 30}])
def test_invalid_bookings_rejected(booking):
    with pytest.raises(ValidationError):
        BookingRequest.model_validate(booking)


def test_partial_booking_keeps_imputation_contract():
    assert BookingRequest(LeadTime=100).model_dump(exclude_none=True) == {"LeadTime": 100}


def test_feedback_requires_real_binary_integer():
    with pytest.raises(ValidationError):
        FeedbackRequest(prediction_id="9825953a-fb7b-4f2c-9068-ceb3d4e52ea6", actual_canceled=True)


def test_errors_survive_database_reinitialization(tmp_path):
    path = tmp_path / "traces.db"
    init_database(path)
    record_error(path, "request-1", "/predict", 422, "7", 3.5, "validation")
    init_database(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT status_code, model_version FROM request_errors").fetchone() == (422, "7")


def test_feedback_metrics_are_gated_and_separated_by_version(tmp_path):
    path = tmp_path / "traces.db"
    init_database(path)
    now = datetime.now(timezone.utc).isoformat()
    # Version 7: TP=1, FP=1, FN=1, TN=1. Version 8 has too little feedback.
    with sqlite3.connect(path) as connection:
        for index, (version, prediction, actual) in enumerate(
            [("7", 1, 1), ("7", 1, 0), ("7", 0, 1), ("7", 0, 0), ("8", 1, 1)]
        ):
            connection.execute("INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                               (str(index), now, version, "{}", prediction, .5, 2.0, actual, now))
    by_version = {row["model_version"]: row for row in feedback_metrics(path, 4, 30)}
    assert by_version["7"]["precision"] == .5
    assert by_version["7"]["recall"] == .5
    assert by_version["7"]["f1"] == .5
    assert by_version["8"]["f1"] is None


def test_old_feedback_excluded(tmp_path):
    path = tmp_path / "traces.db"
    init_database(path)
    with sqlite3.connect(path) as connection:
        connection.execute("INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                           ("old", "2020-01-01", "7", "{}", 1, .9, 2, 1, "2020-01-02"))
    assert feedback_metrics(path, 1, 30) == []
