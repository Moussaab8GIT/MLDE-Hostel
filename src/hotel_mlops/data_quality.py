"""Checks on raw values, before imputing or encoding features."""
import numpy as np
import pandas as pd
from hotel_mlops.schemas import BookingRequest


def validate_values(frame: pd.DataFrame, require_schema: bool = False):
    if frame.empty:
        raise ValueError("Dataset must contain rows.")
    expected = set(BookingRequest.model_fields)
    if require_schema and expected - set(frame.columns):
        raise ValueError(f"Missing booking columns: {sorted(expected - set(frame.columns))}")
    source_only = {"IsCanceled", "ReservationStatus", "ReservationStatusDate", "AssignedRoomType",
                   "BookingChanges", "ADR", "Agent", "Company"}
    if require_schema and set(frame.columns) - expected - source_only:
        raise ValueError(f"Unexpected dataset columns: {sorted(set(frame.columns) - expected - source_only)}")
    # Validate known fields without preprocessing away invalid values. Missing
    # individual values are allowed because the fitted pipeline imputes them.
    for column in expected & set(frame.columns):
        values = frame[column].dropna()
        if column in {"Meal", "Country", "MarketSegment", "DistributionChannel", "ReservedRoomType",
                      "DepositType", "CustomerType", "ArrivalDateMonth"}:
            for value in values.unique():
                BookingRequest.model_validate({column: str(value).strip()})
        else:
            numbers = pd.to_numeric(values, errors="raise")
            if not np.isfinite(numbers).all() or (numbers % 1 != 0).any():
                raise ValueError(f"{column} must contain finite integers or missing values.")
            if len(numbers):
                for value in (numbers.min(), numbers.max()):
                    BookingRequest.model_validate({column: int(value)})
