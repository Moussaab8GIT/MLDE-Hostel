"""Booking contract. Missing fields deliberately retain trained imputation behavior."""
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

Count = Annotated[int, Field(strict=True, ge=0, le=100000)]
Flag = Annotated[int, Field(strict=True, ge=0, le=1)]
Category = Annotated[str, Field(strict=True, min_length=1, max_length=100)]


class BookingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    LeadTime: Count | None = None
    ArrivalDateYear: Annotated[int, Field(strict=True, ge=1900, le=2200)] | None = None
    ArrivalDateMonth: Category | None = None
    ArrivalDateWeekNumber: Annotated[int, Field(strict=True, ge=1, le=53)] | None = None
    ArrivalDateDayOfMonth: Annotated[int, Field(strict=True, ge=1, le=31)] | None = None
    StaysInWeekendNights: Count | None = None
    StaysInWeekNights: Count | None = None
    Adults: Count | None = None
    Children: Count | None = None
    Babies: Count | None = None
    Meal: Category | None = None
    Country: Category | None = None
    MarketSegment: Category | None = None
    DistributionChannel: Category | None = None
    IsRepeatedGuest: Flag | None = None
    PreviousCancellations: Count | None = None
    PreviousBookingsNotCanceled: Count | None = None
    ReservedRoomType: Category | None = None
    DepositType: Category | None = None
    DaysInWaitingList: Count | None = None
    CustomerType: Category | None = None
    RequiredCarParkingSpaces: Count | None = None
    TotalOfSpecialRequests: Count | None = None

    @model_validator(mode="after")
    def require_feature(self):
        if not any(value is not None for value in self.model_dump().values()):
            raise ValueError("Provide at least one non-null booking feature.")
        if self.ArrivalDateMonth is not None:
            import calendar
            if self.ArrivalDateMonth not in list(calendar.month_name)[1:]:
                raise ValueError("ArrivalDateMonth must be an English month name.")
            if self.ArrivalDateYear is not None and self.ArrivalDateDayOfMonth is not None:
                from datetime import date
                date(self.ArrivalDateYear, list(calendar.month_name).index(self.ArrivalDateMonth),
                     self.ArrivalDateDayOfMonth)
        return self


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prediction_id: UUID
    actual_canceled: Flag
