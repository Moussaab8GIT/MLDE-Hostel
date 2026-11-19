from locust import HttpUser, between, task


class CancellationAPIUser(HttpUser):
    wait_time = between(1, 2)

    booking = {
        "LeadTime": 85,
        "ArrivalDateYear": 2015,
        "ArrivalDateMonth": "July",
        "ArrivalDateWeekNumber": 27,
        "ArrivalDateDayOfMonth": 1,
        "StaysInWeekendNights": 0,
        "StaysInWeekNights": 3,
        "Adults": 2,
        "Children": 0,
        "Babies": 0,
        "Meal": "BB",
        "Country": "PRT",
        "MarketSegment": "Online TA",
        "DistributionChannel": "TA/TO",
        "IsRepeatedGuest": 0,
        "PreviousCancellations": 0,
        "PreviousBookingsNotCanceled": 0,
        "ReservedRoomType": "A",
        "DepositType": "No Deposit",
        "DaysInWaitingList": 0,
        "CustomerType": "Transient",
        "RequiredCarParkingSpaces": 0,
        "TotalOfSpecialRequests": 1,
    }

    @task
    def predict_cancellation(self):
        with self.client.post(
            "/predict",
            json=self.booking,
            name="/predict",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if not result.get("prediction_id") or not result.get("model_version"):
                        response.failure("Prediction trace identifiers missing")
                except ValueError:
                    response.failure("Invalid JSON response")
