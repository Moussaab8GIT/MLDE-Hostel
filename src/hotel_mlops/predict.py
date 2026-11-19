from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from hotel_mlops.train import DROP_COLUMNS, ROOT, TARGET, prepare_features, read_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict cancellation risk for bookings in a CSV file."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input booking CSV.")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports" / "predictions.csv", help="Output CSV."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "artifacts" / "cancellation_model.joblib",
        help="Trained model file.",
    )
    args = parser.parse_args()

    model = joblib.load(args.model)
    bookings = read_csv(args.input)
    features = prepare_features(bookings)

    expected_columns = list(model.feature_names_in_)
    missing = [column for column in expected_columns if column not in features.columns]
    if missing:
        raise ValueError(
            "Input is missing model features: " + ", ".join(missing)
        )

    features = features.loc[:, expected_columns]
    prediction = model.predict(features).astype(int)

    classes = list(model.named_steps["classifier"].classes_)
    if 1 in classes:
        cancellation_probability = model.predict_proba(features)[:, classes.index(1)]
    else:
        cancellation_probability = np.zeros(len(features), dtype=float)

    result = pd.DataFrame(
        {
            "row_number": np.arange(1, len(bookings) + 1),
            "predicted_canceled": prediction,
            "cancellation_probability": cancellation_probability,
        }
    )
    if TARGET in bookings.columns:
        result["actual_IsCanceled"] = pd.to_numeric(
            bookings[TARGET], errors="coerce"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Predictions saved to: {args.output}")


if __name__ == "__main__":
    main()
