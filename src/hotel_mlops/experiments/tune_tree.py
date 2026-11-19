from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from hotel_mlops.train import ROOT, TARGET, build_model, prepare_features, read_csv


def main() -> None:
    train_path = ROOT / "data" / "prepared" / "train.csv"
    frame = read_csv(train_path)

    required = {
        TARGET,
        "ArrivalDateYear",
        "ArrivalDateMonth",
        "ArrivalDateDayOfMonth",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"train.csv is missing columns: {sorted(missing)}")

    months = {
        name: number
        for number, name in enumerate(
            ["", "January", "February", "March", "April", "May", "June",
             "July", "August", "September", "October", "November", "December"]
        )
        if name
    }
    month_numbers = (
        frame["ArrivalDateMonth"].astype(str).str.strip().str.capitalize().map(months)
    )
    if month_numbers.isna().any():
        raise ValueError("Some arrival months could not be parsed.")

    dates = pd.to_datetime(
        {
            "year": pd.to_numeric(frame["ArrivalDateYear"], errors="raise"),
            "month": month_numbers,
            "day": pd.to_numeric(frame["ArrivalDateDayOfMonth"], errors="raise"),
        },
        errors="raise",
    )

    # Keep the latest 20% of the training period for validation.
    ordered_dates = dates.sort_values()
    cutoff = ordered_dates.iloc[int(len(frame) * 0.8)]
    fit_mask = dates < cutoff
    validation_mask = dates >= cutoff

    X = prepare_features(frame)
    y = pd.to_numeric(frame[TARGET], errors="raise").astype(int)
    X_fit, y_fit = X.loc[fit_mask], y.loc[fit_mask]
    X_validation, y_validation = X.loc[validation_mask], y.loc[validation_mask]

    numeric_columns = X_fit.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [
        column for column in X_fit.columns if column not in numeric_columns
    ]

    results = []
    best_model = None
    best_result = None
    candidates = [
        (depth, leaf_size, class_weight)
        for depth in (5, 8, 12)
        for leaf_size in (10, 20, 50)
        for class_weight in (None, "balanced")
    ]

    for depth, leaf_size, class_weight in candidates:
        model = build_model(numeric_columns, categorical_columns)
        model.set_params(
            classifier__max_depth=depth,
            classifier__min_samples_leaf=leaf_size,
            classifier__class_weight=class_weight,
        )
        model.fit(X_fit, y_fit)
        prediction = model.predict(X_validation)

        result = {
            "max_depth": depth,
            "min_samples_leaf": leaf_size,
            "class_weight": class_weight,
            "precision_canceled": float(
                precision_score(y_validation, prediction, zero_division=0)
            ),
            "recall_canceled": float(
                recall_score(y_validation, prediction, zero_division=0)
            ),
            "f1_canceled": float(f1_score(y_validation, prediction, zero_division=0)),
        }
        results.append(result)

    # Prefer the best F1 among candidates with recall >= 0.85.
    eligible = [item for item in results if item["recall_canceled"] >= 0.85]
    pool = eligible if eligible else results
    best_result = max(
        pool, key=lambda item: (item["f1_canceled"], item["precision_canceled"])
    )

    best_model = build_model(numeric_columns, categorical_columns)
    best_model.set_params(
        classifier__max_depth=best_result["max_depth"],
        classifier__min_samples_leaf=best_result["min_samples_leaf"],
        classifier__class_weight=best_result["class_weight"],
    )
    best_model.fit(X_fit, y_fit)

    artifacts = ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    model_path = artifacts / "experiments" / "tuned_cancellation_model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = ROOT / "reports" / "evaluation" / "tuning_results.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_path)
    report_path.write_text(
        json.dumps(
            {
                "validation_cutoff": str(cutoff.date()),
                "fit_rows": int(fit_mask.sum()),
                "validation_rows": int(validation_mask.sum()),
                "minimum_recall_preference": 0.85,
                "selected_parameters": best_result,
                "all_candidates": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Validation cutoff: {cutoff.date()}")
    print(
        "Selected tree: "
        f"precision={best_result['precision_canceled']:.3f}, "
        f"recall={best_result['recall_canceled']:.3f}, "
        f"F1={best_result['f1_canceled']:.3f}"
    )
    print(f"Model saved to: {model_path}")
    print(f"Results saved to: {report_path}")
    print("The test.csv file was not used.")


if __name__ == "__main__":
    main()
