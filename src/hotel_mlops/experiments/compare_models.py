from __future__ import annotations

import json

import joblib
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from hotel_mlops.train import ROOT, load_split


def evaluate(model, X_test, y_test):
    prediction = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
        "precision_canceled": float(
            precision_score(y_test, prediction, pos_label=1, zero_division=0)
        ),
        "recall_canceled": float(
            recall_score(y_test, prediction, pos_label=1, zero_division=0)
        ),
        "f1_canceled": float(f1_score(y_test, prediction, pos_label=1, zero_division=0)),
        "confusion_matrix_labels_0_1": confusion_matrix(
            y_test, prediction, labels=[0, 1]
        ).tolist(),
    }


def main():
    baseline_path = ROOT / "artifacts" / "cancellation_model.joblib"
    if not baseline_path.exists():
        raise FileNotFoundError("First run: python manage.py train")

    X_train, y_train = load_split(ROOT / "data" / "prepared" / "train.csv")
    X_test, y_test = load_split(ROOT / "data" / "prepared" / "test.csv")
    if set(X_train.columns) != set(X_test.columns):
        raise ValueError("train.csv et test.csv doivent avoir les mêmes colonnes.")
    X_test = X_test.loc[:, X_train.columns]

    baseline = joblib.load(baseline_path)
    results = {"decision_tree": evaluate(baseline, X_test, y_test)}

    forest = Pipeline(
        steps=[
            ("preprocessing", clone(baseline.named_steps["preprocessing"])),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=14,
                    min_samples_leaf=10,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )
    forest.fit(X_train, y_train)
    results["random_forest"] = evaluate(forest, X_test, y_test)

    artifacts = ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "experiments").mkdir(parents=True, exist_ok=True)
    joblib.dump(forest, artifacts / "experiments" / "random_forest_model.joblib")
    evaluation = ROOT / "reports" / "evaluation"
    evaluation.mkdir(parents=True, exist_ok=True)
    (evaluation / "comparison.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    for name, metrics in results.items():
        print(
            f"{name}: precision={metrics['precision_canceled']:.3f}, "
            f"recall={metrics['recall_canceled']:.3f}, "
            f"F1={metrics['f1_canceled']:.3f}, "
            f"accuracy={metrics['accuracy']:.3f}"
        )
    print(f"Results: {evaluation / 'comparison.json'}")


if __name__ == "__main__":
    main()
