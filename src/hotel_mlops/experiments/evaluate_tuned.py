from __future__ import annotations

import json

import joblib
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from hotel_mlops.train import ROOT, load_split


def evaluate(model, X_test, y_test) -> dict:
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


def main() -> None:
    artifacts = ROOT / "artifacts"
    models = {
        "baseline_tree": artifacts / "cancellation_model.joblib",
        "tuned_tree": artifacts / "experiments" / "tuned_cancellation_model.joblib",
    }

    for name, path in models.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing model {name}: {path}")

    X_test, y_test = load_split(ROOT / "data" / "prepared" / "test.csv")
    results = {}
    for name, path in models.items():
        model = joblib.load(path)
        results[name] = evaluate(model, X_test, y_test)

    report = ROOT / "reports" / "evaluation" / "final_test_comparison.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Evaluation rows: {len(y_test)}")
    for name, metrics in results.items():
        print(
            f"{name}: precision={metrics['precision_canceled']:.3f}, "
            f"recall={metrics['recall_canceled']:.3f}, "
            f"F1={metrics['f1_canceled']:.3f}, "
            f"accuracy={metrics['accuracy']:.3f}"
        )
    print(f"Confusion matrices and full metrics: {report}")


if __name__ == "__main__":
    main()
