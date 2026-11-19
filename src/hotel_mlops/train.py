from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from hotel_mlops.settings import ROOT, CONFIG, MODEL_NAME, MODEL_ALIAS, TRACKING_URI


TARGET = "IsCanceled"
REGISTERED_MODEL_NAME = MODEL_NAME
PRODUCTION_ALIAS = MODEL_ALIAS
# Promotion gates are compared after rounding to three decimals, matching
# the precision reported in the baseline evaluation.
MIN_RECALL_CANCELED = CONFIG["model"]["minimum_recall"]
MIN_F1_CANCELED = CONFIG["model"]["minimum_f1"]
DROP_COLUMNS = {
    "ReservationStatus",
    "ReservationStatusDate",
    "AssignedRoomType",
    "BookingChanges",
    "ADR",
    "Agent",
    "Company",
}


def read_csv(path: Path) -> pd.DataFrame:
    """Read the prepared CSV while handling the source file's padded fields."""
    return pd.read_csv(path, skipinitialspace=True, na_values=["NULL", "null"])


def sha256_file(path: Path) -> str:
    """Hash an input file so an MLflow run identifies its exact dataset version."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove target/leakage columns and normalize categorical values."""
    features = frame.drop(columns=DROP_COLUMNS | {TARGET}, errors="ignore").copy()

    for column in features.select_dtypes(include=["object", "string"]).columns:
        values = features[column].astype("string").str.strip()
        values = values.replace({"": pd.NA, "NULL": pd.NA, "null": pd.NA})
        features[column] = values.astype(object).where(values.notna(), np.nan)

    return features


def load_split(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    frame = read_csv(path)
    if TARGET not in frame.columns:
        raise ValueError(f"{path} must contain the target column {TARGET!r}.")

    target = pd.to_numeric(frame[TARGET], errors="raise")
    if target.isna().any() or not target.isin([0, 1]).all():
        raise ValueError(f"{TARGET} must contain only 0 and 1.")
    from hotel_mlops.data_quality import validate_values
    validate_values(frame)
    return prepare_features(frame), target.astype(int)


def build_model(numeric_columns: list[str], categorical_columns: list[str]) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                DecisionTreeClassifier(
                    max_depth=CONFIG["model"]["max_depth"],
                    min_samples_leaf=CONFIG["model"]["min_samples_leaf"],
                    class_weight="balanced",
                    random_state=CONFIG["model"]["random_state"],
                ),
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a hotel booking cancellation baseline."
    )
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "prepared" / "train.csv")
    parser.add_argument("--test", type=Path, default=ROOT / "data" / "prepared" / "test.csv")
    parser.add_argument(
        "--model", type=Path, default=ROOT / "artifacts" / "cancellation_model.joblib"
    )
    parser.add_argument(
        "--metrics", type=Path, default=ROOT / "reports" / "evaluation" / "metrics.json"
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(CONFIG["model"]["experiment"])

    with mlflow.start_run(run_name="baseline-decision-tree") as run:
        from hotel_mlops.data_quality import validate_values
        validate_values(read_csv(args.train), require_schema=True)
        validate_values(read_csv(args.test), require_schema=True)
        X_train, y_train = load_split(args.train)
        X_test, y_test = load_split(args.test)

        if set(X_train.columns) != set(X_test.columns):
            raise ValueError("train.csv and test.csv must have the same feature columns.")
        X_test = X_test.loc[:, X_train.columns]

        numeric_columns = X_train.select_dtypes(include=["number"]).columns.tolist()
        categorical_columns = [
            column for column in X_train.columns if column not in numeric_columns
        ]
        if not numeric_columns and not categorical_columns:
            raise ValueError("No input features were found.")

        model = build_model(numeric_columns, categorical_columns)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        metrics = {
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "features": list(X_train.columns),
            "accuracy": float(accuracy_score(y_test, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
            "precision_canceled": float(
                precision_score(y_test, predictions, pos_label=1, zero_division=0)
            ),
            "recall_canceled": float(
                recall_score(y_test, predictions, pos_label=1, zero_division=0)
            ),
            "f1_canceled": float(
                f1_score(y_test, predictions, pos_label=1, zero_division=0)
            ),
            "confusion_matrix_labels_0_1": confusion_matrix(
                y_test, predictions, labels=[0, 1]
            ).tolist(),
            "classification_report": classification_report(
                y_test,
                predictions,
                labels=[0, 1],
                target_names=["not_canceled", "canceled"],
                output_dict=True,
                zero_division=0,
            ),
        }

        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        mlflow.set_tags(
            {
                "project": "hotel-cancellation-prediction",
                "model_type": "baseline-decision-tree",
                "split_strategy": "chronological",
            }
        )
        mlflow.log_params(
            {
                "model_type": "DecisionTreeClassifier",
                "max_depth": CONFIG["model"]["max_depth"],
                "min_samples_leaf": CONFIG["model"]["min_samples_leaf"],
                "class_weight": "balanced",
                "random_state": CONFIG["model"]["random_state"],
                "train_dataset": args.train.name,
                "test_dataset": args.test.name,
                "train_dataset_sha256": sha256_file(args.train),
                "test_dataset_sha256": sha256_file(args.test),
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "feature_count": len(X_train.columns),
                "training_code_sha256": sha256_file(Path(__file__)),
                "configuration_sha256": sha256_file(ROOT / "config" / "project.toml"),
            }
        )
        mlflow.log_metrics(
            {
                key: metrics[key]
                for key in (
                    "accuracy",
                    "balanced_accuracy",
                    "precision_canceled",
                    "recall_canceled",
                    "f1_canceled",
                )
            }
        )
        mlflow.log_dict(metrics, "evaluation/metrics.json")
        mlflow.log_artifact(str(ROOT / "config" / "project.toml"), artifact_path="configuration")
        manifest = args.train.parent / "manifest.json"
        if manifest.exists():
            mlflow.log_artifact(str(manifest), artifact_path="data")
        input_example = X_train.head(5).copy()
        for column in input_example.select_dtypes(include=["number"]).columns:
            input_example[column] = input_example[column].astype("float64")
        output_example = model.predict(input_example).astype("float64")
        signature = infer_signature(input_example, output_example)

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="cancellation-model",
            input_example=input_example,
            signature=signature,
            registered_model_name=REGISTERED_MODEL_NAME,
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.compose._column_transformer._RemainderColsList",
                "sklearn.tree._tree.Tree",
            ],
        )

        client = mlflow.MlflowClient()
        candidate_version = model_info.registered_model_version
        if candidate_version is None:
            versions = client.search_model_versions(
                f"name='{REGISTERED_MODEL_NAME}'"
            )
            candidate = next(
                (version for version in versions if version.run_id == run.info.run_id),
                None,
            )
            if candidate is None:
                raise RuntimeError(
                    "MLflow logged the model but did not return its registered version."
                )
            candidate_version = candidate.version

        passed_promotion_gates = (
            round(metrics["recall_canceled"], 3) >= MIN_RECALL_CANCELED
            and round(metrics["f1_canceled"], 3) >= MIN_F1_CANCELED
        )
        client.set_model_version_tag(REGISTERED_MODEL_NAME, candidate_version,
                                     "promotion_gate", "passed" if passed_promotion_gates else "rejected")
        if passed_promotion_gates:
            args.model.parent.mkdir(parents=True, exist_ok=True)
            candidate_path = args.model.with_name(f"candidate-{candidate_version}.joblib")
            joblib.dump(model, candidate_path)
            mlflow.log_artifact(str(candidate_path), artifact_path="deployed-artifact")
            previous = None
            try:
                previous = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, PRODUCTION_ALIAS)
            except mlflow.exceptions.MlflowException as exc:
                if exc.error_code != "RESOURCE_DOES_NOT_EXIST":
                    raise
            if previous:
                mlflow.set_tag("previous_production_version", previous.version)
            client.set_registered_model_alias(REGISTERED_MODEL_NAME, PRODUCTION_ALIAS, candidate_version)
            candidate_path.replace(args.model)
            mlflow.set_tag("promotion_decision", "promoted")
            print(
                f"Promoted model version {candidate_version} to alias "
                f"'{PRODUCTION_ALIAS}'."
            )
        else:
            mlflow.set_tag("promotion_decision", "rejected")
            print(
                "Model not promoted; production alias and deployed artifact were kept. "
                f"Rounded recall={metrics['recall_canceled']:.3f} "
                f"(minimum {MIN_RECALL_CANCELED:.3f}), "
                f"rounded F1={metrics['f1_canceled']:.3f} "
                f"(minimum {MIN_F1_CANCELED:.3f})."
            )

        print(f"Production artifact path: {args.model}")
        print(f"Metrics saved to: {args.metrics}")
        print(f"MLflow run ID: {run.info.run_id}")
        print(f"Registered model: {REGISTERED_MODEL_NAME}, version {candidate_version}")
        print(f"Cancellation recall: {metrics['recall_canceled']:.3f}")
        print(f"Cancellation F1: {metrics['f1_canceled']:.3f}")


if __name__ == "__main__":
    main()
