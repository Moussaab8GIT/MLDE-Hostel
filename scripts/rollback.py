"""Move the production alias to a user-selected version; API restart is explicit."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import mlflow
from hotel_mlops.settings import MODEL_ALIAS, MODEL_NAME, TRACKING_URI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    client = mlflow.MlflowClient(tracking_uri=TRACKING_URI)
    version = client.get_model_version(MODEL_NAME, args.version)
    if version.status != "READY":
        raise RuntimeError(f"Version is not ready: {version.status}")
    client.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, args.version)
    print(f"{MODEL_NAME}@{MODEL_ALIAS} -> {args.version}")
    print("Load it with: docker compose up -d --force-recreate cancellation-api")
