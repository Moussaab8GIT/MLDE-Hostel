"""Project commands. Run `python manage.py --help` from the project folder."""
import runpy
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
# Prefect and training subprocesses must resolve the same source package.
os.environ["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
MODULES = {
    "train": "hotel_mlops.train",
    "predict": "hotel_mlops.predict",
    "ingest": "hotel_mlops.ingest",
    "flow": "hotel_mlops.orchestration.training_flow",
    "serve": "hotel_mlops.orchestration.serve_training",
    "compare": "hotel_mlops.experiments.compare_models",
    "tune": "hotel_mlops.experiments.tune_tree",
    "evaluate": "hotel_mlops.experiments.evaluate_tuned",
}
SCRIPTS = {"rollback": "rollback.py", "load": "measure_load.py", "incident": "invalid_data_demo.py"}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in {"--help", "-h"}:
        print("Usage: python manage.py COMMAND [options]")
        print("Commands: " + ", ".join([*MODULES, *SCRIPTS]))
        raise SystemExit(0)
    command = sys.argv.pop(1)
    if command in MODULES:
        runpy.run_module(MODULES[command], run_name="__main__")
    elif command in SCRIPTS:
        runpy.run_path(str(ROOT / "scripts" / SCRIPTS[command]), run_name="__main__")
    else:
        raise SystemExit(f"Unknown command: {command}")
