"""Run by the user after selecting a Compose resource configuration."""
import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def memory_mib(value):
    match = re.match(r"([\d.]+)\s*([KMGT]?i?B)", value.strip())
    if not match:
        raise ValueError(f"Cannot parse memory usage: {value}")
    factors = {"B": 1 / 1048576, "kB": 1000 / 1048576, "KB": 1000 / 1048576,
               "KiB": 1 / 1024, "MB": 1000000 / 1048576, "MiB": 1,
               "GB": 1000000000 / 1048576, "GiB": 1024}
    return float(match[1]) * factors[match[2]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", choices=["small", "standard"], required=True)
    parser.add_argument("--users", type=int, default=10)
    parser.add_argument("--run-time", default="2m")
    args = parser.parse_args()
    output = ROOT / "reports" / "load" / args.label
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError("Existing result: archive it before repeating this label.")
    container = subprocess.check_output(["docker", "compose", "ps", "-q", "cancellation-api"],
                                        cwd=ROOT, text=True).strip()
    if not container:
        raise RuntimeError("Start cancellation-api before measuring load.")
    inspect = json.loads(subprocess.check_output(["docker", "inspect", container], text=True))[0]
    samples = []
    with (output / "locust.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-m", "locust", "-f", str(ROOT / "tests" / "load" / "locustfile.py"),
                                    "--headless", "-u", str(args.users), "-r", "2",
                                    "--run-time", args.run_time, "--host", "http://127.0.0.1:8000",
                                    "--csv", str(output / "locust"),
                                    "--html", str(output / "locust.html")], cwd=ROOT, stdout=log,
                                   stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                sample = json.loads(subprocess.check_output(
                    ["docker", "stats", "--no-stream", "--format", "{{json .}}", container], text=True))
                samples.append({"timestamp": time.time(), "cpu_percent": float(sample["CPUPerc"].rstrip("%")),
                                "memory_mib": memory_mib(sample["MemUsage"].split("/")[0])})
                time.sleep(1)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=15)
    (output / "resources.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")
    if process.returncode:
        raise RuntimeError(f"Locust exited with {process.returncode}; inspect {output / 'locust.log'}")
    with (output / "locust_stats.csv").open(encoding="utf-8-sig") as source:
        aggregate = next(row for row in csv.DictReader(source) if row["Name"] == "Aggregated")
    summary = {"label": args.label, "users": args.users, "duration": args.run_time,
               "scope": "POST /predict, same booking, user wait 1-2s, includes ramp-up; not saturation",
               "docker_host_config": {key: inspect["HostConfig"].get(key) for key in ("NanoCpus", "Memory")},
               "image_id": inspect["Image"], "locust": aggregate, "resource_samples": len(samples),
               "sampled_peak_cpu_percent": max((s["cpu_percent"] for s in samples), default=None),
               "sampled_peak_memory_mib": max((s["memory_mib"] for s in samples), default=None)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
