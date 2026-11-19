"""Reproducible local ingestion of the historical H1 CSV (no live-data claim)."""
import argparse
import calendar
import json
from pathlib import Path

import pandas as pd

from hotel_mlops.data_quality import validate_values
from hotel_mlops.settings import CONFIG, ROOT
from hotel_mlops.train import load_split, read_csv, sha256_file


def prepare_dataset(source: Path, output: Path) -> dict:
    frame = read_csv(source)
    validate_values(frame, require_schema=True)
    # Validate target before any date parsing, encoding, or output writes.
    load_split(source)
    months = {month: index for index, month in enumerate(calendar.month_name) if month}
    dates = pd.to_datetime({"year": frame["ArrivalDateYear"],
                            "month": frame["ArrivalDateMonth"].str.strip().map(months),
                            "day": frame["ArrivalDateDayOfMonth"]}, errors="raise")
    if dates.isna().any():
        raise ValueError("Arrival dates must all be present.")
    cutoff = dates.sort_values().iloc[int(len(frame) * (1 - CONFIG["data"]["test_fraction"]))]
    splits = {"train": frame.loc[dates < cutoff], "test": frame.loc[dates >= cutoff]}
    for name, split in splits.items():
        if split.empty or set(split["IsCanceled"].unique()) != {0, 1}:
            raise ValueError(f"{name} split must contain both classes and at least one row.")
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / "train.csv", output / "test.csv", output / "manifest.json"]
    if any(path.exists() for path in paths):
        raise FileExistsError("Use a new output directory to preserve previous data versions.")
    manifest = {"source_name": source.name, "source_sha256": sha256_file(source),
                "source_doi": "10.1016/j.dib.2018.11.126", "license": "CC BY 4.0",
                "replay": "Static historical resort-hotel data, not a live feed",
                "split": "arrival date, keep each date in a single partition",
                "cutoff": str(cutoff.date()), "requested_test_fraction": CONFIG["data"]["test_fraction"],
                "partitions": {}}
    for name, split in splits.items():
        path = output / f"{name}.csv"
        split.assign(_date=dates.loc[split.index]).sort_values("_date", kind="stable").drop(
            columns="_date").to_csv(path, index=False)
        manifest["partitions"][name] = {"rows": len(split), "sha256": sha256_file(path)}
    paths[-1].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "prepared")
    args = parser.parse_args()
    print(json.dumps(prepare_dataset(args.source, args.output_dir), indent=2))
