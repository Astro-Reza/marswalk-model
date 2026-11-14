#!/usr/bin/env python
"""Check a generated run before training on it.

Reads every part-*.parquet and meta-*.json under --data, runs the checks in
marswalk_sim/quality.py and writes quality.json. Exit code 1 if any check
fails, so a loop can stop before it trains on bad data.

    python scripts/validate.py --data gs://BUCKET/runs/run-001
    python scripts/validate.py --data data/run-001 --out data/run-001/quality.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import fsspec
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marswalk_sim.quality import dataset_report  # noqa: E402

MARK = {"pass": "ok  ", "warn": "WARN", "fail": "FAIL"}


def load_run(data: str):
    """All rows and shard metadata under a folder (local or gs://)."""
    fs, root = fsspec.core.url_to_fs(data)
    root = root.rstrip("/")
    parts = sorted(set(fs.glob(f"{root}/**/part-*.parquet")) | set(fs.glob(f"{root}/part-*.parquet")))
    if not parts:
        raise SystemExit(f"no part-*.parquet files under {data}")
    frames = []
    for f in parts:
        with fs.open(f, "rb") as fh:
            frames.append(pd.read_parquet(fh))
    metas = []
    for f in sorted(set(fs.glob(f"{root}/**/meta-*.json")) | set(fs.glob(f"{root}/meta-*.json"))):
        with fs.open(f, "r") as fh:
            metas.append(json.load(fh))
    return pd.concat(frames, ignore_index=True), metas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="run folder (local or gs://)")
    ap.add_argument("--out", help="where to write the report (default: quality.json in the working directory)")
    ap.add_argument("--no-fail", action="store_true", help="always exit 0")
    args = ap.parse_args()

    df, metas = load_run(args.data)
    report = dataset_report(df, metas)
    report["data"] = args.data

    for c in report["checks"]:
        print(f"[{MARK[c['level']]}] {c['check']}: {c['detail']}")
    s = report["summary"]
    print(f"\n{s.get('rows', 0)} rows, {s.get('reached_patch', 0)} reached the patch, {s.get('walked', 0)} walked, "
          f"{s.get('falls', 0)} falls, {s.get('maps', 0)} maps")
    for r in s.get("by_slope", []):
        rate = "-" if r["fall_rate"] is None else f"{r['fall_rate']:.0%}"
        cot = "-" if r["cot_net_median"] is None else f"{r['cot_net_median']:.2f}"
        print(f"  slope {r['slope_band_deg']:>10} deg: {r['rows']:5d} rows, {r['walked']:5d} walked, "
              f"falls {rate:>4}, median net cost {cot}")
    print(f"\nVERDICT: {report['verdict']}")

    out = Path(args.out or "quality.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float))
    print(f"wrote {out}")
    if report["verdict"] == "fail" and not args.no_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
