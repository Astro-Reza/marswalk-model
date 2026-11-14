#!/usr/bin/env python
"""Train the step-cost surrogate on generated patches.

Split is by map_id (GroupShuffleSplit-style): train 70 % / validation 15 % /
test 15 % of maps. Validation picks the model size; test is scored once.

    python scripts/train.py --data gs://BUCKET/runs/run-001 --out models/run-001
    python scripts/train.py --data data/run-001 data/run-002 --out models/run-002

Which rows count for each target is decided by marswalk_sim/quality.py, the
same rules scripts/validate.py reports on. Run validate.py first; with
--require-quality this script refuses data that fails it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import fsspec
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (brier_score_loss, log_loss, mean_absolute_error, r2_score,
                             roc_auc_score, root_mean_squared_error)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marswalk_sim.quality import dataset_report, label_mask  # noqa: E402

# What the planner can know before the walk: terrain measured from a DTM,
# surface class (friction), gravity and the astronaut's suit configuration.
FEATURES = [
    "dtm_slope_along_deg", "dtm_slope_cross_deg", "dtm_slope_p90_deg", "dtm_roughness_m",
    "rock_density_per_m2", "rock_height_m", "friction",
    "gravity", "suit_mass_kg", "pack_up_m", "pack_back_m",
]
TARGETS = {
    # name: (column, kind, row filter)
    "cot": ("cot_net_J_per_kg_m", "reg", "walked"),
    "fall": ("fell", "clf", "reached_patch"),
    "mos": ("mos_ml_min_m", "reg", "walked"),
    "speed": ("speed_mps", "reg", "walked"),   # the planner needs time as well as energy
}
GRID = [
    {"learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 20},
    {"learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 20},
    {"learning_rate": 0.10, "max_leaf_nodes": 31, "min_samples_leaf": 10},
    {"learning_rate": 0.05, "max_leaf_nodes": 63, "min_samples_leaf": 10},
]


def load(data) -> pd.DataFrame:
    """Every part-*.parquet under one folder or a list of folders (local or gs://)."""
    frames = []
    for one in [data] if isinstance(data, str) else list(data):
        fs, root = fsspec.core.url_to_fs(one)
        root = root.rstrip("/")
        files = sorted(set(fs.glob(f"{root}/**/part-*.parquet")) | set(fs.glob(f"{root}/part-*.parquet")))
        if not files:
            raise SystemExit(f"no part-*.parquet files under {one}")
        for f in files:
            with fs.open(f, "rb") as fh:
                frames.append(pd.read_parquet(fh))
    return pd.concat(frames, ignore_index=True)


def split_maps(maps: np.ndarray, seed: int, val: float = 0.15, test: float = 0.15) -> dict:
    rng = np.random.default_rng(seed)
    maps = rng.permutation(np.unique(maps))
    n = len(maps)
    n_test = max(1, round(test * n)) if n >= 3 else 0
    n_val = max(1, round(val * n)) if n >= 3 else 0
    return {"test": maps[:n_test].tolist(), "val": maps[n_test:n_test + n_val].tolist(),
            "train": maps[n_test + n_val:].tolist()}


def rows_for(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return df[label_mask(df, rule)]


def fit_one(kind: str, Xtr, ytr, Xva, yva, seed: int):
    best = None
    for params in GRID:
        Model = HistGradientBoostingRegressor if kind == "reg" else HistGradientBoostingClassifier
        mdl = Model(max_iter=400, random_state=seed, **params).fit(Xtr, ytr)
        if kind == "reg":
            score = root_mean_squared_error(yva, mdl.predict(Xva)) if len(yva) else 0.0
        else:
            score = log_loss(yva, mdl.predict_proba(Xva)[:, 1], labels=[0, 1]) if len(yva) else 0.0
        if best is None or score < best[0]:
            best = (score, params, mdl)
    return best


def score(kind: str, mdl, X, y) -> dict:
    if len(y) == 0:
        return {"n": 0}
    if kind == "reg":
        p = mdl.predict(X)
        return {"n": int(len(y)), "rmse": float(root_mean_squared_error(y, p)),
                "mae": float(mean_absolute_error(y, p)), "r2": float(r2_score(y, p)) if len(y) > 1 else None}
    p = mdl.predict_proba(X)[:, 1]
    out = {"n": int(len(y)), "positive_rate": float(np.mean(y)), "brier": float(brier_score_loss(y, p)),
           "accuracy": float(np.mean((p > 0.5) == y))}
    if len(np.unique(y)) == 2:
        out["auc"] = float(roc_auc_score(y, p))
    return out


def planner_lookup(models: dict, ref: pd.DataFrame, gravity: float, suit_mass: float) -> dict:
    """Grid of predictions the web Step Scorer can interpolate.

    `ref` holds the Mars rows the models were trained on. The grid stays inside
    what those rows cover: roughness levels are their quantiles and the features
    that are not on an axis are held at their medians (all listed in the output).
    """
    def median(col: str, default: float) -> float:
        v = ref[col].astype(float).dropna() if col in ref else pd.Series(dtype=float)
        return float(v.median()) if len(v) else default

    slopes = np.arange(-25, 26, 1.0)
    rough = np.array([0.0, 0.01, 0.02, 0.04, 0.06])
    seen = ref["dtm_roughness_m"].astype(float).dropna() if "dtm_roughness_m" in ref else pd.Series(dtype=float)
    if seen.nunique() >= 5:
        rough = np.unique(np.round(seen.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_numpy(), 4))
    excess = 3.0   # how far the steepest tenth of a patch is above its mean slope
    if len(ref) and {"dtm_slope_p90_deg", "dtm_slope_along_deg"} <= set(ref.columns):
        gap = ref["dtm_slope_p90_deg"].astype(float) - ref["dtm_slope_along_deg"].astype(float).abs()
        excess = max(float(gap.median()), 0.0)
    fixed = {"dtm_slope_cross_deg": 0.0, "rock_density_per_m2": median("rock_density_per_m2", 0.1),
             "rock_height_m": median("rock_height_m", 0.1), "friction": median("friction", 0.7),
             "gravity": gravity, "suit_mass_kg": suit_mass,
             "pack_up_m": median("pack_up_m", 0.05), "pack_back_m": median("pack_back_m", 0.15)}
    rows = [dict(fixed, dtm_slope_along_deg=s, dtm_slope_p90_deg=abs(s) + excess, dtm_roughness_m=r)
            for r in rough for s in slopes]
    X = pd.DataFrame(rows)[FEATURES]
    out = {"slope_deg": slopes.tolist(), "roughness_m": rough.tolist(), "gravity": gravity,
           "suit_mass_kg": suit_mass, "fixed_features": fixed, "slope_p90_excess_deg": excess}
    for name, (mdl, kind) in models.items():
        pred = mdl.predict(X) if kind == "reg" else mdl.predict_proba(X)[:, 1]
        out[name] = pred.reshape(len(rough), len(slopes)).round(4).tolist()
    if "fall" in out:
        # "fall" is the chance of falling somewhere along one patch. The planner adds
        # steps of its own length, so also give the chance per metre walked.
        length = median("patch_length_m", float("nan"))
        if not np.isfinite(length):
            length = median("distance_m", 10.0)
        length = max(length, 1.0)
        per_m = 1.0 - (1.0 - np.clip(np.array(out["fall"]), 0.0, 1.0 - 1e-9)) ** (1.0 / length)
        out["fall_exposure_m"] = length
        out["fall_per_m"] = per_m.round(5).tolist()
    walked = ref[label_mask(ref, "walked")] if len(ref) else ref
    if len(walked):
        sl = walked["dtm_slope_along_deg"].astype(float)
        # the cost, speed and stability grids are extrapolated outside this range
        out["walked_slope_range_deg"] = [float(sl.quantile(0.02)), float(sl.quantile(0.98))]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, nargs="+", help="one or more run folders (local or gs://)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--require-quality", action="store_true",
                    help="stop if the data fails the checks in marswalk_sim/quality.py")
    args = ap.parse_args()

    df = load(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"rows_total": int(len(df)), "status_counts": df["status"].value_counts().to_dict(), "data": args.data}
    print(f"loaded {len(df)} rows; status: {report['status_counts']}")

    quality = dataset_report(df)
    failed = [c for c in quality["checks"] if c["level"] == "fail"]
    report["quality"] = {"verdict": quality["verdict"], "failed": failed,
                         "warnings": [c for c in quality["checks"] if c["level"] == "warn"]}
    for c in failed:
        print(f"[quality FAIL] {c['check']}: {c['detail']}")
    if failed and args.require_quality:
        (out / "metrics.json").write_text(json.dumps(report, indent=2, default=float))
        raise SystemExit("data failed the quality checks; see scripts/validate.py")

    usable = df[df["status"].eq("ok")]
    splits = split_maps(usable["map_id"].to_numpy(), args.seed)
    (out / "split.json").write_text(json.dumps(splits, indent=2))
    report["maps"] = {k: len(v) for k, v in splits.items()}
    which = {m: k for k, v in splits.items() for m in v}

    models = {}
    for name, (col, kind, rule) in TARGETS.items():
        if col not in df.columns:
            report[name] = {"skipped": True, "train_rows": 0, "reason": f"no column {col}"}
            continue
        d = rows_for(df, rule).dropna(subset=FEATURES + [col])
        if kind == "clf":
            d = d.assign(**{col: d[col].astype(int)})
        part = d["map_id"].map(which)
        tr, va, te = (d[part.eq(k)] for k in ("train", "val", "test"))
        if len(tr) < 20 or (kind == "clf" and tr[col].nunique() < 2):
            print(f"[{name}] skipped: {len(tr)} training rows")
            report[name] = {"skipped": True, "train_rows": int(len(tr))}
            continue
        sel_score, params, mdl = fit_one(kind, tr[FEATURES], tr[col], va[FEATURES], va[col], args.seed)
        report[name] = {"target": col, "params": params, "val_selection_score": sel_score,
                        "train": score(kind, mdl, tr[FEATURES], tr[col]),
                        "val": score(kind, mdl, va[FEATURES], va[col]),
                        "test": score(kind, mdl, te[FEATURES], te[col])}
        joblib.dump({"model": mdl, "features": FEATURES, "target": col, "kind": kind}, out / f"model_{name}.joblib")
        models[name] = (mdl, kind)
        print(f"[{name}] test: {report[name]['test']}")

    if models:
        mars = usable[usable["gravity"].astype(float) < 5]
        mars = mars[mars["map_id"].map(which).eq("train")]
        lookup = planner_lookup(models, mars, gravity=3.711, suit_mass=40.0)
        if "sim_version" in df.columns:
            lookup["sim_version"] = sorted(df["sim_version"].dropna().astype(str).unique())
        (out / "planner_lookup_mars.json").write_text(json.dumps(lookup))
    (out / "metrics.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"wrote {out}/metrics.json")


if __name__ == "__main__":
    main()
