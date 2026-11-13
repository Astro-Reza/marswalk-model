#!/usr/bin/env python
"""Train the step-cost surrogate on generated patches.

Split is by map_id (GroupShuffleSplit-style): train 70 % / validation 15 % /
test 15 % of maps. Validation picks the model size; test is scored once.

    python scripts/train.py --data gs://BUCKET/runs/run-001 --out models/run-001
    python scripts/train.py --data data/run-001 --out models/run-001
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import fsspec
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (brier_score_loss, log_loss, mean_absolute_error, r2_score,
                             roc_auc_score, root_mean_squared_error)

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
}
GRID = [
    {"learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 20},
    {"learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 20},
    {"learning_rate": 0.10, "max_leaf_nodes": 31, "min_samples_leaf": 10},
    {"learning_rate": 0.05, "max_leaf_nodes": 63, "min_samples_leaf": 10},
]


def load(data: str) -> pd.DataFrame:
    fs, root = fsspec.core.url_to_fs(data)
    files = sorted(fs.glob(f"{root.rstrip('/')}/**/part-*.parquet")) or sorted(fs.glob(f"{root.rstrip('/')}/part-*.parquet"))
    if not files:
        raise SystemExit(f"no part-*.parquet files under {data}")
    frames = []
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
    ok = df["status"].eq("ok")
    if rule == "walked":
        return df[ok & ~df["fell"].astype(bool)]
    return df[ok]


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


def planner_lookup(models: dict, gravity: float, suit_mass: float) -> dict:
    """Grid of predictions the web Step Scorer can interpolate."""
    slopes = np.arange(-25, 26, 1.0)
    rough = np.array([0.0, 0.01, 0.02, 0.04, 0.06])
    rows = [{"dtm_slope_along_deg": s, "dtm_slope_cross_deg": 0.0, "dtm_slope_p90_deg": abs(s) + 3,
             "dtm_roughness_m": r, "rock_density_per_m2": 0.1, "rock_height_m": 0.1, "friction": 0.7,
             "gravity": gravity, "suit_mass_kg": suit_mass, "pack_up_m": 0.05, "pack_back_m": 0.15}
            for r in rough for s in slopes]
    X = pd.DataFrame(rows)[FEATURES]
    out = {"slope_deg": slopes.tolist(), "roughness_m": rough.tolist(), "gravity": gravity,
           "suit_mass_kg": suit_mass}
    for name, (mdl, kind) in models.items():
        pred = mdl.predict(X) if kind == "reg" else mdl.predict_proba(X)[:, 1]
        out[name] = pred.reshape(len(rough), len(slopes)).round(4).tolist()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = load(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"rows_total": int(len(df)), "status_counts": df["status"].value_counts().to_dict()}
    print(f"loaded {len(df)} rows; status: {report['status_counts']}")

    usable = df[df["status"].eq("ok")]
    splits = split_maps(usable["map_id"].to_numpy(), args.seed)
    (out / "split.json").write_text(json.dumps(splits, indent=2))
    report["maps"] = {k: len(v) for k, v in splits.items()}
    which = {m: k for k, v in splits.items() for m in v}

    models = {}
    for name, (col, kind, rule) in TARGETS.items():
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
        lookup = planner_lookup(models, gravity=3.711, suit_mass=40.0)
        (out / "planner_lookup_mars.json").write_text(json.dumps(lookup))
    (out / "metrics.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"wrote {out}/metrics.json")


if __name__ == "__main__":
    main()
