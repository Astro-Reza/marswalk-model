"""GUI-free logic: load datasets, model runs and walker checks, and compute what the tabs show.

Everything here works on plain pandas / numpy objects so it can be tested without Qt.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Same list as scripts/train.py FEATURES (kept in sync by tests/test_gui_core.py).
FEATURES = [
    "dtm_slope_along_deg", "dtm_slope_cross_deg", "dtm_slope_p90_deg", "dtm_roughness_m",
    "rock_density_per_m2", "rock_height_m", "friction",
    "gravity", "suit_mass_kg", "pack_up_m", "pack_back_m",
]
# Slider ranges for the what-if panel: (low, high, default)
FEATURE_RANGES = {
    "dtm_slope_along_deg": (-25.0, 25.0, 0.0),
    "dtm_slope_cross_deg": (-10.0, 10.0, 0.0),
    "dtm_slope_p90_deg": (0.0, 35.0, 5.0),
    "dtm_roughness_m": (0.0, 0.06, 0.01),
    "rock_density_per_m2": (0.0, 0.4, 0.1),
    "rock_height_m": (0.03, 0.25, 0.1),
    "friction": (0.4, 1.0, 0.7),
    "gravity": (3.0, 10.0, 3.711),
    "suit_mass_kg": (0.0, 60.0, 40.0),
    "pack_up_m": (-0.15, 0.25, 0.05),
    "pack_back_m": (0.05, 0.25, 0.15),
}
TARGET_LABELS = {
    "cot": "Net cost of transport (J/kg/m)",
    "fall": "Fall probability",
    "mos": "Min lateral margin of stability (m)",
}


# --------------------------------------------------------------------------- datasets
def load_dataset(path: str) -> pd.DataFrame:
    """Read every part-*.parquet under `path` (local folder or gs:// via fsspec)."""
    import fsspec

    fs, root = fsspec.core.url_to_fs(path)
    root = root.rstrip("/")
    files = sorted(fs.glob(f"{root}/**/part-*.parquet")) or sorted(fs.glob(f"{root}/part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no part-*.parquet files under {path}")
    frames = []
    for f in files:
        with fs.open(f, "rb") as fh:
            frames.append(pd.read_parquet(fh))
    return pd.concat(frames, ignore_index=True)


def dataset_summary(df: pd.DataFrame) -> dict:
    """Headline numbers for the Dataset tab."""
    status = df["status"].value_counts().to_dict() if "status" in df else {}
    ok = df[df["status"].eq("ok")] if "status" in df else df
    fell = ok["fell"].astype(bool) if "fell" in ok else pd.Series(dtype=bool)
    mars = ok[ok["gravity"] < 5] if "gravity" in ok else ok
    walked = ok[~fell] if len(fell) else ok
    out = {
        "rows": int(len(df)),
        "maps": int(df["map_id"].nunique()) if "map_id" in df else 0,
        "status": {str(k): int(v) for k, v in status.items()},
        "ok_rows": int(len(ok)),
        "fall_rate": float(fell.mean()) if len(fell) else None,
        "mars_rows": int(len(mars)),
        "earth_rows": int(len(ok) - len(mars)),
        "walked_rows": int(len(walked)),
    }
    if "cot_net_J_per_kg_m" in walked and walked["cot_net_J_per_kg_m"].notna().any():
        out["cot_median"] = float(walked["cot_net_J_per_kg_m"].median())
    if "walker" in df and df["walker"].notna().any():
        out["walkers"] = sorted(df["walker"].dropna().astype(str).unique().tolist())
    return out


def numeric_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])]
    return [c for c in cols if df[c].notna().any()]


def apply_filters(df: pd.DataFrame, only_ok=True, hide_falls=False, planet="all") -> pd.DataFrame:
    """planet: 'all' | 'mars' | 'earth' (Earth runs are the 10 % gravity sanity check)."""
    out = df
    if only_ok and "status" in out:
        out = out[out["status"].eq("ok")]
    if hide_falls and "fell" in out:
        out = out[~out["fell"].astype(bool)]
    if planet != "all" and "gravity" in out:
        out = out[out["gravity"] < 5] if planet == "mars" else out[out["gravity"] >= 5]
    return out


def minetti_walk(i):
    """Minetti et al. 2002, cost of walking (J/kg/m) at gradient i (rise/run). Same as check_walker.py."""
    i = np.asarray(i, dtype=float)
    return 280.5 * i**5 - 58.7 * i**4 - 76.8 * i**3 + 51.9 * i**2 + 19.6 * i + 2.5


# --------------------------------------------------------------------------- model runs
@dataclass
class ModelRun:
    path: Path
    name: str
    metrics: dict = field(default_factory=dict)
    split: dict = field(default_factory=dict)
    lookup: dict | None = None
    models: dict = field(default_factory=dict)  # name -> (estimator, kind, features)
    warnings: list = field(default_factory=list)


def load_run(path: str | Path, load_models: bool = True) -> ModelRun:
    """Read a folder written by scripts/train.py."""
    p = Path(path)
    if not (p / "metrics.json").exists():
        raise FileNotFoundError(f"{p} has no metrics.json (is this a scripts/train.py output folder?)")
    run = ModelRun(path=p, name=p.name or str(p))
    run.metrics = json.loads((p / "metrics.json").read_text())
    if (p / "split.json").exists():
        run.split = json.loads((p / "split.json").read_text())
    if (p / "planner_lookup_mars.json").exists():
        run.lookup = json.loads((p / "planner_lookup_mars.json").read_text())
    if load_models:
        try:
            import joblib
        except ImportError:  # pragma: no cover
            run.warnings.append("joblib not installed; what-if predictions are unavailable")
            return run
        for name in TARGET_LABELS:
            f = p / f"model_{name}.joblib"
            if f.exists():
                try:
                    blob = joblib.load(f)
                    run.models[name] = (blob["model"], blob["kind"], blob.get("features", FEATURES))
                except Exception as e:  # version mismatch etc.
                    run.warnings.append(f"model_{name}.joblib could not be loaded: {type(e).__name__}: {e}")
    return run


def compare_table(runs: list[ModelRun]) -> pd.DataFrame:
    """One row per run and target, with the test-set scores that matter."""
    rows = []
    for r in runs:
        maps = r.metrics.get("maps", {})
        for name in TARGET_LABELS:
            m = r.metrics.get(name)
            if not m:
                continue
            if m.get("skipped"):
                rows.append({"run": r.name, "target": name, "note": f"skipped ({m.get('train_rows')} train rows)"})
                continue
            t, tr = m.get("test", {}), m.get("train", {})
            rows.append({
                "run": r.name, "target": name, "n_test": t.get("n"),
                "r2_test": t.get("r2"), "rmse_test": t.get("rmse"), "mae_test": t.get("mae"),
                "auc_test": t.get("auc"), "brier_test": t.get("brier"),
                "r2_train": tr.get("r2"), "auc_train": tr.get("auc"),
                "maps_train": maps.get("train"), "maps_test": maps.get("test"),
                "note": overfit_note(m),
            })
    return pd.DataFrame(rows)


def overfit_note(m: dict) -> str:
    """Flag a large train/test gap, small test sets and missing scores."""
    t, tr = m.get("test", {}), m.get("train", {})
    notes = []
    if not t.get("n"):
        return "no test rows"
    if t["n"] < 30:
        notes.append(f"only {t['n']} test rows")
    for k in ("r2", "auc"):
        if tr.get(k) is not None and t.get(k) is not None and tr[k] - t[k] > 0.15:
            notes.append(f"{k} drops {tr[k] - t[k]:.2f} from train to test")
    return "; ".join(notes)


def predict_features(run: ModelRun, values: dict) -> dict:
    """Run every loaded model on one feature row. Returns {target: value}."""
    out = {}
    for name, (mdl, kind, feats) in run.models.items():
        X = pd.DataFrame([{f: values[f] for f in feats}])[feats]
        out[name] = float(mdl.predict_proba(X)[0, 1]) if kind == "clf" else float(mdl.predict(X)[0])
    return out


def test_predictions(run: ModelRun, df: pd.DataFrame, target: str, which: str = "test") -> pd.DataFrame:
    """Actual vs predicted for the rows of the maps the run held out (`which` = train | val | test)."""
    if target not in run.models:
        raise KeyError(f"model_{target} is not loaded")
    col = {"cot": "cot_net_J_per_kg_m", "fall": "fell", "mos": "mos_ml_min_m"}[target]
    mdl, kind, feats = run.models[target]
    d = df[df["status"].eq("ok") & df["map_id"].isin(run.split.get(which, []))]
    if target != "fall":
        d = d[~d["fell"].astype(bool)]
    d = d.dropna(subset=feats + [col])
    if d.empty:
        return pd.DataFrame(columns=["actual", "predicted"])
    pred = mdl.predict_proba(d[feats])[:, 1] if kind == "clf" else mdl.predict(d[feats])
    return pd.DataFrame({"actual": d[col].astype(float).to_numpy(), "predicted": pred,
                         "map_id": d["map_id"].to_numpy()})


# --------------------------------------------------------------------------- walker check
def load_check(path: str | Path) -> dict:
    d = json.loads(Path(path).read_text())
    if "stages" not in d:
        raise ValueError("not a check_walker.py output (no 'stages')")
    return d


def check_stage_table(check: dict) -> pd.DataFrame:
    rows = []
    for tag, s in check["stages"].items():
        rows.append({"stage": tag, "runs": s.get("runs"), "walked": s.get("walked"), "fall_rate": s.get("fall_rate"),
                     "mean_speed_mps": s.get("mean_speed_mps"), "pearson_r_vs_minetti": s.get("pearson_r_vs_minetti"),
                     "cheapest_grade": s.get("cheapest_grade")})
    return pd.DataFrame(rows)


def check_cot_curves(check: dict) -> dict:
    """{stage: (grades, cot)} for the slope sweeps, plus the Minetti reference."""
    out = {}
    for tag, s in check["stages"].items():
        by = s.get("cot_gross_by_grade")
        if by:
            g = sorted(float(k) for k in by)
            out[tag] = (g, [by[f"{x:+.2f}"] for x in g])
    ref = check.get("minetti_reference", {})
    if ref:
        g = sorted(float(k) for k in ref)
        out["minetti"] = (g, [ref[f"{x:+.2f}"] for x in g])
    return out
