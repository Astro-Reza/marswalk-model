"""Data-quality rules for generated patches.

Two things live here so that validation and training can never disagree:
  * label_mask()    which rows may be used to train each target
  * dataset_report() checks on a whole run, each one pass / warn / fail

scripts/validate.py runs the report between generating and training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import __version__, config as C

# What the planner can know before the walk (same list as scripts/train.py FEATURES).
FEATURES = [
    "dtm_slope_along_deg", "dtm_slope_cross_deg", "dtm_slope_p90_deg", "dtm_roughness_m",
    "rock_density_per_m2", "rock_height_m", "friction",
    "gravity", "suit_mass_kg", "pack_up_m", "pack_back_m",
]
REQUIRED = FEATURES + ["map_id", "patch_id", "status", "fell", "distance_m", "speed_mps", "cot_net_J_per_kg_m"]

# Plausible ranges for a walking human model: (low, high). Outside means the
# rollout was not walking in any normal sense, whatever the status says.
LABEL_RANGES = {
    "speed_mps": (0.3, 2.5),
    "cot_net_J_per_kg_m": (0.2, 30.0),
    "stride_time_mean_s": (0.4, 2.5),
    "peak_grf_bw": (0.5, 6.0),
}
SLOPE_BINS = [-90.0, -15.0, -5.0, 5.0, 15.0, 90.0]
SLOPE_BIN_NAMES = ["below -15", "-15 to -5", "-5 to +5", "+5 to +15", "above +15"]

T = {
    "error_rate_warn": 0.0, "error_rate_fail": 0.05,
    "ok_rate_warn": 0.8, "ok_rate_fail": 0.5,
    "walked_rate_warn": 0.5, "walked_rate_fail": 0.2,
    "min_walked_rows": 200,
    "range_warn": 0.05, "range_fail": 0.25,
    "missed_fall_warn": 0.01, "missed_fall_fail": 0.05,
    "slope_mismatch_deg": 2.0, "path_mismatch_deg": 3.0,
    "min_bin_rows": 10, "min_class_rows": 20, "min_maps": 20,
    "earth_cot_range": (0.8, 7.5), "earth_minetti_r": 0.5,
}


def minetti_walk(i):
    """Minetti et al. 2002, cost of walking (J/kg/m) at gradient i (rise/run)."""
    i = np.asarray(i, dtype=float)
    return 280.5 * i**5 - 58.7 * i**4 - 76.8 * i**3 + 51.9 * i**2 + 19.6 * i + 2.5


def _fell(df: pd.DataFrame) -> pd.Series:
    return df["fell"].fillna(True).astype(bool)   # unknown counts as not walked


def label_mask(df: pd.DataFrame, rule: str) -> pd.Series:
    """Rows usable for a target.

    reached_patch : the walker got onto the patch (fall labels)
    walked        : it also stayed up and covered enough distance for a
                    cost / speed / stability label to mean something
    """
    ok = df["status"].eq("ok")
    if rule == "reached_patch":
        return ok
    if rule != "walked":
        raise ValueError(f"unknown rule {rule!r}")
    mask = ok & ~_fell(df)
    if "distance_m" in df:
        mask &= df["distance_m"].astype(float) >= C.MIN_LABEL_DISTANCE_M
    return mask


def slope_table(df: pd.DataFrame) -> list:
    """Per slope band: rows, how many walked, fall rate, median cost."""
    ok = df[label_mask(df, "reached_patch")]
    if ok.empty or "dtm_slope_along_deg" not in ok:
        return []
    band = pd.cut(ok["dtm_slope_along_deg"].astype(float), SLOPE_BINS, labels=SLOPE_BIN_NAMES)
    walked = label_mask(ok, "walked")
    rows = []
    for name in SLOPE_BIN_NAMES:
        sel = band.eq(name)
        w = ok[sel & walked]
        cot = w["cot_net_J_per_kg_m"].astype(float) if "cot_net_J_per_kg_m" in w else pd.Series(dtype=float)
        rows.append({"slope_band_deg": name, "rows": int(sel.sum()), "walked": int(len(w)),
                     "fall_rate": float(_fell(ok[sel]).mean()) if sel.any() else None,
                     "cot_net_median": float(cot.median()) if cot.notna().any() else None})
    return rows


def dataset_report(df: pd.DataFrame, metas: list | None = None) -> dict:
    """Run every check. Returns {"verdict": pass|warn|fail, "checks": [...], "summary": {...}}."""
    checks = []

    def add(name: str, level: str, detail: str) -> None:
        checks.append({"check": name, "level": level, "detail": detail})

    def graded(name: str, value: float, warn: float, fail: float, detail: str, low_is_bad: bool = False) -> None:
        bad_fail = value < fail if low_is_bad else value > fail
        bad_warn = value < warn if low_is_bad else value > warn
        add(name, "fail" if bad_fail else "warn" if bad_warn else "pass", detail)

    n = len(df)
    missing = [c for c in REQUIRED if c not in df.columns]
    if n == 0 or missing:
        add("schema", "fail", "no rows" if n == 0 else f"missing columns: {missing}")
        return {"verdict": "fail", "checks": checks, "summary": {"rows": int(n)}}
    add("schema", "pass", f"{n} rows, {df.shape[1]} columns")

    # ---- lineage: one code version, no repeated patches
    dup = int(df["patch_id"].duplicated().sum())
    add("unique_patches", "fail" if dup else "pass",
        f"{dup} repeated patch_id (two runs with the same seed give the same patches)" if dup else "no repeated patch_id")
    if "sim_version" not in df.columns or df["sim_version"].isna().any():
        add("sim_version", "fail", "rows without sim_version: made before 0.2.0, when a slope was mistaken for a fall "
                                   "and terrain friction had no effect")
    else:
        versions = sorted(df["sim_version"].astype(str).unique())
        if len(versions) > 1:
            add("sim_version", "fail", f"mixed versions {versions}: labels are not comparable")
        elif versions[0] != __version__:
            add("sim_version", "warn", f"data is {versions[0]}, this code is {__version__}")
        else:
            add("sim_version", "pass", versions[0])
    if "walker" in df.columns and df["walker"].nunique() > 1:
        add("one_walker", "warn", f"walkers mixed: {sorted(df['walker'].dropna().astype(str).unique())}. "
                                  "Their costs are not on the same scale; train one model per walker")

    # ---- did the rollouts run
    status = df["status"].astype(str)
    err = float(status.str.startswith("error").mean())
    graded("errors", err, T["error_rate_warn"], T["error_rate_fail"],
           f"{err:.1%} of patches raised an error" + (f", e.g. {status[status.str.startswith('error')].iloc[0][:120]}" if err else ""))
    ok_mask = label_mask(df, "reached_patch")
    ok_rate = float(ok_mask.mean())
    graded("reached_patch", ok_rate, T["ok_rate_warn"], T["ok_rate_fail"],
           f"{ok_rate:.1%} of patches were reached; status counts {status.value_counts().to_dict()}", low_is_bad=True)

    # ---- shards
    if metas:
        not_walking = [m.get("shard_index") for m in metas if m.get("policy_walks_on_flat_earth") is False]
        walking = sum(1 for m in metas if m.get("policy_walks_on_flat_earth"))
        add("calibration", "fail" if not_walking else "pass" if walking == len(metas) else "warn",
            f"walker did not walk on flat Earth ground in shards {not_walking[:20]}" if not_walking
            else f"{walking}/{len(metas)} shards calibrated and walked"
            + ("" if walking == len(metas) else " (the others skipped calibration)"))
        unfinished = [m.get("shard_index") for m in metas if not m.get("finished")]
        add("shards_finished", "warn" if unfinished else "pass",
            f"{len(unfinished)}/{len(metas)} shards did not finish: {unfinished[:20]}" if unfinished
            else f"all {len(metas)} shards finished")
        signs = {(m.get("frame_axis"), m.get("frame_sign")) for m in metas if "frame_sign" in m}
        if len(signs) > 1:
            add("walking_direction", "warn", f"shards disagree on the walking direction: {sorted(signs)}")

    ok = df[ok_mask]
    walked_mask = label_mask(df, "walked")
    walked = df[walked_mask]
    mars_ok = ok[ok["gravity"].astype(float) < 5]
    mars_walked = walked[walked["gravity"].astype(float) < 5]

    # ---- can the walker walk
    if len(mars_ok):
        rate = len(mars_walked) / len(mars_ok)
        graded("mars_walked", rate, T["walked_rate_warn"], T["walked_rate_fail"],
               f"{len(mars_walked)}/{len(mars_ok)} reached Mars patches were walked without a fall ({rate:.1%})",
               low_is_bad=True)
    else:
        add("mars_walked", "fail", "no Mars patch was reached")
    add("enough_rows", "warn" if len(walked) < T["min_walked_rows"] else "pass",
        f"{len(walked)} walked rows (want at least {T['min_walked_rows']} to train the cost model)")

    # ---- are the labels plausible
    for col, (lo, hi) in LABEL_RANGES.items():
        if col not in walked.columns or walked.empty:
            continue
        v = walked[col].astype(float)
        bad = float(((v < lo) | (v > hi) | v.isna()).mean())
        graded(f"range_{col}", bad, T["range_warn"], T["range_fail"],
               f"{bad:.1%} of walked rows outside {lo}..{hi} or empty (median {v.median():.3g})")
    if {"com_height_min_m", "com_height_start_m"} <= set(df.columns) and len(walked):
        low = walked["com_height_min_m"].astype(float) < 0.6 * walked["com_height_start_m"].astype(float)
        graded("missed_falls", float(low.mean()), T["missed_fall_warn"], T["missed_fall_fail"],
               f"{int(low.sum())} rows not marked as falls where the centre of mass dropped below 60 % of its start height")
    if "fall_reason" in ok.columns and _fell(ok).any():
        reasons = ok.loc[_fell(ok), "fall_reason"].astype(str).value_counts()
        turned = reasons.get("turned", 0) / reasons.sum()
        add("fall_reasons", "warn" if turned > 0.5 else "pass",
            f"{reasons.to_dict()}" + (". Most 'falls' are the walker turning away, not falling" if turned > 0.5 else ""))

    # ---- do the features describe the terrain that was walked
    if "slope_deg" in ok.columns and len(ok):
        gap = (ok["dtm_slope_along_deg"].astype(float) - ok["slope_deg"].astype(float)).abs()
        add("slope_feature", "fail" if gap.quantile(0.95) > T["slope_mismatch_deg"] else "pass",
            f"measured slope vs requested slope: 95 % within {gap.quantile(0.95):.2f} deg")
    if "path_slope_deg" in walked.columns and walked["path_slope_deg"].notna().any():
        gap = (walked["path_slope_deg"].astype(float) - walked["dtm_slope_along_deg"].astype(float)).abs()
        add("path_slope", "warn" if gap.median() > T["path_mismatch_deg"] else "pass",
            f"slope under the walker vs patch slope: median gap {gap.median():.2f} deg, 95 % within {gap.quantile(0.95):.2f} deg")

    # ---- coverage: where can the model be trusted
    table = slope_table(df)
    thin = [r["slope_band_deg"] for r in table if r["walked"] < T["min_bin_rows"]] or ([] if table else SLOPE_BIN_NAMES)
    add("slope_coverage", "warn" if thin else "pass",
        f"fewer than {T['min_bin_rows']} walked rows in slope bands {thin}: the cost model extrapolates there" if thin
        else "every slope band has walked rows")
    n_fell = int(_fell(ok).sum())
    n_stood = int(len(ok) - n_fell)
    add("fall_classes", "warn" if min(n_fell, n_stood) < T["min_class_rows"] else "pass",
        f"{n_fell} falls, {n_stood} non-falls among reached patches (want at least {T['min_class_rows']} of each)")
    flat = [f for f in FEATURES if f != "gravity" and ok[f].nunique(dropna=True) < 2]
    if flat and len(ok) > 1:
        add("feature_variation", "warn", f"features that never change: {flat}")
    n_maps = int(walked["map_id"].nunique())
    add("maps", "warn" if n_maps < T["min_maps"] else "pass",
        f"{n_maps} maps with walked rows (the split is by map; want at least {T['min_maps']})")

    # ---- Earth sanity check against Minetti 2002
    earth = walked[walked["gravity"].astype(float) >= 5]
    if "cot_gross_J_per_kg_m" in earth.columns and len(earth) >= 5:
        cot = earth["cot_gross_J_per_kg_m"].astype(float)
        level = earth[earth["dtm_slope_along_deg"].astype(float).abs() < 3.0]
        if len(level) >= 3:
            med = float(level["cot_gross_J_per_kg_m"].astype(float).median())
            lo, hi = T["earth_cot_range"]
            add("earth_level_cost", "pass" if lo <= med <= hi else "warn",
                f"level Earth walking costs {med:.2f} J/kg/m gross (Minetti: 2.5; accepted {lo}..{hi})")
        if len(earth) >= 20:
            ref = minetti_walk(np.tan(np.radians(earth["dtm_slope_along_deg"].astype(float))))
            good = cot.notna().to_numpy()
            r = float(np.corrcoef(cot.to_numpy()[good], ref[good])[0, 1])
            add("earth_slope_curve", "pass" if r >= T["earth_minetti_r"] else "warn",
                f"Earth cost vs slope follows Minetti 2002 with r = {r:.2f} (want {T['earth_minetti_r']} or more)")
    else:
        add("earth_check", "warn", f"only {len(earth)} walked Earth rows: no check against Minetti 2002")

    levels = [c["level"] for c in checks]
    verdict = "fail" if "fail" in levels else "warn" if "warn" in levels else "pass"
    summary = {
        "rows": int(n), "reached_patch": int(len(ok)), "walked": int(len(walked)), "falls": n_fell,
        "maps": int(df["map_id"].nunique()), "mars_rows": int(len(mars_ok)), "earth_rows": int(len(ok) - len(mars_ok)),
        "status": {str(k): int(v) for k, v in status.value_counts().items()},
        "by_slope": table,
    }
    return {"verdict": verdict, "checks": checks, "summary": summary}
