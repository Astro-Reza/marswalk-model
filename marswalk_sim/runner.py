"""Run a walker over one patch and turn the rollout into labels.

Works with any backend that follows the interface described in env.py.
Python 3.8 compatible.
"""
from __future__ import annotations

import time

import numpy as np

from . import config as C
from .terrain import PatchParams, build_heightfield, patch_stats

SETTLE_S = 0.5            # ignore the first half second (start-up transient)
CONTACT_FRAC_BW = 0.10    # foot is "in stance" above 10 % body weight
HALF_FOOT_WIDTH = 0.04    # m, lateral half-width used for margin of stability
LABEL_KEYS = ("distance_m", "speed_mps", "cot_net_J_per_kg_m", "cot_gross_J_per_kg_m", "met_power_net_W",
              "mos_ml_min_m", "mos_ml_mean_m", "mos_ap_mean_m", "stride_time_mean_s", "stride_time_cv",
              "peak_grf_bw", "effort")


def rollout(menv, policy, p: PatchParams, keep_traj: bool = False,
            flat_start: float = C.FLAT_START_M, max_time: float = C.MAX_SIM_TIME_S) -> dict:
    hf = build_heightfield(p, menv.nrow, menv.ncol, flat_start=flat_start)
    obs = menv.reset(p, hf)
    policy.reset()

    g = float(p.gravity)
    bw = menv.total_mass() * g
    s_end = hf.s[-1] - C.END_MARGIN_M
    n_max = int(max_time / menv.dt)

    keys = ("t", "com", "p_pos", "p_neg", "effort", "f_r", "f_l", "foot_r", "foot_l", "toe_r", "toe_l")
    tr = {k: [] for k in keys}
    term = trunc = False
    t0 = time.time()
    for _ in range(n_max):
        obs, _, term, trunc, _ = menv.step(policy(obs))
        pw = menv.muscle_power()
        cf = menv.foot_contacts()
        fp = menv.foot_positions()
        a = menv.act()
        tr["t"].append(float(menv.d.time))
        tr["com"].append(menv.com())
        tr["p_pos"].append(float(pw[pw > 0].sum()))
        tr["p_neg"].append(float(-pw[pw < 0].sum()))
        tr["effort"].append(float(np.mean(a ** 2)) if a is not None and len(a) else 0.0)
        tr["f_r"].append(cf["r"])
        tr["f_l"].append(cf["l"])
        tr["foot_r"].append(fp["r"])
        tr["foot_l"].append(fp["l"])
        tr["toe_r"].append(fp["r_toe"])
        tr["toe_l"].append(fp["l_toe"])
        if term or trunc or menv.frame.s_of(tr["com"][-1]) > s_end:
            break
    wall = time.time() - t0

    tr = {k: np.asarray(v) for k, v in tr.items()}
    labels = compute_labels(menv, tr, g, bw, fell=bool(term), flat_start=flat_start)
    n = len(tr["t"])
    labels.update(sim_steps=n, sim_time_s=float(tr["t"][-1]) if n else 0.0, wall_time_s=wall,
                  reached_end=bool(n and menv.frame.s_of(tr["com"][-1]) > s_end), truncated=bool(trunc))
    row = {}
    for part in (p.to_dict(), patch_stats(hf), hf.meta, labels):
        row.update(part)
    row["total_mass_kg"] = menv.total_mass()
    row["human_mass_kg"] = menv.human_mass
    row["walker"] = getattr(menv, "name", "unknown")
    if keep_traj:
        row["_traj"] = tr
    return row


def compute_labels(menv, tr: dict, g: float, bw: float, fell: bool,
                   flat_start: float = C.FLAT_START_M) -> dict:
    nan = float("nan")
    out = {"fell": fell}
    t, com = tr["t"], tr["com"]
    if len(t) < 3:
        out["status"] = "no_samples"
        return out
    fr = menv.frame
    dt = menv.dt
    s = fr.s_of(com)
    lat = fr.lat_of(com)
    region = (t >= SETTLE_S) & (s >= flat_start + C.BLEND_M)
    min_samples = max(int(0.2 / dt), 5)
    if region.sum() < min_samples:
        out["status"] = "fell_before_patch" if fell else "too_short"
        out.update({k: nan for k in LABEL_KEYS})
        return out
    out["status"] = "ok"

    idx = np.where(region)[0]
    i0, i1 = idx[0], idx[-1]
    dur = t[i1] - t[i0] + dt
    dist = float(s[i1] - s[i0])
    w_pos = tr["p_pos"][region].sum() * dt
    w_neg = tr["p_neg"][region].sum() * dt
    e_net = w_pos / C.EFF_POSITIVE + w_neg / C.EFF_NEGATIVE
    m_h = menv.human_mass
    e_gross = e_net + C.BASAL_W_PER_KG * m_h * dur
    ok_dist = dist > 1.0
    out.update(
        distance_m=dist,
        speed_mps=dist / dur,
        met_power_net_W=e_net / dur,
        cot_net_J_per_kg_m=e_net / (m_h * dist) if ok_dist else nan,
        cot_gross_J_per_kg_m=e_gross / (m_h * dist) if ok_dist else nan,
        effort=float(tr["effort"][region].mean()),
        peak_grf_bw=float(((tr["f_r"] + tr["f_l"])[region]).max() / bw),
    )

    # --- stance, heel strikes, stride timing
    thr = CONTACT_FRAC_BW * bw
    st_r, st_l = tr["f_r"] > thr, tr["f_l"] > thr
    strikes = []
    for st in (st_r, st_l):
        on = np.where(st[1:] & ~st[:-1])[0] + 1
        on = on[region[on]]
        strikes.append(t[on])
    diffs = [np.diff(x) for x in strikes if len(x) > 1]
    intervals = np.concatenate(diffs) if diffs else np.array([])
    intervals = intervals[intervals > 0.3]  # drop contact chatter
    out["stride_time_mean_s"] = float(intervals.mean()) if len(intervals) else nan
    out["stride_time_cv"] = float(intervals.std() / intervals.mean()) if len(intervals) >= 3 else nan
    out["n_heel_strikes"] = int(sum(len(x) for x in strikes))

    # --- margin of stability (Hof 2005): XcoM = CoM + v / omega0
    v = np.gradient(com, dt, axis=0)
    l0 = max(com[0, 2] - menv.ground_z(s[0], lat[0]), 0.5)
    w0 = np.sqrt(g / l0)
    xcom_lat = lat + fr.v_lat(v) / w0
    xcom_s = s + fr.v_s(v) / w0
    foot_lat_r, foot_lat_l = fr.lat_of(tr["foot_r"]), fr.lat_of(tr["foot_l"])
    toe_s_r, toe_s_l = fr.s_of(tr["toe_r"]), fr.s_of(tr["toe_l"])
    single = region & (st_r ^ st_l)
    mos_ml = []
    for i in np.where(single)[0]:
        fl = foot_lat_r[i] if st_r[i] else foot_lat_l[i]
        side = np.sign(fl - lat[i]) or 1.0
        mos_ml.append(side * (fl - xcom_lat[i]) + HALF_FOOT_WIDTH)
    any_st = region & (st_r | st_l)
    mos_ap = []
    for i in np.where(any_st)[0]:
        toes = ([toe_s_r[i]] if st_r[i] else []) + ([toe_s_l[i]] if st_l[i] else [])
        mos_ap.append(max(toes) - xcom_s[i])
    out["mos_ml_min_m"] = float(np.min(mos_ml)) if mos_ml else nan
    out["mos_ml_mean_m"] = float(np.mean(mos_ml)) if mos_ml else nan
    out["mos_ap_mean_m"] = float(np.mean(mos_ap)) if mos_ap else nan
    return out
