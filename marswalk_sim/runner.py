"""Run the walker over one patch and turn the rollout into labels."""
from __future__ import annotations

import time

import numpy as np

from . import config as C
from .env import MarsPatchEnv
from .terrain import PatchParams, build_heightfield, patch_stats

SETTLE_S = 0.5            # ignore the first half second (start-up transient)
CONTACT_FRAC_BW = 0.10    # foot is "in stance" above 10 % body weight
HALF_FOOT_WIDTH = 0.04    # m, lateral half-width used for margin of stability


def rollout(menv: MarsPatchEnv, policy, p: PatchParams, keep_traj: bool = False) -> dict:
    hf = build_heightfield(p, menv.nrow, menv.ncol)
    obs = menv.reset(p, hf)
    policy.reset()

    g = float(p.gravity)
    bw = menv.total_mass() * g
    s_end = hf.s[-1] - C.END_MARGIN_M
    n_max = int(C.MAX_SIM_TIME_S / menv.dt)

    T, COM, PPOS, PNEG, EFF, FR, FL, PR, PL, TR, TL = ([] for _ in range(11))
    term = trunc = False
    t0 = time.time()
    for k in range(n_max):
        obs, _, term, trunc, _ = menv.step(policy(obs))
        pw = menv.muscle_power()
        cf = menv.foot_contacts()
        fp = menv.foot_positions()
        T.append(menv.d.time)
        COM.append(menv.com())
        PPOS.append(pw[pw > 0].sum())
        PNEG.append(-pw[pw < 0].sum())
        EFF.append(float(np.mean(menv.d.act ** 2)) if menv.m.na else 0.0)
        FR.append(cf["r"]); FL.append(cf["l"])
        PR.append(fp["r"]); PL.append(fp["l"]); TR.append(fp["r_toe"]); TL.append(fp["l_toe"])
        if term or trunc or menv.to_s(COM[-1][1]) > s_end:
            break
    wall = time.time() - t0

    T = np.asarray(T); COM = np.asarray(COM)
    traj = dict(t=T, com=COM, p_pos=np.asarray(PPOS), p_neg=np.asarray(PNEG), effort=np.asarray(EFF),
                f_r=np.asarray(FR), f_l=np.asarray(FL), foot_r=np.asarray(PR), foot_l=np.asarray(PL),
                toe_r=np.asarray(TR), toe_l=np.asarray(TL))
    labels = compute_labels(menv, traj, g, bw, fell=bool(term))
    labels.update(sim_steps=len(T), sim_time_s=float(T[-1]) if len(T) else 0.0, wall_time_s=wall,
                  reached_end=bool(len(T) and menv.to_s(COM[-1][1]) > s_end), truncated=bool(trunc))
    row = p.to_dict() | patch_stats(hf) | hf.meta | labels
    row["total_mass_kg"] = menv.total_mass()
    row["human_mass_kg"] = menv.human_mass
    if keep_traj:
        row["_traj"] = traj
    return row


def compute_labels(menv: MarsPatchEnv, tr: dict, g: float, bw: float, fell: bool) -> dict:
    nan = float("nan")
    out = {"fell": fell}
    t, com = tr["t"], tr["com"]
    if len(t) < 3:
        return out | {"status": "no_samples"}
    dt = menv.dt
    s = menv.forward * com[:, 1]
    region = (t >= SETTLE_S) & (s >= C.FLAT_START_M + C.BLEND_M)
    out["status"] = "ok" if region.sum() >= 20 else ("fell_before_patch" if fell else "too_short")
    if region.sum() < 20:
        return out | {k: nan for k in ("distance_m", "speed_mps", "cot_net_J_per_kg_m", "cot_gross_J_per_kg_m",
                                       "met_power_net_W", "mos_ml_min_m", "mos_ml_mean_m", "mos_ap_mean_m",
                                       "stride_time_mean_s", "stride_time_cv", "peak_grf_bw", "effort")}

    idx = np.where(region)[0]
    i0, i1 = idx[0], idx[-1]
    dur = t[i1] - t[i0] + dt
    dist = float(s[i1] - s[i0])
    w_pos = tr["p_pos"][region].sum() * dt
    w_neg = tr["p_neg"][region].sum() * dt
    e_net = w_pos / 0.25 + w_neg / 1.2
    e_gross = e_net + 1.2 * menv.human_mass * dur
    m_h = menv.human_mass
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
    intervals = np.concatenate([np.diff(x) for x in strikes if len(x) > 1]) if any(len(x) > 1 for x in strikes) else np.array([])
    intervals = intervals[intervals > 0.3]  # drop contact chatter
    out["stride_time_mean_s"] = float(intervals.mean()) if len(intervals) else nan
    out["stride_time_cv"] = float(intervals.std() / intervals.mean()) if len(intervals) >= 3 else nan
    out["n_heel_strikes"] = int(sum(len(x) for x in strikes))

    # --- margin of stability (Hof 2005): XcoM = CoM + v / omega0
    v = np.gradient(com, dt, axis=0)
    l0 = max(com[0, 2] - menv.ground_z(s[0], com[0, 0]), 0.5)
    w0 = np.sqrt(g / l0)
    xcom_x = com[:, 0] + v[:, 0] / w0
    xcom_s = s + menv.forward * v[:, 1] / w0
    single = region & (st_r ^ st_l)
    mos_ml = []
    for i in np.where(single)[0]:
        fx = tr["foot_r"][i, 0] if st_r[i] else tr["foot_l"][i, 0]
        side = np.sign(fx - com[i, 0]) or 1.0
        mos_ml.append(side * (fx - xcom_x[i]) + HALF_FOOT_WIDTH)
    any_st = region & (st_r | st_l)
    mos_ap = []
    for i in np.where(any_st)[0]:
        toes = [menv.forward * tr["toe_r"][i, 1]] if st_r[i] else []
        toes += [menv.forward * tr["toe_l"][i, 1]] if st_l[i] else []
        mos_ap.append(max(toes) - xcom_s[i])
    out["mos_ml_min_m"] = float(np.min(mos_ml)) if mos_ml else nan
    out["mos_ml_mean_m"] = float(np.mean(mos_ml)) if mos_ml else nan
    out["mos_ap_mean_m"] = float(np.mean(mos_ap)) if mos_ap else nan
    return out
