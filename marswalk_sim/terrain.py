"""Terrain patches: parameters -> heightfield array.

Coordinates used here:
  s  = distance along the walking direction (m), s = 0 at the start point
  x  = lateral position (m)
The env module maps (s, x) onto the MuJoCo heightfield grid.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np
from scipy.ndimage import gaussian_filter

from . import config as C


@dataclass
class PatchParams:
    slope_deg: float = 0.0
    cross_slope_deg: float = 0.0
    rough_amp_m: float = 0.0
    rough_wavelength_m: float = 1.0
    rock_density_per_m2: float = 0.0
    rock_height_m: float = 0.1
    friction: float = 0.8
    gravity: float = C.G_MARS
    suit_mass_kg: float = 0.0
    pack_up_m: float = 0.0
    pack_back_m: float = 0.1
    map_id: str = "map-0"
    patch_id: str = "patch-0"
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Heightfield:
    """Heights on a regular grid. z[i_s, i_x] in metres."""
    s: np.ndarray          # (n_s,) forward coordinate of each row
    x: np.ndarray          # (n_x,) lateral coordinate of each column
    z: np.ndarray          # (n_s, n_x)
    meta: dict = field(default_factory=dict)


def _ramp(s: np.ndarray, start: float, width: float) -> np.ndarray:
    """0 before `start`, 1 after `start + width`, smooth in between."""
    u = np.clip((s - start) / max(width, 1e-6), 0.0, 1.0)
    return u * u * (3 - 2 * u)


def build_heightfield(p: PatchParams, n_s: int = 100, n_x: int = 100,
                      flat_start: float = C.FLAT_START_M) -> Heightfield:
    rng = np.random.default_rng(p.seed)
    s = np.linspace(-C.BACK_MARGIN_M, 2 * C.HF_HALF_LENGTH_M - C.BACK_MARGIN_M, n_s)
    x = np.linspace(-C.HF_HALF_WIDTH_M, C.HF_HALF_WIDTH_M, n_x)
    S, X = np.meshgrid(s, x, indexing="ij")
    ds, dx = s[1] - s[0], x[1] - x[0]

    s0 = flat_start
    w = _ramp(S, s0, C.BLEND_M)  # terrain features fade in after the run-in

    # 1) slope along the walking direction, continuous at s0
    z = np.tan(np.radians(p.slope_deg)) * np.clip(S - s0, 0.0, None)

    # 2) cross slope (side slope), faded in
    z += np.tan(np.radians(p.cross_slope_deg)) * X * w

    # 3) band-limited roughness: white noise smoothed to the requested wavelength
    if p.rough_amp_m > 0:
        noise = rng.standard_normal(S.shape)
        sigma = (p.rough_wavelength_m / 2.0 / ds, p.rough_wavelength_m / 2.0 / dx)
        noise = gaussian_filter(noise, sigma=sigma, mode="reflect")
        noise /= noise.std() + 1e-12
        z += p.rough_amp_m * noise * w

    # 4) rocks: Gaussian bumps scattered over the patch area
    patch_area = (s[-1] - s0) * (x[-1] - x[0])
    n_rocks = rng.poisson(p.rock_density_per_m2 * max(patch_area, 0.0))
    for _ in range(int(n_rocks)):
        rs = rng.uniform(s0 + C.BLEND_M, s[-1])
        rx = rng.uniform(x[0], x[-1])
        h = p.rock_height_m * rng.uniform(0.5, 1.0)
        r = max(1.5 * h, 0.12)
        z += h * np.exp(-((S - rs) ** 2 + (X - rx) ** 2) / (2 * r * r))

    meta = {"n_rocks": int(n_rocks), "flat_start_m": float(s0)}
    return Heightfield(s=s, x=x, z=z, meta=meta)


def sample_params(rng: np.random.Generator, map_id: str, patch_id: str,
                  shared: dict | None = None) -> PatchParams:
    """Draw one patch. `shared` holds the per-map surface values."""
    R = C.RANGES
    u = lambda k: float(rng.uniform(*R[k]))  # noqa: E731
    shared = shared or {}
    earth = bool(shared.get("earth", False))
    return PatchParams(
        slope_deg=u("slope_deg"),
        cross_slope_deg=u("cross_slope_deg"),
        rough_amp_m=shared.get("rough_amp_m", u("rough_amp_m")),
        rough_wavelength_m=shared.get("rough_wavelength_m", u("rough_wavelength_m")),
        rock_density_per_m2=shared.get("rock_density_per_m2", u("rock_density_per_m2")),
        rock_height_m=shared.get("rock_height_m", u("rock_height_m")),
        friction=shared.get("friction", u("friction")),
        gravity=C.G_EARTH if earth else C.G_MARS,
        suit_mass_kg=0.0 if earth else u("suit_mass_kg"),
        pack_up_m=u("pack_up_m"),
        pack_back_m=u("pack_back_m"),
        map_id=map_id,
        patch_id=patch_id,
        seed=int(rng.integers(0, 2**31 - 1)),
    )


def sample_map_surface(rng: np.random.Generator) -> dict:
    R = C.RANGES
    return {
        "earth": bool(rng.uniform() < C.EARTH_FRACTION),
        "rough_amp_m": float(rng.uniform(*R["rough_amp_m"])),
        "rough_wavelength_m": float(rng.uniform(*R["rough_wavelength_m"])),
        "rock_density_per_m2": float(rng.uniform(*R["rock_density_per_m2"])),
        "rock_height_m": float(rng.uniform(*R["rock_height_m"])),
        "friction": float(rng.uniform(*R["friction"])),
    }


def _plane_stats(s: np.ndarray, x: np.ndarray, z: np.ndarray) -> dict:
    """Best-fit plane, residual roughness and local slope of heights z[i_s, i_x]."""
    gz_s, gz_x = np.gradient(z, s[1] - s[0], x[1] - x[0])
    slope_mag = np.degrees(np.arctan(np.hypot(gz_s, gz_x)))
    S, X = np.meshgrid(s, x, indexing="ij")
    A = np.c_[S.ravel(), X.ravel(), np.ones(S.size)]
    coef, *_ = np.linalg.lstsq(A, z.ravel(), rcond=None)
    resid = z.ravel() - A @ coef
    return {"along": float(np.degrees(np.arctan(coef[0]))), "cross": float(np.degrees(np.arctan(coef[1]))),
            "p90": float(np.percentile(slope_mag, 90)), "max": float(slope_mag.max()),
            "rough": float(resid.std())}


def to_dtm_posts(s: np.ndarray, x: np.ndarray, z: np.ndarray, post_m: float = C.DTM_POST_M):
    """Average the fine grid onto ~post_m cells, like a DTM samples the real surface."""
    ks = max(int(round(post_m / (s[1] - s[0]))), 1)
    kx = max(int(round(post_m / (x[1] - x[0]))), 1)
    ns, nx = len(s) // ks, len(x) // kx
    if ns < 3 or nx < 3:
        raise ValueError("patch is too small for the DTM post spacing")
    zc = z[:ns * ks, :nx * kx].reshape(ns, ks, nx, kx).mean(axis=(1, 3))
    sc = s[:ns * ks].reshape(ns, ks).mean(axis=1)
    xc = x[:nx * kx].reshape(nx, kx).mean(axis=1)
    return sc, xc, zc


def patch_stats(hf: Heightfield) -> dict:
    """Terrain descriptors measured from the heights themselves.

    dtm_*  : measured at DTM post spacing (1 m). This is what the planner can
             compute from a HiRISE DTM, so the surrogate is trained on these.
    fine_* : measured on the full simulation grid (~0.15 m). Rocks and short
             roughness show up here but not in a 1 m DTM; use for analysis only.
    """
    m = hf.s >= hf.meta.get("flat_start_m", C.FLAT_START_M) + C.BLEND_M
    s, z = hf.s[m], hf.z[m]
    fine = _plane_stats(s, hf.x, z)
    dtm = _plane_stats(*to_dtm_posts(s, hf.x, z))
    return {
        "dtm_slope_along_deg": dtm["along"],
        "dtm_slope_cross_deg": dtm["cross"],
        "dtm_slope_p90_deg": dtm["p90"],
        "dtm_slope_max_deg": dtm["max"],
        "dtm_roughness_m": dtm["rough"],
        "dtm_post_m": float(C.DTM_POST_M),
        "fine_slope_p90_deg": fine["p90"],
        "fine_slope_max_deg": fine["max"],
        "fine_roughness_m": fine["rough"],
    }
