"""Play a walk recorded by scripts/record_walk.py in a PyVista window, or save it as a video.

    python scripts/view_walk.py data/kinesis-mars-ft-001/replay                 # first recording, window
    python scripts/view_walk.py data/kinesis-mars-ft-001/replay --index 2
    python scripts/view_walk.py data/kinesis-mars-ft-001/replay/<patch_id>.npz --save walk.mp4

Window: the slider scrubs frames, space plays/pauses, the camera follows the pelvis.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv

# mjtGeom
SPHERE, CAPSULE, ELLIPSOID, CYLINDER, BOX, MESH = 2, 3, 4, 5, 6, 7


def geom_mesh(rec, k: int) -> pv.PolyData | None:
    t, s = int(rec["geom_type"][k]), rec["geom_size"][k]
    if t == SPHERE:
        return pv.Sphere(radius=s[0])
    if t == CAPSULE:
        return pv.Capsule(center=(0, 0, 0), direction=(0, 0, 1), radius=s[0], cylinder_length=2 * s[1])
    if t == ELLIPSOID:
        return pv.ParametricEllipsoid(s[0], s[1], s[2])
    if t == CYLINDER:
        return pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1), radius=s[0], height=2 * s[1])
    if t == BOX:
        return pv.Box(bounds=(-s[0], s[0], -s[1], s[1], -s[2], s[2]))
    if t == MESH:
        mid = int(rec["geom_mesh"][k])
        v, f = rec[f"mesh{mid}_v"], rec[f"mesh{mid}_f"]
        return pv.PolyData(v, np.hstack([np.full((len(f), 1), 3), f]).ravel())
    return None


def pose(xpos, xmat) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = xmat.reshape(3, 3)
    T[:3, 3] = xpos
    return T


def terrain(rec) -> pv.StructuredGrid:
    h = rec["hfield"]
    rx, ry, rz = rec["hfield_size"][:3]
    x, y = np.meshgrid(np.linspace(-rx, rx, h.shape[1]), np.linspace(-ry, ry, h.shape[0]))
    g = pv.StructuredGrid(x, y, h * rz)
    g["height"] = (h * rz).ravel(order="F")
    return g.transform(pose(rec["terrain_pos"], rec["terrain_mat"]), inplace=False)


def build(pl: pv.Plotter, rec):
    pl.set_background("#1b1410", top="#3b2418")
    pl.add_mesh(terrain(rec), scalars="height", cmap="copper", show_scalar_bar=False, smooth_shading=True)
    actors = []
    for k in range(len(rec["geom_type"])):
        m = geom_mesh(rec, k)
        if m is None:
            continue
        a = pl.add_mesh(m, color=rec["geom_rgba"][k][:3], smooth_shading=True)
        actors.append((k, a))
    return actors


def title(rec) -> str:
    orig = "fell" if rec["orig_fell"] else "walked"
    rep = "FELL" if rec["replay_fell"] else "WALKED"
    body = "Mars" if float(rec["gravity"]) < 5 else "Earth"
    return (f"{rec['patch_id']}  |  {body} g={float(rec['gravity']):.2f}  suit {float(rec['suit_mass_kg']):.0f} kg  "
            f"slope {float(rec['slope_deg']):+.1f} deg\nreplay: {rep} ({float(rec['replay_distance_m']):.2f} m on patch)"
            f"   generated run: {orig}")


def show_frame(pl, rec, actors, i: int, follow: bool = True):
    i = int(np.clip(i, 0, len(rec["t"]) - 1))
    for k, a in actors:
        a.user_matrix = pose(rec["xpos"][i, k], rec["xmat"][i, k])
    if follow:
        c = rec["xpos"][i].mean(0)
        pl.camera.focal_point = c
        pl.camera.position = c + np.array([2.5, -3.5, 1.4])
    pl.add_text(f"t = {rec['t'][i]:.2f} s   frame {i + 1}/{len(rec['t'])}", position="lower_left",
                font_size=10, name="clock", color="white")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="a .npz from record_walk.py, or the folder holding them")
    ap.add_argument("--index", type=int, default=0, help="which recording in the folder")
    ap.add_argument("--save", help="write a video (.mp4 or .gif) instead of opening a window")
    args = ap.parse_args()

    p = Path(args.path)
    files = sorted(p.glob("*.npz")) if p.is_dir() else [p]
    if not files:
        raise SystemExit(f"no .npz in {p}")
    if p.is_dir():
        for j, f in enumerate(files):
            print(f"[{j}] {f.name}")
    rec = dict(np.load(files[args.index]))
    n = len(rec["t"])
    fps = max(1, round(1 / float(np.median(np.diff(rec["t"])))))

    pl = pv.Plotter(off_screen=bool(args.save), window_size=(1280, 720))
    actors = build(pl, rec)
    pl.add_text(title(rec), position="upper_left", font_size=10, color="white")

    if args.save:
        (pl.open_gif if args.save.endswith(".gif") else lambda f: pl.open_movie(f, framerate=fps))(args.save)
        for i in range(n):
            show_frame(pl, rec, actors, i)
            pl.write_frame()
        pl.close()
        print(f"wrote {args.save} ({n} frames, {fps} fps)")
        return

    state = {"i": 0, "play": True}
    slider = pl.add_slider_widget(lambda v: (state.update(i=int(v)), show_frame(pl, rec, actors, int(v))),
                                  [0, n - 1], value=0, title="frame", pointa=(0.25, 0.08), pointb=(0.75, 0.08),
                                  style="modern", fmt="%.0f")

    def tick(_step=None):
        if state["play"]:
            state["i"] = (state["i"] + 1) % n
            slider.GetRepresentation().SetValue(state["i"])
            show_frame(pl, rec, actors, state["i"])
            pl.render()

    pl.add_key_event("space", lambda: state.update(play=not state["play"]))
    show_frame(pl, rec, actors, 0)
    pl.add_timer_event(max_steps=10 ** 9, duration=int(1000 / fps), callback=tick)
    pl.show(title="Marswalk replay")


if __name__ == "__main__":
    main()
