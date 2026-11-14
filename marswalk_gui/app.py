"""Qt (PySide6) interface for analysing Marswalk model development.

Run:  python -m marswalk_gui [--data DIR] [--model DIR ...] [--check FILE]
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QObject, Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QMainWindow, QMessageBox,
                               QPushButton, QSlider, QSplitter, QStatusBar, QTableView, QTabWidget, QTextEdit,
                               QVBoxLayout, QWidget)

from marswalk_sim import config as C
from marswalk_sim.terrain import PatchParams, build_heightfield, patch_stats

from . import core, theme


# --------------------------------------------------------------------------- shared widgets
class PandasModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame | None = None, max_rows: int = 5000):
        super().__init__()
        self.max_rows = max_rows
        self.set_frame(df if df is not None else pd.DataFrame())

    def set_frame(self, df: pd.DataFrame):
        self.beginResetModel()
        self.df = df.head(self.max_rows).reset_index(drop=True)
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.df)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else self.df.shape[1]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        v = self.df.iat[index.row(), index.column()]
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        if isinstance(v, (float, np.floating)):
            return f"{v:.4g}"
        return str(v)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        return str(self.df.columns[section]) if orientation == Qt.Orientation.Horizontal else str(section)


def make_table(df: pd.DataFrame | None = None) -> tuple[QTableView, PandasModel]:
    view, model = QTableView(), PandasModel(df)
    view.setModel(model)
    view.setAlternatingRowColors(True)
    view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    view.horizontalHeader().setStretchLastSection(True)
    return view, model


class Plot(FigureCanvasQTAgg):
    def __init__(self):
        super().__init__(Figure(figsize=(6, 4), facecolor=theme.SURFACE_BASE, tight_layout=True))

    def axes(self, nrows=1, ncols=1, **kw):
        self.figure.clear()
        ax = self.figure.subplots(nrows, ncols, **kw)
        for a in np.atleast_1d(ax).ravel():
            theme.style_axes(a)
        return ax

    def message(self, text: str):
        self.figure.clear()
        self.figure.text(0.5, 0.5, text, ha="center", va="center", color=theme.TEXT_MUTED, wrap=True)
        self.draw_idle()


def muted(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("muted")
    lab.setWordWrap(True)
    return lab


class SliderRow(QWidget):
    """Float slider with a value label."""
    changed = Signal()

    def __init__(self, label: str, lo: float, hi: float, value: float, decimals: int = 2):
        super().__init__()
        self.lo, self.hi, self.decimals, self.steps = lo, hi, decimals, 1000
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.name = QLabel(label)
        self.name.setMinimumWidth(150)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, self.steps)
        self.value_label = QLabel()
        self.value_label.setMinimumWidth(60)
        lay.addWidget(self.name)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.value_label)
        self.set_value(value)
        self.slider.valueChanged.connect(self._moved)

    def value(self) -> float:
        return self.lo + (self.hi - self.lo) * self.slider.value() / self.steps

    def set_value(self, v: float):
        self.slider.blockSignals(True)
        self.slider.setValue(int(round((v - self.lo) / (self.hi - self.lo) * self.steps)))
        self.slider.blockSignals(False)
        self.value_label.setText(f"{self.value():.{self.decimals}f}")

    def _moved(self):
        self.value_label.setText(f"{self.value():.{self.decimals}f}")
        self.changed.emit()


# --------------------------------------------------------------------------- shared state
class State(QObject):
    dataset_changed = Signal()
    runs_changed = Signal()
    check_changed = Signal()
    message = Signal(str)

    def __init__(self):
        super().__init__()
        self.df: pd.DataFrame | None = None
        self.data_path = ""
        self.synthetic = False
        self.runs: list[core.ModelRun] = []
        self.check: dict | None = None

    def set_dataset(self, df, path, synthetic=False):
        self.df, self.data_path, self.synthetic = df, path, synthetic
        self.dataset_changed.emit()

    def add_run(self, run: core.ModelRun, synthetic=False):
        self.runs = [r for r in self.runs if r.path != run.path] + [run]
        if synthetic:
            run.name = f"{run.name} (synthetic)"
        self.runs_changed.emit()

    def remove_run(self, idx: int):
        if 0 <= idx < len(self.runs):
            del self.runs[idx]
            self.runs_changed.emit()


def run_combo(state: State) -> QComboBox:
    """A combo box kept in sync with state.runs."""
    box = QComboBox()

    def refill():
        cur = box.currentText()
        box.clear()
        box.addItems([r.name for r in state.runs])
        i = box.findText(cur)
        box.setCurrentIndex(i if i >= 0 else max(box.count() - 1, 0))

    state.runs_changed.connect(refill)
    refill()
    return box


def guarded(fn):
    """Show an error box instead of crashing the app from a slot."""
    def wrapper(self, *a, **k):
        try:
            return fn(self, *a, **k)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            QMessageBox.warning(self, "Marswalk analysis", f"{type(e).__name__}: {e}")
    return wrapper


# --------------------------------------------------------------------------- tab 1: dataset
class DatasetTab(QWidget):
    def __init__(self, state: State):
        super().__init__()
        self.state = state
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.path = QLineEdit()
        self.path.setPlaceholderText("Folder with part-*.parquet (local path or gs://bucket/runs/<RUN_ID>)")
        browse, load, demo = QPushButton("Browse…"), QPushButton("Load"), QPushButton("Load synthetic demo")
        load.setObjectName("primary")
        browse.clicked.connect(self.browse)
        load.clicked.connect(self.load)
        demo.clicked.connect(self.load_demo)
        for w in (self.path, browse, load, demo):
            top.addWidget(w, 1 if w is self.path else 0)
        lay.addLayout(top)
        split = QSplitter(Qt.Orientation.Horizontal)
        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        split.addWidget(self.summary)
        tabs = QTabWidget()
        self.rows_view, self.rows_model = make_table()
        self.cols_view, self.cols_model = make_table()
        tabs.addTab(self.rows_view, "Rows")
        tabs.addTab(self.cols_view, "Columns")
        split.addWidget(tabs)
        split.setSizes([320, 700])
        lay.addWidget(split, 1)
        state.dataset_changed.connect(self.refresh)
        self.refresh()

    @guarded
    def browse(self, *_):
        d = QFileDialog.getExistingDirectory(self, "Dataset folder")
        if d:
            self.path.setText(d)
            self.load()

    @guarded
    def load(self, *_):
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            df = core.load_dataset(self.path.text().strip())
        finally:
            QApplication.restoreOverrideCursor()
        synthetic = bool(df.get("walker", pd.Series(dtype=str)).astype(str).eq("synthetic-demo").any())
        self.state.set_dataset(df, self.path.text().strip(), synthetic)

    @guarded
    def load_demo(self, *_):
        from .demo import make_demo
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            out = Path(tempfile.mkdtemp(prefix="marswalk-demo-"))
            data, model = make_demo(out)
            df = core.load_dataset(str(data))
            run = core.load_run(model)
        finally:
            QApplication.restoreOverrideCursor()
        self.path.setText(str(data))
        self.state.add_run(run, synthetic=True)
        self.state.set_dataset(df, str(data), synthetic=True)

    def refresh(self):
        df = self.state.df
        if df is None:
            self.summary.setHtml("<p>No dataset loaded.</p><p>Load a folder written by <code>scripts/generate.py</code>, "
                                 "or try the synthetic demo.</p>")
            self.rows_model.set_frame(pd.DataFrame())
            self.cols_model.set_frame(pd.DataFrame())
            return
        s = core.dataset_summary(df)
        banner = ("<p style='color:#F2A47E'><b>SYNTHETIC DEMO DATA.</b> Not simulation output, not Mars measurements.</p>"
                  if self.state.synthetic else "")
        status = "".join(f"<li>{k}: {v}</li>" for k, v in s["status"].items())
        fr = f"{s['fall_rate']:.1%}" if s["fall_rate"] is not None else "n/a"
        html = (f"{banner}<h3>{Path(self.state.data_path).name or self.state.data_path}</h3>"
                f"<p>{s['rows']} patches in {s['maps']} maps.</p><p>Status:</p><ul>{status}</ul>"
                f"<p>Usable (status ok): {s['ok_rows']}<br>Mars gravity: {s['mars_rows']}, Earth check: {s['earth_rows']}"
                f"<br>Fall rate: {fr}<br>Walked without falling: {s['walked_rows']}</p>")
        if "cot_median" in s:
            html += f"<p>Median net cost of transport (walked): {s['cot_median']:.2f} J/kg/m</p>"
        if "walkers" in s:
            html += f"<p>Walker: {', '.join(s['walkers'])}</p>"
        if s["ok_rows"] and s["fall_rate"] is not None and s["fall_rate"] > 0.5:
            html += "<p style='color:#FFD60A'><b>Caution:</b> more than half of the patches ended in a fall. See README, “If the walker does not walk”.</p>"
        self.summary.setHtml(html)
        self.rows_model.set_frame(df)
        cols = pd.DataFrame({"column": df.columns, "dtype": [str(t) for t in df.dtypes],
                             "filled %": [round(100 * df[c].notna().mean(), 1) for c in df.columns]})
        num = df.select_dtypes("number")
        cols["min"] = cols["column"].map(num.min())
        cols["max"] = cols["column"].map(num.max())
        self.cols_model.set_frame(cols)


# --------------------------------------------------------------------------- tab 2: explorer
class ExplorerTab(QWidget):
    KINDS = ["Scatter", "Binned mean (Y vs X)", "Histogram of X"]

    def __init__(self, state: State):
        super().__init__()
        self.state = state
        lay = QHBoxLayout(self)
        ctl = QVBoxLayout()
        form = QFormLayout()
        self.kind, self.x, self.y, self.color = QComboBox(), QComboBox(), QComboBox(), QComboBox()
        self.kind.addItems(self.KINDS)
        form.addRow("Plot", self.kind)
        form.addRow("X", self.x)
        form.addRow("Y", self.y)
        form.addRow("Colour", self.color)
        ctl.addLayout(form)
        box = QGroupBox("Filter")
        fl = QVBoxLayout(box)
        self.only_ok = QCheckBox("Only status = ok")
        self.only_ok.setChecked(True)
        self.hide_falls = QCheckBox("Hide runs that fell")
        self.planet = QComboBox()
        self.planet.addItems(["all", "mars", "earth"])
        for w in (self.only_ok, self.hide_falls, self.planet):
            fl.addWidget(w)
        ctl.addWidget(box)
        self.info = muted("")
        ctl.addWidget(self.info)
        ctl.addStretch(1)
        lay.addLayout(ctl)
        self.plot = Plot()
        lay.addWidget(self.plot, 1)
        for w in (self.kind, self.x, self.y, self.color, self.planet):
            w.currentIndexChanged.connect(self.draw)
        for w in (self.only_ok, self.hide_falls):
            w.toggled.connect(self.draw)
        state.dataset_changed.connect(self.reload)
        self.reload()

    def reload(self):
        df = self.state.df
        for b in (self.x, self.y, self.color):
            b.blockSignals(True)
            b.clear()
        if df is not None:
            cols = core.numeric_columns(df)
            self.x.addItems(cols)
            self.y.addItems(cols)
            self.color.addItems(["(none)"] + cols)
            for box, want in ((self.x, "dtm_slope_along_deg"), (self.y, "cot_net_J_per_kg_m"), (self.color, "fell")):
                i = box.findText(want)
                if i >= 0:
                    box.setCurrentIndex(i)
        for b in (self.x, self.y, self.color):
            b.blockSignals(False)
        self.draw()

    def draw(self, *_):
        df = self.state.df
        if df is None or not self.x.count():
            self.plot.message("Load a dataset first (Dataset tab).")
            return
        d = core.apply_filters(df, self.only_ok.isChecked(), self.hide_falls.isChecked(), self.planet.currentText())
        xc, yc, cc = self.x.currentText(), self.y.currentText(), self.color.currentText()
        kind = self.kind.currentText()
        cols = [xc] + ([yc] if kind != self.KINDS[2] else [])
        d = d.dropna(subset=cols)
        self.info.setText(f"{len(d)} of {len(df)} rows shown")
        if d.empty:
            self.plot.message("No rows match the filter.")
            return
        ax = self.plot.axes()
        ax.set_xlabel(xc)
        if kind == self.KINDS[2]:
            ax.hist(d[xc].astype(float), bins=30, color=theme.BRAND_SOFT, edgecolor=theme.SURFACE_INSET)
            ax.set_ylabel("patches")
        elif kind == self.KINDS[1]:
            x, y = d[xc].astype(float), d[yc].astype(float)
            bins = np.linspace(x.min(), x.max() + 1e-9, 13)
            g = y.groupby(pd.cut(x, bins), observed=True)
            mid = [iv.mid for iv in g.mean().index]
            ax.errorbar(mid, g.mean().values, yerr=g.std().fillna(0).values, color=theme.BRAND_SOFT,
                        ecolor=theme.BORDER_DEFAULT, marker="o", capsize=3)
            ax.set_ylabel(f"{yc} (mean ± sd per bin)")
        else:
            ax.set_ylabel(yc)
            if cc == "fell":
                f = d["fell"].astype(bool)
                ax.scatter(d.loc[~f, xc], d.loc[~f, yc], s=14, c=theme.ICE, alpha=0.7, label="walked")
                ax.scatter(d.loc[f, xc], d.loc[f, yc], s=14, c=theme.STATUS_DANGER, alpha=0.8, label="fell")
                ax.legend(facecolor=theme.SURFACE_PANEL, labelcolor=theme.TEXT_PRIMARY, edgecolor=theme.BORDER_SUBTLE)
            elif cc and cc != "(none)":
                sc = ax.scatter(d[xc], d[yc], s=14, c=d[cc].astype(float), cmap="magma", alpha=0.85)
                cb = self.plot.figure.colorbar(sc, ax=ax)
                cb.set_label(cc, color=theme.TEXT_SECONDARY)
                cb.ax.tick_params(colors=theme.TEXT_SECONDARY)
            else:
                ax.scatter(d[xc], d[yc], s=14, c=theme.BRAND_SOFT, alpha=0.8)
        self.plot.draw_idle()


# --------------------------------------------------------------------------- tab 3: terrain
class TerrainTab(QWidget):
    def __init__(self):
        super().__init__()
        lay = QHBoxLayout(self)
        ctl = QVBoxLayout()
        R = C.RANGES
        self.rows = {
            "slope_deg": SliderRow("Slope (deg)", *R["slope_deg"], 0.0, 1),
            "cross_slope_deg": SliderRow("Side slope (deg)", *R["cross_slope_deg"], 0.0, 1),
            "rough_amp_m": SliderRow("Roughness amp (m)", *R["rough_amp_m"], 0.02, 3),
            "rough_wavelength_m": SliderRow("Wavelength (m)", *R["rough_wavelength_m"], 1.0, 2),
            "rock_density_per_m2": SliderRow("Rocks per m²", *R["rock_density_per_m2"], 0.1, 2),
            "rock_height_m": SliderRow("Rock height (m)", *R["rock_height_m"], 0.1, 2),
            "seed": SliderRow("Seed", 0, 99, 1, 0),
        }
        for r in self.rows.values():
            r.changed.connect(self.draw)
            ctl.addWidget(r)
        self.stats = QLabel()
        self.stats.setTextFormat(Qt.TextFormat.RichText)
        ctl.addWidget(self.stats)
        ctl.addWidget(muted("Patch built by marswalk_sim.terrain. The measured stats are the surrogate model's "
                            "inputs (what a HiRISE DTM would give the planner)."))
        ctl.addStretch(1)
        left = QWidget()
        left.setLayout(ctl)
        left.setMaximumWidth(420)
        lay.addWidget(left)
        self.plot = Plot()
        lay.addWidget(self.plot, 1)
        self.draw()

    def draw(self, *_):
        v = {k: r.value() for k, r in self.rows.items()}
        p = PatchParams(**{k: v[k] for k in v if k != "seed"}, seed=int(round(v["seed"])))
        hf = build_heightfield(p)
        st = patch_stats(hf)
        fig = self.plot.figure
        fig.clear()
        a1 = theme.style_axes(fig.add_subplot(1, 2, 1))
        a2 = theme.style_axes(fig.add_subplot(1, 2, 2))
        im = a1.imshow(hf.z, origin="lower", cmap="magma", aspect="auto",
                       extent=[hf.x[0], hf.x[-1], hf.s[0], hf.s[-1]])
        a1.set_xlabel("lateral x (m)")
        a1.set_ylabel("forward s (m)")
        a1.set_title("Height (m)")
        a1.grid(False)
        cb = fig.colorbar(im, ax=a1)
        cb.ax.tick_params(colors=theme.TEXT_SECONDARY)
        mid = hf.z.shape[1] // 2
        a2.plot(hf.s, hf.z[:, mid], color=theme.BRAND_SOFT)
        a2.axvline(hf.meta["flat_start_m"], color=theme.BORDER_DEFAULT, linestyle="--", linewidth=1)
        a2.set_xlabel("forward s (m)")
        a2.set_ylabel("height at x = 0 (m)")
        a2.set_title("Centre line (dashed: start)")
        self.plot.draw_idle()
        self.stats.setText(
            "<b>Measured from heights</b><br>"
            f"slope along: {st['dtm_slope_along_deg']:+.1f}°<br>slope across: {st['dtm_slope_cross_deg']:+.1f}°<br>"
            f"slope p90: {st['dtm_slope_p90_deg']:.1f}°<br>slope max: {st['dtm_slope_max_deg']:.1f}°<br>"
            f"roughness (residual sd): {st['dtm_roughness_m'] * 100:.2f} cm<br>rocks placed: {hf.meta['n_rocks']}")


# --------------------------------------------------------------------------- tab 4: model runs
class ModelTab(QWidget):
    def __init__(self, state: State):
        super().__init__()
        self.state = state
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        add, rem = QPushButton("Add model folder…"), QPushButton("Remove selected")
        add.setObjectName("primary")
        add.clicked.connect(self.add)
        rem.clicked.connect(self.remove)
        top.addWidget(add)
        top.addWidget(rem)
        top.addWidget(muted("Folders written by scripts/train.py. Add several to compare runs."), 1)
        lay.addLayout(top)
        self.runs = QListWidget()
        self.runs.setMaximumHeight(80)
        lay.addWidget(self.runs)
        self.table, self.table_model = make_table()
        self.table.setMaximumHeight(210)
        lay.addWidget(self.table)
        row = QHBoxLayout()
        self.run, self.target, self.which = run_combo(state), QComboBox(), QComboBox()
        self.target.addItems(list(core.TARGET_LABELS))
        self.which.addItems(["test", "val", "train"])
        for lab, w in (("Run", self.run), ("Target", self.target), ("Maps", self.which)):
            row.addWidget(QLabel(lab))
            row.addWidget(w)
        row.addStretch(1)
        lay.addLayout(row)
        self.note = muted("")
        lay.addWidget(self.note)
        self.plot = Plot()
        lay.addWidget(self.plot, 1)
        for w in (self.run, self.target, self.which):
            w.currentIndexChanged.connect(self.draw)
        state.runs_changed.connect(self.refresh)
        state.dataset_changed.connect(self.draw)
        self.refresh()

    @guarded
    def add(self, *_):
        d = QFileDialog.getExistingDirectory(self, "Model folder (has metrics.json)")
        if d:
            self.state.add_run(core.load_run(d))

    def remove(self, *_):
        self.state.remove_run(self.runs.currentRow())

    def refresh(self):
        self.runs.clear()
        for r in self.state.runs:
            self.runs.addItem(f"{r.name}   —   {r.path}")
        self.table_model.set_frame(core.compare_table(self.state.runs))
        self.table.resizeColumnsToContents()
        self.draw()

    def draw(self, *_):
        i = self.run.currentIndex()
        if not (0 <= i < len(self.state.runs)):
            self.plot.message("Add a model folder to see predictions against the simulated labels.")
            self.note.setText("")
            return
        run, target = self.state.runs[i], self.target.currentText()
        self.note.setText("; ".join(run.warnings))
        if self.state.df is None:
            self.plot.message("Load the dataset the model was trained on (Dataset tab) to compare predictions.")
            return
        if target not in run.models:
            self.plot.message(f"No loaded model for “{target}” in this run (skipped in training or failed to load).")
            return
        res = core.test_predictions(run, self.state.df, target, self.which.currentText())
        if res.empty:
            self.plot.message(f"No {self.which.currentText()} rows for this target. Is this the dataset the run was trained on?")
            return
        ax = self.plot.axes()
        if target == "fall":
            f = res["actual"].astype(bool)
            bins = np.linspace(0, 1, 21)
            ax.hist(res.loc[~f, "predicted"], bins=bins, color=theme.ICE, alpha=0.8, label="walked")
            ax.hist(res.loc[f, "predicted"], bins=bins, color=theme.STATUS_DANGER, alpha=0.8, label="fell")
            ax.set_xlabel("predicted fall probability")
            ax.set_ylabel("patches")
            ax.legend(facecolor=theme.SURFACE_PANEL, labelcolor=theme.TEXT_PRIMARY, edgecolor=theme.BORDER_SUBTLE)
        else:
            ax.scatter(res["actual"], res["predicted"], s=14, c=theme.BRAND_SOFT, alpha=0.8)
            lo, hi = float(min(res.min(numeric_only=True))), float(max(res.max(numeric_only=True)))
            ax.plot([lo, hi], [lo, hi], color=theme.BORDER_DEFAULT, linestyle="--", linewidth=1)
            ax.set_xlabel("simulated")
            ax.set_ylabel("predicted")
        ax.set_title(f"{core.TARGET_LABELS[target]}: {self.which.currentText()} maps, n = {len(res)}")
        self.plot.draw_idle()


# --------------------------------------------------------------------------- tab 5: what-if
class WhatIfTab(QWidget):
    def __init__(self, state: State):
        super().__init__()
        self.state = state
        lay = QHBoxLayout(self)
        ctl = QVBoxLayout()
        self.run = run_combo(state)
        ctl.addWidget(self.run)
        self.rows = {}
        for f in core.FEATURES:
            lo, hi, d = core.FEATURE_RANGES[f]
            self.rows[f] = SliderRow(f.replace("dtm_", "").replace("_", " "), lo, hi, d, 3 if hi < 1 else 1)
            self.rows[f].changed.connect(self.draw)
            ctl.addWidget(self.rows[f])
        self.tie = QCheckBox("Slope p90 follows |slope| + 3° (as in the planner lookup)")
        self.tie.setChecked(True)
        self.tie.toggled.connect(self.draw)
        ctl.addWidget(self.tie)
        self.out = QLabel()
        self.out.setTextFormat(Qt.TextFormat.RichText)
        ctl.addWidget(self.out)
        ctl.addStretch(1)
        left = QWidget()
        left.setLayout(ctl)
        left.setMaximumWidth(460)
        lay.addWidget(left)
        self.plot = Plot()
        lay.addWidget(self.plot, 1)
        self.run.currentIndexChanged.connect(self.draw)
        state.runs_changed.connect(self.draw)
        self.draw()

    def values(self) -> dict:
        v = {f: r.value() for f, r in self.rows.items()}
        if self.tie.isChecked():
            v["dtm_slope_p90_deg"] = abs(v["dtm_slope_along_deg"]) + 3
        return v

    def draw(self, *_):
        self.rows["dtm_slope_p90_deg"].setEnabled(not self.tie.isChecked())
        i = self.run.currentIndex()
        if not (0 <= i < len(self.state.runs)) or not self.state.runs[i].models:
            self.out.setText("")
            self.plot.message("Add a model folder (Model runs tab) to get predictions.")
            return
        run, v = self.state.runs[i], self.values()
        if self.tie.isChecked():
            self.rows["dtm_slope_p90_deg"].set_value(min(v["dtm_slope_p90_deg"], 35.0))
        pred = core.predict_features(run, v)
        lines = []
        if "cot" in pred:
            lines.append(f"Net cost of transport: <b>{pred['cot']:.2f}</b> J/kg/m")
        if "fall" in pred:
            lines.append(f"Fall probability: <b>{pred['fall']:.0%}</b>")
        if "mos" in pred:
            lines.append(f"Min lateral margin of stability: <b>{pred['mos']:.3f}</b> m")
        self.out.setText("<br>".join(lines))
        slopes = np.arange(-25, 25.5, 1.0)
        sweeps = {}
        for s in slopes:
            row = dict(v, dtm_slope_along_deg=float(s))
            if self.tie.isChecked():
                row["dtm_slope_p90_deg"] = abs(s) + 3
            for k, val in core.predict_features(run, row).items():
                sweeps.setdefault(k, []).append(val)
        ax = self.plot.axes(1, 2)
        if "cot" in sweeps:
            ax[0].plot(slopes, sweeps["cot"], color=theme.BRAND_SOFT, label="model (this setting)")
            ax[0].plot(slopes, core.minetti_walk(np.tan(np.radians(slopes))), color=theme.BORDER_DEFAULT,
                       linestyle="--", label="Minetti 2002 (Earth, gross)")
            ax[0].plot([v["dtm_slope_along_deg"]], [pred["cot"]], "o", color=theme.TEXT_PRIMARY)
            ax[0].legend(facecolor=theme.SURFACE_PANEL, labelcolor=theme.TEXT_PRIMARY, edgecolor=theme.BORDER_SUBTLE,
                         fontsize=8)
        ax[0].set_xlabel("slope along (deg)")
        ax[0].set_ylabel("cost of transport (J/kg/m)")
        if "fall" in sweeps:
            ax[1].plot(slopes, sweeps["fall"], color=theme.STATUS_DANGER)
            ax[1].plot([v["dtm_slope_along_deg"]], [pred["fall"]], "o", color=theme.TEXT_PRIMARY)
        ax[1].set_ylim(0, 1)
        ax[1].set_xlabel("slope along (deg)")
        ax[1].set_ylabel("fall probability")
        self.plot.draw_idle()


# --------------------------------------------------------------------------- tab 6: planner lookup
class LookupTab(QWidget):
    def __init__(self, state: State):
        super().__init__()
        self.state = state
        lay = QVBoxLayout(self)
        row = QHBoxLayout()
        self.run, self.target = run_combo(state), QComboBox()
        self.target.addItems(list(core.TARGET_LABELS))
        for lab, w in (("Run", self.run), ("Target", self.target)):
            row.addWidget(QLabel(lab))
            row.addWidget(w)
        row.addWidget(muted("Grid exported to planner_lookup_mars.json (Mars gravity, 40 kg suit)."), 1)
        lay.addLayout(row)
        self.plot = Plot()
        lay.addWidget(self.plot, 1)
        self.run.currentIndexChanged.connect(self.draw)
        self.target.currentIndexChanged.connect(self.draw)
        state.runs_changed.connect(self.draw)
        self.draw()

    def draw(self, *_):
        i, t = self.run.currentIndex(), self.target.currentText()
        if not (0 <= i < len(self.state.runs)):
            self.plot.message("Add a model folder (Model runs tab).")
            return
        lk = self.state.runs[i].lookup
        if not lk or t not in lk:
            self.plot.message(f"This run has no planner lookup for “{t}”.")
            return
        slopes, rough, z = np.array(lk["slope_deg"]), np.array(lk["roughness_m"]), np.array(lk[t])
        a1, a2 = self.plot.axes(1, 2)
        im = a1.imshow(z, origin="lower", aspect="auto", cmap="magma",
                       extent=[slopes[0], slopes[-1], 0, len(rough) - 1])
        a1.set_yticks(range(len(rough)))
        a1.set_yticklabels([f"{r * 100:g} cm" for r in rough])
        a1.set_xlabel("slope along (deg)")
        a1.set_ylabel("roughness")
        a1.set_title(core.TARGET_LABELS[t])
        a1.grid(False)
        cb = self.plot.figure.colorbar(im, ax=a1)
        cb.ax.tick_params(colors=theme.TEXT_SECONDARY)
        cmap = [theme.BRAND_SOFT, theme.BRAND, theme.ICE, theme.TEXT_SECONDARY, theme.BRAND_DEEP]
        for k, r in enumerate(rough):
            a2.plot(slopes, z[k], color=cmap[k % len(cmap)], label=f"{r * 100:g} cm")
        if t == "cot":
            a2.plot(slopes, core.minetti_walk(np.tan(np.radians(slopes))), color=theme.BORDER_DEFAULT, linestyle="--",
                    label="Minetti (Earth)")
        a2.set_xlabel("slope along (deg)")
        a2.set_ylabel(core.TARGET_LABELS[t])
        a2.legend(facecolor=theme.SURFACE_PANEL, labelcolor=theme.TEXT_PRIMARY, edgecolor=theme.BORDER_SUBTLE, fontsize=8,
                  title="roughness", title_fontsize=8)
        self.plot.draw_idle()


# --------------------------------------------------------------------------- tab 7: walker check
class CheckTab(QWidget):
    def __init__(self, state: State):
        super().__init__()
        self.state = state
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.path = QLineEdit()
        self.path.setPlaceholderText("check_walker.json from scripts/check_walker.py")
        browse, load = QPushButton("Browse…"), QPushButton("Load")
        load.setObjectName("primary")
        browse.clicked.connect(self.browse)
        load.clicked.connect(self.load)
        for w in (self.path, browse, load):
            top.addWidget(w, 1 if w is self.path else 0)
        lay.addLayout(top)
        self.verdict = QLabel("")
        self.verdict.setWordWrap(True)
        lay.addWidget(self.verdict)
        self.table, self.table_model = make_table()
        self.table.setMaximumHeight(190)
        lay.addWidget(self.table)
        self.plot = Plot()
        lay.addWidget(self.plot, 1)
        state.check_changed.connect(self.refresh)
        self.refresh()

    @guarded
    def browse(self, *_):
        f, _ = QFileDialog.getOpenFileName(self, "check_walker.json", filter="JSON (*.json)")
        if f:
            self.path.setText(f)
            self.load()

    @guarded
    def load(self, *_):
        self.state.check = core.load_check(self.path.text().strip())
        self.state.check_changed.emit()

    def refresh(self):
        c = self.state.check
        if not c:
            self.verdict.setText("")
            self.table_model.set_frame(pd.DataFrame())
            self.plot.message("Load check_walker.json to see whether the pretrained walker walks.")
            return
        self.verdict.setText(f"<b>{c.get('walker')}</b> ({c.get('policy')}): {c.get('verdict')}")
        self.table_model.set_frame(core.check_stage_table(c))
        self.table.resizeColumnsToContents()
        curves = core.check_cot_curves(c)
        if not curves:
            self.plot.message("No slope sweeps in this file (the walker stopped at an earlier stage).")
            return
        ax = self.plot.axes()
        colors = {"4_sweep_earth": theme.ICE, "5_sweep_mars_suit": theme.BRAND_SOFT}
        for tag, (g, y) in curves.items():
            if tag == "minetti":
                ax.plot(g, y, color=theme.BORDER_DEFAULT, linestyle="--", label="Minetti 2002 (Earth)")
            else:
                ax.plot(g, y, "o-", color=colors.get(tag, theme.TEXT_SECONDARY), label=tag)
        ax.set_xlabel("grade (rise/run)")
        ax.set_ylabel("gross cost of transport (J/kg/m)")
        ax.legend(facecolor=theme.SURFACE_PANEL, labelcolor=theme.TEXT_PRIMARY, edgecolor=theme.BORDER_SUBTLE)
        self.plot.draw_idle()


# --------------------------------------------------------------------------- window
class MainWindow(QMainWindow):
    def __init__(self, state: State | None = None):
        super().__init__()
        self.state = state or State()
        self.setWindowTitle("Marswalk model analysis")
        self.resize(1280, 840)
        tabs = QTabWidget()
        self.dataset_tab = DatasetTab(self.state)
        self.check_tab = CheckTab(self.state)
        for w, name in ((self.dataset_tab, "Dataset"), (ExplorerTab(self.state), "Explorer"), (TerrainTab(), "Terrain patch"),
                        (ModelTab(self.state), "Model runs"), (WhatIfTab(self.state), "What-if"),
                        (LookupTab(self.state), "Planner lookup"), (self.check_tab, "Walker check")):
            tabs.addTab(w, name)
        self.tabs = tabs
        central = QWidget()
        lay = QVBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        self.banner = QLabel("SYNTHETIC DEMO DATA — not simulation output, not Mars measurements")
        self.banner.setObjectName("banner")
        self.banner.hide()
        lay.addWidget(self.banner)
        lay.addWidget(tabs, 1)
        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())
        self.state.dataset_changed.connect(self._status)
        self.state.runs_changed.connect(self._status)

    def _status(self):
        s = self.state
        self.banner.setVisible(s.synthetic or any("(synthetic)" in r.name for r in s.runs))
        n = len(s.df) if s.df is not None else 0
        self.statusBar().showMessage(f"{n} patches loaded · {len(s.runs)} model run(s)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m marswalk_gui", description=__doc__)
    ap.add_argument("--data", help="dataset folder (part-*.parquet)")
    ap.add_argument("--model", action="append", default=[], help="model folder from scripts/train.py (repeatable)")
    ap.add_argument("--check", help="check_walker.json")
    args = ap.parse_args(argv)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setStyleSheet(theme.QSS)
    win = MainWindow()
    try:
        if args.data:
            df = core.load_dataset(args.data)
            win.state.set_dataset(df, args.data, bool(df.get("walker", pd.Series(dtype=str)).astype(str).eq("synthetic-demo").any()))
        for m in args.model:
            win.state.add_run(core.load_run(m))
        if args.check:
            win.state.check = core.load_check(args.check)
            win.state.check_changed.emit()
    except Exception as e:  # noqa: BLE001
        QMessageBox.warning(win, "Marswalk analysis", f"{type(e).__name__}: {e}")
    win.show()
    return app.exec()
