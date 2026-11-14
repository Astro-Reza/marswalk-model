"""Colours from claude/DESIGN.md (Marswalk design system), kept in one place."""
SURFACE_BASE = "#060606"
SURFACE_RAISED = "#1E1E1E"
SURFACE_PANEL = "#222222"
SURFACE_INSET = "#272727"
BORDER_DEFAULT = "#848484"
BORDER_SUBTLE = "#2A2A2A"      # text-primary at ~8% on surface-base
TEXT_PRIMARY = "#FFFFFF"
TEXT_SECONDARY = "#BDBDBD"
TEXT_MUTED = "#848484"
BRAND = "#D9663A"
BRAND_SOFT = "#F2A47E"
BRAND_DEEP = "#8C3B20"
STATUS_DANGER = "#FF2A1F"      # used here only for "the walker fell"
ICE = "#8FD3FF"                # data-link-high, used as the neutral second series colour

import matplotlib

matplotlib.rcParams.update({"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10, "legend.fontsize": 8})

QSS = f"""
QWidget {{ background: {SURFACE_BASE}; color: {TEXT_PRIMARY}; font-size: 13px; }}
QTabWidget::pane {{ border: 1px solid {BORDER_SUBTLE}; }}
QTabBar::tab {{ background: {SURFACE_BASE}; color: {TEXT_SECONDARY}; padding: 8px 16px; border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: {TEXT_PRIMARY}; border-bottom: 2px solid {BRAND}; }}
QPushButton {{ background: {SURFACE_RAISED}; border: 1px solid {BORDER_DEFAULT}; padding: 6px 14px; border-radius: 4px; }}
QPushButton:hover {{ background: {SURFACE_INSET}; }}
QPushButton#primary {{ background: {BRAND}; border: 1px solid {BRAND}; color: {TEXT_PRIMARY}; }}
QPushButton#primary:hover {{ background: {BRAND_DEEP}; }}
QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox, QTextEdit, QListWidget {{ background: {SURFACE_INSET}; border: 1px solid {BORDER_SUBTLE}; padding: 4px; }}
QTableView {{ background: {SURFACE_RAISED}; alternate-background-color: {SURFACE_PANEL}; gridline-color: {BORDER_SUBTLE}; }}
QHeaderView::section {{ background: {SURFACE_PANEL}; color: {TEXT_SECONDARY}; border: 0; padding: 4px; }}
QGroupBox {{ border: 1px solid {BORDER_SUBTLE}; margin-top: 10px; padding-top: 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; color: {TEXT_SECONDARY}; }}
QLabel#muted {{ color: {TEXT_MUTED}; }}
QLabel#banner {{ background: {BRAND_DEEP}; padding: 6px 10px; }}
QSlider::groove:horizontal {{ height: 4px; background: {SURFACE_INSET}; }}
QSlider::handle:horizontal {{ background: {TEXT_PRIMARY}; width: 12px; margin: -5px 0; border-radius: 6px; }}
QStatusBar {{ color: {TEXT_SECONDARY}; }}
"""


def style_axes(ax):
    ax.set_facecolor(SURFACE_INSET)
    for s in ax.spines.values():
        s.set_color(BORDER_DEFAULT)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(TEXT_SECONDARY)
    ax.yaxis.label.set_color(TEXT_SECONDARY)
    ax.title.set_color(TEXT_PRIMARY)
    ax.grid(True, color=BORDER_SUBTLE, linewidth=0.6)
    return ax
