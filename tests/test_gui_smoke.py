"""Builds the whole window offscreen and draws every tab. Skipped when PySide6 / matplotlib are missing."""
import os

import pytest

pytest.importorskip("PySide6.QtWidgets")
pytest.importorskip("matplotlib")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from marswalk_gui import core, theme  # noqa: E402
from marswalk_gui.app import MainWindow, State  # noqa: E402
from marswalk_gui.demo import make_demo  # noqa: E402


@pytest.fixture(scope="module")
def app():
    a = QApplication.instance() or QApplication([])
    a.setStyleSheet(theme.QSS)
    return a


def test_all_tabs_draw_empty_and_loaded(app, tmp_path):
    win = MainWindow(State())
    for i in range(win.tabs.count()):  # nothing loaded: every tab must show a message, not crash
        win.tabs.setCurrentIndex(i)
        app.processEvents()
    data, model = make_demo(tmp_path)
    win.state.add_run(core.load_run(model), synthetic=True)
    win.state.set_dataset(core.load_dataset(str(data)), str(data), synthetic=True)
    for i in range(win.tabs.count()):
        win.tabs.setCurrentIndex(i)
        app.processEvents()
    assert not win.banner.isHidden()  # synthetic data must be flagged
    assert "384 patches" in win.statusBar().currentMessage()
    assert win.dataset_tab.rows_model.rowCount() == 384
