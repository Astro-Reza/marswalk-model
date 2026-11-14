"""Marswalk walking-model data generator (MyoSuite myoLeg on Mars terrain patches)."""
# Written to every row as sim_version. Raise it whenever labels or features change
# meaning, so data from different versions is never mixed by accident.
__version__ = "0.2.0"

import sys as _sys

if _sys.platform == "win32":
    # pyarrow (used by pandas) ships an old msvcp140.dll. If it is loaded first,
    # Windows hands that copy to MuJoCo, whose DLL then fails to start
    # ("WinError 1114"). Loading MuJoCo first avoids it.
    try:
        import mujoco  # noqa: F401
    except (ImportError, OSError):  # not installed, or pandas came first
        pass
