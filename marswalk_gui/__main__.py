import sys

from PySide6 import QtWidgets  # noqa: F401  load Qt before pyarrow (via pandas): the other order crashes on Windows

from .app import main

sys.exit(main())
