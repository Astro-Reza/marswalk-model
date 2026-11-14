# Imported by pytest before any test module. On Windows MuJoCo has to be loaded
# before pandas/pyarrow (see marswalk_sim/__init__.py), whatever order the tests run in.
import marswalk_sim  # noqa: F401
