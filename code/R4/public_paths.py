from pathlib import Path
import os
PACKAGE_ROOT=Path(__file__).resolve().parents[2]
def package_path(relative):
    return PACKAGE_ROOT/relative
def data_path(relative):
    root=os.environ.get("DATA_ROOT")
    if not root:raise RuntimeError("Set DATA_ROOT to your independently obtained datasets/cache/weights directory.")
    return Path(root)/relative
