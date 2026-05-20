"""将 vendor 依赖加入 Python 路径（GFPGAN 与 CodeFormer 的 basicsr 需分开加载）。"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
_VENDOR = os.path.join(_ROOT, "vendor")

_BASICSR = os.path.join(_VENDOR, "BasicSR")
_GFPGAN = os.path.join(_VENDOR, "GFPGAN")
_CODEFORMER = os.path.join(_VENDOR, "CodeFormer")


def _clear_basicsr_modules():
    for key in list(sys.modules.keys()):
        if key == "basicsr" or key.startswith("basicsr."):
            del sys.modules[key]


def setup_gfpgan_paths():
    """加载 GFPGAN（依赖 vendor/BasicSR）。"""
    for p in (_GFPGAN, _BASICSR):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


def setup_codeformer_paths():
    """加载 CodeFormer（使用其自带的 basicsr，须单独加载）。"""
    _clear_basicsr_modules()
    if _CODEFORMER not in sys.path:
        sys.path.insert(0, _CODEFORMER)
