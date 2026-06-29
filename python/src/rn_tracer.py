import numpy as np
from pathlib import Path

_tracing = False
_iter = 0
_feature = 0
_path = None

def init_tracer(path: str):
    """docstring for init_traacer"""
    _path = path
    Path(_path).mkdir(parents=True, exist_ok=True)

def set_iter(iter: int):
    if _tracing:
        global _iter
        _iter = iter

def set_feature(feature: int):
    if _tracing:
        global _feature
        _feature = feature

def register_op(d: dict):
    if _tracing:
        for k, v in d.items():
            t = v.cpu().numpy()
            fmt = "%d"
            np.savetxt(f"{_path}/{k}.txt", t, fmt=fmt)
