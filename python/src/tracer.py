import numpy as np
from pathlib import Path

from torchhd.tensors.map import MAPTensor

_tracing = False
_query = 0
_iter = 0
_feature = 0
_path = None

def init(path: str):
    """docstring for init_traacer"""
    global _tracing, _path
    _path = path
    Path(_path).mkdir(parents=True, exist_ok=True)
    _tracing = True

def set_query(query: int):
    if _tracing:
        global _query
        _query = query

def set_iter(iter: int):
    if _tracing:
        global _iter
        _iter = iter

def set_feature(feature: int):
    if _tracing:
        global _feature
        _feature = feature

def register_input(d: dict):
    """docstring for register_input"""
    if _tracing:
        for k, v in d.items():
            t = v.cpu().numpy()

            if type(v) is MAPTensor:
                t[t==-1] = 0 # Detect MAP HVs and change them to binary

            fmt = "%d"
            p = f"{_path}/q_{_query}/input"
            Path(p).mkdir(parents=True, exist_ok=True)

            np.savetxt(f"{p}/{k}.txt", t, delimiter="", fmt=fmt)

def register_op(d: dict):
    if _tracing:
        op = d.pop("name")
        for k, v in d.items():
            t = v.cpu().numpy()

            if type(v) is MAPTensor:
                t[t==-1] = 0 # Detect MAP HVs and change them to binary

            fmt = "%d"
            p = f"{_path}/q_{_query}/i_{_iter}/f_{_feature}/{op}"
            Path(p).mkdir(parents=True, exist_ok=True)

            np.savetxt(f"{p}/{k}.txt", t, delimiter="", fmt=fmt)
