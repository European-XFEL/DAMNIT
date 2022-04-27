import logging
from pathlib import Path
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


# RectROI = namedtuple("RectROI", ["x", "y", "width", "height", "image"],
#                      defaults=[None])

@dataclass
class RectROI:
    x: int
    y: int
    width: int
    height: int
    image: np.ndarray = None

class Parameter:
    def __init__(self, title=None):
        self.func = None
        self.title = title

    def __call__(self, func):
        self.func = func
        self.name = func.__name__
        return self


class Variable:
    def __init__(self, title=None, summary=None):
        self.func = None
        self.title = title
        self.summary = summary

    def __call__(self, func):
        self.func = func
        self.name = func.__name__
        return self


class ContextFile:
    def __init__(self, vars, params):
        self.vars = vars
        self.params = params

    @classmethod
    def from_py_file(cls, path: Path):
        code = path.read_bytes()
        d = {}
        exec(code, d)
        vars = {v.name: v for v in d.values() if isinstance(v, Variable)}
        params = {p.name: p for p in d.values() if isinstance(p, Parameter)}
        log.debug("Loaded context from %s: %d variables and %d parameters",
                  path, len(vars), len(params))
        return cls(vars, params)
