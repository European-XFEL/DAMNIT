import logging
from pathlib import Path

log = logging.getLogger(__name__)


class Variable:
    def __init__(self, title=None, summary=None, data="raw"):
        self.func = None
        self.title = title
        self.summary = summary

        if data not in ["raw", "proc"]:
            raise ValueError(f"Error in Variable declaration: the 'data' argument is '{data}' but it should be either 'raw' or 'proc'")
        self.data = data

    def __call__(self, func):
        self.func = func
        self.name = func.__name__
        return self


class ContextFile:
    def __init__(self, vars):
        self.vars = vars

    @classmethod
    def from_py_file(cls, path: Path):
        code = path.read_bytes()
        d = {}
        exec(code, d)
        vars = {v.name: v for v in d.values() if isinstance(v, Variable)}
        log.debug("Loaded context from %s: %d variables", path, len(vars))
        return cls(vars)
