"""This module is made available by manipulating sys.path

We aim to maintain compatibility with older Python 3 versions (currently 3.9+)
than the DAMNIT code in general, to allow running context files in other Python
environments.
"""
import re
from enum import Enum

class RunData(Enum):
    RAW = "raw"
    PROC = "proc"
    ALL = "all"

class Variable:
    def __init__(self, title=None, summary=None, data=None, cluster=False):
        self.func = None
        self.title = title
        self.summary = summary

        if data is not None and data not in ["raw", "proc"]:
            raise ValueError(f"Error in Variable declaration: the 'data' argument is '{data}' but it should be either 'raw' or 'proc'")
        else:
            # Store the users original setting, this is used later to determine
            # whether raw-data variables that depend on proc-data variables can
            # automatically be promoted.
            self._data = data

        self.cluster = cluster

    def __call__(self, func):
        self.func = func
        self.name = func.__name__
        return self

    @classmethod
    def from_data(cls, var_data, title=None, **kwargs):
        name = re.sub(r"[\W_]+", "_", title.lower()).strip("_")
        var = cls(title=title, **kwargs)(lambda: var_data)
        var.name = name

        return var

    @property
    def data(self):
        """
        Return the RunData of the Variable.
        """
        return RunData.RAW if self._data is None else RunData(self._data)

    def arg_dependencies(self):
        """
        Get all direct dependencies of this Variable. Returns a dict of argument name
        to variable name.
        """
        return { arg_name: annotation.removeprefix("var#")
                 for arg_name, annotation in self.annotations().items()
                 if annotation.startswith("var#") }

    def annotations(self):
        """
        Get all annotated arguments of this Variable (including meta arguments,
        unlike `Variable.dependencies()`).

        Returns a dict of argument names to their annotations.
        """
        if self.func is None:
            raise RuntimeError(f"Variable '{self.title}' is not initialized with a function")

        return getattr(self.func, '__annotations__', {})

