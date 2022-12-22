"""This module is made available by manipulating sys.path

We aim to maintain compatibility with older Python 3 versions (currently 3.9+)
than the DAMNIT code in general, to allow running context files in other Python
environments.
"""
import re
from enum import Enum

class VariableBase:

    def __new__(cls, *args, **kwargs):
        if cls == VariableBase:
            raise TypeError(f"only children of '{cls.__name__}' may be instantiated")
        return super().__new__(cls)

    def __init__(self, name=None, func=None, title=None, description=None, attributes={}):
        self.func = func
        self.name = name
        self.title = title
        self.description = description
        self.attributes = attributes

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if value and not re.fullmatch('[a-zA-Z_]\w+', value, flags=re.A):
            raise ValueError(f"Error in variable: the variable name '{value}' is not of the form '[a-zA-Z_]\w+'")
        self._name = value


class UserEditableVariable(VariableBase):

    def __init__(self, _name, title, variable_type, description=None, attributes={}):
        super().__init__(name=_name, title=title, description=description, attributes=attributes)

        self.variable_type = variable_type


class RunData(Enum):
    RAW = "raw"
    PROC = "proc"
    ALL = "all"

class Variable(VariableBase):

    def __init__(self, title=None, summary=None, data=None, cluster=False, description=None, attributes={}):
        super().__init__(title=title, description=description, attributes=attributes)

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
        return getattr(self.func, '__annotations__', {})


