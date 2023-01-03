"""This module is made available by manipulating sys.path

We aim to maintain compatibility with older Python 3 versions (currently 3.9+)
than the DAMNIT code in general, to allow running context files in other Python
environments.
"""
import re
from functools import wraps
from enum import Enum

import pandas as pd

types_map = {
  'bool' : pd.BooleanDtype(),
  'int' : pd.Int32Dtype(),
  'float' : pd.Float32Dtype(),
  'str' : pd.StringDtype(),
}

def get_type_from_name(type_name):

    if type_name not in types_map:
        raise ValueError(f"The type {type_name} is not valid available types are {', '.join(list(types_map.keys()))}")

    return types_map[type_name]


class VariableBase:

    def __new__(cls, *args, **kwargs):
        if cls == VariableBase:
            raise TypeError(f"only children of '{cls.__name__}' may be instantiated")
        return super().__new__(cls)

    def __init__(self, name=None, func=None, title=None, description=None, attributes={}):
        self.store_result = False
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


class UserEditableVariable(VariableBase):

    def __init__(self, _name, title, variable_type, description=None, attributes={}):
        super().__init__(name=_name, title=title, description=description, attributes=attributes)

        self.variable_type = variable_type
        self.func = self.get_data_func()

    def get_data_func(self):

        variable_name = self.name

        def retrieve_data(inputs, proposal_id : 'meta#proposal', run_id : 'meta#run_number'):
            db = inputs['db_conn']
            res = db.execute(
                f'SELECT {variable_name} FROM runs WHERE proposal=:proposal_id AND runnr=:run_id',
                {
                    'proposal_id' : proposal_id,
                    'run_id' : run_id
                }
            )
            # add cast
            return res.fetchone()[0]

        return retrieve_data

        self.func = retrieve_data


class RunData(Enum):
    RAW = "raw"
    PROC = "proc"
    ALL = "all"

class Variable(VariableBase):

    def __init__(self, title=None, summary=None, data=None, cluster=False, description=None, attributes={}):
        super().__init__(title=title, description=description, attributes=attributes)

        self.store_result = True
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

        @wraps(func)
        def get_default_inputs(inputs, **kwargs):
            return func(inputs['run_data'], **kwargs)

        self.func = get_default_inputs
        self.name = func.__name__
        return self

    @property
    def data(self):
        """
        Return the RunData of the Variable.
        """
        return RunData.RAW if self._data is None else RunData(self._data)


