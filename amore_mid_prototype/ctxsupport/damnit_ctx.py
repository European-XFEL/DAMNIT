"""This module is made available by manipulating sys.path

We aim to maintain compatibility with older Python 3 versions (currently 3.9+)
than the DAMNIT code in general, to allow running context files in other Python
environments.
"""
import re
import abc
import collections

from functools import wraps
from enum import Enum

import pandas as pd

class ValueType(abc.ABC):

    @abc.abstractmethod
    def type_instance():
        pass

    @abc.abstractmethod
    def type_name():
        pass

    def __str__(self):
        return self.type_name

    @abc.abstractmethod
    def description():
        pass

    @abc.abstractmethod
    def examples():
        pass

    @classmethod
    def unwrap(cls, x):
        if len(x) == 1:
            return x.to_numpy().item()
        return list(x)

    @classmethod
    def convert(cls, data, unwrap=False):
        is_string = isinstance(data, str)
        is_sequence = hasattr(data, "__len__") and not is_string

        if not is_sequence:
            data = [data]

        res = pd.Series(data).convert_dtypes().astype(cls.type_instance)

        return cls.unwrap(res) if unwrap else res

class BooleanValueType(ValueType):

    type_instance = pd.BooleanDtype()

    type_name = "boolean"

    description = "A value type that can be used to denote truth values."

    examples = ["True", "T", "true", "1", "False", "F", "f", "0"]

    _valid_values = {
        "true": True,
        "yes": True,
        "1": True,
        "false": False,
        "no": False,
        "0": False
    }

    @classmethod
    def _map_strings_to_values(cls, to_convert, valid_strings):
        res = valid_strings.str.startswith(to_convert.lower())
        n_matches = res.sum()
        if n_matches == 1:
            return cls._valid_values[valid_strings[res.argmax()]]
        else:
            raise ValueError(f"Value \"{to_convert}\" matches {'more than one' if n_matches > 0 else 'none'} of the allowed ones ({', '.join(valid_strings)})")

    @classmethod
    def convert(cls, data, unwrap=False):
        is_string = isinstance(data, str)
        is_sequence = hasattr(data, "__len__") and not is_string

        if not is_sequence:
            data = [data]

        to_convert = pd.Series(data).convert_dtypes()
        res = None

        match to_convert.dtype:
            case "object":
                raise ValueError("The input array contains mixed object types")
            case "string":
                valid_strings = pd.Series(cls._valid_values.keys(), dtype="string")
                res = to_convert.map(lambda x: cls._map_strings_to_values(x, valid_strings))
            case _:
                res = to_convert.astype(cls.type_instance)

        return cls.unwrap(res) if unwrap else res


class IntegerValueType(ValueType):

    type_instance = pd.Int32Dtype()

    type_name = "integer"

    description = "A value type that can be used to count whole number of elements or classes."

    examples = ["-7", "-2", "0", "10", "34"]

class NumberValueType(ValueType):

    type_instance = pd.Float32Dtype()

    type_name = "number"

    description = "A value type that can be used to represent decimal numbers."

    examples = ["-34.1e10", "-7.1", "-4", "0.0", "3.141592653589793", "85.4E7"]

class StringValueType(ValueType):

    type_instance = pd.StringDtype()

    type_name = "string"

    description = "A value type that can be used to represent text."

    examples = ["Broken", "Dark frame", "test_frame"]

types_map = { tt.type_name : tt for tt in [BooleanValueType(), IntegerValueType(), NumberValueType(), StringValueType()] }

def get_type_from_typename(type_name):

    if type_name not in types_map:
        raise ValueError(f"The type \"{type_name}\" is not valid available types are {', '.join(list(types_map.keys()))}")

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

    def get_type_class(self):
        return get_type_from_typename(self.variable_type)


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


