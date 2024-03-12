from dataclasses import dataclass
from typing import Optional


class ValueType:
    # These are intended to be overridden in subclasses.
    type_name = None

    description = None

    examples = None

    def __str__(self):
        return self.type_name

    @classmethod
    def parse(cls, input: str):
        return input

    @classmethod
    def from_db_value(cls, value):
        return value


class BooleanValueType(ValueType):
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
            raise ValueError(
                f"Value \"{to_convert}\" matches {'more than one' if n_matches > 0 else 'none'} of the allowed ones ({', '.join(valid_strings)})")

    @classmethod
    def parse(cls, input: str):
        try:
            return cls._valid_values[input.lower()]
        except KeyError:
            raise ValueError("{input!r} not one of true/false/1/0/yes/no")

    @classmethod
    def from_db_value(cls, value):
        if value is None:
            return None
        return bool(value)


class IntegerValueType(ValueType):
    type_name = "integer"

    description = "A value type that can be used to count whole number of elements or classes."

    examples = ["-7", "-2", "0", "10", "34"]

    @classmethod
    def parse(cls, input: str):
        return int(input)

class NumberValueType(ValueType):
    type_name = "number"

    description = "A value type that can be used to represent decimal numbers."

    examples = ["-34.1e10", "-7.1", "-4", "0.0", "3.141592653589793", "85.4E7"]

    @classmethod
    def parse(cls, input: str):
        return float(input)


class StringValueType(ValueType):
    type_name = "string"

    description = "A value type that can be used to represent text."

    examples = ["Broken", "Dark frame", "test_frame"]


value_types_by_name = {tt.type_name: tt for tt in [
    BooleanValueType(), IntegerValueType(), NumberValueType(), StringValueType()
]}


@dataclass
class UserEditableVariable:
    name: str  # Computer friendly ID: 'xgm_intensity'
    title: str  # Human friendly title: 'XGM intensity (μJ)
    variable_type: str   # e.g. 'integer', 'string' - see above
    description: Optional[str] = ""
    attributes: Optional[str] = ""

    def get_type_class(self):
        return value_types_by_name[self.variable_type]
