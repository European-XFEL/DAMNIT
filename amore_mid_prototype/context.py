import inspect
import logging
from enum import Enum
from pathlib import Path
from graphlib import CycleError, TopologicalSorter

log = logging.getLogger(__name__)


class Variable:
    def __init__(self, title=None, summary=None, data="raw", heavy=False):
        self.func = None
        self.title = title
        self.summary = summary

        if data not in ["raw", "proc"]:
            raise ValueError(f"Error in Variable declaration: the 'data' argument is '{data}' but it should be either 'raw' or 'proc'")
        self.data = data
        self.heavy = heavy

    def __call__(self, func):
        self.func = func
        self.name = func.__name__
        return self

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

        return inspect.get_annotations(self.func)

class RunData(Enum):
    RAW = "raw"
    PROC = "proc"
    ALL = "all"


class ContextFile:
    def __init__(self, vars, code):
        self.vars = vars
        self.code = code

        try:
            self.ordered_vars()
        except CycleError as e:
            # Tweak the error message to make it clearer
            raise CycleError(f"These Variables have cyclical dependencies, which is not allowed: {e.args[1]}") from e

    def ordered_vars(self):
        """
        Return a tuple of variables in the context file, topologically sorted.
        """
        vars_graph = { name: set(var.arg_dependencies().values()) for name, var in self.vars.items() }

        # Sort and return
        ts = TopologicalSorter(vars_graph)
        return tuple(ts.static_order())

    def all_dependencies(self, *variables):
        """
        Return a set of names of all dependencies (direct and indirect) of the
        passed Variable's.
        """
        dependencies = set()

        for var in variables:
            var_deps = set(var.arg_dependencies().values())
            dependencies |= var_deps

            if len(var_deps) > 0:
                for dep_name in var_deps:
                    dependencies |= self.all_dependencies(self.vars[dep_name])

        return dependencies

    @classmethod
    def from_py_file(cls, path: Path):
        code = path.read_bytes()
        log.debug("Loading context from %s", path)
        return ContextFile.from_str(code)

    @classmethod
    def from_str(cls, code: str):
        d = {}
        exec(code, d)
        vars = {v.name: v for v in d.values() if isinstance(v, Variable)}
        log.debug("Loaded %d variables", len(vars))
        return cls(vars, code)

    def filter(self, run_data=RunData.ALL, heavy=True, name_matches=()):
        new_vars = {}
        for name, var in self.vars.items():
            title = var.title or name

            # If this is being triggered by a migration/calibration message for
            # raw/proc data, then only process the Variable's that require that data.
            data_match = run_data == RunData.ALL or var.data == run_data.value
            # Skip data tagged heavy unless we're in a dedicated Slurm job
            heavy_match = heavy or not var.heavy
            # Skip Variables that don't match the match list
            name_match = (len(name_matches) == 0
                          or any(m.lower() in title.lower() for m in name_matches))

            if data_match and heavy_match and name_match:
                new_vars[name] = var

        # Add back any dependencies of the selected variables
        new_vars.update({name: self.vars[name] for name in self.all_dependencies(*new_vars.values())})

        return ContextFile(new_vars, self.code)
