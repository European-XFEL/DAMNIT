import inspect
import logging

from pathlib import Path
from graphlib import CycleError, TopologicalSorter

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
