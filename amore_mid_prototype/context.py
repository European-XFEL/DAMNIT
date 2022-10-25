import inspect
import logging
from enum import Enum
from pathlib import Path
from graphlib import CycleError, TopologicalSorter

log = logging.getLogger(__name__)


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

        return inspect.get_annotations(self.func)

class ContextFile:
    def __init__(self, vars, code):
        self.vars = vars
        self.code = code

        # Check for cycles
        try:
            self.ordered_vars()
        except CycleError as e:
            # Tweak the error message to make it clearer
            raise CycleError(f"These Variables have cyclical dependencies, which is not allowed: {e.args[1]}") from e

        # Check for raw-data variables that depend on proc-data variables
        raw_vars = { name: var for name, var in self.vars.items()
                     if var.data == RunData.RAW }
        for name, var in raw_vars.items():
            dependencies = self.all_dependencies(var)
            proc_dependencies = [dep for dep in dependencies
                                 if self.vars[dep].data == RunData.PROC]

            if len(proc_dependencies) > 0:
                # If we have a variable that depends on proc data but didn't
                # explicitly set `data`, then promote this variable to use proc
                # data.
                if var._data == None:
                    var._data = RunData.PROC.value
                # Otherwise, if the user explicitly requested raw data but the
                # variable depends on proc data, that's a problem with the
                # context file and we raise an exception.
                elif var._data == RunData.RAW.value:
                    raise RuntimeError(f"Variable '{name}' is triggered by migration of raw data by data='raw', "
                                       f"but depends on these Variables that require proc data: {', '.join(proc_dependencies)}\n"
                                       f"Either remove data='raw' for '{name}' or change it to data='proc'")

        # Check that no variables have duplicate titles
        titles = [var.title for var in self.vars.values() if var.title is not None]
        duplicate_titles = set([title for title in titles if titles.count(title) > 1])
        if len(duplicate_titles) > 0:
            bad_variables = [name for name, var in self.vars.items()
                             if var.title in duplicate_titles]
            raise RuntimeError(f"These Variables have duplicate titles between them:{', '.join(bad_variables)}")

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
        code = path.read_text()
        log.debug("Loading context from %s", path)
        return ContextFile.from_str(code)

    @classmethod
    def from_str(cls, code: str):
        d = {}
        exec(code, d)
        vars = {v.name: v for v in d.values() if isinstance(v, Variable)}
        log.debug("Loaded %d variables", len(vars))
        return cls(vars, code)

    def filter(self, run_data=RunData.ALL, cluster=True, name_matches=()):
        new_vars = {}
        for name, var in self.vars.items():
            title = var.title or name

            # If this is being triggered by a migration/calibration message for
            # raw/proc data, then only process the Variable's that require that data.
            data_match = run_data == RunData.ALL or var.data == run_data
            # Skip data tagged cluster unless we're in a dedicated Slurm job
            cluster_match = cluster or not var.cluster
            # Skip Variables that don't match the match list
            name_match = (len(name_matches) == 0
                          or any(m.lower() in title.lower() for m in name_matches))

            if data_match and cluster_match and name_match:
                new_vars[name] = var

        # Add back any dependencies of the selected variables
        new_vars.update({name: self.vars[name] for name in self.all_dependencies(*new_vars.values())})

        return ContextFile(new_vars, self.code)
