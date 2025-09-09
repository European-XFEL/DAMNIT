import fnmatch
import logging
from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from typing import Any

from damnit_ctx import RunData, Variable

log = logging.getLogger(__name__)


class PipelineErrors(RuntimeError):
    def __init__(self, problems):
        self.problems = problems

    def __str__(self):
        return "\n".join(self.problems)


class Pipeline:
    def __init__(self, variables: dict[str, Variable]):
        self.vars = variables

        # Check for cycles
        try:
            ordered_names = self.ordered(self.vars)
        except CycleError as e:
            raise CycleError(
                f"These Variables have cyclical dependencies, which is not allowed: {e.args[1]}"
            ) from e

        # 'Promote' variables to match characters of their dependencies
        for name in ordered_names:
            var = self.vars[name]
            deps = [self.vars[dep] for dep in self.all_dependencies(var)]
            if var._data is None and any(v.data == RunData.PROC for v in deps):
                var._data = RunData.PROC.value

            if any(v.cluster for v in deps):
                var.cluster = True

    def to_dict(self, inc_transient=False) -> dict[str, Any]:
        """Get a dict of variable metadata.

        args:
            inc_transient (bool): include transient Variables in the dict
        """
        return {
            name: v.to_dict()
            for (name, v) in self.vars.items()
            if not v.transient or inc_transient
        }

    def check(self):
        problems = []
        for name, var in self.vars.items():
            problems.extend(var.check())
            if var.data != RunData.RAW:
                continue
            proc_dependencies = [dep for dep in self.all_dependencies(var)
                                 if self.vars[dep].data == RunData.PROC]

            if proc_dependencies:
                if var.data == RunData.RAW:
                    problems.append(
                        f"Variable {name} is triggered by migration of raw data (data='raw'), "
                        f"but depends on these Variables that require proc data: {', '.join(proc_dependencies)}\n"
                        f"Remove data='raw' for {name} or change it to data='proc'"
                    )

        # Check that no variables have duplicate titles
        titles = [var.title for var in self.vars.values() if var.title is not None]
        duplicate_titles = set([title for title in titles if titles.count(title) > 1])
        if len(duplicate_titles) > 0:
            bad_variables = [name for name, var in self.vars.items()
                             if var.title in duplicate_titles]
            problems.append(
                f"These Variables have duplicate titles between them: {', '.join(bad_variables)}"
            )

        # Check that all mymdc dependencies are valid
        for name, var in self.vars.items():
            mymdc_args = var.arg_dependencies("mymdc#")
            for arg_name, annotation in mymdc_args.items():
                if annotation not in ["sample_name", "run_type", "techniques"]:
                    problems.append(f"Argument '{arg_name}' of variable '{name}' has an invalid MyMdC dependency: '{annotation}'")

        if problems:
            raise PipelineErrors(problems)

    def ordered(self, variables: dict[str, Variable]) -> tuple[str]:
        """Return topologically sorted variables."""
        ts = TopologicalSorter()

        for name, var in variables.items():
            ts.add(name, *self.dependencies(var))

        return tuple(ts.static_order())

    def dependencies(self, variable: Variable) -> set[str]:
        """return a set of names of direct dependencies of the passed Variable
        """
        dependencies = set()
        for dependency in variable.arg_dependencies().values():
            # expand matching patterns to match all variable dependencies
            deps = fnmatch.filter(self.vars, dependency)
            if len(deps) == 0:
                raise KeyError(f"Missing dependency: {dependency!r} for {variable.name!r}")
            dependencies.update(deps)
        return dependencies

    def all_dependencies(self, *variables):
        """
        Return a set of names of all dependencies (direct and indirect) of the
        passed Variable's.
        """
        dependencies = set()

        for var in variables:
            var_deps = self.dependencies(var)
            dependencies |= var_deps

            if len(var_deps) > 0:
                for dep_name in var_deps:
                    dependencies |= self.all_dependencies(self.vars[dep_name])

        return dependencies

    # def _add_dependencies(self, variables: list[Variable]) -> list[Variable]:
    #     res = variables.copy()
    #     to_check = set(v.name for v in variables)

    #     while to_check:
    #         name = to_check.pop()
    #         if name not in self.vars:
    #             raise KeyError

    #         var = self.vars[name]
    #         for dep in self.dependencies(var):
    #             if dep not in res:
    #                 res[dep] = self.vars[dep].copy()
    #                 to_check.add(dep)

    #     return res

    def filter(
        self, 
        variables: list[str],
        *,
        run_data: RunData = RunData.ALL,
        cluster: bool | None = None,
        names: list[str] | None = None,
        patterns: list[str] | None = None,
    ) -> 'Pipeline':
        vars = [v.copy() for name, v in self.vars.items() if name in variables]

        # Filter
        vars = [v for v in vars if v.data == run_data]
        if cluster is not None:
            vars = [v for v in vars if vars.cluster == cluster]
        if names is not None:
            vars = [v for v in vars if vars.name in names]
        if patterns is not None:
            vars = [v for v in vars if any(p.lower() in v.title.lower() for p in patterns)]

        # Add dependencies
        vars += [self.vars[v] for v in self.all_dependencies(*vars)]
        return Pipeline({v.name: v for v in vars})


class ContextFile:
    def __init__(self, pipeline: Pipeline, code: str):
        self.pipe = pipeline
        self.code = code

    @classmethod
    def from_py_file(cls, path: Path):
        code = path.read_text()
        log.debug("Loading context from %s", path)
        return ContextFile.from_str(code, str(path.absolute()))

    @classmethod
    def from_str(cls, code: str, path='<string>'):
        d = {}
        codeobj = compile(code, path, 'exec')
        exec(codeobj, d)
        vars = {v.name: v for v in d.values() if isinstance(v, Variable)}
        log.debug("Loaded %d variables", len(vars))
        return cls(Pipeline(vars), code)
