"""Exception classes specific to DAMNIT context execution."""

def _format_dependency_errors(dep_errors):
    def format_entry(dep_name, exc, prefix, is_last):
        connector = "└ " if is_last else "├ "
        line_prefix = f"{prefix}{connector}"
        child_prefix = f"{prefix}{'   ' if is_last else '│  '}"

        if isinstance(exc, DependencyError):
            lines = [f"{line_prefix}({dep_name}) skipped"]
            if not exc.dep_errors:
                lines.append(f"{child_prefix}<no details>")
                return lines
            last_idx = len(exc.dep_errors)
            for idx, (sub_dep, sub_exc) in enumerate(exc.dep_errors, start=1):
                lines.extend(format_entry(
                    sub_dep,
                    sub_exc,
                    child_prefix,
                    idx == last_idx,
                ))
            return lines

        msg = str(exc)
        if not msg:
            return [f"{line_prefix}({dep_name}) failed: {type(exc).__name__}"]

        msg_lines = msg.splitlines()
        summary = f"{type(exc).__name__}: {msg_lines[0]}"
        lines = [f"{line_prefix}({dep_name}) failed: {summary}"]
        lines.extend(f"{child_prefix}{line}" for line in msg_lines[1:])
        return lines

    lines = ["Skipped due to missing dependency:"]
    last_idx = len(dep_errors)
    for idx, (dep, exc) in enumerate(dep_errors, start=1):
        lines.extend(format_entry(dep, exc, "", idx == last_idx))
    return "\n".join(lines)


class DependencyError(RuntimeError):
    def __init__(self, dep_errors):
        self.dep_errors = list(dep_errors)
        super().__init__(_format_dependency_errors(self.dep_errors))


class ContextFileErrors(RuntimeError):
    def __init__(self, problems):
        self.problems = problems

    def __str__(self):
        return "\n".join(self.problems)


class Skip(Exception):
    pass


class GroupError(Exception):
    pass
