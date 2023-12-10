from pathlib import Path
import shlex
import subprocess

from extra_data.read_machinery import find_proposal


class Bubblewrap:
    """A class representing a sandbox environment using Bubblewrap.

    Bubblewrap is a sandboxing tool that creates a restricted environment for processes,
    this class provides methods to configure and build a bubblewrap sandbox for running
    a context file such that it only has access to data from the relevant proposal.

    Attributes:
        command (list[str]): The base command for running in the sandbox.
        command_binds (list[tuple[str, str]]): List of bind mounts commands.
    """

    def __init__(self):
        self.command = [
            "bwrap",
            "--disable-userns", # Disable creation of user namespaces in sandbox
            "--die-with-parent",  # Kill sandbox if parent process dies
            "--unshare-all",  # Unshare all namespaces
            "--share-net",  # Share network namespace
            "--dev /dev",  # Bind mount /dev
            "--tmpfs /tmp",  # Mount tmpfs on /tmp
            "--dir /gpfs",  # Create empty directory at /gpfs
        ]

        self.command_binds: list[tuple[str, str]] = []

        for path in (
            "/bin",
            "/etc/resolv.conf",
            "/lib",
            "/lib64",
            "/sbin",
            "/usr",
        ):
            self.add_bind(Path(path), ro=True)

        if Path("/gpfs/exfel/sw/software").exists():
            self.add_bind(Path("/gpfs/exfel/sw/software"), ro=True)

    def add_bind(
        self, source: Path, dest: Path | None = None, ro: bool = False
    ) -> None:
        """Adds a bind mount to the sandbox.

        Args:
            source (Path): The source path to be bind mounted.
            dest (Path, optional): The destination path in the sandbox. If not provided, the source path is used.
            ro (bool, optional): Whether the bind mount should be read-only. Defaults to False.

        Raises:
            ValueError: If the source path is not absolute.
        """
        if not source.is_absolute():
            raise ValueError("Source path must be absolute")

        if dest is None:
            dest = source

        self.command_binds.append(
            (
                f"--{'ro-' if ro else ''}bind",
                f"{shlex.quote(str(source))} {shlex.quote(str(dest))}",
            )
        )

    def add_bind_proposal(self, proposal_id: int) -> None:
        """Adds bind mounts for a proposal directory and its contents.

        Args:
            proposal_id (int): The ID of the proposal.

        Raises:
            FileNotFoundError: If the proposal directory is not found.
        """
        proposal_dir = Path(find_proposal(f"p{proposal_id:06d}"))

        self.add_bind(proposal_dir)

        for path in proposal_dir.iterdir():
            self.add_bind(path.resolve())

    def add_bind_venv(self, python_exec: Path) -> None:
        """Adds all paths required by a virtual environment to the sandbox.

        This function will use the given python executable to first call `sys.prefix` to
        check if the executable is in a venv, if it is then `sysconfig.get_paths()` is
        used to find required paths and add them paths as read-only binds.

        Args:
            python_exec (Path): The path to the Python executable.

        Raises:
            subprocess.CalledProcessError: If the command to get the virtual environment paths fails.
        """
        venv = subprocess.check_output(
            [python_exec, "-c", "import sys; print(sys.prefix != sys.base_prefix)"]
        ).decode("utf-8")

        if venv == "False":
            return

        paths = subprocess.check_output(
            [
                python_exec,
                "-c",
                'import sysconfig; print(" ".join(v for v in sysconfig.get_paths().values()))',
            ]
        ).decode("utf-8")

        for path in paths.split():
            path = Path(path)
            self.add_bind(path, ro=True)
            if path.is_symlink():
                self.add_bind(path.resolve(), ro=True)

    def build_command(self, command: str | list[str]) -> list[str]:
        """Builds the final command for running in the sandbox.

        Args:
            command (str or list[str]): The command to be executed in the sandbox.

        Returns:
            list[str]: The final command for running in the sandbox.
        """
        _command = self.command.copy()

        for bind in self.command_binds:
            _command.extend(bind)

        _command.extend(command if isinstance(command, list) else [command])

        return _command
