"""Local Git checkpoints of context files and their support files."""
import fcntl
import getpass
import logging
import os
from pathlib import Path
import socket
import subprocess

log = logging.getLogger(__name__)

HISTORY_DIR = '.damnit-history.git'
LOCK_FILE = '.damnit-history.lock'
_METADATA_PATHS = {HISTORY_DIR, LOCK_FILE, '.git'}


def checkpoint_context(directory: Path) -> str | None:
    """Checkpoint inputs, returning HEAD; warn and return None on failure.

    The private index belongs only to DAMNIT. A lock serializes checkpoint
    writers and is inherited by Git, but does not prevent users editing files
    while Git reads them.
    """
    try:
        directory = Path(directory).resolve()
        history = directory / HISTORY_DIR
        if history.is_symlink():
            raise OSError(f'History directory must not be a symlink: {history}')
        index_lock = history / 'index.lock'
        env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
        env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull)
        username = getpass.getuser()
        cmd = [
            'git', f'--git-dir={history}', f'--work-tree={directory}',
            '-c', f'safe.directory={directory}',
            '-c', f'user.name={username}',
            '-c', f'user.email={username}@{socket.gethostname()}',
            '-c', f'core.hooksPath={os.devnull}', '-c', 'commit.gpgSign=false',
        ]

        def git(*args, input=None, check=True):
            command = ['git'] if args[0] == 'init' else cmd
            try:
                return subprocess.run(
                    [*command, *args], cwd=directory, env=env, input=input,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    check=check, timeout=30, pass_fds=(lock.fileno(),),
                )
            except subprocess.TimeoutExpired:
                # run() has killed and waited for Git; we still hold the lock.
                index_lock.unlink(missing_ok=True)
                raise

        # Keep this file in place: unlinking it would allow two distinct locks.
        fd = os.open(
            directory / LOCK_FILE, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o666
        )
        with os.fdopen(fd, 'r+b') as lock:
            if os.fstat(lock.fileno()).st_uid == os.getuid():
                os.fchmod(lock.fileno(), 0o666)
            fcntl.flock(lock, fcntl.LOCK_EX)
            # Surviving Git writers retain the flock, so any leftover is stale.
            index_lock.unlink(missing_ok=True)
            if not (history / 'HEAD').is_file():
                git('init', '--bare', '--shared=0666', '--template=', str(history))
            exclude_file = history / 'info/exclude'
            if not exclude_file.exists():
                exclude_file.parent.mkdir(exist_ok=True, mode=0o777)
                if exclude_file.parent.stat().st_uid == os.getuid():
                    exclude_file.parent.chmod(0o777)
                template = Path(__file__).with_suffix('.gitignore')
                exclude_file.write_text(template.read_text(encoding='utf-8'), encoding='utf-8')
                exclude_file.chmod(0o666)

            # Metadata is excluded even when absent or negated in .gitignore.
            # Git rejects explicit pathspecs for ignored files, even exclusions.
            ignored_paths = git(
                'check-ignore', '--no-index', '-z', '--stdin', check=False,
                input=b'\0'.join(os.fsencode(p) for p in sorted(_METADATA_PATHS)) + b'\0',
            )
            if ignored_paths.returncode not in (0, 1):
                ignored_paths.check_returncode()
            already_ignored = {
                os.fsdecode(p) for p in ignored_paths.stdout.split(b'\0') if p
            }
            pathspecs = ['.'] + [
                f':(top,exclude,literal){p}'
                for p in sorted(_METADATA_PATHS - already_ignored)
            ]
            git('add', '-A', '--pathspec-from-file=-', '--pathspec-file-nul',
                input=b'\0'.join(os.fsencode(p) for p in pathspecs) + b'\0')

            # Ignore rules also apply to files tracked by earlier checkpoints.
            ignored = git('ls-files', '-ci', '--exclude-standard', '-z').stdout
            tracked = git('ls-files', '-c', '-z').stdout.split(b'\0')
            removed = set(ignored.split(b'\0')) - {b''}
            metadata = {os.fsencode(p) for p in _METADATA_PATHS}
            removed.update(
                filename for filename in tracked
                if filename.split(b'/', 1)[0] in metadata
            )
            if removed:
                git('update-index', '--force-remove', '-z', '--stdin',
                    input=b'\0'.join(sorted(removed)) + b'\0')
            context_path = directory / 'context.py'
            if context_path.is_file() or context_path.is_symlink():
                git('add', '-f', '--', 'context.py')
            diff = git('diff', '--cached', '--quiet', check=False)
            if diff.returncode == 1:
                git('commit', '-m', 'Checkpoint context files')
            elif diff.returncode != 0:
                diff.check_returncode()
            revision = git('rev-parse', 'HEAD').stdout.decode().strip()
        log.info('Context checkpoint: %s', revision)
        return revision
    except (OSError, subprocess.SubprocessError) as error:
        detail = getattr(error, 'stderr', None)
        if isinstance(detail, bytes):
            detail = detail.decode(errors='replace').strip()
        log.warning(
            'Could not checkpoint context in %s: %s', directory, detail or error
        )
        return None
