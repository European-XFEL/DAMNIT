from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
from pathlib import Path
import pickle
import subprocess
from unittest.mock import MagicMock

import pytest

from damnit.backend.context_history import checkpoint_context


def git(directory, *args, private=True):
    command = ['git']
    if private:
        command += [f'--git-dir={directory / ".damnit-history.git"}',
                    f'--work-tree={directory}']
    return subprocess.check_output(command + list(args), cwd=directory).decode().strip()


def tracked(directory):
    return set(git(directory, 'ls-tree', '-r', '--name-only', 'HEAD').splitlines())


@pytest.fixture
def context_dir(tmp_path):
    (tmp_path / 'context.py').write_text('x = 1\n')
    return tmp_path


def test_checkpoints_changes_and_deletions(context_dir):
    directory = context_dir
    first = checkpoint_context(directory)
    assert first and git(directory, 'rev-parse', 'HEAD') == first
    assert checkpoint_context(directory) == first
    assert git(directory, 'rev-list', '--count', 'HEAD') == '1'
    (directory / 'context.py').write_text('x = 2\n')
    (directory / 'helpers.py').write_text('offset = 3\n')
    (directory / 'calibration.csv').write_text('1,2\n')
    second = checkpoint_context(directory)
    assert second and second != first
    assert tracked(directory) == {'context.py', 'helpers.py', 'calibration.csv'}
    (directory / 'helpers.py').unlink()
    (directory / 'calibration.csv').write_text('3,4\n')
    third = checkpoint_context(directory)
    assert third and third != second
    assert tracked(directory) == {'context.py', 'calibration.csv'}
    assert git(directory, 'show', f'{first}:context.py') == 'x = 1'
    assert git(directory, 'show', 'HEAD:calibration.csv') == '3,4'
    (directory / 'calibration.csv').write_text('5,6\n')
    fourth = checkpoint_context(directory)
    assert fourth and fourth != third
    (directory / 'context.py').unlink()
    fifth = checkpoint_context(directory)
    assert fifth and fifth != fourth
    assert tracked(directory) == {'calibration.csv'}


def test_default_runtime_environment_and_cache_files_excluded(context_dir):
    excluded = [
        'runs.sqlite', 'runs.sqlite-wal', 'runs.sqlite-shm',
        'runs.sqlite3', 'runs.sqlite3-wal',
        'process_logs/output.txt', 'extracted_data/calibration.csv',
        'supervisord.conf', 'supervisord.pid', 'supervisord.log',
        'context.pickle', '__pycache__/helpers.pyc', 'helpers.pyc',
        'helpers.pyo', 'output.log', 'output.log.1', 'output.out',
        'output.h5', 'output.hdf5', '.tmp_ctx.py', '.tmp/scratch.txt',
        '.venv/lib/site.py', 'venv/lib/site.py', 'env/lib/site.py',
        'ENV/lib/site.py', '.pixi/envs/default/site.py', '.conda/history',
        '__pypackages__/module.py', 'site-packages/module.py',
        'dist-packages/module.py', '.pytest_cache/results', '.mypy_cache/results',
        '.ruff_cache/results', '.tox/test/site.py', '.nox/test/site.py',
        '.cache/results', '.pyre/results', '.pytype/results',
        '.coverage', '.coverage.host.1234', 'htmlcov/index.html',
        '.hypothesis/examples/result',
        'notebooks/.ipynb_checkpoints/analysis-checkpoint.ipynb',
        'build/module.py', 'dist/module.py', '.eggs/module.py',
        'module.egg-info/PKG-INFO', 'module.egg/module.py',
    ]
    for filename in excluded:
        file = context_dir / filename
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text('runtime\n')
    (context_dir / 'helpers.py').write_text('offset = 2\n')
    assert checkpoint_context(context_dir)
    assert tracked(context_dir) == {'context.py', 'helpers.py'}
    for filename in excluded:
        (context_dir / filename).write_text('updated runtime\n')
    checkpoint_context(context_dir)
    assert git(context_dir, 'rev-list', '--count', 'HEAD') == '1'


def test_custom_environment_and_installation_uses_explicit_ignores(context_dir):
    (context_dir / '.gitignore').write_text(
        'custom-python/\ncustom-conda/\ninstallation/\n'
    )
    for filename in [
        'custom-python/pyvenv.cfg', 'custom-python/lib/site.py',
        'custom-conda/conda-meta/history', 'custom-conda/lib/site.py',
        'installation/damnit/__init__.py', 'installation/damnit/backend/db.py',
    ]:
        file = context_dir / filename
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text('environment\n')
    (context_dir / 'helpers.py').write_text('offset = 2\n')
    assert checkpoint_context(context_dir)
    assert tracked(context_dir) == {'.gitignore', 'context.py', 'helpers.py'}


def test_gitignore_opt_in_and_newly_ignored_files(context_dir):
    git(context_dir, 'init', private=False)
    (context_dir / '.gitignore').write_text(
        'context.py\nprivate.csv\n!calibration.h5\n!output.log\n'
        '!runs.sqlite\n!damnit.log\n!supervisord.log\n'
        '!extracted_data/\n!extracted_data/**\n'
        '!.damnit-history.git/\n!.damnit-history.git/**\n'
        '!.damnit-history.lock\n!.git/\n!.git/**\n'
    )
    for filename in ['calibration.h5', 'output.log', 'private.csv', 'offsets.csv',
                     'runs.sqlite', 'damnit.log', 'supervisord.log',
                     'extracted_data/secret.csv']:
        path = context_dir / filename
        path.parent.mkdir(exist_ok=True)
        path.write_text('values\n')
    assert checkpoint_context(context_dir)
    assert tracked(context_dir) == {
        '.gitignore', 'context.py', 'calibration.h5', 'output.log', 'offsets.csv',
        'runs.sqlite', 'damnit.log', 'supervisord.log', 'extracted_data/secret.csv',
    }
    with (context_dir / '.gitignore').open('a') as f:
        f.write('offsets.csv\n')
    assert checkpoint_context(context_dir)
    assert 'offsets.csv' not in tracked(context_dir)
    assert (context_dir / 'offsets.csv').exists()


def test_existing_history_exclude_file_preserved(context_dir):
    from damnit.backend import context_history

    assert checkpoint_context(context_dir)
    exclude_file = context_dir / '.damnit-history.git/info/exclude'
    template = Path(context_history.__file__).with_name('context_history.gitignore')
    assert exclude_file.read_bytes() == template.read_bytes()
    customized = '# Custom history rules\nprivate.csv\n'
    exclude_file.write_text(customized)
    (context_dir / 'private.csv').write_text('secret\n')
    (context_dir / 'calibration.h5').write_text('values\n')
    assert checkpoint_context(context_dir)
    assert exclude_file.read_text() == customized
    assert tracked(context_dir) == {'context.py', 'calibration.h5'}


@pytest.mark.parametrize('ignored_directory', ['.venv', 'process_logs', 'scratch'])
def test_ignored_directories_not_probed(context_dir, monkeypatch, ignored_directory):
    ignored = context_dir / ignored_directory
    ignored.mkdir()
    (ignored / 'support.py').write_text('unreadable\n')
    (context_dir / 'helpers.py').write_text('offset = 2\n')
    expected = {'context.py', 'helpers.py'}
    if ignored_directory == 'scratch':
        (context_dir / '.gitignore').write_text('scratch/\n')
        expected.add('.gitignore')

    def deny_probes(original):
        def probe(path, *args, **kwargs):
            if path.is_relative_to(ignored):
                raise PermissionError(f'Cannot inspect {path}')
            return original(path, *args, **kwargs)
        return probe

    # Python 3.13 propagates permission errors from these marker probes. Git
    # should prune ignored directories without Python inspecting their contents.
    monkeypatch.setattr(Path, 'is_file', deny_probes(Path.is_file))
    monkeypatch.setattr(Path, 'is_dir', deny_probes(Path.is_dir))
    assert checkpoint_context(context_dir)
    assert tracked(context_dir) == expected


def test_existing_repository_untouched(context_dir):
    git(context_dir, 'init', private=False)
    git(context_dir, 'add', 'context.py', private=False)
    git(context_dir, '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
        'commit', '-m', 'Manual commit', private=False)
    (context_dir / 'context.py').write_text('x = 2\n')
    git(context_dir, 'add', 'context.py', private=False)
    head = git(context_dir, 'rev-parse', 'HEAD', private=False)
    index = (context_dir / '.git/index').read_bytes()
    index_lock = context_dir / '.git/index.lock'
    index_lock.write_bytes(b'External Git operation\n')
    assert checkpoint_context(context_dir)
    assert git(context_dir, 'rev-parse', 'HEAD', private=False) == head
    assert (context_dir / '.git/index').read_bytes() == index
    assert index_lock.read_bytes() == b'External Git operation\n'
    assert '.git' not in tracked(context_dir)


def test_inherited_git_environment_isolated(context_dir, tmp_path, monkeypatch):
    foreign = tmp_path.parent / f'{tmp_path.name}-foreign'
    foreign.mkdir()
    git(foreign, 'init', private=False)
    index = foreign / 'foreign-index'
    monkeypatch.setenv('GIT_DIR', str(foreign / '.git'))
    monkeypatch.setenv('GIT_WORK_TREE', str(foreign))
    monkeypatch.setenv('GIT_INDEX_FILE', str(index))
    monkeypatch.setenv('GIT_OBJECT_DIRECTORY', str(foreign / '.git/objects'))
    assert checkpoint_context(context_dir)
    assert not index.exists()
    assert not (foreign / '.git/refs/heads/master').exists()
    assert not (foreign / '.git/refs/heads/main').exists()


@pytest.mark.parametrize('executor_type', [ThreadPoolExecutor, ProcessPoolExecutor])
@pytest.mark.parametrize('stale_lock', [False, True])
def test_concurrent_checkpoints(context_dir, executor_type, stale_lock):
    if stale_lock:
        first = checkpoint_context(context_dir)
        assert first
        (context_dir / '.damnit-history.git/index.lock').write_bytes(b'interrupted staging\n')
        (context_dir / 'context.py').write_text('x = 2\n')
        (context_dir / 'helpers.py').write_text('offset = 3\n')
    options = {}
    if executor_type is ProcessPoolExecutor:
        options['mp_context'] = multiprocessing.get_context('spawn')
    with executor_type(max_workers=4, **options) as executor:
        commits = list(executor.map(checkpoint_context, [context_dir] * 8))
    assert commits[0] and len(set(commits)) == 1
    assert git(context_dir, 'rev-list', '--count', 'HEAD') == ('2' if stale_lock else '1')
    assert not (context_dir / '.damnit-history.git/index.lock').exists()
    if stale_lock:
        assert commits[0] != first
        assert git(context_dir, 'show', f'{first}:context.py') == 'x = 1'
        assert git(context_dir, 'show', 'HEAD:context.py') == 'x = 2'
        assert git(context_dir, 'show', 'HEAD:helpers.py') == 'offset = 3'


def test_staging_timeout_recovers_for_next_checkpoint(context_dir, monkeypatch, caplog):
    from damnit.backend import context_history

    first = checkpoint_context(context_dir)
    assert first
    (context_dir / 'context.py').write_text('x = 2\n')
    (context_dir / 'helpers.py').write_text('offset = 3\n')
    (context_dir / '.gitattributes').write_text('helpers.py filter=slow\n')
    git(context_dir, 'config', 'filter.slow.clean', 'sleep 1; cat')
    run = subprocess.run
    index_lock = context_dir / '.damnit-history.git/index.lock'
    timeout_locks = []

    def shorten_staging_timeout(command, **kwargs):
        if 'add' in command:
            kwargs['timeout'] = 0.2
        try:
            return run(command, **kwargs)
        except subprocess.TimeoutExpired:
            timeout_locks.append(index_lock.exists())
            raise

    with monkeypatch.context() as patch:
        patch.setattr(context_history.subprocess, 'run', shorten_staging_timeout)
        assert checkpoint_context(context_dir) is None
    assert 'timed out' in caplog.text
    assert timeout_locks == [True]  # Real Git left a lock before timeout cleanup.
    assert not index_lock.exists()
    git(context_dir, 'config', '--remove-section', 'filter.slow')
    second = checkpoint_context(context_dir)
    assert second and second != first
    assert not (context_dir / '.damnit-history.git/index.lock').exists()
    assert git(context_dir, 'rev-list', '--count', 'HEAD') == '2'
    assert git(context_dir, 'show', f'{first}:context.py') == 'x = 1'
    assert git(context_dir, 'show', 'HEAD:context.py') == 'x = 2'
    assert git(context_dir, 'show', 'HEAD:helpers.py') == 'offset = 3'


def test_symlinks_and_explicitly_ignored_embedded_repositories(context_dir):
    outside = context_dir.parent / f'{context_dir.name}-external.csv'
    outside.write_text('outside\n')
    (context_dir / 'calibration.csv').symlink_to(outside)
    nested = context_dir / 'nested'
    nested.mkdir()
    git(nested, 'init', private=False)
    (nested / 'private.py').write_text('secret = 1\n')
    (context_dir / '.gitignore').write_text('nested/\n')
    first = checkpoint_context(context_dir)
    assert first
    assert tracked(context_dir) == {'.gitignore', 'context.py', 'calibration.csv'}
    assert git(context_dir, 'ls-tree', 'HEAD', 'calibration.csv').startswith('120000 blob')
    assert git(context_dir, 'show', 'HEAD:calibration.csv') == str(outside)
    outside.write_text('edited outside\n')
    assert checkpoint_context(context_dir) == first


def test_custom_environment_marker_requires_explicit_ignore(context_dir):
    directory = context_dir / 'custom-environment'
    directory.mkdir()
    (directory / 'support.py').write_text('offset = 2\n')
    assert checkpoint_context(context_dir)
    assert 'custom-environment/support.py' in tracked(context_dir)
    (directory / 'pyvenv.cfg').write_text('home = /usr/bin\n')
    assert checkpoint_context(context_dir)
    assert 'custom-environment/support.py' in tracked(context_dir)
    (context_dir / '.gitignore').write_text('custom-environment/\n')
    assert checkpoint_context(context_dir)
    assert tracked(context_dir) == {'context.py', '.gitignore'}


def test_context_directory_environment_uses_explicit_ignores(context_dir):
    (context_dir / '.gitignore').write_text(
        'pyvenv.cfg\nbin/\ninclude/\nlib/\nshare/\n'
    )
    for filename in ['pyvenv.cfg', 'bin/python', 'include/header.h',
                     'lib/python3.13/site-packages/module.py', 'share/man/tool']:
        path = context_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('environment\n')
    (context_dir / 'helpers.py').write_text('offset = 2\n')
    assert checkpoint_context(context_dir)
    assert tracked(context_dir) == {'.gitignore', 'context.py', 'helpers.py'}


def test_history_metadata_symlink_rejected(context_dir, caplog):
    target = context_dir / 'unrelated'
    target.mkdir()
    (context_dir / '.damnit-history.git').symlink_to(target, target_is_directory=True)
    assert checkpoint_context(context_dir) is None
    assert 'symlink' in caplog.text
    assert not list(target.iterdir())


def test_history_permissions_allow_shared_writers(context_dir):
    assert checkpoint_context(context_dir)
    for filename in ['.damnit-history.lock', '.damnit-history.git/info/exclude']:
        assert (context_dir / filename).stat().st_mode & 0o666 == 0o666
    assert (context_dir / '.damnit-history.git').stat().st_mode & 0o222 == 0o222


def test_git_missing_warns_without_raising(context_dir, monkeypatch, caplog):
    monkeypatch.setenv('PATH', '')
    assert checkpoint_context(context_dir) is None
    assert caplog.records and any(record.levelname == 'WARNING' for record in caplog.records)


def test_permission_error_warns_without_raising(context_dir, monkeypatch, caplog):
    from damnit.backend import context_history

    def denied(*args, **kwargs):
        raise PermissionError('read-only context directory')

    monkeypatch.setattr(context_history.os, 'open', denied)
    assert checkpoint_context(context_dir) is None
    assert 'read-only' in caplog.text


@pytest.mark.parametrize('error', [
    subprocess.CalledProcessError(1, ['git'], stderr=b'Git failed'),
    subprocess.TimeoutExpired(['git'], 30),
])
def test_git_failure_warns_without_raising(context_dir, monkeypatch, caplog, error):
    from damnit.backend import context_history

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(context_history.subprocess, 'run', fail)
    assert checkpoint_context(context_dir) is None
    assert any(record.levelname == 'WARNING' for record in caplog.records)


def test_editor_context_validation_does_not_checkpoint(context_dir, monkeypatch):
    from damnit.backend import extract_data

    def evaluate(command, **kwargs):
        Path(command[-1]).write_bytes(pickle.dumps((None, None)))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(extract_data.subprocess, 'run', evaluate)
    extract_data.get_context_file(context_dir / 'context.py')
    assert not (context_dir / '.damnit-history.git').exists()


@pytest.mark.parametrize('run', [False, True])
@pytest.mark.parametrize('git_available', [False, True])
def test_extractor_checkpoints_before_context_load(
    context_dir, monkeypatch, caplog, run, git_available
):
    from damnit.backend import extract_data

    monkeypatch.chdir(context_dir)
    db = MagicMock()
    db.metameta = {'proposal': 1234}
    db.path = context_dir / 'runs.sqlite'
    monkeypatch.setattr(extract_data, 'DamnitDB', lambda: db)
    monkeypatch.setattr(extract_data, 'kafka_producer', MagicMock())
    if not git_available:
        monkeypatch.setenv('PATH', '')

    def load_context(*args, **kwargs):
        if git_available:
            assert git(context_dir, 'show', 'HEAD:context.py') == 'x = 1'
        else:
            assert 'Could not checkpoint context' in caplog.text
        return MagicMock(), None

    monkeypatch.setattr(extract_data, 'get_context_file', load_context)
    if run:
        extract_data.RunExtractor(1234, 42)
    else:
        extract_data.Extractor()


def test_invalid_context_preserved(context_dir, monkeypatch):
    from damnit.backend import extract_data

    (context_dir / 'context.py').write_text('invalid python !!!\n')
    monkeypatch.chdir(context_dir)
    db = MagicMock()
    db.metameta = {'proposal': 1234}
    db.path = context_dir / 'runs.sqlite'
    monkeypatch.setattr(extract_data, 'DamnitDB', lambda: db)
    monkeypatch.setattr(extract_data, 'kafka_producer', MagicMock())
    monkeypatch.setattr(extract_data, 'get_context_file',
                        lambda *a, **kw: (None, ('syntax error',)))
    with pytest.raises(RuntimeError, match='syntax error'):
        extract_data.Extractor()
    assert git(context_dir, 'show', 'HEAD:context.py') == 'invalid python !!!'
