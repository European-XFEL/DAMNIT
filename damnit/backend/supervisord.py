import os
import time
import shutil
import socket
import secrets
import logging
import subprocess
import configparser
from pathlib import Path

from .db import db_path, DamnitDB


log = logging.getLogger(__name__)


def wait_until(condition, timeout=1):
    """
    Re-evaluate `condition()` until it either returns true or we've waited
    longer than `timeout`.
    """
    slept_for = 0
    sleep_interval = 0.2

    while slept_for < timeout and not condition():
        time.sleep(sleep_interval)
        slept_for += sleep_interval

    if slept_for >= timeout:
        raise TimeoutError("Condition timed out")

def get_supervisord_address():
    """
    Find an available hostname and port for supervisord to bind to.
    """
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)

    sock = socket.socket()
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()

    return hostname, port

def backend_is_running(root_path: Path, timeout=1):
    config_path = root_path / "supervisord.conf"
    supervisorctl_status = ["supervisorctl", "-c", str(config_path), "status", "damnit"]

    # If supervisord isn't running or the program is stopped, the status
    # will return something non-zero. Whereas if it's still starting it will
    # return 0.
    if subprocess.run(supervisorctl_status).returncode != 0:
        return False

    n_polls = 10 if timeout > 0 else 1
    for _ in range(n_polls):
        stdout = subprocess.run(supervisorctl_status,
                                text=True, capture_output=True).stdout
        if "RUNNING" in stdout:
            return True

        time.sleep(timeout / n_polls)

    return False

def write_supervisord_conf(root_path):
    # Find an available address
    hostname, port = get_supervisord_address()
    username = secrets.token_hex(32)
    password = secrets.token_hex(32)

    # Create supervisord.conf
    config = configparser.ConfigParser()
    with open(Path(__file__).parent / "supervisord.conf", 'r') as f:
        config.read_file(f)
    config["inet_http_server"]["port"] = str(port)
    config["inet_http_server"]["username"] = username
    config["inet_http_server"]["password"] = password
    config["program:damnit"]["directory"] = str(root_path)
    config["supervisorctl"]["serverurl"] = f"http://{hostname}:{port}"
    config["supervisorctl"]["username"] = username
    config["supervisorctl"]["password"] = password

    config_path = root_path / "supervisord.conf"
    with open(config_path, "w") as f:
        config.write(f)

    if config_path.stat().st_uid == os.getuid():
        os.chmod(config_path, 0o666)

def start_backend(root_path: Path, try_again=True):
    config_path = root_path / "supervisord.conf"
    if not config_path.is_file():
        write_supervisord_conf(root_path)

    supervisorctl = ["supervisorctl", "-c", str(config_path)]
    rc = subprocess.run([*supervisorctl, "status", "damnit"]).returncode

    # 4 means that supervisorctl couldn't connect to supervisord and we
    # need to start supervisord.
    if rc == 4:
        # Write a new config file to make sure that the hostname and port are valid
        write_supervisord_conf(root_path)

        supervisord = ["supervisord", "-c", config_path]
        cmd = subprocess.run(supervisord,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True)

        if cmd.returncode != 0:
            log.error(f"Couldn't start supervisord, tried to run command: {' '.join(supervisord)}\n"
                      f"Return code: {cmd.returncode}"
                      f"Command output: {cmd.stdout}")
            return False

        if try_again:
            return start_backend(root_path, try_again=False)
    elif rc == 3:
        # 3 means it's stopped and we need to start the program
        cmd = subprocess.run([*supervisorctl, "start", "damnit"])
        if cmd.returncode != 0:
            log.error(f"Couldn't start supervisord, tried to run command: {' '.join(cmd)}\n"
                      f"Return code: {cmd.returncode}"
                      f"Command output: {cmd.stdout}")
            return False
    elif rc > 0:
        log.error(f"Unrecognized return code from supervisorctl: {rc}")
        return False

    # Make sure the PID and log file are writable by everyone in case a
    # different user restarts supervisord.
    pid_path = root_path / "supervisord.pid"
    log_path = root_path / "supervisord.log"
    try:
        wait_until(lambda: pid_path.is_file() and log_path.is_file(), timeout=5)
    except TimeoutError:
        log.error("supervisord did not start up properly")
        return False
    else:
        os.chmod(pid_path, 0o666)
        os.chmod(log_path, 0o666)

    return True

def initialize_and_start_backend(root_path, proposal=None, context_file_src=None, user_vars_src=None):
    # Ensure the directory exists
    root_path.mkdir(parents=True, exist_ok=True)
    if root_path.stat().st_uid == os.getuid():
        os.chmod(root_path, 0o777)

    # If the database doesn't exist, create it
    if new_db := not db_path(root_path).is_file():
        if proposal is None:
            raise ValueError("Must pass a proposal number to `initialize_and_start_backend()` if the database doesn't exist yet.")

        # Initialize database
        db = DamnitDB.from_dir(root_path)
        db.metameta["proposal"] = proposal
        db.metameta["context_python"] = "/gpfs/exfel/sw/software/euxfel-environment-management/environments/202502/.pixi/envs/default/bin/python"
    else:
        # Otherwise, load the proposal number
        db = DamnitDB.from_dir(root_path)
        proposal = db.metameta["proposal"]

    context_path = root_path / "context.py"
    # Copy initial context file if necessary
    if not context_path.is_file():
        if context_file_src is not None:
            shutil.copyfile(context_file_src, context_path)
        else:
            context_path.touch()
        os.chmod(context_path, 0o666)

    # Copy user editable variables if requested
    if new_db and (user_vars_src is not None):
        prev_db = DamnitDB(user_vars_src)
        for var in prev_db.get_user_variables().values():
            db.add_user_variable(var)

    # Start backend
    return start_backend(root_path)
