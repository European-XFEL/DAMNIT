"""Utilities for running services"""

import os
import errno
import socket

def notify(message):
    if not message:
        raise ValueError("notify() requires a message")

    if not (socket_path := os.environ.get("NOTIFY_SOCKET")):
        return  # Not started by systemd

    if socket_path[0] not in ("/", "@"):
        raise OSError(errno.EAFNOSUPPORT, "Unsupported socket type")

    # Handle abstract socket.
    if socket_path[0] == "@":
        socket_path = "\0" + socket_path[1:]

    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM | socket.SOCK_CLOEXEC) as sock:
        sock.connect(socket_path)
        sock.sendall(message)

def notify_ready():
    """Tell systemd the service has started successfully"""
    notify(b"READY=1")
