import socket

def get_supervisord_address(default_port=2322):
    """Find an available port to bind to."""
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)

    port = default_port
    with socket.socket() as s:
        while True:
            try:
                s.bind((ip, port))
            except OSError:
                port += 1
            else:
                break

    return hostname, port
