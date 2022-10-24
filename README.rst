AMORE prototype for MID
=======================

Installation::

    # Make an environment for AMORE
    git clone https://git.xfel.eu/amore/mid-2832.git
    cd mid-2832
    conda env create -f environment.yml

    conda activate amore
    pip install .

Usage:

Copy ``context.py`` to new folder (e.g. within proposal usr or scratch).
This 'context dir' is also where it will save selected data. Edit ``context.py``
to define what data is interesting.

To listen for new runs and extract data::

    conda activate amore
    cd /path/to/proposal/scratch/amore  # Context directory
    amore-proto proposal 1234
    amore-proto listen .

The backend will write its logs to stdout, and also a file called ``amore.log``
in the same directory.

To launch the GUI overview (e.g. on `max-display <https://max-display.desy.de:3443/>`_)::

    conda activate amore
    amore-proto gui

You can open the context dir inside the GUI ('autoconfiguration'), or pass it
at the command line.

Managing the backend
--------------------
The GUI is capable of initializing a database for a proposal and starting the
backend automatically. We do this by abusing tmux as a service manager, which is
possible because:

- tmux servers support Unix sockets.
- tmux is designed to run 'asynchronously', without a user having to be logged
  in with a terminal open.

So the rules are:

- The backend is run inside a tmux session named ``AMORE``.
- The tmux server is bound to a socket named ``amore-tmux.sock`` inside the
  database directory.
- If the socket exists, the GUI assumes that the backend is running. The backend
  is responsible for deleting the socket when it closes (tmux does not support
  that).
- If the socket does not exist, the GUI starts a tmux session (on the machine it
  is currently running on, this may be changed later) and runs the backend in
  it.

So lets say you're running the GUI on FastX, and the backend is now started on
``max-exfl-display002``. If you open a terminal and ``cd`` to the database
directory you'll see the tmux socket::

    $ cd /gpfs/path/to/proposal/usr/Shared/amore
    $ ls
    amore.log  amore-tmux.sock  context.py  extracted_data  runs.sqlite

You can list the sessions running under the socket::

    $ tmux -S amore-tmux.sock ls
    AMORE: 1 windows (created Thu Jun  2 13:21:13 2022) [190x41]

Or you can attach to the session (hit ``Ctrl + B, d`` to detach)::

    $ tmux -S amore-tmux.sock a

Troubleshooting
^^^^^^^^^^^^^^^
If you run ``tmux -S amore-tmux.sock ls`` and see ``failed to connect to
server``, the problem is likely that you're on the wrong machine. While the
sockets are accessible from anywhere thanks to GPFS, you still need to be on the
same machine as the tmux server process to communicate with it. The backend
prints the hostname it's running on when it starts; so by looking at
``amore.log`` for lines like ``Running on max-exfl-display002.desy.de under user
wrigleyj, PID 67427``, you should be able to figure out the right machine.

However, if you are on the right machine and the socket is still unusable (and
you verified with e.g. ``ps -aux | grep tmux | grep amore`` that tmux is still
running), then you probably need to tell tmux to recreate the socket. Why this
happens `I don't know
<https://stackoverflow.com/questions/9668763/why-am-i-getting-a-failed-to-connect-to-server-message-from-tmux-when-i-try-to>`_,
but you can fix it by sending ``USR1`` to the tmux process::

    kill -s USR1 <pid>

Tmux will then recreate the socket and you should be able to access it as
usual.

Kafka
-----
The GUI is updated by Kafka messages sent by the backend. Currently we use
XFEL's internal Kafka broker at ``exflwebstor01.desy.de:9102``, but this is only
accessible inside the control network.

AMORE can run offline, but if you want updates from the backend and you're
running AMORE outside the network and not using a VPN, you'll first have to
forward the broker port to your machine::

    ssh -L 9102:exflwebstor01.desy.de:9102 max-exfl.desy.de

And then set the ``AMORE_BROKER`` variable::

    export AMORE_BROKER=localhost:9102

AMORE will then connect to the broker at that address.

Deployment on Maxwell
---------------------
AMORE is deployed in a module on Maxwell::

    $ module load exfel amore

There was an idea in the beginning to potentially have multiple submodules for
different instruments depending on how many people worked on features for
instruments simultaneously, but so far we've just stuck with a single submodule
for MID, ``exfel amore/mid``, which is loaded by default with ``exfel
amore``. To update the module:

..  code-block:: bash

    $ ssh xsoft@max-exfl.desy.de

    # Helper command to cd into the module directory and activate its environment
    $ amoremod mid
    $ git pull # Or whatever command is necessary to update the code

    # Only necessary if updating dependencies since AMORE is installed in editable mode
    $ pip install -e .
