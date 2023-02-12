AMORE prototype for MID
=======================

Installation::

    # Make an environment for AMORE
    git clone https://git.xfel.eu/amore/mid-2832.git
    cd mid-2832
    conda create -n amore python

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
backend automatically, using `supervisor <http://supervisord.org>`_ under the
hood.

In a nutshell:

- ``supervisord`` will manage the backend using a configuration file named
  ``supervisord.conf`` stored in the database directory. It's configured to
  listen for commands over HTTP on a certain port with a certain
  username/password. ``supervisord`` will save its logs to ``supervisord.log``.
- It can be controlled with ``supervisorctl`` on any machine using the same
  config file.

So lets say you're running the GUI on FastX, and the backend is now started. If
you open a terminal and ``cd`` to the database directory you'll see::

    $ cd /gpfs/path/to/proposal/usr/Shared/amore
    $ ls
    amore.log  context.py  extracted_data  runs.sqlite  supervisord.conf  supervisord.log

You could get the status of the backend with::

    $ supervisorctl -c supervisord.conf status damnit
    damnit                           RUNNING   pid 3793870, uptime 0:00:20

And you could restart it with::

    $ supervisorctl -c supervisord.conf restart damnit
    damnit: stopped
    damnit: started

    $ supervisorctl -c supervisord.conf status damnit
    damnit                           RUNNING   pid 3793880, uptime 0:00:04

Kafka
-----
The GUI is updated by Kafka messages sent by the backend. Currently we use
XFEL's internal Kafka broker at ``exflwebstor01.desy.de:9102``, but this is only
accessible inside the control network.

AMORE can run offline, but if you want updates from the backend and you're
running AMORE outside the network and not using a VPN, you'll first have to
forward the broker port to your machine::

    ssh -L 9102:exflwebstor01.desy.de:9102 max-exfl.desy.de

And then add a line in your ``/etc/hosts`` file to resolve
``exflwebstor01.desy.de`` to ``localhost``::

    127.0.0.1 exflwebstor01.desy.de

And then AMORE should be able to use XFELs broker.

If you want to use a specific broker you can set the ``AMORE_BROKER`` variable::

    export AMORE_BROKER=localhost:9102


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
