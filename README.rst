AMORE prototype for MID
=======================

Installation::

    # Make a virtualenv
    module load maxwell anaconda-python/3.8
    git clone https://git.xfel.eu/amore/mid-2832.git
    cd mid-2832
    python -m venv env

    env/bin/pip install .

Usage:

Copy ``context.py`` to new folder (e.g. within proposal usr or scratch).
This 'context dir' is also where it will save selected data. Edit ``context.py``
to define what data is interesting.

To listen for new runs and extract data::

    source env/bin/activate
    cd /path/to/proposal/scratch/amore  # Context directory
    amore-proto listen

To launch the GUI overview (e.g. on `max-display <https://max-display.desy.de:3443/>`_)::

    source env/bin/activate
    amore-proto gui

You can open the context dir inside the GUI ('autoconfiguration'), or pass it
at the command line.

Kafka
------
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
