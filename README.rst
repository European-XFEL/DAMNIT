AMORE prototype for MID
=======================

Installation::

    # Make a virtualenv
    module load maxwell anaconda-python/3.8
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
