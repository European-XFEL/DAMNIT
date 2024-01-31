# DAMNIT

[![Documentation Status](https://readthedocs.org/projects/damnit/badge/?version=latest)](https://damnit.readthedocs.io/en/latest/?badge=latest)

## Installation
```bash
# Make an environment for DAMNIT
git clone https://github.com/European-XFEL/DAMNIT.git
cd DAMNIT
conda create -n amore python

conda activate amore
pip install .
```

## Usage
Copy `context.py` to new folder (e.g. within proposal usr or scratch).
This 'context dir' is also where it will save selected data. Edit `context.py`
to define what data is interesting.

To listen for new runs and extract data:
```bash
conda activate amore
cd /path/to/proposal/scratch/amore  # Context directory
amore-proto proposal 1234
amore-proto listen .
```

The backend will write its logs to stdout, and also a file called `amore.log`
in the same directory.

To launch the GUI overview (e.g. on [max-exfl-display](https://max-exfl-display.desy.de:3389)):
```bash
conda activate amore
amore-proto gui
```

You can open the context dir inside the GUI ('autoconfiguration'), or pass it
at the command line.

## Kafka
The GUI is updated by Kafka messages sent by the backend. Currently we use
XFEL's internal Kafka broker at `exflwgs06.desy.de:9091`, but this is only
accessible inside the DESY network.

DAMNIT can run offline, but if you want updates from the backend and you're
running DAMNIT outside the network and not using a VPN, you'll first have to
forward the broker port to your machine:
```bash
ssh -L 9091:exflwgs06.desy.de:9091 max-exfl.desy.de
```

And then add a line in your `/etc/hosts` file to resolve `exflwgs06.desy.de`
to `localhost` (remember to remove it afterwards!):
```
127.0.0.1 exflwgs06.desy.de
```

And then DAMNIT should be able to use XFELs broker. If you want to use a specific
broker you can set the `AMORE_BROKER` variable:
```bash
export AMORE_BROKER=localhost:9091
```

DAMNIT will then connect to the broker at that address.

## Deployment on Maxwell
DAMNIT is deployed in a module on Maxwell:
```bash
$ module load exfel amore
```

There was an idea in the beginning to potentially have multiple submodules for
different instruments depending on how many people worked on features for
instruments simultaneously, but so far we've just stuck with a single submodule
for MID, `exfel amore/mid`, which is loaded by default with `exfel
amore`. To update the module:
```bash
$ ssh xsoft@max-exfl.desy.de

# Helper command to cd into the module directory and activate its environment
$ amoremod mid
$ git pull # Or whatever command is necessary to update the code

# Only necessary if updating dependencies since DAMNIT is installed in editable mode
$ pip install -e .
```
