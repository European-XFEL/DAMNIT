# DAMNIT

Welcome to the DAMNIT user documentation! The name 'DAMNIT' is
definitely-not-a-backronym for *Data And Metadata iNspection Interactive
Thing*. It's a tool developed by the [Data Analysis
group](https://www.xfel.eu/data_analysis) at the European XFEL to provide users
with a way to automatically create an overview of their experiment, hopefully
replacing the manually-created spreadsheets that are often used.

Sneak peak:
![](static/beauty-shot.png)

Note: DAMNIT was previously named AMORE, and there still places in the
application where you'll see this name used instead of DAMNIT.

## Design overview
There are two parts to DAMNIT: the frontend GUI and the backend. The GUI is
currently written in PyQt and will be moved to a web interface at some
point. The backend is a service that runs on XFEL's offline cluster for each
proposal, so one instance of the backend for one proposal is completely separate
from any others and can be started by any user.

To fill up the table you see in the screenshot above, the backend executes what
we call a *context file*, which is simply a Python file that contains *variable*
definitions. A *variable* in DAMNIT is a 'thing' that you want to track during
the experiment, and the values for each variable for each run can either be
generated automatically [from the context file](backend.md), or entered manually
[into the GUI](gui.md#adding-user-editable-variables).

When a new run is taken, the files are migrated from the online cluster to the
offline cluster and this triggers the backend to execute any variables in the
context file that use `raw` data. If calibrated data is used, the offline
calibration pipeline will also trigger the backend to execute any variables that
require `proc` data. When the variables are processed they will be saved in
DAMNIT's internal database, and in HDF5 files (which may be used directly by the
user).

Whenever the backend finishes processing some variables it sends updates to any
open GUIs, so the effect is that you see an automatically-updating table of
runs with variables coming from code that you can modify in the context file;
which gives an enormous amount of flexibility to customize the variables for
your experiment.
