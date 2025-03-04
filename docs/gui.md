# GUI

## Open the GUI
1. Start by opening a [FastX](https://max-exfl-display.desy.de:3389) XFCE
   session. You can also SSH to `max-exfl.desy.de` with X forwarding enabled
   but that's not recommended because it can be slow.
2. Open a terminal and run:
   ```bash
   $ module load exfel amore
   $ damnit gui
   ```
3. If you want to work with the shared database for a proposal you can enter the
   proposal number directly (it will prompt you to create one if it doesn't
   already exist):
   ![](static/damnit-open-by-proposal.gif)

   Otherwise if you're playing around and want to make a test database, you can
   create one manually like this:

```bash
# Make a directory for the database
$ mkdir mydb
$ cd mydb

# Initialize the database for a certain proposal
$ damnit proposal 1234
# Start the backend and create a default context file
$ damnit listen . --daemonize

# Start the GUI
$ damnit gui .
```

## Exploring variables
The interface is organized around a central table of runs that contain
*variables*, which are the things computed by the functions in the context file
(or entered by users). Each variable gets its own column, and variables can be
hidden or shown with the checkboxes on the right.

Features:

- Hide/show and move columns using the controls on the right:
  ![](static/columns.gif)
- To add a comment about a specific run (e.g. "Very good signal, we're really
  nailing this science thing."), click and edit its `Comment` column.
- To add an additional comment not related to a specific run (e.g. "Beam went
  down. Devastated."), write something in the text input widget at the bottom
  and click the `Additional comment` button. These comments are arranged by
  time, which you can also adjust before adding one.
- Variables are 'summarized' in the table but they can be arrays. If you
  double-click on a variable for a row and it has train-resolved data, it will
  be plotted. For example if you double-click on a variable for the XGM
  intensity of a run, you might see something like:
  ![](static/inspect-arrays.png)

- To plot one variable vs another, select some variables in the bottom right
  hand corner and click one of the plotting buttons:
  ![](static/plotting-controls.png)
  Click the 'vs.' button to swap the axes. The `Plot summary for all runs`
  button will plot summary variables vs each other, for example you can plot the
  run number vs the XGM intensity to see a plot of the beam intensity over the
  experiment.
  ![](static/plot-summaries.png)

- But you can also plot variables with train-resolved data against each other
  within a certain run, which could be useful to visualize scans:
  ![](static/plot-for-selected-runs.gif)
- There is also some (very basic) support for histogramming single variables
  with the `Histogram` button (click to enable/disable it):
  ![](static/histogramming.gif)

- The table's `Cells` can be highlighted with custom background colors and text
  formatting for emphasis, see [Cell configuration](backend.md#cell) for more
  information.


## Adding user-editable variables
Ideally it would be possible to define all variables as code, but sometimes
that's just not possible. Hence, DAMNIT allows you to create user-editable
variables straight from the GUI which you can edit directly. This is good for
anything that cannot be computed automatically from saved files, such as sample
information or a note about the run.

You can add a user-editable variable from the menu in the top left:
![](static/user-editable-variables.gif)

## Tag Filtering
Variables can be tagged in the context file to organize them into logical groups
(e.g., by detector, data type, or processing stage). The GUI provides an easy
way to filter variables based on these tags by using the "Variables by Tag" button.

Tags are preserved in the database and are automatically updated when you modify
them in the context file. For information on how to define tags for variables,
see the [Variables](backend.md#variables) section.

## Exporting
All the data in the run table can be exported to an Excel/CSV file, with the
caveat that images will not be exported (they'll be replaced with an `<image>`
string):
![](static/export.gif)
