# Changelog

## [0.1.4]

Only API change: added a deserialize_plotly option to read() to allow reading the Plotly JSON string.

Added:

- (Re)process runs from the GUI (!285).
- Add support for optionally returning the Plotly JSON string (!328).

Changed:

- Check context file without blocking GUI thread (!291).
- Defer some imports in damnit.api (!281).
- Evaluate the context file in the database directory when validating (!301).
- Update default Python for context file to 202402 environment (!315).
- xarray plot cbar range limit (!312).
- Refactor GUI plotting code (!316).

Fixed:

- reprocessing from the GUI when the context dir is not the CWD (!302).
- Add back log message on successful variable computation (!303).
- allow plotting variable with Cell's custom summary value (!306).
- passing through --var options to cluster job (!320).

[Full Changelog](https://github.com/European-XFEL/DAMNIT/compare/0.1.3...0.1.4)

## [0.1.3]

Added:

- Log timing after each successful variable computation (!247).
- Show tooltip with full size thumbnail for image items (!263).
- Save plotly figure (!262).
- Add an option to force software OpenGL (!268).
- Templates for setting up new context files (!248).
- Table cell formatting with a Cell class (!221).
- Add a database config option to skip starting a listener (!277).
- Add back --direct option to run reprocessing directly on local node (!287).
- Plot xarray (!289).

Changed:

- Preserve context file path for tracebacks (!249).
- Improve error handling when opening variables in the GUI (!257).
- Evaluate non-cluster variables in Slurm using Solaris cluster (!256).
- Use Slurm jobs for reprocessing (!270).
- HDF5 array compression (!273).
- Simplify testing CLI by passing argv in (!279).
- Defer imports for context file runner code (!280).
- Add support for Python 3.12 (!283).
- Enable coverage reporting in CI (!293).

Fixed:

- the API examples (!244)
- Various fixes & improvements to image results (!241).
- Handle standalone comments properly when plotting (!254).
- Explicitly specify maxwell cluster when submitting larger jobs (!266).
- renaming data vars in xarray DataSet objects before saving to HDF5 (!272).
- Prevent variable results with dtypes that cannot be converted to HDF5 (!271).
- returning e.g. bool from variable functions (!276).
- Try to fix broken tests (!278).
- Promote variables to cluster=True if their dependencies are (!286).
- Save QAction's as class variables (!290).

[Full Changelog](https://github.com/European-XFEL/DAMNIT/compare/0.1.2...0.1.3)

## [0.1.2]

Fixed:

- API support for comments and timestamps (!242).

[Full Changelog](https://github.com/European-XFEL/DAMNIT/compare/0.1.1...0.1.2)

## [0.1.1]

Fixed:

- API dependencies (!239)

[Full Changelog](https://github.com/European-XFEL/DAMNIT/compare/0.1...0.1.1)

## [0.1]

Initial release of the DAMNIT Python API.

Added:

- Dialog to select proposal/directory when starting GUI (!9).
- Add support for mock runs with --mock for the reprocess command (!19).
- Slurm improvements (!21).
- Set the aspect ratio of images automatically based on their shape (!22).
- Allow setting default arguments for dependencies (!23).
- Jump to run, auto-scroll and run-verticalHeader (!73).
- Zulip support (!76).
- Add support for exporting the table to a CSV/Excel file (!24).
- Improve support for executing a context file in another environment (!111).
- Make per-run logs accessible from the GUI (!128).
- Run DAMNIT on the online cluster (!133).
- Feat/add container cd (!142).
- Database v1 (!134).
- Add summary method (e.g. 'mean', 'sum') to database (!167).
- Migrate the `time_comments` and `variables` tables (!169).
- Add a migration for the short-lived intermediate v1 format (!176).
- Add support for deleting variables (!180).
- Add maximize / minimize buttons to plot dialog (!190).
- Store variables from context file in database (!203).
- Add the ability to pre-create runs in the database (!209).
- Add a Python console for debugging (!214).
- Inspect 2D summarized data (!219).
- Add support for returning Axes from variables (!223).
- Mymdc proxy (!226).
- Create a workflow for automatically publishing new releases (!238).

Changed:

- Migrate CI to GitHub workflow (!2).
- Switch to mpl-pan-zoom (!17).
- Swap the plotting widgets axis order so they make sense (!88).
- Set a default summary for the number of pulses (!82).
- Skip non-existent runs when reprocessing (!89).
- Convert - to _ for config keys in db-config command (!116).
- Reorganise run-processing logs (!118).
- Add reload button to the editor (!119).
- Add a newline to the log before reprocessing a run (!122).
- Log the --match argument during reprocessing as well (!126).
- Refuse to open old databases (except to migrate) (!166).
- Listen for incoming events on EuXFEL Kafka broker (!181).
- Refactor GUI table machinery (!192).
- Refactor: separate ctxrunner from database (!193).
- Delete variable in GUI without reloading entire table (!217).
- Skip over Variables returning unexpected types (!220).
- Refactor: base GUI table on QStandardItemModel (!225).

Fixed:

- Make slurm jobs respect --match (!64).
- Ensure that proc variables have access to raw data in the run object (!65).
- Only attempt to change file permissions when the user is the owner (!67).
- Handle all exceptions when checking for proc data (!104).
- --mock (!112).
- a context file example (!173).
- call to get_column_states() while exporting to Zulip (!201).
- filtering of disabled runs (!202).
- getting POSIX timestamps from files (!105).
- Handle standalone comments properly (!170).
- migration of old DataArray's (!204).
- migration with scalar coordinates from DataArrays (!189).
- relative heights of the editor widgets (!123).
- selecting rows for sending table to Zulip (!207).
- sending updates containing PNG thumbnails (!182).
- Don't try to find max_diff from empty datasets (!187).
- some warnings that occur when running the tests (!197).
- sorting by column in the GUI (!206).
- the DPI calculation for creating thumbnails (!175).
- the FastX link (!75).
- the instructions for connecting to XFELs broker offline (!20).
- hidden header when loading a new proposal in a running GUI (!227).
- kafka event for offline correction (!136).
- setup/open context dir on the online cluster (!236).
- subtraction fails on arrays of boolean (!195).

[Full Changelog](https://github.com/European-XFEL/DAMNIT/commits/0.1)


[0.1.4]: https://github.com/European-XFEL/DAMNIT/releases/tag/0.1.4
[0.1.3]: https://github.com/European-XFEL/DAMNIT/releases/tag/0.1.3
[0.1.2]: https://github.com/European-XFEL/DAMNIT/releases/tag/0.1.2
[0.1.1]: https://github.com/European-XFEL/DAMNIT/releases/tag/0.1.1
[0.1]: https://github.com/European-XFEL/DAMNIT/releases/tag/0.1
