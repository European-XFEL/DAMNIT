# Changelog

## [0.1.4]

2024-09-03

Only API change: added a deserialize_plotly option to read() to allow reading the Plotly JSON string.

Added:

- API: Add support for optionally returning the Plotly JSON string (!328).
- GUI: (Re)process runs from the GUI (!285).

Changed:

- GUI: Check context file without blocking GUI thread (!291).
- GUI: xarray plot cbar range limit (!312).
- Update default Python for context file to 202402 environment (!315).

Fixed:

- Add back log message on successful variable computation (!303).
- allow plotting variable with Cell's custom summary value (!306).
- passing through --var options to cluster job (!320).

[Full Changelog](https://github.com/European-XFEL/DAMNIT/compare/0.1.3...0.1.4)

## [0.1.3]

2024-07-25

Added:

- GUI: Show tooltip with full size thumbnail for image items (!263).
- GUI: Add an option to force software OpenGL (!268).
- GUI: Templates for setting up new context files (!248).
- GUI: Table cell formatting with a Cell class (!221).
- GUI: Plot xarray (!289).
- Log timing after each successful variable computation (!247).
- Save plotly figure (!262).
- Add a database config option to skip starting a listener (!277).
- Add back --direct option to run reprocessing directly on local node (!287).

Changed:

- Preserve context file path for tracebacks (!249).
- Improve error handling when opening variables in the GUI (!257).
- Evaluate non-cluster variables in Slurm using Solaris cluster (!256).
- Use Slurm jobs for reprocessing (!270).
- HDF5 array compression (!273).
- Add support for Python 3.12 (!283).

Fixed:

- the API examples (!244)
- Various fixes & improvements to image results (!241).
- Handle standalone comments properly when plotting (!254).
- Explicitly specify maxwell cluster when submitting larger jobs (!266).
- renaming data vars in xarray DataSet objects before saving to HDF5 (!272).
- Prevent variable results with dtypes that cannot be converted to HDF5 (!271).
- returning e.g. bool from variable functions (!276).
- Promote variables to cluster=True if their dependencies are (!286).

[Full Changelog](https://github.com/European-XFEL/DAMNIT/compare/0.1.2...0.1.3)

## [0.1.2]

2024-04-26

Fixed:

- API support for comments and timestamps (!242).

[Full Changelog](https://github.com/European-XFEL/DAMNIT/compare/0.1.1...0.1.2)

## [0.1.1]

2024-04-24

Fixed:

- API dependencies (!239)

[Full Changelog](https://github.com/European-XFEL/DAMNIT/compare/0.1...0.1.1)

## [0.1]

2024-04-23

Initial release of the DAMNIT Python API.

Added:

- GUI: Dialog to select proposal/directory when starting GUI (!9).
- GUI: Set the aspect ratio of images automatically based on their shape (!22).
- GUI: Jump to run, auto-scroll and run-verticalHeader (!73).
- GUI: Zulip support (!76).
- GUI: Add support for exporting the table to a CSV/Excel file (!24).
- GUI: Make per-run logs accessible from the GUI (!128).
- GUI: Add support for deleting variables (!180).
- GUI: Add maximize / minimize buttons to plot dialog (!190).
- GUI: Add the ability to pre-create runs in the database (!209).
- GUI: Add a Python console for debugging (!214).
- GUI: Inspect 2D summarized data (!219).
- Add support for mock runs with --mock for the reprocess command (!19).
- Allow setting default arguments for dependencies (!23).
- Run DAMNIT on the online cluster (!133).
- Add summary method (e.g. 'mean', 'sum') to database (!167).
- Migrate the `time_comments` and `variables` tables (!169).
- Add a migration for the short-lived intermediate v1 format (!176).
- Store variables from context file in database (!203).
- Improve support for executing a context file in another environment (!111).
- Add support for returning Axes from variables (!223).
- Mymdc proxy (!226).

Changed:

- GUI: Swap the plotting widgets axis order so they make sense (!88).
- GUI: Add reload button to the editor (!119).
- GUI: Delete variable in GUI without reloading entire table (!217).
- Skip non-existent runs when reprocessing (!89).
- Reorganise run-processing logs (!118).

Fixed:

- GUI: call to get_column_states() while exporting to Zulip (!201).
- GUI: filtering of disabled runs (!202).
- GUI: Handle standalone comments properly (!170).
- GUI: relative heights of the editor widgets (!123).
- GUI: selecting rows for sending table to Zulip (!207).
- GUI: sorting by column in the GUI (!206).
- GUI: hidden header when loading a new proposal in a running GUI (!227).
- Make slurm jobs respect --match (!64).
- Ensure that proc variables have access to raw data in the run object (!65).
- Only attempt to change file permissions when the user is the owner (!67).
- Handle all exceptions when checking for proc data (!104).
- --mock (!112).
- getting POSIX timestamps from files (!105).
- migration of old DataArray's (!204).
- migration with scalar coordinates from DataArrays (!189).
- sending updates containing PNG thumbnails (!182).
- Don't try to find max_diff from empty datasets (!187).
- the DPI calculation for creating thumbnails (!175).
- kafka event for offline correction (!136).
- setup/open context dir on the online cluster (!236).
- subtraction fails on arrays of boolean (!195).
- Skip over Variables returning unexpected types (!220).

[Full Changelog](https://github.com/European-XFEL/DAMNIT/commits/0.1)


[0.1.4]: https://github.com/European-XFEL/DAMNIT/releases/tag/0.1.4
[0.1.3]: https://github.com/European-XFEL/DAMNIT/releases/tag/0.1.3
[0.1.2]: https://github.com/European-XFEL/DAMNIT/releases/tag/0.1.2
[0.1.1]: https://github.com/European-XFEL/DAMNIT/releases/tag/0.1.1
[0.1]: https://github.com/European-XFEL/DAMNIT/releases/tag/0.1