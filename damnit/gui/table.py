import json
import logging
import time
from base64 import b64encode
from itertools import groupby
from numbers import Number

import numpy as np
from PyQt6 import QtCore, QtGui
from PyQt6.QtCore import QProcess, Qt

from ..backend.db import (
    BlobTypes, DamnitDB, ReducedData, blob2complex, blob2numpy
)
from ..backend.extraction_control import ExtractionJobTracker
from ..util import timestamp2str
from .roles import LINE_DATA_ROLE, PROVENANCE_ROLE, UNITS_ROLE
from .util import StatusbarStylesheet

log = logging.getLogger(__name__)

THUMBNAIL_SIZE = (100, 35)  # w, h (pixels)
STATIC_COLUMN_IDS = ['Status', 'proposal', 'run', 'start_time', 'comment']


def best_text_color(background: QtGui.QColor) -> QtGui.QColor:
    brightness = (
        background.red() * 299
        + background.green() * 587
        + background.blue() * 114
    ) / 1000
    return QtGui.QColor(Qt.GlobalColor.black) if brightness >= 186 else QtGui.QColor(Qt.GlobalColor.white)


class DamnitTableModel(QtGui.QStandardItemModel):
    value_changed = QtCore.pyqtSignal(int, int, str, object)
    column_removed = QtCore.pyqtSignal(str)  # column_id
    time_comment_changed = QtCore.pyqtSignal(int, str)
    run_visibility_changed = QtCore.pyqtSignal(int, bool)

    def __init__(self, db: DamnitDB, column_settings: dict, parent, settings_by_id=True):
        self.column_ids, self.column_titles = self._load_columns(db, column_settings, settings_by_id)
        self.column_descriptions = db.variable_descriptions()
        n_run_rows = db.conn.execute("SELECT count(*) FROM run_info").fetchone()[0]
        log.info(f"Table will have {n_run_rows} runs")

        super().__init__(n_run_rows, len(self.column_ids), parent)
        self.setHorizontalHeaderLabels(self.column_titles)
        self._main_window = parent
        self.is_sorted_by = ""
        self.is_sorted_order = None
        self.db = db
        self.column_index = {c: i for (i, c) in enumerate(self.column_ids)}
        self.run_index = {}  # {(proposal, run): row}
        if settings_by_id:
            self.column_settings_loaded = {
                col_id: vis for col_id, vis in column_settings.items()
                if col_id in self.column_index
            }
        else:
            title_to_id = dict(zip(self.column_titles, self.column_ids))
            self.column_settings_loaded = {
                title_to_id[title]: vis
                for title, vis in column_settings.items()
                if title in title_to_id
            }

        self.processing_jobs = QtExtractionJobTracker(self)
        self.processing_jobs.run_jobs_changed.connect(self.update_processing_status)

        self._bold_font = QtGui.QFont()
        self._bold_font.setBold(True)

        # Empty spaces are not editable by default
        proto = QtGui.QStandardItem()
        proto.setEditable(False)
        self.setItemPrototype(proto)

        self.user_variables = db.get_user_variables()
        self.editable_columns = {"comment"} | {
            vv.name for vv in self.user_variables.values()
        }
        for col_id in self.editable_columns:
            col_ix = self.find_column(col_id)
            for r in range(self.rowCount()):
                # QStandardItem is editable by default
                self.setItem(r, col_ix, QtGui.QStandardItem())

        # Set up status column with checkboxes for runs
        checkbox_proto = self.itemPrototype().clone()
        checkbox_proto.setCheckable(True)
        checkbox_proto.setCheckState(Qt.CheckState.Checked)
        for r in range(n_run_rows):
            self.setItem(r, 0, checkbox_proto.clone())

        self._load_from_db()

    @staticmethod
    def _load_columns(db: DamnitDB, col_settings, settings_by_id=True):
        t0 = time.perf_counter()
        col_id_to_title = {
            "run": "Run",
            "proposal": "Proposal",
            "start_time": "Timestamp",
            "comment": "Comment",
        } | dict(
            db.conn.execute("""SELECT name, title FROM variables WHERE title NOT NULL""")
        )
        non_static_cols = sorted(
            set(db.variable_names()) - {"comment"},
            key=lambda col: (col_id_to_title.get(col, col).casefold(), col.casefold()),
        )
        column_ids = STATIC_COLUMN_IDS + non_static_cols

        # Column settings previously stored human friendly titles - convert to IDs
        if settings_by_id:
            saved_col_order = list(col_settings)
        else:
            col_title_to_id = {t: n for (n, t) in col_id_to_title.items()}
            saved_col_order = [col_title_to_id.get(c, c) for c in col_settings]

        # Strip missing columns
        saved_col_order = [col for col in saved_col_order if col in column_ids]

        # Sort columns such that all static columns (proposal, run, etc) are at
        # the beginning, followed by all the columns that have saved settings,
        # followed by all the other columns (i.e. comment_id and any new columns
        # added in between the last save and now).
        sorted_cols = column_ids[:5]
        # Static columns are saved too to store their visibility, but we filter
        # them out here because they've already been added to the list.
        sorted_cols.extend([col for col in saved_col_order if col not in sorted_cols])
        # Add all other unsaved columns
        sorted_cols.extend([col for col in non_static_cols if col not in saved_col_order])

        column_titles = [col_id_to_title.get(c, c) for c in sorted_cols]

        t1 = time.perf_counter()
        log.info(f"Got columns in {t1 - t0:.3f} s")
        return sorted_cols, column_titles

    def _apply_provenance_style(self, item, provenance: str):
        if provenance is None or provenance == "context.py":
            return
        item.setData(provenance, role=PROVENANCE_ROLE)
        tip = item.toolTip() or ""
        if tip.startswith("<"):
            sep = "<br/>"
        else:
            sep = "\n" if tip else ""
        item.setToolTip(f"{tip}{sep}Provenance: {provenance}")

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.ToolTipRole:
            if 0 <= section < len(self.column_ids):
                return self.column_descriptions.get(self.column_ids[section])
            return None
        return super().headerData(section, orientation, role)

    def text_item(self, value, display=None, attrs=None):
        if display is None:
            if value is None:
                display = ''
            else:
                display = prettify_notation(value)

        item = self.itemPrototype().clone()
        # We will use UserRole data for sorting
        item.setData(value, role=Qt.ItemDataRole.UserRole)
        item.setData(display, role=Qt.ItemDataRole.DisplayRole)

        units = attrs.get("units") if attrs else None
        if (
            isinstance(units, str) and units != ""
            and isinstance(value, Number)
        ):
            item.setData(units, role=UNITS_ROLE)

        return item

    def image_item(self, png_data: bytes):
        item = self.itemPrototype().clone()
        item.setData(self.generateThumbnail(png_data), role=Qt.ItemDataRole.DecorationRole)
        item.setToolTip(
            f'<img src="data:image/png;base64,{b64encode(png_data).decode()}">'
        )
        return item

    def line_item(self, line_data: np.ndarray):
        item = self.itemPrototype().clone()
        item.setData(line_data, role=LINE_DATA_ROLE)
        item.setData("", role=Qt.ItemDataRole.DisplayRole)
        return item

    def comment_item(self, text):
        item = self.text_item(text)
        item.setToolTip(text)
        item.setEditable(True)
        return item

    def error_item(self, attrs):
        item = self.itemPrototype().clone()
        msg = attrs['error']
        match attrs.get('error_cls', ''):
            case 'Skip':
                colour = 'lightgrey'
                msg = "Skipped: " + msg
            case 'SourceNameError':  # Typically an issue with data, not code
                colour = 'lightgrey'
            case cls:
                colour = 'orange'
                msg = f'{msg}\n(see processing log for details)'
                if cls != 'DependencyError':
                    msg = f"{cls}: {msg}"
        item.setToolTip(msg)
        item.setData(QtGui.QColor(colour), Qt.ItemDataRole.DecorationRole)
        return item

    def new_item(self, value, column_id, max_diff, attrs, summary_type=None, provenance=None):
        if summary_type == "trendline":
            item = self.line_item(blob2numpy(value))
        elif summary_type == "numpy":
            arr = blob2numpy(value)
            item = self.text_item(f"{arr.dtype}: {arr.shape}")
        elif is_png_bytes(value):
            item = self.image_item(value)
        elif column_id == 'comment':
            item = self.comment_item(value)
        elif column_id == 'start_time':
            item = self.text_item(value, timestamp2str(value))
        elif 'error' in attrs:
            item = self.error_item(attrs)
        else:
            item = self.text_item(value, attrs=attrs)
            item.setEditable(column_id in self.editable_columns)
            bold = attrs.get("bold")
            if bold is None:
                bold = (max_diff is not None) and max_diff > 1e-9
            if bold:
                item.setFont(self._bold_font)
            if (bg := attrs.get('background')) is not None:
                bg_color = QtGui.QColor(*bg)
                item.setBackground(QtGui.QBrush(bg_color))
                item.setForeground(QtGui.QBrush(best_text_color(bg_color)))

        self._apply_provenance_style(item, provenance)
        return item

    def _load_from_db(self):
        t0 = time.perf_counter()

        row_headers = []
        row_ix = -1

        for row_ix, (prop, run, ts) in enumerate(self.db.conn.execute("""
            SELECT proposal, run, start_time FROM run_info ORDER BY proposal, run
        """).fetchall()):
            row_headers.append(str(run))
            self.run_index[(prop, run)] = row_ix
            self.setItem(row_ix, 1, self.text_item(prop))
            self.setItem(row_ix, 2, self.text_item(run))
            self.setItem(row_ix, 3, self.text_item(ts, timestamp2str(ts)))

        for (prop, run), grp in groupby(self.db.conn.execute("""
            SELECT proposal, run, name, value, max_diff, summary_type, attributes, provenance FROM run_variables
            ORDER BY proposal, run
        """).fetchall(), key=lambda r: r[:2]):  # Group by proposal & run
            try:
                row_ix = self.run_index[(prop, run)]
            except KeyError:
                # This can happen if an entry gets into run_variables without
                # a corresponding run_info entry. Add a row for it.
                status_item = self.itemPrototype().clone()
                status_item.setCheckable(True)
                status_item.setCheckState(Qt.CheckState.Checked)
                self.appendRow([status_item, self.text_item(prop), self.text_item(run)])
                row_ix = self.run_index[(prop, run)] = self.rowCount() - 1

            for *_, name, value, max_diff, summary_type, attr_json, provenance in grp:
                try:
                    col_ix = self.column_index[name]
                except KeyError:
                    # Shouldn't happen, but make sure we can still launch the GUI
                    log.warning("Column %r not found (loading p%s r%s)",
                                name, prop, run)
                    continue
                if name in self.user_variables:
                    value = self.user_variables[name].get_type_class().from_db_value(value)
                if summary_type == "complex":
                    value = blob2complex(value)
                attrs = json.loads(attr_json) if attr_json else {}
                self.setItem(row_ix, col_ix, self.new_item(value, name, max_diff, attrs, summary_type, provenance))

        self.setVerticalHeaderLabels(row_headers)
        t1 = time.perf_counter()
        log.info(f"Filled rows in {t1 - t0:.3f} s")

    def has_column(self, name, by_title=False):
        if by_title:
            return name in self.column_titles
        else:
            return name in self.column_index

    def find_column(self, name, by_title=False):
        if by_title:
            try:
                return self.column_titles.index(name)
            except ValueError:  # Convert to KeyError, matching dict
                raise KeyError(name)
        else:
            return self.column_index[name]

    def column_title(self, col_ix):
        return self.column_titles[col_ix]

    def column_id(self, col_ix):
        return self.column_ids[col_ix]

    def column_title_to_id(self, title):
        return self.column_id(self.find_column(title, by_title=True))

    def computed_columns(self, by_title=False):
        for i, col_id in enumerate(self.column_ids[5:], start=5):
            if col_id not in self.editable_columns:
                if by_title:
                    yield self.column_titles[i]
                else:
                    yield col_id

    def find_row(self, proposal, run):
        return self.run_index[(proposal, run)]

    def row_to_proposal_run(self, row_ix):
        prop_col, run_col = 1, 2
        prop_it, run_it = self.item(row_ix, prop_col), self.item(row_ix, run_col)
        if prop_it is None:
            return None, None
        return prop_it.data(Qt.ItemDataRole.UserRole), run_it.data(Qt.ItemDataRole.UserRole)

    def precreate_runs(self, n_runs: int):
        proposal = self.db.metameta["proposal"]
        start_run = max(
            [r for (p, r) in self.run_index if p == proposal], default=0
        ) + 1

        for run in range(start_run, start_run + n_runs):
            # To precreate the run we add it to the `run_info` table, and
            # the `run_variables` table with an empty comment. Adding it to
            # both ensures that the run will show up in the `runs` view.
            self.db.ensure_run(proposal, run)
            self.db.set_variable(proposal, run, "comment", ReducedData(None), provenance="")

            self.insert_run_row(proposal, run, {})

    def insert_columns(
        self,
        before: int,
        titles,
        column_ids=None,
        column_descriptions=None,
        editable=False
    ):
        if column_ids is None:
            column_ids = titles
        else:
            assert len(column_ids) == len(titles)

        if column_descriptions is not None:
            assert len(column_descriptions) == len(column_ids)
            for col_id, desc in zip(column_ids, column_descriptions):
                self.column_descriptions[col_id] = desc

        self.column_ids[before:before] = column_ids
        self.column_titles[before:before] = titles
        self.column_index = {c: i for (i, c) in enumerate(self.column_ids)}
        if editable:
            self.editable_columns.update(column_ids)

        self.insertColumns(before, len(column_ids))

        for i, title in enumerate(titles, start=before):
            self.setHorizontalHeaderItem(before, QtGui.QStandardItem(title))

    def removeColumn(self, column: int, parent=QtCore.QModelIndex()):
        column_id = self.column_ids[column]
        del self.column_ids[column]
        del self.column_titles[column]
        self.column_descriptions.pop(column_id, None)
        self.column_index = {c: i for (i, c) in enumerate(self.column_ids)}
        super().removeColumn(column, parent)
        self.column_removed.emit(column_id)

    def insert_run_row(self, proposal, run, items: dict):
        status_item = self.itemPrototype().clone()
        status_item.setCheckable(True)
        status_item.setCheckState(Qt.CheckState.Checked)
        row = [status_item, self.text_item(proposal), self.text_item(run)]

        for column_id in self.column_ids[3:]:
            if (item := items.get(column_id, None)) is None:
                if column_id in self.editable_columns:
                    item = QtGui.QStandardItem()  # Editable by default
                else:
                    item = None
            row.append(item)
        # We add new rows at the end, a QSortFilterProxyModel shows them in
        # the correct position given the current sort.
        self.run_index[(proposal, run)] = row_ix = self.rowCount()
        self.appendRow(row)
        self.setVerticalHeaderItem(row_ix, QtGui.QStandardItem(str(run)))
        return row_ix

    def handle_run_values_changed(self, proposal, run, values: dict):
        known_col_ids = set(self.column_ids)
        new_col_ids = [c for c in values if c not in known_col_ids]

        if new_col_ids:
            log.info("New columns for table: %s", new_col_ids)
            # TODO: retrieve titles for new columns
            self.insert_columns(self.columnCount(), new_col_ids)

        try:
            row_ix = self.find_row(proposal, run)
            if not values:
                # No need to update anything else
                return
            placeholders = ", ".join(["?"] * len(values))
            query = (
                "SELECT name, value, max_diff, summary_type, attributes, provenance "
                "FROM run_variables WHERE proposal=? AND run=? "
                f"AND name IN ({placeholders})"
            )
            params = (proposal, run, *values.keys())
        except KeyError:
            row_ix = None
            query = (
                "SELECT name, value, max_diff, summary_type, attributes, provenance "
                "FROM run_variables WHERE proposal=? AND run=?"
            )
            params = (proposal, run)

        items = {}
        for (
                name, value, max_diff, summary_type, attr_json, provenance
        ) in self.db.conn.execute(query, params).fetchall():
            items[name] = self.new_item(
                value,
                column_id=name,
                max_diff=(max_diff or 0),
                attrs=(json.loads(attr_json) if attr_json else {}),
                summary_type=summary_type,
                provenance=provenance,
            )

        if 'start_time' in values or row_ix is None:
            row = self.db.conn.execute(
                "SELECT start_time FROM run_info WHERE proposal=? AND run=?",
                (proposal, run)
            ).fetchone()
            if row is not None:
                items['start_time'] = self.new_item(row[0], 'start_time', 0, {})

        col_id_to_ix = {c: i for (i, c) in enumerate(self.column_ids)}

        if row_ix is not None:
            log.debug("Update existing row %s for run %s", row_ix, run)
            for column_id, item in items.items():
                col_ix = col_id_to_ix[column_id]
                self.setItem(row_ix, col_ix, item)
        else:
            self.insert_run_row(proposal, run, items)

    def handle_variable_set(self, var_info: dict):
        col_id = var_info['name']
        title = var_info['title'] or col_id
        description = var_info.get('description')
        try:
            col_ix = self.find_column(col_id)
        except KeyError:
            # New column
            end = self.columnCount()
            self.insert_columns(
                end, [title], [col_id], [description],
                editable=var_info['type'] is not None
            )
        else:
            # Update existing column
            old_title = self.column_title(col_ix)
            if title != old_title:
                self.column_titles[col_ix] = title
                self.setHorizontalHeaderItem(col_ix, QtGui.QStandardItem(title))

            self.column_descriptions[col_id] = description

    def handle_processing_state_set(self, info):
        self.processing_jobs.on_processing_state_set(info)

    def handle_processing_finished(self, info):
        self.processing_jobs.on_processing_finished(info)

    def update_processing_status(self, proposal, run, jobs_for_run):
        """Show/hide the processing indicator for the given run"""
        try:
            row_ix = self.find_row(proposal, run)
        except KeyError:
            if jobs_for_run:
                row_ix = self.insert_run_row(proposal, run, {})
            else:
                return

        running = [j for j in jobs_for_run if j['status'] == 'RUNNING']

        row_header_item = self.verticalHeaderItem(row_ix)
        if running:
            row_header_item.setData(f"{run} ⚙️", Qt.ItemDataRole.DisplayRole)
            if len(running) == 1:
                info = running[0]
                msg = f"Processing on {info['username']}@{info['hostname']}"
                if job_id := info['slurm_job_id']:
                    msg += f" (Slurm job {job_id})"
                row_header_item.setToolTip(msg)
            else:
                row_header_item.setToolTip(f"Processing in {len(running)} jobs")
        elif jobs_for_run:
            # Jobs in the list but not running must be pending
            row_header_item.setData(f"{run} ⋮", Qt.ItemDataRole.DisplayRole)
            row_header_item.setToolTip("Processing is queued")
        else:
            row_header_item.setData(f"{run}", Qt.ItemDataRole.DisplayRole)
            row_header_item.setToolTip("")

    def add_editable_column(self, name):
        if name == "Status":
            return
        self.editable_columns.add(name)

    def remove_editable_column(self, name):
        if name == "comment":
            return
        self.editable_columns.remove(name)

    @staticmethod
    def generateThumbnail(data: bytes) -> QtGui.QPixmap:
        """
        Helper function to generate a thumbnail from PNG data.
        """
        pixmap = QtGui.QPixmap()
        pixmap.loadFromData(data, "PNG")
        if pixmap.width() > THUMBNAIL_SIZE[0] or pixmap.height() > THUMBNAIL_SIZE[1]:
            pixmap = pixmap.scaled(*THUMBNAIL_SIZE, Qt.AspectRatioMode.KeepAspectRatio)
        return pixmap

    def numbers_for_plotting(self, *cols, by_title=True):
        col_ixs = [self.find_column(c, by_title) for c in cols]
        res = [[]  for _ in cols]
        for r in range(self.rowCount()):
            status_item = self.item(r, 0)
            if status_item is None or status_item.checkState() != Qt.CheckState.Checked:
                continue

            vals = [self.get_value_at_rc(r, ci) for ci in col_ixs]
            if all(isinstance(val, (int, float)) for val in vals):
                for res_list, val in zip(res, vals):
                    res_list.append(val)

        return res

    def get_value_at(self, index):
        """Get the value for programmatic use, not for display"""
        return self.itemFromIndex(index).data(Qt.ItemDataRole.UserRole)

    def get_value_at_rc(self, row, col):
        item = self.item(row, col)
        return item.data(Qt.ItemDataRole.UserRole) if item is not None else None

    # QStandardItemModel assumes empty cells (no item) can be edited, regardless
    # of itemPrototype. This override prevents editing empty cells.
    def flags(self, model_index):
        # Not using itemFromIndex() here, as it creates & inserts items
        if model_index.isValid() and model_index.model() is self:
            itm = self.item(model_index.row(), model_index.column())
            if itm is not None:
                return itm.flags()

        return Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsDragEnabled |Qt.ItemFlag.ItemIsDropEnabled

    def setData(self, index, value, role=None) -> bool:
        if not index.isValid():
            return False

        if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
            # A cell was edited in the table
            changed_column = self.column_id(index.column())
            if changed_column == "comment":
                if not super().setData(index, value, role):
                    return False
                parsed = value
            else:
                variable_type_class = self.user_variables[changed_column].get_type_class()

                try:
                    parsed = variable_type_class.parse(value) if value != '' else None
                except Exception:
                    self._main_window.show_status_message(
                        f'Value {value!r} is not valid for the {variable_type_class} column "{self.column_titles[index.column()]}"',
                        timeout=5000,
                        stylesheet=StatusbarStylesheet.ERROR
                    )
                    return False
                else:
                    if not super().setData(index, parsed, Qt.ItemDataRole.UserRole):
                        return False

                    if parsed is None:
                        display = ''
                    elif isinstance(parsed, float):
                        display = prettify_notation(value)
                    else:
                        display = str(parsed)
                    if not super().setData(index, display, role):
                        return False

            prop, run = self.row_to_proposal_run(index.row())
            self.value_changed.emit(int(prop), int(run), changed_column, parsed)

            return True

        elif role == Qt.ItemDataRole.CheckStateRole:
            if not super().setData(index, value, role):
                return False
            # CHeckboxes are only on the status column
            self.run_visibility_changed.emit(
                index.row(), (value == Qt.CheckState.Checked)
            )
            return True

        return super().setData(index, value, role)

    def dataframe_for_export(self, column_ids, rows=None, drop_image_cols=False):
        """Create a cleaned-up dataframe to be saved as a spreadsheet"""
        import pandas as pd
        column_ixs = [self.column_index[col_id] for col_id in column_ids]

        if rows is None:
            rows = range(self.rowCount())

        if drop_image_cols:
            filtered_ixs = []
            for col_ix in column_ixs:
                for row_ix in range(self.rowCount()):
                    item = self.item(row_ix, col_ix)
                    if item and is_png_bytes(item.data(Qt.ItemDataRole.UserRole)):
                        break
                else:
                    filtered_ixs.append(col_ix)
            column_ixs = filtered_ixs

        # Put the dtype options in an order so we can promote columns to more
        # general types based on their values.
        col_dtype_options = [
            pd.Float64Dtype(),  # Used if no values in selection
            pd.BooleanDtype(),
            pd.Int64Dtype(),
            pd.Float64Dtype(),
            pd.StringDtype(),   # For everything that's not covered above
        ]
        value_types = [type(None), bool, int, float]

        cols_dict = {}
        for col_ix in column_ixs:
            values = []
            col_type_ix = 0
            for row_ix in rows:
                item = self.item(row_ix, col_ix)
                if self.column_ids[col_ix] == 'start_time':
                    # Include timestamp as string
                    val = item and item.data(Qt.ItemDataRole.DisplayRole)
                else:
                    val = item and item.data(Qt.ItemDataRole.UserRole)
                    if val is None and item and item.data(Qt.ItemDataRole.DecorationRole):
                        val = "<image>"
                    if val is None and item and item.data(LINE_DATA_ROLE) is not None:
                        val = "<trendline>"

                values.append(val)
                for i, pytype in enumerate(value_types):
                    if isinstance(val, pytype):
                        col_type_ix = max(col_type_ix, i)
                        break
                else:
                    col_type_ix = 4

            cols_dict[self.column_titles[col_ix]] = pd.Series(
                values, dtype=col_dtype_options[col_type_ix]
            )

        df = pd.DataFrame(cols_dict)

        row_labels = [self.headerData(r, Qt.Orientation.Vertical) for r in rows]
        df.index = row_labels

        return df


class QtExtractionJobTracker(ExtractionJobTracker, QtCore.QObject):
    run_jobs_changed = QtCore.pyqtSignal(int, int, object) # prop, run, jobs

    def __init__(self, parent):
        super().__init__()
        QtCore.QObject.__init__(self, parent)

        # Check for crashed Slurm jobs every 2 minutes
        self.slurm_check_timer = QtCore.QTimer(self)
        self.slurm_check_timer.timeout.connect(self.check_slurm_jobs)
        self.slurm_check_timer.start(120_000)

    def squeue_check_jobs(self, cmd, jobs_to_check):
        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.ForwardedErrorChannel)

        def done():
            proc.deleteLater()
            if proc.exitStatus() != QProcess.NormalExit or proc.exitCode() != 0:
                log.warning("Error calling squeue")
                return
            stdout = bytes(proc.readAllStandardOutput()).decode()
            self.process_squeue_output(stdout, jobs_to_check)

        proc.finished.connect(done)
        proc.start(cmd[0], cmd[1:])

    def on_run_jobs_changed(self, proposal, run):
        jobs = [i for i in self.jobs.values()
                if i['proposal'] == proposal and i['run'] == run]
        self.run_jobs_changed.emit(proposal, run, jobs)


def prettify_notation(value):
    if isinstance(value, float):
        if value % 1 == 0 and abs(value) < 10_000:
            # If it has no decimal places, display it as an int
            return f"{int(value)}"
        elif 0.0001 < abs(value) < 10_000:
            # If it's an easily-recognized range for numbers, display as a float
            return f"{value:.4f}"
        else:
            # Otherwise, display in scientific notation
            return f"{value:.3e}"
    return f"{value}"

def is_png_bytes(obj):
    return isinstance(obj, bytes) and BlobTypes.identify(obj) is BlobTypes.png

def is_numpy_bytes(obj):
    return isinstance(obj, bytes) and BlobTypes.identify(obj) is BlobTypes.numpy
