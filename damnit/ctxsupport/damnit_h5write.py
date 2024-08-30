import os
import fcntl
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from queue import Queue
from threading import Thread

import h5py
import h5netcdf
import xarray as xr
from xarray.backends import H5NetCDFStore
from xarray.backends.api import dump_to_store

log = logging.getLogger(__name__)

@dataclass
class ToWrite:
    name: str
    data: object
    attrs: dict
    compression_opts: dict = field(default_factory=dict)

@dataclass
class SummaryToWrite(ToWrite):
    pass


class WriterThread(Thread):
    def __init__(self, file_path, reduced_only=False):
        super().__init__(daemon=True)
        self.file_path = file_path
        self.reduced_only = reduced_only

        self.lock_fd = os.open(file_path, os.O_RDWR | os.O_CLOEXEC | os.O_CREAT)
        if os.stat(file_path).st_uid == os.getuid():
            os.chmod(file_path, 0o666)
        self.have_lock = False
        self.queue = Queue()
        self.abort = False
        self.n_reduced = 0
        self.n_main = 0

    def stop(self, abort=False):
        if abort:
            self.abort = True
        self.queue.put(None)

    def get_lock(self):
        while True:
            if self.abort:
                raise SystemExit(0)  # exit the thread with no traceback
            try:
                fcntl.lockf(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.have_lock = True
                return
            except (PermissionError, BlockingIOError):
                time.sleep(1)

    @contextmanager
    def locked_h5_access(self):
        self.get_lock()
        try:
            with h5py.File(self.file_path, 'r+') as h5f:
                with h5netcdf.File(h5f.id, 'r+') as h5ncf:
                    yield h5f, h5ncf
        finally:
            self.have_lock = False
            # Closing the file above has already released the lock; this is how
            # POSIX process-associated locks work (see lockf & fcntl man pages).
            # We'll do this as well to ensure the lock is released, just in case
            # anything does not behave as expected.
            fcntl.lockf(self.lock_fd, fcntl.LOCK_UN)

    def run(self):
        try:
            while True:
                if (item := self.queue.get()) is None:
                    return

                assert isinstance(item, ToWrite)

                with self.locked_h5_access() as (h5f, ncf):
                    if isinstance(item, SummaryToWrite):
                        path = f'.reduced/{item.name}'
                        if path in h5f:
                            del h5f[path]
                        ds = h5f.create_dataset(
                            path, data=item.data, **item.compression_opts
                        )
                        ds.attrs.update(item.attrs)
                        self.n_reduced += 1
                    else:
                        if item.name in h5f:
                            del h5f[item.name]

                        # Create the group and set attributes
                        h5f.require_group(item.name).attrs.update(item.attrs)

                        if isinstance(item.data, (xr.Dataset, xr.DataArray)):
                            write_xarray_object(item.data, item.name, ncf)
                        else:
                            path = f"{item.name}/data"
                            h5f.create_dataset(
                                path, data=item.data, **item.compression_opts
                            )
                        self.n_main += 1
        finally:
            os.close(self.lock_fd)
            self.lock_fd = -1

            log.info("Written %d data & %d summary variables to %s",
                     self.n_main, self.n_reduced, self.file_path)


def write_xarray_object(obj, group, ncf: h5netcdf.File):
    """Write an xarray DataArray/Dataset into an h5netcdf File"""
    if isinstance(obj, xr.DataArray):
        obj = dataarray_to_dataset_for_netcdf(obj)
    store = H5NetCDFStore(ncf, group=group, mode='a', autoclose=False)
    dump_to_store(obj, store)
    # Don't close the store object - that would also close the file

def dataarray_to_dataset_for_netcdf(self: xr.DataArray):
    # From xarray (DataArray.to_netcdf() method), under Apache License 2.0
    # Copyright 2014-2023, xarray Developers
    from xarray.backends.api import DATAARRAY_NAME, DATAARRAY_VARIABLE

    if self.name is None:
        # If no name is set then use a generic xarray name
        dataset = self.to_dataset(name=DATAARRAY_VARIABLE)
    elif self.name in self.coords or self.name in self.dims:
        # The name is the same as one of the coords names, which netCDF
        # doesn't support, so rename it but keep track of the old name
        dataset = self.to_dataset(name=DATAARRAY_VARIABLE)
        dataset.attrs[DATAARRAY_NAME] = self.name
    else:
        # No problems with the name - so we're fine!
        dataset = self.to_dataset()

    return dataset


@contextmanager
def writer_threads(paths, reduced_paths):
    threads = [
        WriterThread(path) for path in paths
    ] + [
        WriterThread(path, reduced_only=True) for path in reduced_paths
    ]
    error = False
    for thread in threads:
        thread.start()
    try:
        yield threads
    except:
        error = True
        raise
    finally:
        for thread in threads:
            thread.stop(abort=error)
        for thread in threads:
            # If there was no error, give threads a generous amount of time
            # to do any further writes.
            thread.join(timeout=(5 if error else 120))
            if thread.is_alive():
                log.warning("HDF5 writer thread for %s did not stop properly",
                            thread.file_path)
