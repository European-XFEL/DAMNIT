# API

The data that DAMNIT saves in its database and HDF5 files can be accessed
through our Python API.

## Quick start

You can open a database by proposal number, in which case it will look in
`usr/Shared/amore`, or by an absolute path:
```python
from damnit import Damnit

db = Damnit(1234) # This would also work: Damnit("/my/path/to/amore")
```

The run table can be read into a dataframe
with [Damnit.table()][damnit.api.Damnit.table]:
```python
# Use with_titles to name the columns by the variable titles rather than their
# names.
df = db.table(with_titles=True)
```

The variables themselves can be read by indexing `db`:
```python
run_vars = db[100] # Index by run number
run_vars.keys()    # Get all available variables for this run

myvar = db[100, "myvar"] # Equivalent to run_vars["myvar"]
data = myvar.read()
summary = myvar.summary()
```

You can also write to [user-editable
variables](gui.md#adding-user-editable-variables):
```python
run_vars["myvar"] = 42

# An alternative style would be:
myvar.write(42)
```

## API reference

::: damnit.Damnit

::: damnit.RunVariables
    options:
      merge_init_into_class: no

::: damnit.VariableData
    options:
      merge_init_into_class: no
