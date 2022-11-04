When DAMNIT is set to run the context file in a separate Python environment
from the application, the Python code in this folder runs in the target Python
environment.

This folder is added to sys.path, so the context file can use
`from ctxsupport import Variable`.
