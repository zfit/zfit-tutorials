

Unfortunately, this is not solved very well: the interactive buttons
in the jupyter book require the notebooks to be still preseved in order
to be executable. So we copy (rsync)
the jupyter notebooks every time from the main folders
using a git pre-commit hook. If you want to add a new folder
(next to "introduction" etc), make sure to add it in the
sync_jupyter_notebooks.py to be synced.
