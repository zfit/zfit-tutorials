#!/usr/bin/env python
import filecmp
import shutil

# sphinx can't handle relative pathes, so add repo rsynced. Ugly!
from pathlib import Path


project_dir = Path(__file__).parents[1]
tutorial_path = Path(project_dir).joinpath("_website", "tutorials")
tutorial_path.mkdir(exist_ok=True)
for folder in ["introduction", "components", "guides", "TensorFlow"]:
    sourcepath = project_dir.joinpath(folder)
    targetpath = tutorial_path.joinpath(folder)
    targetpath.mkdir(exist_ok=True)
    comparison = filecmp.dircmp(sourcepath, targetpath, ignore=[".ipynb_checkpoints"])
    # only update if missing or changed
    if (
        comparison.left_only
        or comparison.right_only
        or comparison.diff_files
        or comparison.funny_files
    ):
        shutil.copytree(sourcepath, targetpath, dirs_exist_ok=True)


folder = "_static"
sourcepath = project_dir.joinpath(folder)
targetpath = project_dir / "_website" / folder
targetpath.mkdir(exist_ok=True)
shutil.copytree(sourcepath, targetpath, dirs_exist_ok=True)
