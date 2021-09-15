# type: ignore

"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""
import atexit
import os

# -- Project information -----------------------------------------------------
import shutil
from pathlib import Path

project = "zfit"
package = "tutorials"
repo_name = "repo tutorials"
copyright = "2021, zfit"
author = "zfit"


# sphinx can't handle relative pathes, so add repo as symlink
project_dir = Path(__file__).parents[1]
tutorial_path = Path(project_dir).joinpath("_website", "tutorials")
# if tutorial_path.exists():
#     shutil.rmtree(tutorial_path)
tutorial_path.mkdir(exist_ok=True)
# tutorial_path.mkdir(parents=True, exist_ok=True)
for folder in ["introduction", "components", "guides", "TensorFlow"]:
    folderpath = project_dir.joinpath(folder)
    for target in os.listdir(folderpath):
        if not target.startswith("_"):
            targetpath = tutorial_path.joinpath(folder, target)
            if not targetpath.exists():
                sourcepath = folderpath.joinpath(target)
                os.symlink(sourcepath, targetpath)

# -- General configuration ---------------------------------------------------

html_logo = str(project_dir.joinpath("_website", "images", "zfit-logo_400x168.png"))
source_suffix = {
    ".ipynb": "myst-nb",
    ".md": "myst-nb",
    ".rst": "restructuredtext",
}

# The master toctree document.
master_doc = "index"
modindex_common_prefix = [
    f"{package}.",
]

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_panels",
    "sphinx_thebe",
    "sphinx_togglebutton",
]
exclude_patterns = [
    "**.ipynb_checkpoints",
    "*build",
    "adr*",
    "tests",
]

html_copy_source = True  # needed for download notebook button
html_favicon = str(project_dir.joinpath("_website", "images", "zfit-favicon.png"))
html_show_copyright = False
html_show_sourcelink = False
html_show_sphinx = False
html_sourcelink_suffix = ""
html_static_path = ["_static"]
html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": f"https://github.com/zfit/{repo_name}",
    "repository_branch": "master",
    "path_to_docs": "_website",
    "use_download_button": True,
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "colab_url": "https://colab.research.google.com",
        "notebook_interface": "jupyterlab",
        "thebe": True,
        "thebelab": True,
    },
    "theme_dev_mode": True,
}
html_title = "tutorials"
panels_add_bootstrap_css = False  # wider page width with sphinx-panels
pygments_style = "sphinx"
todo_include_todos = False

# Cross-referencing configuration
default_role = "py:obj"
primary_domain = "py"
# nitpicky = True  # warn if cross-references are missing
# nitpick_ignore = [
#     ("py:class", "tensorflow.keras.losses.Loss"),
#     ("py:class", "tensorflow.python.keras.losses.Loss"),
#     ("py:obj", "Loss"),
# ]

# Settings for copybutton
copybutton_prompt_is_regexp = True
copybutton_prompt_text = r">>> |\.\.\. "  # doctest

# Settings for myst_nb
execution_timeout = -1
nb_output_stderr = "remove"
nb_render_priority = {
    "html": (
        "application/vnd.jupyter.widget-view+json",
        "application/javascript",
        "text/html",
        "image/svg+xml",
        "image/png",
        "image/jpeg",
        "text/markdown",
        "text/latex",
        "text/plain",
    )
}

jupyter_execute_notebooks = "cache"
execution_excludepatterns = [
    "B2KstLL.py",
    "*_website*",
    "kstmumu_tutorial.py",
    "API.ipynb",
    "Summary.ipynb",
]
jupyter_cache_path = project_dir.joinpath("_website", ".jupiter_cache")
jupyter_cache_path.mkdir(exist_ok=True)
if jupyter_execute_notebooks == "cache":
    jupyter_cache = str(jupyter_cache_path)

# Settings for myst-parser
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "dollarmath",
    "smartquotes",
]
myst_update_mathjax = False

# Settings for Thebe cell output
thebe_config = {
    "repository_url": html_theme_options["repository_url"],
    "repository_branch": html_theme_options["repository_branch"],
}
