# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import importlib
import os
import sys
from datetime import date

sys.path.insert(0, os.path.abspath('../'))

project = 'OAMC'
copyright = f'{date.today().year}, Nicolas Ebeling'
author = 'Nicolas Ebeling'
release = importlib.metadata.version("oamc")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx_copybutton',
    'myst_parser',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_title = f'OAMC {importlib.metadata.version("oamc")} Docs'
html_last_updated_fmt = '%A, %B %d, %Y at %H:%M:%S'
html_show_sourcelink = False
html_theme_options = {
"icon_links": [
    {
        "name": "GitHub",
        "url": "https://github.com/nicolasebeling/oamc",
        "icon": "fa-brands fa-square-github",
        "type": "fontawesome",
    }
],
"footer_start": [
    "copyright",
    "last-updated",
],
"footer_end": [
    # "sphinx-version",
    # "theme-version",
],
}
