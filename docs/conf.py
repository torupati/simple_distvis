import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

# Configuration file for the Sphinx documentation builder.
project = "kalman1d-velocity-motion"
copyright = "2024, Your Name"
author = "Your Name"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "alabaster"
html_static_path = ["_static"]
