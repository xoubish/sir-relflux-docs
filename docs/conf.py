import os, sys
from datetime import datetime
# make ../src importable
sys.path.insert(0, os.path.abspath("../src"))

project = "SIR RelativeFlux"
author = "Euclid SIR Team"
copyright = f"{datetime.now():%Y}, {author}"
release = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxext.opengraph",
]

myst_enable_extensions = ["colon_fence", "deflist"]

templates_path = ["_templates"]
exclude_patterns = ["_build"]
html_theme = "furo"
html_title = project
html_static_path = ["_static"]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"

ogp_site_url = "https://<your-gh-pages-url>/"
ogp_image = "https://<your-gh-pages-url>/_static/social-card.png"
