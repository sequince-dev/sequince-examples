"""Sphinx configuration; generated outputs are mandatory."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from docs_pipeline import validate  # noqa: E402

validate()
project = "Sequince examples"
author = "Sequince contributors"
extensions = ["sphinx.ext.intersphinx"]
html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/sequince-dev/sequince-examples",
    "use_repository_button": True,
}
exclude_patterns = ["_build"]
intersphinx_mapping = {
    "sequince": (
        "https://sequince.readthedocs.io/en/latest/",
        os.environ.get("SEQUINCE_INVENTORY"),
    ),
}
