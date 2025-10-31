# Configuration file for the Sphinx documentation builder.
from __future__ import annotations

import pathlib, sys
from datetime import datetime

# -------------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.is_dir():
    sys.path.insert(0, str(SRC))  # support src/ layout

# -------------------------------------------------------------------------
# Project metadata no import
# -------------------------------------------------------------------------
try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # local fallback if needed
    import tomli as tomllib  # type: ignore

project = "rfmix-reader"  # default; will be overwritten if found in pyproject
author = "Kynon J.M. Benjamin"
_meta = {}
try:
    with open(ROOT / "pyproject.toml", "rb") as f:
        _meta = tomllib.load(f).get("project", {})
        project = _meta.get("name", project) or project
        a = _meta.get("authors", [])
        a_names = [x.get("name", "") for x in a if isinstance(x, dict)]
        author = ", ".join([n for n in a_names if n]) or author
except Exception:
    pass

# Version: prefer installed distribution; fall back to pyproject version
def _read_version() -> str:
    from importlib.metadata import version as _v, PackageNotFoundError
    # Try both normalized distribution names, since import name may differ
    candidates = [project]
    if project.replace("-", "_") not in candidates:
        candidates.append(project.replace("-", "_"))
    if project.replace("_", "-") not in candidates:
        candidates.append(project.replace("_", "-"))
    for dist in candidates:
        try:
            return _v(dist)
        except PackageNotFoundError:
            continue
    return (_meta or {}).get("version", "0.0.0")

release = _read_version()
version = release

current_year = datetime.utcnow().year
copyright = f"{current_year}, {author}"


# -------------------------------------------------------------------------
# Sphinx configuration
# -------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.duration",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
]

# RST only
source_suffix = [".rst"]
root_doc = "index"  # Sphinx 8+ update

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# Autosummary / Autodoc
autosummary_generate = True
autosectionlabel_prefix_document = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
python_use_unqualified_type_names = True
napoleon_numpy_docstring = True
napoleon_google_docstring = True
napoleon_attr_annotations = True

# Mock heavy deps during autodoc import
autodoc_mock_imports = [
    "psutil", "zarr",
    "torch", "cupy", "cudf", "cuml", "dask_cuda", "numba",
    "scanpy", "anndata", "cyvcf2", "pysam"
]

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "dask": ("https://docs.dask.org/en/latest/", None),
    "rapids": ("https://docs.rapids.ai/api/cudf/stable/", None),
}

# HTML
html_theme = "sphinx_rtd_theme"
html_title = f"{project} {release}"
html_sidebars = {"**": ["relations.html", "searchbox.html"]}
htmlhelp_basename = "rfmix-readerdoc"

# Ensure _static exists
_static = pathlib.Path(__file__).resolve().parent / "_static"
_static.mkdir(exist_ok=True)
html_static_path = ["_static"]

# Man/EPUB
man_pages = [(root_doc, "rfmix-reader", "rfmix-reader Documentation", [author], 1)]
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright
epub_exclude_files = ["search.html"]

# Quality gate
nitpicky = True
nitpick_ignore = [
    ("py:class", "numpy.ndarray"),
    ("py:class", "pandas.DataFrame"),
    ("py:class", "pandas.Series"),
]
