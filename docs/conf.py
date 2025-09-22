# Configuration file for the Sphinx documentation builder.

# -------------------------------------------------------------------------
# Mock CUDA to avoid errors on ReadTheDocs or CPU-only environments
# -------------------------------------------------------------------------
import sys

try:
    import torch
    # If torch imports but no GPU is available, patch CUDA funcs
    if not torch.cuda.is_available():
        import types
        torch.cuda = types.SimpleNamespace(
            is_available=lambda: False,
            device_count=lambda: 0,
            get_device_properties=lambda _: None,
            empty_cache=lambda: None,
        )
except ImportError:
    # If torch itself is missing, stub it entirely
    import types
    torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: False,
            device_count=lambda: 0,
            get_device_properties=lambda _: None,
            empty_cache=lambda: None,
        )
    )
    sys.modules["torch"] = torch
    sys.modules["torch.cuda"] = torch.cuda

# -------------------------------------------------------------------------
# Project information
# -------------------------------------------------------------------------
import rfmix_reader
import sphinx_rtd_theme


def get_version():
    return rfmix_reader.__version__

# -- General configuration ---------------------------------------------------

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

templates_path = ["_templates"]
autosummary_generate = True
autosectionlabel_prefix_document = True
napoleon_numpy_docstring = True

source_suffix = ".rst"
main_doc = "index"

project = "rfmix-reader"
copyright = "2024, Kynon J.M. Benjamin"
author = "Kynon J.M. Benjamin"

version = get_version()
release = version

exclude_patterns = ["_build", "conf.py"]
pygments_style = "default"
todo_include_todos = False

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_sidebars = {"**": ["relations.html", "searchbox.html"]}
htmlhelp_basename = "rfmix-readerdoc"

man_pages = [(main_doc, "rfmix-reader", "rfmix-reader Documentation", [author], 1)]

epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright

epub_exclude_files = ["search.html"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "xarray": ("http://xarray.pydata.org/en/stable/", None),
    "pandas": ("http://pandas.pydata.org/pandas-docs/stable/", None),
    "dask": ("http://docs.dask.org/en/latest/", None),
    "rapids": ("https://docs.rapids.ai/api/cudf/stable/", None),
}
