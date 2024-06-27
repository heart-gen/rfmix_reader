# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sphinx_rtd_theme

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

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

templates_path = ['_templates']
autosummary_generate = False
autosectionlabel_prefix_document = True
napoleon_numpy_docstring = True

source_suffix = ".rst"
main_doc = "index"

project = 'rfmix-reader'
copyright = '2024, Kynon JM Benjamin'
author = 'Kynon JM Benjamin'

version = '0.1.13'
release = version

exclude_patterns = ["_build", "conf.py"]
pygments_style = "default"
todo_include_todos = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

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
