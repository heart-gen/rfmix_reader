from setuptools import setup

def build(setup_kwargs):
    setup_kwargs.update(
        setup_requires=['cffi'],
        cffi_modules="build_ext.py:ffibuilder",
        zip_safe=False,
    )
