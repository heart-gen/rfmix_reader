from setuptools import setup

def build(setup_kwargs):
    setup_kwargs.update(
        setup_requires=['cffi'],
        cffi_modules=["build_ext.py:ffibuilder"],
        install_requires=["cffi"],
        zip_safe=False,
        include_package_data=True,
    )
