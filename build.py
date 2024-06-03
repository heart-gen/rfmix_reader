from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

ext_modules = [
    Extension(
        "rfmix_reader.fb_reader",
        sources=["rfmix_reader/fb_reader.c"],
        include_dirs=["rfmix_reader"]
    ),
]


class CustomBuildExt(build_ext):
    def build_extensions(self):
        # Initialize compiler options if not already initialized
        compile_options = getattr(self.compiler, 'compile_options', None)
        if compile_options is None:
            self.compiler.compile_options = []

        # Add optimization flag
        self.compiler.compile_options.extend(['-O3'])

        # Proceed with the standard build process
        build_ext.build_extensions(self)


def build(setup_kwargs):
    setup_kwargs.update({
        "ext_modules": ext_modules,
        "cmdclass": {"build_ext": CustomBuildExt},
        "zip_safe": False,
    })
