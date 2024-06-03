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
    # Customize the compiler options here if needed
    def build_extensions(self):
        self.compiler.initialize()
        self.compiler.compile_options.extend(['-O3'])
        build_ext.build_extensions(self)

def build(setup_kwargs):
    setup_kwargs.update({
        "ext_modules": ext_modules,
        "cmdclass": {"build_ext": CustomBuildExt},
        "zip_safe": False,
    })
