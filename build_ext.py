"""
Adapted from `build_ext.py` script in the `pandas-plink` package.
Source: https://github.com/limix/pandas-plink/blob/main/build_ext.py
"""
from cffi import FFI
from sysconfig import get_paths
from os.path import dirname, abspath, join

def read_lines(file_path):
    with open(file_path, "r") as f:
        content = f.read()
    return content

# Get the current directory
current_dir = join(dirname(abspath(__file__)))
# Create an instance of FFI
ffibuilder = FFI()
# Set Unicode mode to False
ffibuilder.set_unicode(False)

# Configure pyconfig.h
python_include_path = get_paths()['include']
pyconfig_64_path = join(python_include_path, 'pyconfig-64.h')

# Read the header file
h_file = join(current_dir, "rfmix_reader", "_fb_reader.h")
h_content = read_lines(h_file)

# Parse the header file
ffibuilder.cdef(f"#include {pyconfig_64_path}\n{h_content}")

# Read the C source file
c_file = join(current_dir, "rfmix_reader", "_fb_reader.c")
c_content = read_lines(c_file)
# Set the source code and language
ffibuilder.set_source("rfmix_reader.fb_reader",
                      c_content, language="c")


if __name__ == "__main__":
    # Compile the module
    ffibuilder.compile(verbose=True)
