"""
Adapted from `build_ext.py` script in the `pandas-plink` package.
Source: https://github.com/limix/pandas-plink/blob/main/build_ext.py
"""
from cffi import FFI
from os.path import dirname, abspath, join

def read_lines(file_path):
    with open(file_path, "r") as f:
        content = f.read()
    return content


def main():
    # Get the current directory
    current_dir = join(dirname(abspath(__file__)))
    # Create an instance of FFI
    ffibuilder = FFI()
    # Set Unicode mode to False
    ffibuilder.set_unicode(False)

    # Read the header file
    h_file = join(current_dir, "rfmix_reader", "_fb_reader.h")
    h_content = read_lines(h_file)
    # Parse the header file
    ffibuilder.cdef(h_content)

    # Read the C source file
    c_file = join(current_dir, "rfmix_reader", "_fb_reader.c")
    c_content = read_lines(c_file)
    # Set the source code and language
    ffibuilder.set_source("rfmix_reader.fb_reader",
                          c_content, language="c")
    return ffibuilder


if __name__ == "__main__":
    # Parse files
    ffibuilder = main()
    # Compile the module
    ffibuilder.compile(verbose=True)
