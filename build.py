"""
Adapted from `build_ext.py` script in the `pandas-plink` package.
Source: https://github.com/limix/pandas-plink/blob/main/build_ext.py
"""
from cffi import FFI
from os.path import join, dirname, abspath

def read_file(file_path):
    try:
        with open(file_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except IOError as e:
        raise IOError(f"Error reading file {file_path}: {e}")


def main():
    # Determine directory for scripts
    folder = dirname(abspath(__file__))
    # Read C header and source files
    header = read_file(join(folder, "rfmix_reader",
                            "include", "fb_reader.h"))
    source = read_file(join(folder, "rfmix_reader",
                            "fb_reader.c"))
    # Initialize the FFI builder
    ffibuilder = FFI()
    ffibuilder.set_unicode(False)
    # Define C functions and types declared in header files
    ffibuilder.cdef(header)
    # Set source for FFI builder
    ffibuilder.set_source("rfmix_reader.fb_reader",
                          source, language="c")
    return ffibuilder


if __name__ == "__main__":
    ffibuilder = main()
    ffibuilder.compile(verbose=True)
