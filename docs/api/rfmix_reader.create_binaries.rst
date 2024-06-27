rfmix\_reader.create\_binaries
==============================

.. function:: create_binaries(file_prefix: str, binary_dir: str = "./binary_files")

    Create binary files from fullband (FB) TSV files.

    This function identifies FB TSV files based on a given prefix, creates a directory
    for binary files if it doesn't exist, and converts the identified TSV files to binary format.

    :param file_prefix: The prefix used to identify the relevant FB TSV files.
    :type file_prefix: str
    :param binary_dir: The directory where the binary files will be stored. Defaults to "./binary_files".
    :type binary_dir: str, optional

    :returns: None

    :raises FileNotFoundError: If no files matching the given prefix are found.
    :raises PermissionError: If there are insufficient permissions to create the binary directory.
    :raises IOError: If there's an error during the file conversion process.

    Example
    -------
    ::

        create_binaries("sample_data_", "./output_binaries")

    Notes
    -----
    - This function relies on helper functions `get_prefixes` and `_generate_binary_files`.
    - Ensure that the necessary permissions are available to create directories and files.
    - Creates a directory for binary files if it doesn't exist.
    - Converts identified FB TSV files to binary format.
    - Prints messages about the creation process.

    Dependencies
    ------------
    - `get_prefixes`: Function to get file prefixes.
    - `_generate_binary_files`: Function to convert TSV files to binary format.
    - `os.makedirs`: For creating directories.
