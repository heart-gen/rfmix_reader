rfmix\_reader.delete\_files\_or\_directories
============================================

.. function:: delete_files_or_directories(path_patterns)

    Deletes the specified files or directories using the 'rm -rf' command.

    This function takes a list of path patterns, finds all matching files
    or directories, and deletes them using the 'rm -rf' command. It prints
    a message for each deleted path and handles errors gracefully.

    :param path_patterns: A list of file or directory path patterns to delete. These patterns can include wildcards.
    :type path_patterns: list of str

    :returns: None

    Example
    -------
    ::

        delete_files_or_directories(['/tmp/test_dir/*', '/tmp/old_files/*.log'])

    Notes
    -----
    - This function uses the `glob` module to find matching paths
      and the `subprocess` module to execute the 'rm -rf' command.
    - Ensure that the paths provided are correct and that you have
      the necessary permissions to delete the specified files or
      directories.
    - Use this function with caution as it will permanently delete
      the specified files or directories.
    - Deletes files or directories that match the specified patterns.
    - Prints messages indicating the deletion status of each path.
    - Prints error messages if a path cannot be deleted.
