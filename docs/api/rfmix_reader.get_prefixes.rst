rfmix\_reader.get\_prefixes
===========================

.. function:: get_prefixes(file_prefix: str, verbose: bool = True)

    Retrieve and clean file prefixes for specified file types.

    This function searches for files with a given prefix, cleans
    the prefixes, and constructs a list of dictionaries mapping
    specific file types to their corresponding file paths.

    :param file_prefix: The prefix used to identify relevant files. This can be a directory or a common prefix for the files.
    :type file_prefix: str
    :param verbose: If True, show progress information; if False, suppress progress information. Default is True.
    :type verbose: bool, optional

    :returns: A list of dictionaries where each dictionary maps file types (e.g., "fb.tsv", "rfmix.Q") to their corresponding file paths.
    :rtype: list of dict

    :raises FileNotFoundError: If no files matching the given prefix are found.

    Example
    -------
    Given a directory structure::

        /data/
            chr1.fb.tsv
            chr1.rfmix.Q
            chr2.fb.tsv
            chr2.rfmix.Q

    Calling get_prefixes("/data/") will return::

        [
            {'fb.tsv': '/data/chr1.fb.tsv', 'rfmix.Q': '/data/chr1.rfmix.Q'},
            {'fb.tsv': '/data/chr2.fb.tsv
