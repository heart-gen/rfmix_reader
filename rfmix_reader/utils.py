from __future__ import annotations

import shutil
from tqdm import tqdm
from os import makedirs, remove
from pathlib import Path
from re import search as rsearch
from numpy import float32
from typing import Any, Callable, Dict, List, Optional, Sequence
from multiprocessing import Pool, cpu_count
from os.path import basename, join, exists, isdir
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pandas import DataFrame

def _read_file(fn: Sequence[Any], read_func: Callable, pbar=None) -> List[Any]:
    """
    Apply a reader function across multiple input files.

    Parameters:
    ----------
    fn : list of str
        Paths to files (e.g., one per chromosome).
    read_func : callable
        Function that accepts a file path and returns a parsed object
        (e.g., DataFrame or Dask array).
    pbar : tqdm, optional
        Progress bar to update after each file is processed

    Returns:
    -------
    list
        List of objects returned by `read_func`, one per file.
    """
    data = []
    for file_name in fn:
        data.append(read_func(file_name))
        if pbar:
            pbar.update(1)
    return data


def _normalize_chrom_label(label: str) -> str:
    """Normalize chromosome labels by stripping a ``chr`` prefix and lowering."""

    label = label.lower()
    return label[3:] if label.startswith("chr") else label


def _extract_chrom_from_path(path: str) -> Optional[str]:
    """Best-effort extraction of a chromosome label from a file path."""

    base = basename(path).lower()
    match = rsearch(r"chr([a-z0-9]+)", base)
    if match:
        return match.group(1)

    fallback = rsearch(r"(?:[_\.])([0-9xy]+)(?:[^a-z0-9]|$)", base)
    if fallback:
        return fallback.group(1)

    return None


def filter_file_maps_by_chrom(
    file_maps: List[dict], chrom: Optional[str], *, kind: str = "dataset",
) -> List[dict]:
    """
    Filter file maps produced by :func:`get_prefixes` to a single chromosome.

    Parameters
    ----------
    file_maps
        List of dictionaries mapping suffixes to file paths.
    chrom
        Target chromosome label. When :data:`None`, the input is returned
        unchanged.
    kind
        Label used in error messages to clarify what is being filtered.
    """

    if chrom is None:
        return file_maps

    target = _normalize_chrom_label(str(chrom))
    filtered: List[dict] = []

    for fmap in file_maps:
        paths = list(fmap.values())
        chrom_label = _extract_chrom_from_path(paths[0]) if paths else None
        if chrom_label is None:
            continue
        if _normalize_chrom_label(chrom_label) == target:
            filtered.append(fmap)

    if not filtered:
        raise FileNotFoundError(
            f"No {kind} files found for chromosome '{chrom}'."
        )

    return filtered


def filter_paths_by_chrom(
    paths: List[str], chrom: Optional[str], *, kind: str = "VCF"
) -> List[str]:
    """Filter a list of file paths down to those matching ``chrom``."""

    if chrom is None:
        return paths

    target = _normalize_chrom_label(str(chrom))
    filtered: List[str] = []

    for path in paths:
        chrom_label = _extract_chrom_from_path(path)
        if chrom_label is None:
            continue
        if _normalize_chrom_label(chrom_label) == target:
            filtered.append(path)

    if not filtered:
        raise FileNotFoundError(
            f"No {kind} files found for chromosome '{chrom}'."
        )

    return filtered


def set_gpu_environment():
    """
    Reviews and prints the properties of available GPUs.

    This function checks the number of GPUs available on the system.
    If no GPUs are found, it prints a message indicating that no GPUs
    are available. If GPUs are found, it iterates through each GPU
    and prints its properties, including the name, total memory in gigabytes,
    and CUDA capability.

    The function relies on two external functions:

    - `device_count()`:
      Returns the number of GPUs available.
    - `get_device_properties(device_id)`:
      Returns the properties of the GPU with the given device ID.

    Raises
    ------
    Any exceptions raised by `device_count` or `get_device_properties`
    will propagate up to the caller.

    Dependencies
    ------------
    - torch.cuda.device_count: Counts the numer of GPU devices
    - torch.cuda.get_device_propoerties: Get device properties

    Example
    -------
    GPU 0: NVIDIA GeForce RTX 3080
      Total memory: 10.00 GB
      CUDA capability: 8.6
    GPU 1: NVIDIA GeForce RTX 3070
      Total memory: 8.00 GB
      CUDA capability: 8.6
    """
    try:
        from torch.cuda import device_count, get_device_properties
    except ImportError:
        print("PyTorch is not installed; GPU information is unavailable.")
        return
    num_gpus = device_count()
    if num_gpus == 0:
        print("No GPUs available.")
    else:
        for num in range(num_gpus):
            gpu_properties = get_device_properties(num)
            total_memory = gpu_properties.total_memory / (1024 ** 3)
            print(f"GPU {num}: {gpu_properties.name}")
            print(f"  Total memory: {total_memory:.2f} GB")
            print(f"  CUDA capability: {gpu_properties.major}.{gpu_properties.minor}")


_MODE_SUFFIXES: Dict[str, List[str]] = {
    "rfmix": ["fb.tsv", "fb.tsv.gz", "rfmix.Q", "rfmix.Q.gz"],
    "msp": ["msp.tsv", "msp.tsv.gz", "rfmix.Q", "rfmix.Q.gz"],
    "flare": ["anc.vcf.gz", "global.anc.gz"],
}
_ALL_SUFFIXES: List[str] = sorted(
    {sfx for sfxs in _MODE_SUFFIXES.values() for sfx in sfxs}, key=len, reverse=True
)


def _suffix_key(sfx: str) -> str:
    """Normalised file-map key for a suffix (``fb.tsv.gz`` -> ``fb.tsv``)."""
    return sfx[:-3] if sfx.endswith(".gz") else sfx


def _strip_known_suffix(path: str, suffixes: Sequence[str]) -> Optional[str]:
    """Return ``path`` without its (longest) known suffix, or ``None``."""
    for sfx in sorted(suffixes, key=len, reverse=True):
        if path.endswith("." + sfx):
            return path[: -len(sfx) - 1]
    return None


def _chrom_sort_key(prefix: str):
    """Sort prefixes by numeric chromosome first (chr2 before chr10), then name."""
    label = _extract_chrom_from_path(prefix)
    if label is None:
        return (2, 0, prefix)
    if label.isdigit():
        return (0, int(label), prefix)
    return (1, 0, label + prefix)


def _clean_prefixes(prefixes: Sequence[str], suffixes: Optional[Sequence[str]] = None) -> List[str]:
    """
    Reduce a list of file paths to unique, sorted path prefixes.

    Each path is stripped of its (longest) known output suffix, e.g.
    ``/out/cohort.v2_chr1.fb.tsv.gz`` -> ``/out/cohort.v2_chr1``.  Paths that
    do not end in a known suffix (logs, indexes, ...) are dropped.

    Parameters
    ----------
    prefixes : sequence of str
        File paths.
    suffixes : sequence of str, optional
        Suffixes to recognise.  Default: every suffix of every mode.
    """
    suffixes = list(suffixes) if suffixes else _ALL_SUFFIXES
    cleaned = []
    for path in prefixes:
        stem = _strip_known_suffix(str(path), suffixes)
        if stem is not None:
            cleaned.append(stem)
    return sorted(dict.fromkeys(cleaned), key=_chrom_sort_key)


def _discover_prefixes(file_prefix: str, suffixes: Sequence[str]) -> List[str]:
    """
    Find output-file prefixes for ``file_prefix``.

    ``file_prefix`` may be a directory (every file inside is considered), a
    complete file path, or a path prefix (``/out/run_`` matches
    ``/out/run_chr1.fb.tsv``, ``/out/run_chr2.fb.tsv``, ...).
    """
    p = Path(file_prefix)
    if p.is_dir():
        candidates = [str(x) for x in p.iterdir() if x.is_file()]
    elif p.is_file():
        candidates = [str(p)]
    else:
        candidates = [str(x) for x in p.parent.glob(p.name + "*") if x.is_file()]
    return _clean_prefixes(candidates, suffixes)


def _build_file_maps(prefixes: Sequence[str], suffixes: Sequence[str]) -> List[Dict[str, str]]:
    """Map each prefix to ``{normalised suffix: existing path}``; plain files win over ``.gz``."""
    fn = []
    for pfx in prefixes:
        filemap: Dict[str, str] = {}
        for sfx in suffixes:
            key = _suffix_key(sfx)
            if key in filemap:
                continue
            candidate = f"{pfx}.{sfx}"
            if exists(candidate):
                filemap[key] = candidate
        if filemap:
            fn.append(filemap)
    return fn


def get_prefixes(file_prefix: str, mode: str = "rfmix", verbose: bool = True) -> List[Dict[str, str]]:
    """
    Locate RFMix / FLARE output files and group them per chromosome.

    Parameters
    ----------
    file_prefix : str
        A directory containing the outputs, a single output file, or a common
        path prefix of the outputs (``"/out/run_"``).
    mode : {"rfmix", "msp", "flare"}
        - ``"rfmix"``: ``<prefix>.fb.tsv[.gz]`` and ``<prefix>.rfmix.Q[.gz]``
        - ``"msp"``:   ``<prefix>.msp.tsv[.gz]`` and ``<prefix>.rfmix.Q[.gz]``
        - ``"flare"``: ``<prefix>.anc.vcf.gz`` and ``<prefix>.global.anc.gz``
    verbose : bool, optional
        Print the order in which multiple file sets are read.

    Returns
    -------
    list of dict
        One dict per prefix mapping the normalised suffix (``"fb.tsv"``,
        ``"rfmix.Q"``, ``"msp.tsv"``, ``"anc.vcf"``, ``"global.anc"``) to the
        existing file path.  Only prefixes that have the primary file
        (first suffix of the mode) are returned.  Sorted by chromosome.

    Raises
    ------
    FileNotFoundError
        If no primary files are found.
    ValueError
        If ``mode`` is unknown.
    """
    if mode not in _MODE_SUFFIXES:
        raise ValueError(
            f"Invalid mode: {mode}. Choose from {list(_MODE_SUFFIXES.keys())}."
        )
    suffixes = _MODE_SUFFIXES[mode]
    primary = _suffix_key(suffixes[0])

    prefixes = _discover_prefixes(file_prefix, suffixes)
    fn = [m for m in _build_file_maps(prefixes, suffixes) if primary in m]
    if not fn:
        raise FileNotFoundError(
            f"No valid {mode.upper()} files found for prefix: {file_prefix}"
        )

    if len(fn) > 1 and verbose:
        names = [basename(m[primary]) for m in fn]
        print(f"Multiple {mode.upper()} file sets read in this order: {names}")
    return fn


def _text_to_binary(input_file: str, output_file: str):
    """
    Converts a text file to a binary file, skipping the first two rows
    and processing the remaining lines.

    This function reads an input text file, skips the first two rows,
    and processes each subsequent line. It extracts data starting from
    the fifth column, converts it to a NumPy array of type `float32`, and
    writes the binary representation of this data to an output file.

    Parameters
    ----------
    input_file (str): The path to the input text file.
    output_file (str): The path to the output binary file.

    Example
    -------
    Given an input file `data.txt` with the following content:
        Header1 Header2 Header3 Header4 Header5 Header6
        Header1 Header2 Header3 Header4 Header5 Header6
        1 2 3 4 5.0 6.0
        7 8 9 10 11.0 12.0

    The function will skip the first two header rows and process the
    remaining lines, extracting data starting from the fifth column.
    The resulting binary file will contain the binary representation
    of the following data:
        [5.0, 6.0]
        [11.0, 12.0]

    Note
    ----
    Ensure that the input file exists and is formatted correctly.
    The function assumes that the data to be processed starts from
    the fifth column of each line.

    Raises
    ------
    FileNotFoundError: If the input file does not exist.
    IOError: If there is an error reading from the input file or
             writing to the output file.
    """
    import pandas as pd

    input_file = Path(input_file); output_file = Path(output_file)

    # Pandas' C parser is 10-50x faster than Python line-by-line splitting for
    # wide float files (e.g. 9 GB .fb.tsv with ~175k rows × 2000 columns).
    # We skip the 2 header rows, drop the 4 metadata columns (chrom, pos, gpos,
    # snp_idx — which may contain strings), and write each chunk as a raw
    # float32 binary block in a single .tofile() call.
    # No global dtype= here because the metadata columns contain strings.
    CHUNK = 10_000
    with open(output_file, 'wb') as outfile:
        for chunk in pd.read_csv(
            input_file, sep=r"\s+", header=None, skiprows=2,
            compression="infer", chunksize=CHUNK,
        ):
            chunk.iloc[:, 4:].to_numpy(dtype=float32).tofile(outfile)


def _process_file(args):
    """
    Process a single file by converting it from text to binary format.

    This function takes a tuple of arguments containing a file path
    and a temporary directory path. It constructs an output file path
    in the temporary directory and calls the _text_to_binary function
    to perform the conversion.

    Parameters
    ----------
    args (tuple): A tuple containing two elements:
        - file_path (str): The path to the input text file to be
                           processed.
        - temp_dir (str): The path to the temporary directory
                          where the output will be stored.

    Returns
    -------
    None

    Side Effects
    ------------
    Creates a new binary file in the specified temporary directory.
    The output file name is derived from the input file name, with
    the extension changed to '.bin'.

    Example
    -------
    If args is ('/path/to/input/data.txt', '/tmp/processing/'), and
    assuming _text_to_binary is properly implemented, this function will:
    1. Create an output file path: '/tmp/processing/data.bin'
    2. Call _text_to_binary to convert '/path/to/input/data.txt' to
       '/tmp/processing/data.bin'
    """
    file_path, temp_dir = args
    file_path = Path(file_path)
    outname = basename(file_path).split(".")[0] + ".bin"
    output_file = join(temp_dir, outname)
    _text_to_binary(file_path, output_file)


def _generate_binary_files(fb_files, binary_dir, verbose: bool = True):
    """
    Convert multiple FB (Fullband) files to binary format using parallel processing.

    This function takes a list of FB file paths and a binary directory path, then
    converts each FB file to a binary format. It utilizes multiprocessing to speed up
    the conversion process by distributing the work across multiple CPU cores.

    Parameters
    ----------
    fb_files (list of str): A list of file paths to the FB files that
                            need to be converted.
    binary_dir (str): The path to the binary directory where the
                    output binary files will be stored.

    Returns
    -------
    None

    Performance
    -----------
    The function automatically determines the optimal number of CPU
    cores to use for parallel processing, which is the minimum of
    available CPU cores and the number of input files.

    Example
    -------
    _generate_binary_files(['/path/to/file1.fb.tsv', '/path/to/file2.fb.tsv'],
                            '/tmp/output/')

    Notes
    -----
    - The function uses the tqdm library to display a progress bar.
    - Any exceptions raised during the processing of individual files
      will be handled by the multiprocessing Pool and may interrupt
      the entire process.

    Side Effects
    ------------
    - Creates binary files in the specified binary directory for
      each input FB file.
    - Prints a message indicating the start of the conversion process.
    - Displays a progress bar during the conversion process.
    """
    if verbose:
        print("Converting fb files to binary!")
    # Determine the number of CPU cores to use
    num_cores = min(cpu_count(), len(fb_files))
    # Create a list of arguments for each file
    args_list = [(file_path, binary_dir) for file_path in fb_files]
    if num_cores <= 1:
        # Single file (or single core): convert in-process; no worker pool.
        for args in tqdm(args_list, total=len(fb_files), disable=not verbose):
            _process_file(args)
        return
    with Pool(num_cores) as pool:
        list(tqdm(pool.imap(_process_file, args_list),
                  total=len(fb_files), disable=not verbose))


def delete_files_or_directories(path_patterns):
    """
    Delete files or directories matching the given glob patterns.

    Parameters
    ----------
    path_patterns : list of str
        Glob patterns (``recursive=True``, so ``**`` is honoured).

    Notes
    -----
    Directories are removed recursively.  A message is printed for each
    deleted path; errors are reported and do not stop the remaining
    deletions.  Use with care.
    """
    from glob import glob

    for pattern in path_patterns:
        match_paths = glob(pattern, recursive=True)
        if not match_paths:
            print(f"Path does not exist: {pattern}")
        for path in match_paths:
            try:
                if isdir(path) and not Path(path).is_symlink():
                    shutil.rmtree(path)
                else:
                    remove(path)
                print(f"Deleted: {path}")
            except OSError as e:
                print(f"Error deleting {path}: {e}")


def get_pops(g_anc: DataFrame):
    """
    Extract population names from an RFMix Q-matrix DataFrame.

    This function removes the 'sample_id' and 'chrom' columns from
    the input DataFrame and returns the remaining column names, which
    represent population names.

    Parameters
    ----------
    g_anc (pd.DataFrame): A DataFrame containing RFMix Q-matrix data.
        Expected to have 'sample_id' and 'chrom' columns, along with
        population columns.

    Returns
    -------
    np.ndarray: An array of population names extracted from the column names.

    Example
    -------
    If g_anc has columns ['sample_id', 'chrom', 'pop1', 'pop2', 'pop3'],
    this function will return ['pop1', 'pop2', 'pop3'].

    Note
    ----
    This function assumes that all columns other than 'sample_id' and 'chrom'
    represent population names.
    """
    return g_anc.drop(["sample_id", "chrom"], axis=1).columns.values


def get_sample_names(g_anc: DataFrame):
    """
    Extract unique sample IDs from an RFMix Q-matrix DataFrame and
    convert to Arrow array.

    This function retrieves unique values from the 'sample_id' column
    of the input DataFrame and converts them to a PyArrow array.

    Parameters
    ----------
    g_anc (pd.DataFrame): A DataFrame containing RFMix Q-matrix data.
        Expected to have a 'sample_id' column.

    Returns
    -------
    pa.Array: A PyArrow array containing unique sample IDs.

    Example
    -------
    If g_anc has a 'sample_id' column with values ['sample1', 'sample2',
    'sample1', 'sample3'], this function will return a PyArrow array
    containing ['sample1', 'sample2', 'sample3'].

    Note
    ----
    This function assumes that the 'sample_id' column exists in the
    input DataFrame. It uses PyArrow on GPU for efficient memory
    management and interoperability with other data processing libraries.
    """
    if hasattr(g_anc, "to_pandas"):
        return g_anc.sample_id.unique().to_arrow()
    else:
        return g_anc.sample_id.unique()


def create_binaries(
        file_prefix: str, binary_dir: str = "./binary_files",
        chrom: Optional[str] = None, verbose: bool = True,
):
    """
    Convert RFMix ``.fb.tsv`` files into the raw float32 binaries used by
    :func:`read_rfmix_fb`.

    Parameters
    ----------
    file_prefix : str
        Directory, file, or path prefix identifying the ``.fb.tsv`` files
        (see :func:`get_prefixes`).
    binary_dir : str, optional
        Output directory, created if needed.  Default ``"./binary_files"``.
    chrom : str, optional
        Only convert the file for this chromosome.
    verbose : bool, optional
        Print progress.

    Raises
    ------
    FileNotFoundError
        If no ``.fb.tsv`` files are found.
    RuntimeError
        If both ``<prefix>.fb.tsv`` and ``<prefix>.fb.tsv.gz`` exist, which
        makes the binary name ambiguous.
    OSError
        On permission or I/O errors.
    """
    fn = filter_file_maps_by_chrom(
        get_prefixes(file_prefix, "rfmix", False), chrom, kind="RFMix"
    )

    fb_files = []
    for f in fn:
        fb_path = f["fb.tsv"]  # normalized key, may be plain or gzipped file
        prefix = fb_path.replace(".fb.tsv.gz", "").replace(".fb.tsv", "")
        if Path(f"{prefix}.fb.tsv").exists() and Path(f"{prefix}.fb.tsv.gz").exists():
            raise RuntimeError(
                f"Both compressed and uncompressed FB files found for prefix {prefix}"
            )
        fb_files.append(fb_path)

    makedirs(binary_dir, exist_ok=True)
    if verbose:
        print(f"Created binary files at: {binary_dir}")
    _generate_binary_files(fb_files, binary_dir, verbose=verbose)
    if verbose:
        print(f"Successfully converted {len(fb_files)} files to binary format.")
