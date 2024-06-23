from tqdm import tqdm
from glob import glob
from os import makedirs
from pathlib import Path
from numpy import float32, array
from multiprocessing import Pool, cpu_count
from subprocess import run, CalledProcessError
from os.path import basename, dirname, join, exists
from torch.cuda import device_count, get_device_properties

__all__ = [
    "set_gpu_environment",
    "delete_files_or_directories",
    "get_prefixes", "create_binaries"
]

def set_gpu_environment():
    """
    Reviews and prints the properties of available GPUs.

    This function checks the number of GPUs available on the system. 
    If no GPUs are found, it prints a message indicating that no GPUs
    are available. If GPUs are found, it iterates through each GPU
    and prints its properties, including the name, total memory in gigabytes,
    and CUDA capability.

    The function relies on two external functions:
    - `device_count()`: Returns the number of GPUs available.
    - `get_device_properties(device_id)`: Returns the properties of the GPU
      with the given device ID.

    Example output
    --------------
    GPU 0: NVIDIA GeForce RTX 3080
      Total memory: 10.00 GB
      CUDA capability: 8.6
    GPU 1: NVIDIA GeForce RTX 3070
      Total memory: 8.00 GB
      CUDA capability: 8.6

    Note
    ----
    Ensure that the `device_count` and `get_device_properties` functions 
    are defined and accessible in the scope where this function is called.

    Raises
    ------
    Any exceptions raised by `device_count` or `get_device_properties`
    will propagate up to the caller.
    """
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


def _clean_prefixes(prefixes: list[str]):
    """
    Clean and filter a list of file prefixes.

    Parameters:
    ----------
    prefixes (list): A list of file prefixes (paths).

    Returns:
    -------
    list: A list of unique, cleaned file prefixes without the file extensions.

    Notes:
    -----
        - The function removes any prefixes that end with ".logs".
        - It also removes any duplicate prefixes after cleaning.
    """
    cleaned_prefixes = []
    for prefix in prefixes:
        # Split the prefix into directory and base name
        dir_path = dirname(prefix)
        base_name = basename(prefix)
        # Remove the file extensions from the base name
        base = base_name.split(".")[0]        
        # Skip prefixes that end with ".logs"
        if base.startswith("chr"):
            cleaned_prefix = join(dir_path, base)
            cleaned_prefixes.append(cleaned_prefix)
    # Remove duplicate prefixes
    return list(set(cleaned_prefixes))


def get_prefixes(file_prefix: str):
    # Get file prefixes
    file_prefixes = sorted([str(x) for x in Path(file_prefix).glob("chr*")])
    if len(file_prefixes) == 1:
        file_prefixes = sorted(glob(join(file_prefix, "*")))
    file_prefixes = sorted(_clean_prefixes(file_prefixes))
    fn = [{s: f"{fp}.{s}" for s in ["fb.tsv", "rfmix.Q"]} for fp in file_prefixes]
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
    with open(input_file, 'r') as infile, open(output_file, 'wb') as outfile:
        # Skip the first two rows
        next(infile)
        next(infile)
        # Process and write each line individually
        for line in infile:
            data = array(line.split()[4:], dtype=float32)
            # Write the binary data to the output file
            data.tofile(outfile)


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
    input_file = file_path
    output_file = join(temp_dir,
                       basename(file_path).split(".")[0] + ".bin")
    _text_to_binary(input_file, output_file)


def _generate_binary_files(fb_files, binary_dir):
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

    Side Effects
    ------------
    - Creates binary files in the specified binary directory for
      each input FB file.
    - Prints a message indicating the start of the conversion process.
    - Displays a progress bar during the conversion process.

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
    """
    print("Converting fb files to binary!")
    # Determine the number of CPU cores to use
    num_cores = min(cpu_count(), len(fb_files))
    # Create a list of arguments for each file
    args_list = [(file_path, binary_dir) for file_path in fb_files]
    with Pool(num_cores) as pool:
        list(tqdm(pool.imap(_process_file, args_list),
                  total=len(fb_files)))


def delete_files_or_directories(path_patterns):
    """
    Deletes the specified files or directories using the 'rm -rf' command.

    This function takes a list of path patterns, finds all matching files
    or directories, and deletes them using the 'rm -rf' command. It prints
    a message for each deleted path and handles errors gracefully.

    Parameters
    ----------
    path_patterns (list of str): A list of file or directory path
                                 patterns to delete. These patterns
                                 can include wildcards.

    Returns
    -------
    None

    Side Effects
    ------------
    - Deletes files or directories that match the specified patterns.
    - Prints messages indicating the deletion status of each path.
    - Prints error messages if a path cannot be deleted.

    Example
    -------
    delete_files_or_directories(['/tmp/test_dir/*', '/tmp/old_files/*.log'])

    Notes
    -----
    - This function uses the 'glob' module to find matching paths
      and the 'subprocess' module to execute the 'rm -rf' command.
    - Ensure that the paths provided are correct and that you have
      the necessary permissions to delete the specified files or
      directories.
    - Use this function with caution as it will permanently delete
      the specified files or directories.
    """
    for pattern in path_patterns:
        match_paths = glob(pattern, recursive=True)
        for path in match_paths:
            if exists(path):
                try:
                    # Use subprocess to call 'rm -rf' on the path
                    run(['rm', '-rf', path], check=True)
                    print(f"Deleted: {path}")
                except CalledProcessError as e:
                    print(f"Error deleting {path}: {e}")
            else:
                print(f"Path does not exist: {path}")


def create_binaries(file_prefix: str, binary_dir: str):
    fn = get_prefixes(file_prefix)
    fb_files = [f["fb.tsv"] for f in fn]
    makedirs(binary_dir, exist_ok=True)
    print(f"Created binary files at: {binary_dir}")
    _generate_binary_files(fb_files, binary_dir)

