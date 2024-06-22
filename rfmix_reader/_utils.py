from tqdm import tqdm
from glob import glob
from numpy import float32, array
from os.path import join, basename, exists
from multiprocessing import Pool, cpu_count
from subprocess import run, CalledProcessError
from torch.cuda import (
    device_count,
    get_device_properties,
)

__all__ = [
    "set_gpu_environment",
    "generate_binary_files",
    "delete_files_or_directories",
]

def set_gpu_environment():
    """
    This function reviews GPU properties.
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


def _text_to_binary(input_file, output_file):
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
    file_path, temp_dir = args
    input_file = file_path
    output_file = join(temp_dir,
                       basename(file_path).split(".")[0] + ".bin")
    _text_to_binary(input_file, output_file)


def generate_binary_files(fb_files, temp_dir):
    print("Converting fb files to binary!")
    # Determine the number of CPU cores to use
    num_cores = min(cpu_count(), len(fb_files))
    # Create a list of arguments for each file
    args_list = [(file_path, temp_dir) for file_path in fb_files]
    with Pool(num_cores) as pool:
        list(tqdm(pool.imap(_process_file, args_list),
                  total=len(fb_files)))


def delete_files_or_directories(path_patterns):
    """
    Deletes the specified files or directories using 'rm -rf'.
    
    Parameters:
    paths (list): List of file or directory paths to delete.
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
