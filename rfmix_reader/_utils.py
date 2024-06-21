from torch.cuda import (
    device_count,
    get_device_properties,
)

__all__ = [
    "set_gpu_environment",
    "generate_binary_files",
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
    from numpy import float32, array
    with open(input_file, 'r') as infile, open(output_file, 'wb') as outfile:
        # Skip the first two rows
        next(infile)
        next(infile)
        # Read the remaining lines into a list
        lines = infile.readlines()
        # Convert the list of lines to a 2D NumPy array
        data = array([line.split()[4:] for line in lines], dtype=float32)
        # Write the binary data to the output file
        data.tofile(outfile)


def _process_file(file_path, temp_dir):
    from os.path import join, basename
    input_file = file_path
    output_file = join(temp_dir,
                       basename(file_path).split(".")[0] + ".bin")
    _text_to_binary(input_file, output_file)
    

def generate_binary_files(fb_files, temp_dir):
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor
    print("Converting fb files to binary!")
    with tqdm(total=len(fb_files)) as pbar:
        with ThreadPoolExecutor(max_workers=5) as executor:
            for _ in executor.map(_process_file, fb_files, [temp_dir] * len(fb_files)):
                pbar.update()

