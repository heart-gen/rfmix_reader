from torch.cuda import (
    device_count,
    get_device_properties,
)

__all__ = ["set_gpu_environment"]

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


