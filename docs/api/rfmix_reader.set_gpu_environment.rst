rfmix\_reader.set\_gpu\_environment
===================================

.. function:: set_gpu_environment()

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
    - `torch.cuda.device_count`: Counts the number of GPU devices
    - `torch.cuda.get_device_properties`: Gets device properties
    
    Example
    -------
    ::

        GPU 0: NVIDIA GeForce RTX 3080
          Total memory: 10.00 GB
          CUDA capability: 8.6
        GPU 1: NVIDIA GeForce RTX 3070
          Total memory: 8.00 GB
          CUDA capability: 8.6
