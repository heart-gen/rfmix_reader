__all__ = ["BinaryFileNotFoundError"]

class BinaryFileNotFoundError(FileNotFoundError):
    """
    Custom exception raised when a required binary file is not found.

    This exception provides detailed information about the missing file
    and offers suggestions for resolving the issue.

    Attributes
    ----------
        binary_fn (str): The name of the missing binary file.
        binary_dir (str): The directory where the binary file was expected.

    Example usage
    -------------
    raise BinaryFileNotFoundError(binary_fn, binary_dir)
    """

    def __init__(self, binary_fn, binary_dir):
        self.binary_fn = binary_fn
        self.binary_dir = binary_dir
        super().__init__(self._format_message())

    def _format_message(self):
        """
        Formats the error message with detailed information and suggestions.

        Returns:
            str: A formatted error message.
        """
        return f"""
        ## Error: Binary File Not Found

        The file '{self.binary_fn}' could not be found in the specified directory.

        ### Possible Solutions:

        1. Check the 'binary_dir' setting:
           - Current value: '{self.binary_dir}'
           - Ensure this path points to the location of your binary files.

        2. Generate missing binary files:
           - Use the following function in your RFMix working directory:
             `create_binaries(file_prefix, binary_dir)`

        ### Additional Information:

        - Make sure all required binary files are present in the specified directory.
        - Verify file permissions and path accessibility.
        
        If the problem persists, please review your file structure and RFMix configuration.
        """
