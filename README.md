# rfmix-reader
`rfmix-reader` is a Python package designed to efficiently read and process output 
files generated by RFMix, a popular tool for estimating local ancestry in admixed 
populations. The package employs a lazy loading approach, which minimizes memory 
consumption by reading only the loci that are accessed by the user, rather than 
loading the entire dataset into memory at once.

## Install
`rfmix-reader` can be installed using [pip](https://pypi.python.org/pypi/pip):

```bash
pip install rfmix-reader
```

## Key Features

**Lazy Loading**
- Reads data on-the-fly as requested, reducing memory footprint.
- Ideal for working with large RFMix output files that may not fit entirely in memory.

**Efficient Data Access**
- Provides convenient access to specific loci or regions of interest.
- Allows for selective loading of data, enabling faster processing times.

**Seamless Integration**
- Designed to work seamlessly with existing Python data analysis workflows.
- Facilitates downstream analysis and manipulation of RFMix output data.

Whether you're working with large-scale genomic datasets or have limited 
computational resources, RFMix-reader offers an efficient and memory-conscious 
solution for reading and processing RFMix output files. Its lazy loading approach 
ensures optimal resource utilization, making it a valuable tool for researchers 
and bioinformaticians working with admixed population data.

## Usage
This works similarly to [`pandas-plink`]():

```python
from rfmix_reader import read_rfmix

file_path = "examples/two_popuations/"
loci, rf_q, admix = read_rfmix(file_path)
```

## Authors
* [Kynon JM Benjamin](https://github.com/Krotosbenjamin)

## Citation

Please cite: XXXX.
