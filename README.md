# RFMix-reader
`RFMix-reader` is a Python package designed to efficiently read and process output
files generated by [`RFMix`](https://github.com/slowkoni/rfmix), a popular tool 
for estimating local ancestry in admixed populations. The package employs a lazy
loading approach, which minimizes memory consumption by reading only the loci that
are accessed by the user, rather than loading the entire dataset into memory at 
once. Additionally, we leverage GPU acceleration to improve computational speed.

## Install
`rfmix-reader` can be installed using [pip](https://pypi.python.org/pypi/pip):

```bash
pip install rfmix-reader
```

**GPU Acceleration:**
`rfmix-reader` leverages GPU acceleration for improved performance. To use this
functionality, you will need to install the following libraries for your specific
CUDA version:
- `RAPIDS`: Refer to official installation guide [here](https://docs.rapids.ai/install)
- `PyTorch`: Installation instructions can be found [here](https://pytorch.org/)

**Additional Notes:**
- We have not tested installation with `Docker` or `Conda` environemnts. Compatibility
  may vary.
- If you do not have GPU, you can still use the basic functionality of `rfmix-reader`.
  This is still much faster than processing the files with stardard scripting.


## Key Features
**Lazy Loading**
- Reads data on-the-fly as requested, reducing memory footprint.
- Ideal for working with large RFMix output files that may not fit entirely in memory.

**Efficient Data Access**
- Provides convenient access to specific loci or regions of interest.
- Allows for selective loading of data, enabling faster processing times.

**Seamless Integration**
- Designed to work seamlessly with existing Python data analysis workflows.
- Facilitates downstream analysis and manipulation of `RFMix` output data.

**Loci Imputation**
- Designed to impute local ancestry loci to a larger genotype data genomic positions.
- Array-based data for ease of integration with downstream analysis.

Whether you are working with large-scale genomic datasets or have limited
computational resources, `RFMix-reader` offers an efficient and memory-conscious
solution for reading and processing `RFMix` output files. Its lazy loading approach
ensures optimal resource utilization, making it a valuable tool for researchers
and bioinformaticians working with admixed population data.

## Simulation Data
Simulation data is available for testing two and three population admixture on 
Synapse: [syn61691659](https://www.synapse.org/Synapse:syn61691659).

## Usage
This works similarly to [`pandas-plink`]():

### Two Population Admixture Example
This is a two-part process.

#### Generate Binary Files
To reduce computational time and memory, we leverage binary files.
While `RFMix` does not generate these directly, we provide a function
for their creation: `create_binaries`. This function can also be invoked 
via the command line:
`create-binaries [-h] [--version] [--binary_dir BINARY_DIR] file_prefix`.

```python
from rfmix_reader import create_binaries

# Generate binary files
file_path = "examples/two_popuations/out/"
binary_dir = "./binary_files"
create_binaries(file_path, binary_dir=binary_dir)
```

You can also do this on the fly.

```python
from rfmix_reader import read_rfmix

file_path = "examples/two_popuations/out/"
binary_dir = "./binary_files"
loci, rf_q, admix = read_rfmix(file_path, binary_dir=binary_dir,
                               generate_binary=True)
```

We do not have this turned on by default, as it is the
rate limiting step. It can take upwards of 20 to 25 minutes
to run depending on `*fb.tsv` file size.

#### Main Function
Once binary files are generated, you can the main function
to process the RFMix results. With GPU this takes less than
5 minutes.

```python
from rfmix_reader import read_rfmix

file_path = "examples/two_popuations/out/"
loci, rf_q, admix = read_rfmix(file_path)
```
**Note:** `./binary_files` is the default for `binary_dir`,
so this is an optional parameter.

### Three Population Admixture Example
`RFMix-reader` is adaptable for as many population admixtures as
needed.

```python
from rfmix_reader import read_rfmix

file_path = "examples/three_popuations/out/"
binary_dir = "./binary_files"
loci, rf_q, admix = read_rfmix(file_path, binary_dir=binary_dir,
                               generate_binary=True)
```

### Loci Imputation
Imputing local ancestry loci information to genotype variant locations improves
integration of the local ancestry information with genotype data. As such, we provide
the `interpolate_array` function to efficiently interpolate missing values when local
ancestry loci information is converted to more variable genotype variant locations. 
It leverages the power of [`Zarr`](https://zarr.readthedocs.io/en/stable/index.html) 
arrays, making it suitable for handling substantial datasets while managing memory 
usage effectively.

#### Features
- **CUDA Acceleration**: Uses CUDA for performance enhancement when available; 
  otherwise, it defaults to `NumPy`.
- **Chunk Processing**: Processes data in manageable chunks to optimize memory usage,
  making it ideal for large datasets.
- **Progress Monitoring**: Displays progress through a `tqdm` progress bar, providing
  real-time feedback during execution.
- **Column-wise Interpolation**: Employs the `_interpolate_col` function to perform 
  interpolation along each column of the dataset.

#### Example Usage
```python
import pandas as pd
import dask.array as da

# Outer merged dataframe of loci and variant locations
# "i" is from the loci information; "chrom" and "pos" from both dataframes
variant_loci_df = pd.DataFrame({'chrom': ['1', '1', '1', '1'], 
                                'pos': [100, 200, 300, 400], 
                                'i': [1, NA, NA, 2]})

# Dask array of admixture data, which will have few rows than variant_loci_df
admix = da.random.random((2, 3)) # Random data here

# This expands the Dask array (admix) and interpolates missing data
# Default chunk_size = 50000 assuming variant_loci_df 6-9M rows. 
# Adjust this based on variant_loci_df size.
z = interpolate_array(variant_loci_df, admix, '/path/to/output', chunk_size=1)

# Check the shape of the resulting Zarr array, which should have the same
# row numbers as variant_loci_df
print(z.shape)  # Output: (4, 3)
```

#### Example Preprocessing Functions
The helper functions `_load_genotypes` and `_load_admix` are designed to facilitate
the loading of loci and genotype data for constructing the `variant_loci_df`
DataFrame.

1. **`_load_genotypes(plink_prefix_path)`**: This function uses the `tensorqtl` 
   library to read genotype data from PLINK files (`PGEN`). It returns both the 
   loaded genotype data and a DataFrame containing variant information, which 
   includes chromosome and position details. The chromosome identifiers are 
   formatted to include the "chr" prefix for consistency.
2. **`_load_admix(prefix_path, binary_dir)`**: This function employs the 
   `rfmix_reader` library to load local ancestry data from specified paths. It 
   reads the ancestry information into a suitable format for further processing, 
   enabling integration with genotype data.

These functions ensure accurate loading and formatting of variant and local ancestry 
data, streamlining subsequent analyses.

```python
def _load_genotypes(plink_prefix_path):
    from tensorqtl import pgen
    pgr = pgen.PgenReader(plink_prefix_path)
    variant_df = pgr.variant_df
    variant_df.loc[:, "chrom"] = "chr" + variant_df.chrom
    return pgr.load_genotypes(), variant_df

def _load_admix(prefix_path, binary_dir):
    from rfmix_reader import read_rfmix
    return read_rfmix(prefix_path, binary_dir=binary_dir)

def __testing__():
	basename = "/projects/b1213/large_projects/brain_coloc_app/input"
    # Local ancestry
    prefix_path = f"{basename}/local_ancestry_rfmix/_m/"
    binary_dir = f"{basename}/local_ancestry_rfmix/_m/binary_files/"
    loci, _, admix = _load_admix(prefix_path, binary_dir)
    loci.rename(columns={"chromosome": "chrom",
                         "physical_position": "pos"},
                inplace=True)
    # Variant data
    plink_prefix = f"{basename}/genotypes/TOPMed_LIBD"
    _, variant_df = _load_genotypes(plink_prefix)
    variant_df = variant_df.drop_duplicates(subset=["chrom", "pos"],
                                            keep='first')
	# Keep all locations for more accurate imputation
    variant_loci_df = variant_df.merge(loci.to_pandas(), on=["chrom", "pos"],
                                       how="outer", indicator=True)\
                                .loc[:, ["chrom", "pos", "i", "_merge"]]
    data_path = f"{basename}/local_ancestry_rfmix/_m"
    z = interpolate_array(variant_loci_df, admix, data_path)
	# Match variant data genomic positions
    arr_geno = arr_mod.array(variant_loci_df[~(variant_loci_df["_merge"] == "right_only")].index)
    new_admix = z[arr_geno.get(), :]
```

**Note**: Following imputation, `variant_df` will include genomic positions for
both local ancestry and genotype data.

## Author(s)
* [Kynon JM Benjamin](https://github.com/Krotosbenjamin)

## Citation
If you use this software in your work, please cite it.
[![DOI](https://zenodo.org/badge/807052842.svg)](https://zenodo.org/doi/10.5281/zenodo.12629787)

Benjamin, K. J. M. (2024). RFMix-reader (Version v0.1.15) [Computer software]. 
https://github.com/heart-gen/rfmix_reader

Kynon JM Benjamin. "RFMix-reader: Accelerated reading and processing for
local ancestry studies." *bioRxiv*. 2024.
DOI: [10.1101/2024.07.13.603370](https://www.biorxiv.org/content/10.1101/2024.07.13.603370v2).

## Funding
This work was supported by grants from the National Institutes of Health,
National Institute on Minority Health and Health Disparities (NIMHD) 
K99MD016964 / R00MD016964.
