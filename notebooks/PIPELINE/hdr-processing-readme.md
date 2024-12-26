# HDR Image Processing Pipeline

This package provides tools for High Dynamic Range (HDR) image processing, including camera response function estimation, radiance map computation, and image denoising.

## Table of Contents
- [Installation](#installation)
- [Prerequisites](#prerequisites)
- [Core Functions](#core-functions)
- [Usage Examples](#usage-examples)
- [Available Weighting Functions](#available-weighting-functions)
- [File Structure](#file-structure)
- [Tips and Best Practices](#tips-and-best-practices)

## Installation

```bash
# Clone the repository
git clone [repository-url]


# Install required packages
pip install numpy scipy matplotlib reportlab
```

## Prerequisites

The following Python packages are required:
- NumPy
- SciPy
- Matplotlib
- ReportLab
- logging

## Core Functions

### 1. `process_hdr_images`
Main function to process HDR images from multiple exposures. This function takes .h5 files as raw data with keys appended to the end of file names that correspond to excitation wavelengths and emission wavelengths used. 

Dark current and saturation points must be precalculated and imported using 

```python
processed_data = process_hdr_images(
    directory="path/to/data",
    experiment_title="experiment_name",
    base_data_folder="data_folder",
    coefficients_dict={
        'Smax': smax_array,
        'Sd': sd_array,
        'b': bias_array
    },
    smoothing_lambda=1000,
    weighting_function=debevec_weight,
    num_sets=None
)
```

Parameters:
- `directory`: Path to the main data directory
- `experiment_title`: Name of the experiment
- `base_data_folder`: Name of the folder containing the data
- `coefficients_dict`: Dictionary containing camera coefficients
- `smoothing_lambda`: Smoothing parameter for response curve (default: 1000)
- `weighting_function`: Function to weight pixel values (default: debevec_weight)
- `num_sets`: Number of sets to process (optional)

### 2. `computeRadianceMap`
Computes the radiance map from multiple exposures.

```python
radiance_map = computeRadianceMap(
    images,
    exposure_times,
    Zmax_precomputed,
    smoothing_lambda=1000,
    return_all=False,
    crf=None,
    weighting_function=debevec_weight
)
```


## Available Weighting Functions

Several weighting functions are provided for different HDR reconstruction approaches:

1. `debevec_weight`: Standard Debevec weighting function
2. `robertson_weight`: Gaussian weighting function
3. `broadhat_weight`: Broad hat weighting function
4. `linear_weight`: Linear weighting function
5. `no_weight`: No weighting (constant weight of 1)
6. `square_weight`: Square function with threshold

Usage:
```python
from hdr_processing import debevec_weight, robertson_weight

# Use in process_hdr_images
processed_data = process_hdr_images(..., weighting_function=robertson_weight)
```

## File Structure

Expected directory structure:
```
experiment_folder/
├── base_data_folder/
│   ├── processed_data/
│   │   ├── image_set1_clip.npy
│   │   ├── image_set1_denoise.npy
│   │   ├── image_set1_clip_denoise.npy
│   │   └── ...
│   └── final_data/
│       ├── image_set1_radiance_map.npy
│       └── ...
```

## Usage Examples

### Basic HDR Processing

```python
import numpy as np
from hdr_processing import process_hdr_images, debevec_weight

# Load camera coefficients
coefficients = {
    'Smax': np.load('smax_coefficients.npy'),
    'Sd': np.load('sd_coefficients.npy'),
    'b': np.load('bias_coefficients.npy')
}

# Process HDR images
processed_data = process_hdr_images(
    directory='experiment_folder',
    experiment_title='test_experiment',
    base_data_folder='data',                 #HDR output folder
    coefficients_dict=coefficients
)

# Access results
radiance_map = processed_data[0]['radiance_map']
response_curve = processed_data[0]['response_curve']
```

### Custom Processing with Different Weighting

```python
from hdr_processing import robertson_weight

# Process with Robertson weighting
processed_data = process_hdr_images(
    directory='experiment_folder',
    experiment_title='test_experiment',
    base_data_folder='data',
    coefficients_dict=coefficients,
    weighting_function=broadhat_weight
)
```

## Tips and Best Practices

1. **Data Preparation**
   - Ensure images are properly aligned
   - Remove any corrupted or invalid frames
   - Verify exposure times are correct

2. **Memory Management**
   - Use `num_sets` parameter to process subsets of data if memory is limited
   - Consider using numpy's memory mapping for large datasets

3. **Performance Optimization**
   - Precompute Zmax values once and reuse
   - Use appropriate smoothing_lambda values (typically 100-1000)
   - Consider using faster weighting functions for large datasets

4. **Error Handling**
   - Check input data shapes match expected dimensions
   - Verify coefficient arrays match image dimensions
   - Monitor log output for warnings and errors

## Troubleshooting

Common issues and solutions:

1. **Broadcasting Errors**
   - Ensure Zmax values are properly shaped for weighting functions
   - Check array dimensions match in computation steps

2. **Memory Errors**
   - Reduce number of samples in sampleIntensities
   - Process fewer sets at once using num_sets parameter

3. **Invalid Results**
   - Verify exposure times are in correct units
   - Check camera coefficients are properly loaded
   - Ensure images are properly preprocessed

For additional support or bug reports, please open an issue in the repository.
