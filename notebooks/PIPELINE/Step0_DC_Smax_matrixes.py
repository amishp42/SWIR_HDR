import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from reportlab.lib.pagesizes import letter, landscape
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
from reportlab.lib.utils import ImageReader
import io
from scipy.optimize import curve_fit
from scipy import stats
from datetime import datetime

# Data Import Functions
def import_reflectance_data(directory, file_pattern=None):
    """
    Import reflectance measurement data from H5 files in the specified directory.
    """
    if file_pattern:
        reflectance_files = [f for f in os.listdir(directory) 
                            if f.startswith(file_pattern) and f.endswith('.h5')]
    else:
        reflectance_files = [f for f in os.listdir(directory) if f.endswith('.h5')]
    
    if not reflectance_files:
        raise ValueError(f"No H5 files found in {directory}" + 
                       (f" matching pattern '{file_pattern}'" if file_pattern else ""))
    
    try:
        reflectance_files.sort(key=lambda x: float(x.split('_')[-1][:-3]))
    except (IndexError, ValueError) as e:
        print("Warning: Could not sort files by exposure time from filenames.")
        print("Files will be processed in filesystem order.")
    
    reflectance_data = []
    exposure_times = []
    first_shape = None
    
    for file in reflectance_files:
        file_path = os.path.join(directory, file)
        try:
            with h5py.File(file_path, 'r') as h5f:
                if 'Cube' in h5f and 'Images' in h5f['Cube']:
                    reflectance = h5f['Cube']['Images'][()]
                else:
                    raise ValueError(f"Required dataset 'Cube/Images' not found in {file}")
                
                if 'TimeExposure' in h5f['Cube']:
                    exposure_time = h5f['Cube']['TimeExposure'][()].item()
                else:
                    raise ValueError(f"Required dataset 'Cube/TimeExposure' not found in {file}")
                
                if first_shape is None:
                    first_shape = reflectance.shape
                elif reflectance.shape != first_shape:
                    raise ValueError(f"Inconsistent dimensions in {file}. " +
                                  f"Expected {first_shape}, got {reflectance.shape}")
                
                reflectance_data.append(reflectance)
                exposure_times.append(exposure_time)
                
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            raise
    
    reflectance_array = np.array(reflectance_data)
    exposure_times = np.array(exposure_times)
    reflectance_array = reflectance_array.squeeze()
    
    print(f"Loaded {len(reflectance_files)} files")
    print(f"Shape of reflectance_array: {reflectance_array.shape}")
    print(f"Shape of exposure_times: {exposure_times.shape}")
    print(f"Exposure time range: {exposure_times.min():.2e} to {exposure_times.max():.2e} seconds")
    
    return reflectance_array, exposure_times

# Curve Fitting Functions
def linear_to_asymptote(t, slope, intercept, Smax, smoothness):
    """
    S-curve function that's linear before transitioning to asymptote.
    """
    linear_term = slope * t + intercept
    exp_term = np.exp(-smoothness * (linear_term - Smax))
    exp_term = np.clip(exp_term, -1e15, 1e15)  # Prevent overflow
    return Smax - (1/smoothness) * np.log1p(exp_term)

def sigmoid_curve(t, a, b, c, d):
    """
    Generalized sigmoidal function for log-linear fitting.
    
    Args:
        t: exposure times
        a: amplitude parameter
        b: center point (in log space)
        c: steepness parameter
        d: vertical offset
    
    Returns:
        Sigmoidal curve values
    """
    return a / (1 + np.exp(-c * (np.log10(t) - b))) + d


def log_linear_range_sigmoid(b, c):
    """
    Calculate Slinear point for sigmoid fit based on parameters.
    
    Args:
        b: center point in log space
        c: steepness parameter
    
    Returns:
        float: value representing Slinear point in log space
    """
    # Point where sigmoid reaches ~95% of its maximum value
    return b + 1.317 * c

def is_approximately_monotonic(y, tolerance=0.01):
    """
    Check if the sequence is approximately monotonic.
    """
    diff = np.diff(y)
    negative_diffs = diff[diff < 0]
    if len(negative_diffs) == 0:
        return True
    
    max_allowed_negative = tolerance * np.max(y)
    return np.all(np.abs(negative_diffs) <= max_allowed_negative)

def find_monotonic_range(y, tolerance=0.01):
    """
    Find the largest approximately monotonic range from the beginning of the sequence.
    """
    for i in range(len(y), 0, -1):
        if is_approximately_monotonic(y[:i], tolerance):
            return i
    return 1

def find_log_linear_range(exposure_times, pixel_data, threshold=0.01):
    """
    Find the linear range in log space.
    """
    log_times = np.log10(exposure_times)
    log_data = np.log10(pixel_data)
    
    window_size = min(5, len(log_times) // 2)
    slopes = []
    for i in range(len(log_times) - window_size + 1):
        x = log_times[i:i+window_size]
        y = log_data[i:i+window_size]
        slope, _ = np.polyfit(x, y, 1)
        slopes.append(slope)
    
    slopes = np.array(slopes)
    mean_slope = np.median(slopes)
    deviations = np.abs(slopes - mean_slope) / mean_slope
    nonlinear_points = np.where(deviations > threshold)[0]
    
    if len(nonlinear_points) > 0:
        end_idx = nonlinear_points[0] + window_size//2
    else:
        end_idx = len(exposure_times)
    
    return 0, min(end_idx, len(exposure_times))

# Main Analysis Function
def analyze_light_response(light_array, exposure_times, threshold=0.01, fit_method='linear'):
    """
    Analyze light response data to find per-pixel Slinear and Smax values.
    """
    if fit_method not in ['linear', 'log_linear']:
        raise ValueError("fit_method must be either 'linear' or 'log_linear'")
    
    height, width = light_array.shape[1:]
    Slinear = np.zeros((height, width))
    Smax = np.zeros((height, width))
    
    fit_params = {
        'slope': np.zeros((height, width), dtype=np.float64),
        'intercept': np.zeros((height, width), dtype=np.float64),
        'smoothness': np.zeros((height, width), dtype=np.float64)
    }
    
    fit_quality = {
        'r_squared': np.zeros((height, width)),
        'fit_error': np.zeros((height, width), dtype=bool),
        'is_monotonic': np.zeros((height, width), dtype=bool)
    }
    
    for i in range(height):
        for j in range(width):
            pixel_data = light_array[:, i, j]
            
            if not is_approximately_monotonic(pixel_data, tolerance=0.05):
                fit_quality['fit_error'][i, j] = True
                continue
                
            fit_quality['is_monotonic'][i, j] = True
            
            try:
                if fit_method == 'log_linear':
                    # Initial parameter guesses for sigmoid
                    max_val = np.max(pixel_data)
                    min_val = np.min(pixel_data)
                    range_val = max_val - min_val
                    
                    # Convert to log space for initial fit
                    log_times = np.log10(exposure_times)
                    median_time = np.median(log_times)
                    
                    p0 = [
                        range_val,        # amplitude
                        median_time,      # center point in log space
                        0.5,             # steepness (smaller initial value)
                        min_val          # offset
                    ]
                    
                    # Ensure bounds prevent negative offsets and unrealistic values
                    bounds = (
                        [0.1*range_val, np.min(log_times), 0.1, min_val],     # lower bounds
                        [2.0*range_val, np.max(log_times), 2.0, max_val]      # upper bounds
                    )
                    
                    try:
                        popt, pcov = curve_fit(
                            sigmoid_curve,
                            exposure_times,
                            pixel_data,
                            p0=p0,
                            bounds=bounds,
                            method='trf',
                            maxfev=10000
                        )
                        
                        # Store parameters for sigmoid fit
                        fit_params['slope'][i, j] = popt[0]  # amplitude
                        fit_params['intercept'][i, j] = popt[1]  # center
                        fit_params['smoothness'][i, j] = popt[2]  # steepness
                        
                        # Smax is the upper asymptote of sigmoid
                        Smax[i, j] = popt[0] + popt[3]
                        
                        # Calculate Slinear using the new formula
                        log_slinear = log_linear_range_sigmoid(popt[1], popt[2])
                        Slinear[i, j] = sigmoid_curve(10**log_slinear, *popt)
                        
                    except Exception as e:
                        print(f"Sigmoid fitting failed for pixel ({i}, {j}): {str(e)}")
                        fit_quality['fit_error'][i, j] = True
                        continue
                else:
                    # Linear fitting
                    initial_slope = (pixel_data[5] - pixel_data[0]) / (exposure_times[5] - exposure_times[0])
                    p0 = [initial_slope, pixel_data[0], np.max(pixel_data) * 1.2, 0.1]
                    bounds = ([0, 0, np.max(pixel_data), 0], [np.inf, np.inf, np.inf, 10])
                    
                    popt, _ = curve_fit(
                        linear_to_asymptote,
                        exposure_times,
                        pixel_data,
                        p0=p0,
                        bounds=bounds,
                        method='trf',
                        maxfev=10000
                    )
                
                # Store parameters
                fit_params['slope'][i, j] = popt[0]
                fit_params['intercept'][i, j] = popt[1]
                Smax[i, j] = popt[0] + popt[3]
                print(Smax[i,j])
                fit_params['smoothness'][i, j] = popt[3]
                
                # Calculate Slinear
                if fit_method == 'log_linear':
                    log_linear_response = 10**(popt[0] * np.log10(exposure_times) + popt[1])
                    actual_response = sigmoid_curve(exposure_times, *popt)
                else:
                    linear_response = popt[0] * exposure_times + popt[1]
                    actual_response = linear_to_asymptote(exposure_times, *popt)
                
                relative_deviation = np.abs(
                    actual_response - (log_linear_response if fit_method == 'log_linear' else linear_response)
                ) / (log_linear_response if fit_method == 'log_linear' else linear_response)
                
                nonlinear_points = np.where(relative_deviation > threshold)[0]
                if len(nonlinear_points) > 0:
                    Slinear[i, j] = actual_response[nonlinear_points[0]]
                else:
                    Slinear[i, j] = Smax[i, j]
                
                # Calculate R-squared
                y_pred = sigmoid_curve(exposure_times, *popt) if fit_method == 'log_linear' else \
                        linear_to_asymptote(exposure_times, *popt)
                ss_res = np.sum((pixel_data - y_pred) ** 2)
                ss_tot = np.sum((pixel_data - np.mean(pixel_data)) ** 2)
                fit_quality['r_squared'][i, j] = 1 - (ss_res / ss_tot)
                
            except Exception as e:
                print(f"Fitting failed for pixel ({i}, {j}): {str(e)}")
                fit_quality['fit_error'][i, j] = True
                Slinear[i, j] = np.max(pixel_data)
                Smax[i, j] = np.max(pixel_data) * 1.2
    
    return Slinear, Smax, fit_params, fit_quality

# Visualization Functions
def select_random_pixels(height, width, n_pixels, seed=None):
    """
    Select random pixel coordinates.
    """
    if seed is not None:
        np.random.seed(seed)
    
    i_coords = np.random.randint(0, height, n_pixels, dtype=np.int32)
    j_coords = np.random.randint(0, width, n_pixels, dtype=np.int32)
    return [(int(i), int(j)) for i, j in zip(i_coords, j_coords)]

def plot_pixel_responses(light_array, exposure_times, pixel_coords, 
                        Slinear, Smax, fit_params, fit_method='linear', random_pixels=None):
    """
    Plot light response curves for selected pixels.
    """
    if random_pixels is not None:
        height, width = light_array.shape[1:]
        pixel_coords = select_random_pixels(height, width, random_pixels)
    
    if not pixel_coords:
        raise ValueError("No pixel coordinates provided or selected")
    
    n_pixels = len(pixel_coords)
    fig, axes = plt.subplots(n_pixels, 2, figsize=(15, 5*n_pixels))
    
    if n_pixels == 1:
        axes = axes.reshape(1, -1)
    
    x_fine = np.logspace(np.log10(min(exposure_times)), np.log10(max(exposure_times)), 1000)
    
    for idx, (i, j) in enumerate(pixel_coords):
        i, j = int(i), int(j)
        pixel_data = light_array[:, i, j]
        
        try:
            if fit_method == 'log_linear':
                # Parameters for sigmoid fit
                amplitude = float(fit_params['slope'][i, j])
                center = float(fit_params['intercept'][i, j])
                steepness = float(fit_params['smoothness'][i, j])
                offset = float(Smax[i, j]) - amplitude
                
                popt = [amplitude, center, steepness, offset]
                y_fine = sigmoid_curve(x_fine, *popt)
                
            else:
                # Parameters for linear fit
                slope = float(fit_params['slope'][i, j])
                intercept = float(fit_params['intercept'][i, j])
                smax_val = float(Smax[i, j])
                smoothness = float(fit_params['smoothness'][i, j])
                
                popt = [slope, intercept, smax_val, smoothness]
                y_fine = linear_to_asymptote(x_fine, *popt)
                
            # Linear scale plot
            axes[idx, 0].scatter(exposure_times, pixel_data, label='Data')
            axes[idx, 0].plot(x_fine, y_fine, 'r-', label='Fitted Curve')
            axes[idx, 0].axhline(y=Slinear[i, j], color='b', linestyle=':', 
                                label='Slinear')
            axes[idx, 0].axhline(y=Smax[i, j], color='r', linestyle=':', 
                                label='Smax')
            axes[idx, 0].set_xlabel('Exposure Time (s)')
            axes[idx, 0].set_ylabel('Pixel Value')
            axes[idx, 0].set_title(f'Pixel ({i}, {j}) - Linear Scale')
            axes[idx, 0].legend()
            
            # Log scale plot
            axes[idx, 1].scatter(exposure_times, pixel_data, label='Data')
            axes[idx, 1].plot(x_fine, y_fine, 'r-', label='Fitted Curve')
            axes[idx, 1].axhline(y=Slinear[i, j], color='b', linestyle=':', 
                                label='Slinear')
            axes[idx, 1].axhline(y=Smax[i, j], color='r', linestyle=':', 
                                label='Smax')
            axes[idx, 1].set_xlabel('Exposure Time (s)')
            axes[idx, 1].set_ylabel('Pixel Value')
            axes[idx, 1].set_title(f'Pixel ({i}, {j}) - Log Scale')
            axes[idx, 1].set_xscale('log')
            axes[idx, 1].legend()
        
        except Exception as e:
            print(f"Error plotting pixel ({i}, {j}): {str(e)}")
            continue
    
    plt.tight_layout()
    return fig


def import_darkcount_data(directory):
    """
    Import darkcount data from H5 files in the specified directory.
    """
    # use this version if there is only darkcount data that can all be considered together in the folder (will consider all the files)
    darkcount_files = [f for f in os.listdir(directory) if f.endswith('.h5')]
    # use this version if it's a folder with mixed data -- include the consistent first part of the darkcount file names
    #darkcount_files = [f for f in os.listdir(directory) if f.startswith('darkcount') and f.endswith('.h5')]
    darkcount_files.sort(key=lambda x: float(x.split('_')[-1][:-3]))  # Sort by the number before .h5

    darkcount_data = []
    exposure_times = []

    for file in darkcount_files:
        file_path = os.path.join(directory, file)
        with h5py.File(file_path, 'r') as h5f:
            darkcount = h5f['Cube']['Images'][()]
            exposure_time = h5f['Cube']['TimeExposure'][()].item()
            darkcount_data.append(darkcount)
            exposure_times.append(exposure_time)

    darkcount_array = np.array(darkcount_data)
    darkcount_array = darkcount_array.squeeze()  # This will remove the extra dimension
    exposure_times = np.array(exposure_times)
    
    print(f"Shape of darkcount_array: {darkcount_array.shape}")
    print(f"Shape of exposure_times: {exposure_times.shape}")

    return darkcount_array, exposure_times

def analyze_darkcount_data(darkcount_array, exposure_times):
    """
    Analyze darkcount data and return summary statistics.
    """
    mean_values = np.mean(darkcount_array, axis=(1, 2))
    std_values = np.std(darkcount_array, axis=(1, 2))
    min_values = np.min(darkcount_array, axis=(1, 2))
    max_values = np.max(darkcount_array, axis=(1, 2))

    low_outliers = np.sum(darkcount_array < (mean_values[:, np.newaxis, np.newaxis] - 2 * std_values[:, np.newaxis, np.newaxis]), axis=(1, 2))
    high_outliers = np.sum(darkcount_array > (mean_values[:, np.newaxis, np.newaxis] + 2 * std_values[:, np.newaxis, np.newaxis]), axis=(1, 2))

    total_pixels = darkcount_array.shape[1] * darkcount_array.shape[2]
    low_outlier_percentages = (low_outliers / total_pixels) * 100
    high_outlier_percentages = (high_outliers / total_pixels) * 100

    return {
        'mean': mean_values,
        'std': std_values,
        'min': min_values,
        'max': max_values,
        'low_outliers': low_outlier_percentages,
        'high_outliers': high_outlier_percentages
    }

def plot_darkcount_images(darkcount_array, exposure_times):
    """
    Create plots of darkcount images with correct aspect ratio.
    """
    num_images = len(exposure_times)
    cols = min(3, num_images)
    rows = (num_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 15*rows/cols))
    axes = axes.flatten() if num_images > 1 else [axes]

    for i, (img, time) in enumerate(zip(darkcount_array, exposure_times)):
        im = axes[i].imshow(img, cmap='viridis', aspect='equal')
        axes[i].set_title(f'Exposure: {time:.4g} s')
        plt.colorbar(im, ax=axes[i])

    for i in range(num_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)

    return buf

def is_monotonic(y):
    return np.all(np.diff(y) >= 0)

def is_approximately_monotonic(y, tolerance=0.01):
    """
    Check if the sequence is approximately monotonic.
    
    Args:
    y : array-like
        The sequence to check.
    tolerance : float, optional
        The relative tolerance for non-monotonicity. Default is 0.05 (5%).
    
    Returns:
    bool
        True if the sequence is approximately monotonic, False otherwise.
    """
    diff = np.diff(y)
    negative_diffs = diff[diff < 0]
    if len(negative_diffs) == 0:
        return True
    
    # Calculate the maximum allowed negative difference
    max_allowed_negative = tolerance * np.max(y)
    
    return np.all(np.abs(negative_diffs) <= max_allowed_negative)

def find_monotonic_range(y, tolerance=0.01):
    """
    Find the largest approximately monotonic range from the beginning of the sequence.
    
    Args:
    y : array-like
        The sequence to check.
    tolerance : float, optional
        The relative tolerance for non-monotonicity. Default is 0.05 (5%).
    
    Returns:
    int
        The index where the approximately monotonic range ends.
    """
    for i in range(len(y), 0, -1):
        if is_approximately_monotonic(y[:i], tolerance):
            return i
    return 1


def plot_mean_darkcount(exposure_times, mean_values):
    """
    Create plots of mean darkcount value vs exposure time (linear and log scales).
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))
    
    # Linear scale
    ax1.scatter(exposure_times, mean_values)
    ax1.plot(exposure_times, mean_values, '--', alpha=0.5)  # Added line
    ax1.set_xlabel('Exposure Time (s)')
    ax1.set_ylabel('Mean Dark Current')
    ax1.set_title('Mean Dark Current vs Exposure Time (Linear Scale)')
    
    # Log scale
    ax2.scatter(exposure_times, mean_values)
    ax2.plot(exposure_times, mean_values, '--', alpha=0.5)  # Added line
    ax2.set_xlabel('Exposure Time (s)')
    ax2.set_ylabel('Mean Dark Current')
    ax2.set_title('Mean Dark Current vs Exposure Time (Log Scale)')
    ax2.set_xscale('log')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf

def plot_mean_darkcount_boxplot(exposure_times, darkcount_array):
    """
    Create box plots of dark current values vs exposure time (linear and log scales).
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))
    
    # Linear scale
    bp1 = ax1.boxplot([img.flatten() for img in darkcount_array], positions=exposure_times, widths=0.05*exposure_times)
    ax1.plot(exposure_times, [np.mean(img) for img in darkcount_array], '--', color='red', alpha=0.5)  # Added line
    ax1.set_xlabel('Exposure Time (s)')
    ax1.set_ylabel('Dark Current')
    ax1.set_title('Dark Current Distribution vs Exposure Time (Linear Scale)')
    
    # Log scale
    bp2 = ax2.boxplot([img.flatten() for img in darkcount_array], positions=exposure_times, widths=0.05*exposure_times)
    ax2.plot(exposure_times, [np.mean(img) for img in darkcount_array], '--', color='red', alpha=0.5)  # Added line
    ax2.set_xlabel('Exposure Time (s)')
    ax2.set_ylabel('Dark Current')
    ax2.set_title('Dark Current Distribution vs Exposure Time (Log Scale)')
    ax2.set_xscale('log')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf

def plot_specific_pixel_intensities(exposure_times, darkcount_array):
    """
    Plot specific pixel intensities across exposure times.
    """
    # Get specific pixel intensities
    brightest_pixel = darkcount_array[0].max()
    dimmest_pixel = darkcount_array[-1].min()
    median_pixel = np.median(darkcount_array[len(darkcount_array)//2])
    
    brightest_intensities = [img.max() for img in darkcount_array]
    dimmest_intensities = [img.min() for img in darkcount_array]
    median_intensities = [np.median(img) for img in darkcount_array]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))
    
    # Linear scale
    ax1.plot(exposure_times, brightest_intensities, 'r-', label='Brightest Pixel')
    ax1.plot(exposure_times, dimmest_intensities, 'b-', label='Dimmest Pixel')
    ax1.plot(exposure_times, median_intensities, 'g-', label='Median Pixel')
    ax1.scatter(exposure_times, brightest_intensities, c='r', marker='o')
    ax1.scatter(exposure_times, dimmest_intensities, c='b', marker='s')
    ax1.scatter(exposure_times, median_intensities, c='g', marker='^')
    ax1.set_xlabel('Exposure Time (s)')
    ax1.set_ylabel('Pixel Intensity')
    ax1.set_title('Specific Pixel Intensities vs Exposure Time (Linear Scale)')
    ax1.legend()
    
    # Log scale
    ax2.plot(exposure_times, brightest_intensities, 'r-', label='Brightest Pixel')
    ax2.plot(exposure_times, dimmest_intensities, 'b-', label='Dimmest Pixel')
    ax2.plot(exposure_times, median_intensities, 'g-', label='Median Pixel')
    ax2.scatter(exposure_times, brightest_intensities, c='r', marker='o')
    ax2.scatter(exposure_times, dimmest_intensities, c='b', marker='s')
    ax2.scatter(exposure_times, median_intensities, c='g', marker='^')
    ax2.set_xlabel('Exposure Time (s)')
    ax2.set_ylabel('Pixel Intensity')
    ax2.set_title('Specific Pixel Intensities vs Exposure Time (Log Scale)')
    ax2.set_xscale('log')
    ax2.legend()
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf

def linear_fit(t, slope, intercept):
    return slope * t + intercept

def calculate_slinear(exposure_times, darkcount_array, Sd, b, Smax, smooth, threshold=0.1, num_points=1000):
    num_pixels = darkcount_array.shape[1] * darkcount_array.shape[2]
    Slinear = np.zeros(num_pixels)
    
    # Create a finer time resolution
    fine_times = np.linspace(exposure_times.min(), exposure_times.max(), num_points)
    
    for i in range(num_pixels):
        sd = Sd[i // darkcount_array.shape[2], i % darkcount_array.shape[2]]
        bias = b[i // darkcount_array.shape[2], i % darkcount_array.shape[2]]
        smax = Smax[i // darkcount_array.shape[2], i % darkcount_array.shape[2]]
        smoothness = smooth[i // darkcount_array.shape[2], i % darkcount_array.shape[2]]
        
        linear_response = sd * fine_times + bias
        asymptotic_response = linear_to_asymptote(fine_times, sd, bias, smax, smoothness)
        
        relative_deviation = np.abs(asymptotic_response - linear_response) / linear_response
        nonlinear_indices = np.where(relative_deviation > threshold)[0]
        
        if len(nonlinear_indices) > 0:
            Slinear[i] = asymptotic_response[nonlinear_indices[0]]
        else:
            Slinear[i] = smax
    
    return Slinear.reshape(darkcount_array.shape[1], darkcount_array.shape[2])

def model_dark_current(darkcount_array, exposure_times, linear_range, global_popt):
    """Model dark current for each pixel using a two-step fitting process."""

    print("Starting model_dark_current function")
    print(f"Shape of darkcount_array: {darkcount_array.shape}")
    print(f"Number of exposure times: {len(exposure_times)}")
    print(f"Linear range: {linear_range}")
    print(f"Global popt: {global_popt}")

    linear_mask = (exposure_times >= linear_range[0]) & (exposure_times <= linear_range[1])
    
    if not np.any(linear_mask):
        print("Warning: No exposure times fall within the calculated linear range.")
        print(f"Linear range: {linear_range}")
        print(f"Exposure times: {exposure_times}")
        linear_mask = np.ones_like(exposure_times, dtype=bool)
    
    linear_exposure_times = exposure_times[linear_mask]
    linear_darkcount_data = darkcount_array[linear_mask]
    
    if linear_darkcount_data.size == 0:
        raise ValueError("No data points in the linear range. Check your linear_range and exposure_times.")
    
    num_pixels = darkcount_array.shape[1] * darkcount_array.shape[2]
    darkcount_data_reshaped = darkcount_array.reshape(darkcount_array.shape[0], -1)
    
    Sd = np.zeros(num_pixels)
    b = np.zeros(num_pixels)
    Smax = np.zeros(num_pixels)
    smooth = np.zeros(num_pixels)
    
    global_fit_count = 0
    global_Smax = global_popt[2]
    global_smoothness = global_popt[3]
    fit_failure_reasons = []
    non_monotonic_count = 0

    for i in range(num_pixels):
        if i % 10000 == 0:
            print(f"Processing pixel {i}/{num_pixels}")
        pixel_data = darkcount_data_reshaped[:, i]
        
        # Check if pixel response is approximately monotonic
        if not is_approximately_monotonic(pixel_data, tolerance=0.05):
            non_monotonic_count += 1
            if non_monotonic_count <= 5:
                print(f"Non-monotonic pixel found: {i}")
                print(f"Pixel data: {pixel_data}")
            # Find the largest approximately monotonic range from the beginning
            monotonic_range = find_monotonic_range(pixel_data, tolerance=0.05)
            linear_pixel_data = pixel_data[:monotonic_range]
            linear_times = exposure_times[:monotonic_range]
        else:
            linear_pixel_data = pixel_data[linear_mask]
            linear_times = linear_exposure_times
        
        # Step 1: Linear fit in the linear range
        try:
            popt_linear, _ = curve_fit(linear_fit, linear_times, linear_pixel_data)
            Sd[i], b[i] = popt_linear
        except:
            Sd[i], b[i], _, _, _ = stats.linregress(linear_exposure_times, linear_pixel_data)
        
        # Step 2: Asymptotic fit using fixed Sd and b
        try:
            max_pixel_value = np.max(pixel_data)
            initial_Smax_guess = max(global_Smax, max_pixel_value * 1.1)
            
            if is_approximately_monotonic(pixel_data, tolerance=0.01):
                popt_asymptotic, _ = curve_fit(
                    lambda t, Smax, smoothness: linear_to_asymptote(t, Sd[i], b[i], Smax, smoothness),
                    exposure_times, pixel_data,
                    p0=[initial_Smax_guess, global_smoothness],
                    bounds=([max_pixel_value, 0], [np.inf, 10]),
                    method='trf',
                    max_nfev=10000
                )
                Smax[i], smooth[i] = popt_asymptotic
            else:
                # For non-monotonic pixels, use the maximum value as Smax
                Smax[i] = max_pixel_value
                smooth[i] = global_smoothness

            # Relaxed sanity check on Smax
            if Smax[i] > global_Smax * 2 or Smax[i] < max_pixel_value:
                raise ValueError(f"Unreasonable Smax: {Smax[i]}")
        except Exception as e:
            Smax[i] = global_Smax
            smooth[i] = global_smoothness
            global_fit_count += 1
            if len(fit_failure_reasons) < 10:
                fit_failure_reasons.append(f"Pixel {i}: Asymptotic fitting failed - {str(e)}")

    Sd = Sd.reshape(darkcount_array.shape[1], darkcount_array.shape[2])
    b = b.reshape(darkcount_array.shape[1], darkcount_array.shape[2])
    Smax = Smax.reshape(darkcount_array.shape[1], darkcount_array.shape[2])
    smooth = smooth.reshape(darkcount_array.shape[1], darkcount_array.shape[2])
    
    print(f"Finished processing all pixels")
    print(f"Total non-monotonic pixels: {non_monotonic_count}")
    print(f"Total global fits: {global_fit_count}")
    print("Sanity check of fitted parameters:")
    print(f"Sd range: {np.min(Sd)} to {np.max(Sd)}")
    print(f"b range: {np.min(b)} to {np.max(b)}")
    print(f"Smax range: {np.min(Smax)} to {np.max(Smax)}")
    print(f"smooth range: {np.min(smooth)} to {np.max(smooth)}")
    print(f"Number of Smax values equal to global Smax: {np.sum(Smax == global_Smax)}")
    print(f"Number of smooth values equal to global smoothness: {np.sum(smooth == global_smoothness)}")
    print(f"Number of non-monotonic pixels: {non_monotonic_count}")

     # Calculate Slinear
    Slinear = calculate_slinear(exposure_times, darkcount_array, Sd, b, Smax, smooth)
    
    return Sd, b, Smax, smooth, Slinear, global_fit_count, global_Smax, fit_failure_reasons, non_monotonic_count

def linear_to_asymptote(t, slope, intercept, Smax, smoothness):
    linear_term = slope * t + intercept
    exp_term = np.exp(-smoothness * (linear_term - Smax))
    exp_term = np.clip(exp_term, -1e15, 1e15)  # Prevent overflow
    return Smax - (1/smoothness) * np.log1p(exp_term)

def find_linear_range(exposure_times, mean_values, popt, threshold=0.01):
    """
    Find the linear range of the response curve, focusing on the initial linear part.
    """
    slope, intercept, Smax, smoothness = popt
    linear_prediction = slope * exposure_times + intercept
    actual_response = linear_to_asymptote(exposure_times, *popt)
    
    relative_deviation = np.abs(actual_response - linear_prediction) / linear_prediction
    
    # Find where the relative deviation exceeds the threshold
    nonlinear_indices = np.where(relative_deviation > threshold)[0]
    
    if len(nonlinear_indices) > 0:
        linear_range_end = exposure_times[nonlinear_indices[0]]
    else:
        linear_range_end = exposure_times[-1]
    
    # For the start, find where the relative deviation goes below the threshold
    linear_start_indices = np.where(relative_deviation < threshold)[0]
    if len(linear_start_indices) > 0:
        linear_range_start = exposure_times[linear_start_indices[0]]
    else:
        linear_range_start = exposure_times[0]
    
    return linear_range_start, linear_range_end

def fit_s_curve(exposure_times, mean_values):
    """
    Fit the data to the linear-to-asymptote equation and determine the linear range.
    """
    # Improved initial guesses
    initial_slope = (mean_values[5] - mean_values[0]) / (exposure_times[5] - exposure_times[0])
    p0 = [
        initial_slope,  # slope
        mean_values[0],  # intercept
        np.max(mean_values),  # Smax
        0.1  # smoothness
    ]
    
    # Add bounds to prevent unrealistic values
    bounds = ([0, 0, np.max(mean_values)*0.95, 0], [np.inf, np.inf, np.inf, 10])
    
    popt, _ = curve_fit(linear_to_asymptote, exposure_times, mean_values, 
                        p0=p0, bounds=bounds, maxfev=10000)
    
    return popt  # Just return popt, we'll calculate linear_range separately

def plot_s_curve_fit(exposure_times, mean_values, popt, linear_range):
    """
    Plot the original data, fitted curve, and linear range.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    x_fine = np.linspace(min(exposure_times), max(exposure_times), 1000)
    
    print(f"Debug - x_fine shape: {x_fine.shape}")
    print(f"Debug - popt in plot_s_curve_fit: {popt}")
    print(f"Debug - type of popt: {type(popt)}")
    print(f"Debug - linear_range: {linear_range}")
    
    if len(popt) != 4:
        print("Error: popt does not contain 4 parameters as expected.")
        y_fine = None
    else:
        try:
            y_fine = linear_to_asymptote(x_fine, *popt)
            print("Debug - y_fine calculated successfully")
        except Exception as e:
            print(f"Debug - Error in linear_to_asymptote: {str(e)}")
            y_fine = None
    
    for ax in (ax1, ax2):
        ax.scatter(exposure_times, mean_values, label='Data')
        if y_fine is not None:
            ax.plot(x_fine, y_fine, 'r-', label='Fitted Curve')
        ax.axvline(linear_range[0], color='g', linestyle='--', label='Linear Range Start')
        ax.axvline(linear_range[1], color='g', linestyle='--', label='Linear Range End')
        ax.set_xlabel('Exposure Time (s)')
        ax.set_ylabel('Mean Dark Current')
        ax.legend()
    
    ax1.set_title('Fit and Linear Range (Linear Scale)')
    ax2.set_title('Fit and Linear Range (Log Scale)')
    ax2.set_xscale('log')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf

def generate_darkcount_report(darkcount_array, exposure_times, analysis_results, output_dir, experiment_name, popt, linear_range):
    """
    Generate a PDF report of the darkcount analysis.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"{experiment_name}_darkcount_analysis_report_{timestamp}.pdf")

    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    # Add margins
    margin = 50
    content_width = width - 2*margin
    content_height = height - 2*margin

    # First page: Title and Table
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, height - margin, "Darkcount Analysis Report")

    # Table of summary statistics
    data = [['Exposure Time (s)', 'Mean', 'Std Dev', 'Min', 'Max', 'Low Outliers (%)', 'High Outliers (%)']]
    for i, time in enumerate(exposure_times):
        data.append([
            f"{time:.4g}",
            f"{analysis_results['mean'][i]:.2f}",
            f"{analysis_results['std'][i]:.2f}",
            f"{analysis_results['min'][i]:.2f}",
            f"{analysis_results['max'][i]:.2f}",
            f"{analysis_results['low_outliers'][i]:.2f}",
            f"{analysis_results['high_outliers'][i]:.2f}"
        ])

    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))

    # Calculate table dimensions and position
    table_width, table_height = table.wrapOn(c, content_width, content_height)
    table_x = margin
    table_y = height - margin - 30 - table_height  # 30 is space for title
    table.drawOn(c, table_x, table_y)

   # Second page: Darkcount images
    c.showPage()
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, height - margin, "Darkcount Images")

    img_buf = plot_darkcount_images(darkcount_array, exposure_times)
    img = ImageReader(img_buf)
    img_width, img_height = img.getSize()
    aspect = img_width / img_height
    c.drawImage(img, margin, margin, width=content_width, height=content_width/aspect)

    # Third page: Original mean darkcount vs exposure time plots
    c.showPage()
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, height - margin, "Mean Dark Current vs Exposure Time")

    plot_buf = plot_mean_darkcount(exposure_times, analysis_results['mean'])
    plot_img = ImageReader(plot_buf)
    c.drawImage(plot_img, margin, margin, width=content_width, height=content_height - 50)

    # Fourth page: Mean darkcount vs exposure time box plots
    c.showPage()
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, height - margin, "Dark Current Distribution vs Exposure Time")

    plot_buf = plot_mean_darkcount_boxplot(exposure_times, darkcount_array)
    plot_img = ImageReader(plot_buf)
    c.drawImage(plot_img, margin, margin, width=content_width, height=content_height - 50)

    # Fifth page: Specific pixel intensities plots
    c.showPage()
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, height - margin, "Specific Pixel Intensities vs Exposure Time")

    plot_buf = plot_specific_pixel_intensities(exposure_times, darkcount_array)
    plot_img = ImageReader(plot_buf)
    c.drawImage(plot_img, margin, margin, width=content_width, height=content_height - 50)


    # Model dark current
    Sd, b, Smax, smooth, Slinear, global_fit_count, global_Smax, fit_failure_reasons, non_monotonic_count = model_dark_current(darkcount_array, exposure_times, linear_range, popt)
    
    # Save model parameters
    save_model_parameters(Sd, b, Smax, smooth, Slinear, output_dir, experiment_name)
    

    c.showPage()
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, height - margin, "Curve Fit and Linear Range")

    plot_buf = plot_s_curve_fit(exposure_times, analysis_results['mean'], popt, linear_range)
    plot_img = ImageReader(plot_buf)
    c.drawImage(plot_img, margin, margin, width=content_width, height=content_height - 70)

    # Add fit parameters and linear range information
    c.setFont("Helvetica", 10)
    c.drawString(margin, margin - 20, f"Fit parameters: slope={popt[0]:.2f}, intercept={popt[1]:.2f}, Smax={popt[2]:.2f}, smoothness={popt[3]:.2f}")
    c.drawString(margin, margin - 40, f"Linear range: {linear_range[0]:.2f}s to {linear_range[1]:.2f}s")

    # Add model parameters page
    c.showPage()
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, height - margin, "Dark Current Model Parameters")

    plot_buf = plot_model_parameters(Sd, b, Smax, smooth, Slinear, global_fit_count, global_Smax, fit_failure_reasons, non_monotonic_count)
    plot_img = ImageReader(plot_buf)
    c.drawImage(plot_img, margin, margin, width=content_width, height=content_height - 50)

    c.save()
    print(f"Report saved as {filename}")
    print(f"Model parameters (Sd, b, and Smax) saved in {output_dir}")

def sort_by_exposure_time(darkcount_array, exposure_times):
    """
    Sort darkcount_array and exposure_times by ascending exposure times.
    """
    sorted_indices = np.argsort(exposure_times)
    sorted_darkcount_array = darkcount_array[sorted_indices]
    sorted_exposure_times = exposure_times[sorted_indices]
    return sorted_darkcount_array, sorted_exposure_times

def plot_model_parameters(Sd, b, Smax, smooth, Slinear, global_fit_count, global_Smax, fit_failure_reasons, non_monotonic_count):
    """Plot histograms of the model parameters and include summary statistics."""
    fig, axs = plt.subplots(3, 2, figsize=(12, 15))
    axs = axs.ravel()  # Flatten the 3x2 array to make indexing easier
    
    axs[0].hist(Sd.flatten(), bins=50)
    axs[0].set_xlabel('Sd (Dark Current Slope)')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title('Distribution of Sd')
    
    axs[1].hist(b.flatten(), bins=50)
    axs[1].set_xlabel('b (Bias)')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title('Distribution of b')
    
    axs[2].hist(Smax.flatten(), bins=50)
    axs[2].set_xlabel('Smax (Saturation Level)')
    axs[2].set_ylabel('Frequency')
    axs[2].set_title('Distribution of Smax')

    axs[3].hist(smooth.flatten(), bins=50)
    axs[3].set_xlabel('smoothness (fit parameter)')
    axs[3].set_ylabel('Frequency')
    axs[3].set_title('Distribution of smoothness')
    
    axs[4].hist(Slinear.flatten(), bins=50)
    axs[4].set_xlabel('Slinear (Linear Saturation Point)')
    axs[4].set_ylabel('Frequency')
    axs[4].set_title('Distribution of Slinear')
    
    # Use the sixth quadrant for summary statistics
    axs[5].axis('off')
    individual_fit_count = Smax.size - global_fit_count
    summary_text = (
        f"Summary Statistics:\n\n"
        f"Sd: {np.mean(Sd):.2f} ± {np.std(Sd):.2f}\n"
        f"b: {np.mean(b):.2f} ± {np.std(b):.2f}\n"
        f"smooth: {np.mean(smooth):.2f} ± {np.std(smooth):.2f}\n"
        f"Smax (all): {np.mean(Smax):.2f} ± {np.std(Smax):.2f}\n"
        f"Smax (individual fits): {np.mean(Smax[Smax != global_Smax]):.2f} ± {np.std(Smax[Smax != global_Smax]):.2f}\n"
        f"Slinear: {np.mean(Slinear):.2f} ± {np.std(Slinear):.2f}\n"
        f"Global Smax: {global_Smax:.2f}\n\n"
        f"Pixels with individual Smax fit: {individual_fit_count} out of {Smax.size}\n"
        f"Pixels using global Smax: {global_fit_count} out of {Smax.size}\n\n"
        f"Non-monotonic pixels: {non_monotonic_count} out of {Smax.size}\n\n"
        f"Sample fit failure reasons:\n" + "\n".join(fit_failure_reasons[:5])
    )
    axs[5].text(0.05, 0.95, summary_text, transform=axs[5].transAxes, 
                verticalalignment='top', fontsize=8)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf

def save_model_parameters(Sd, b, Smax, smooth, Slinear, output_dir, experiment_name):
    """Save Sd, b, Smax, smooth, and Slinear as NPY files."""
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, f'{experiment_name}_Sd.npy'), Sd.astype(np.float32))
    np.save(os.path.join(output_dir, f'{experiment_name}_b.npy'), b.astype(np.float32))
    np.save(os.path.join(output_dir, f'{experiment_name}_Smax.npy'), Smax.astype(np.float32))
    np.save(os.path.join(output_dir, f'{experiment_name}_smooth.npy'), smooth.astype(np.float32))
    np.save(os.path.join(output_dir, f'{experiment_name}_Slinear.npy'), Slinear.astype(np.float32))

    print(Slinear.shape)
    print(f"Model parameters saved in {output_dir}")
    print(f"Saved files: {experiment_name}_Sd.npy, {experiment_name}_b.npy, {experiment_name}_Smax.npy, {experiment_name}_smooth.npy, {experiment_name}_Slinear.npy")