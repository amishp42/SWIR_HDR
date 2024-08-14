import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import cv2
import io
import logging
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Image, Paragraph, PageBreak, Spacer, Flowable, BaseDocTemplate, PageTemplate, Frame
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from matplotlib.gridspec import GridSpec

logger = logging.getLogger(__name__)

"""
This module contains functions for High Dynamic Range (HDR) image processing,
including camera response function estimation, radiance map computation,
and saving the linear radiance map as a npy array. 8-bit HDR image generation 
and presentation is handled in a separate code module.
"""

class FilenamePlaceholder(Flowable):
    def __init__(self, filename):
        Flowable.__init__(self)
        self.filename = filename
    def draw(self):
        self.canv.setFont('Helvetica', 9)
        self.canv.drawString(0, 0, f"File: {self.filename}")

def load_clipped_denoised_data(directory, experiment_title, base_data_folder="data", extract=False):
    """
    Load clipped and denoised data from the final_data folder.
    
    Args:
    directory (str): The main directory containing the data folders.
    experiment_title (str): The title of the experiment.
    base_data_folder (str): The name of the base data folder.
    extract (bool): If True, extract data into separate standard arrays.
    
    Returns:
    dict: A dictionary containing the loaded data for each laser-filter combination.
    """
    experiment_folder = os.path.basename(os.path.normpath(directory))
    final_data_folder = os.path.join(base_data_folder, experiment_folder, "final_data")
    
    data_dict = {}
    
    for file in os.listdir(final_data_folder):
        if file.endswith("_clipped_denoised.npy"):
            key = file[:-21]  # Remove '_clipped_denoised.npy'
            file_path = os.path.join(final_data_folder, file)
            
            loaded_data = np.load(file_path)
            
            if extract:
                # Extract data into separate standard arrays
                images = loaded_data['image']
                exposure_times = loaded_data['exposure_time']
                data_dict[key] = {
                    'images': images,
                    'exposure_times': exposure_times
                }
            else:
                # Keep data in the structured array format
                data_dict[key] = loaded_data
    
    return data_dict

def precompute_zmax(Slinear, Sd, bias, exposure_times):
    """
    Precompute Zmax for all pixels and exposure times.
    
    Args:
    Slinear (numpy.ndarray): Array of linear saturation levels for each pixel.
    Sd (numpy.ndarray): Array of dark current slopes for each pixel.
    bias (numpy.ndarray): Array of bias values for each pixel.
    exposure_times (numpy.ndarray): Array of exposure times.
    
    Returns:
    numpy.ndarray: Array of Zmax values with shape [num_exposure_times, height, width]
    """
    num_exposures = len(exposure_times)
    height, width = Slinear.shape
    
    # Reshape arrays for broadcasting
    Slinear = Slinear.reshape(1, height, width)
    Sd = Sd.reshape(1, height, width)
    bias = bias.reshape(1, height, width)
    exposure_times = exposure_times.reshape(num_exposures, 1, 1)
    
    # Compute Zmax for all pixels and exposure times
    Zmax = Slinear - (Sd * exposure_times + bias)
    
    return Zmax

def adaptive_weight(z, Zmax):
    """
    Compute the weighting function for each pixel.
    """
    Zmin = 0  # Assuming the minimum possible pixel value is 0
    
    # Ensure z and Zmax are scalars or have the same shape
    if np.isscalar(z) and np.isscalar(Zmax):
        middle = (Zmax + Zmin) / 2
        if z <= middle:
            return (z - Zmin) / (middle - Zmin)
        else:
            return (Zmax - z) / (Zmax - middle)
    else:
        z = np.atleast_2d(z)
        Zmax = np.atleast_2d(Zmax)
        if z.shape != Zmax.shape:
            Zmax = np.full_like(z, Zmax)
        
        weight = np.zeros_like(z, dtype=np.float32)
        middle = (Zmax + Zmin) / 2
        weight[z <= middle] = (z[z <= middle] - Zmin) / (middle[z <= middle] - Zmin)
        weight[z > middle] = (Zmax[z > middle] - z[z > middle]) / (Zmax[z > middle] - middle[z > middle])
        return weight

def computeRadianceMap(images, exposure_times, Zmax_precomputed, smoothing_lambda=1000, return_all=False):
    """Calculate an unscaled radiance map for each pixel from the response curve."""
    logger.info("Starting computeRadianceMap function")
    
    intensity_samples, log_exposures, z_min, z_max = sampleIntensities(images, exposure_times, Zmax_precomputed)
    logger.info("Finished sampleIntensities function")
    
    response_curve = computeResponseCurve(intensity_samples, log_exposures, exposure_times, smoothing_lambda, adaptive_weight, z_min, z_max, Zmax_precomputed)
    response_curve = savgol_filter(response_curve, window_length=51, polyorder=3)

    num_images, height, width = images.shape
    radiance_map = np.zeros((height, width), dtype=np.float32)
    sum_weights = np.zeros((height, width), dtype=np.float32)

    for i in range(num_images):
        w = adaptive_weight(images[i], Zmax_precomputed[i] if Zmax_precomputed.ndim == 3 else Zmax_precomputed)
        
        # Convert images to integer indices
        indices = np.clip(np.round(images[i] - z_min).astype(int), 0, len(response_curve) - 1)
        
        radiance_map += w * (response_curve[indices] - np.log(exposure_times[i]))
        sum_weights += w

    sum_weights[sum_weights == 0] = 1e-6
    radiance_map /= sum_weights

    logger.info("Finished computeRadianceMap function")

    if return_all:
        return radiance_map, response_curve, z_min, z_max, intensity_samples, log_exposures
    else:
        return radiance_map
            
def estimate_radiance(images, exposure_times, Zmax_precomputed):
    num_images, height, width = images.shape
    radiance = np.zeros((height, width))
    weight_sum = np.zeros((height, width))
    
    for i in range(num_images):
        weights = adaptive_weight(images[i], Zmax_precomputed[i])
        radiance += weights * images[i].astype(float) / exposure_times[i]
        weight_sum += weights
    
    return radiance / np.maximum(weight_sum, 1e-6)

def sampleIntensities(images, exposure_times, Zmax_precomputed, num_samples=50000):
    """Sample pixel intensities from the exposure stack."""
    num_images, height, width = images.shape
    z_min = 0  # Assuming the minimum is always 0 after dark current subtraction
    z_max = np.max(Zmax_precomputed)  # Use the maximum value across all Zmax_precomputed
    logger.info(f"z_max: {z_max}; z_min: {z_min}")

    # Ensure num_samples is an integer
    num_samples = int(num_samples)
    logger.info(f"num_samples: {num_samples}")

    # Use the updated estimate_radiance function that takes Zmax_precomputed
    radiance = estimate_radiance(images, exposure_times, Zmax_precomputed)

    # Logarithmic binning   
    num_bins = min(num_samples // num_images, int(z_max - z_min + 1))
    num_bins = max(1, int(num_bins))  # Ensure at least one bin
    logger.info(f"num_bins: {num_bins}")

    bins = np.logspace(np.log10(z_min + 1), np.log10(z_max + 1), num_bins + 1) - 1
    bins = np.unique(bins.astype(int))
    logger.info(f"Number of unique bins: {len(bins)}")

    sampled_pixels = []
    for i, img in enumerate(images):
        for j in range(len(bins) - 1):
            bin_mask = (img >= bins[j]) & (img < bins[j+1])
            pixels_in_bin = np.where(bin_mask)
            if len(pixels_in_bin[0]) > 0:
                num_to_sample = min(len(pixels_in_bin[0]), num_samples // (num_images * len(bins)))
                num_to_sample = max(1, int(num_to_sample))  # Ensure at least 1 sample
                logger.info(f"Sampling {num_to_sample} pixels from bin {j} in image {i}")
                sampled_indices = np.random.choice(len(pixels_in_bin[0]), num_to_sample, replace=False)
                sampled_pixels.extend(list(zip(pixels_in_bin[0][sampled_indices], pixels_in_bin[1][sampled_indices], [i]*num_to_sample)))

    logger.info(f"Total sampled pixels: {len(sampled_pixels)}")

    intensity_samples = np.zeros((len(sampled_pixels), num_images), dtype=np.float32)
    log_exposures = np.zeros((len(sampled_pixels), num_images))

    for i, (row, col, img_idx) in enumerate(sampled_pixels):
        for j in range(num_images):
            intensity_samples[i, j] = images[j, row, col]
            log_exposures[i, j] = np.log(radiance[row, col] * exposure_times[j] + 1e-10)

    # Relaxed validity criteria using Zmax_precomputed
    valid_samples = np.sum((intensity_samples > 0) & (intensity_samples < Zmax_precomputed[:, row, col]), axis=1) >= num_images // 2
    intensity_samples = intensity_samples[valid_samples]
    log_exposures = log_exposures[valid_samples]

    logger.info(f"Number of valid samples: {intensity_samples.shape[0]}")

    return intensity_samples, log_exposures, z_min, z_max

def computeResponseCurve(intensity_samples, log_exposures, exposure_times, smoothing_lambda, weighting_function, z_min, z_max, Zmax_precomputed):
    """Find the camera response curve for a single color channel."""
    num_samples, num_images = intensity_samples.shape
    intensity_range = int(np.max(Zmax_precomputed)) - z_min + 1

    data_constraints = num_samples * num_images
    smoothness_constraints = intensity_range - 2
    monotonicity_constraints = intensity_range - 1
    total_constraints = data_constraints + smoothness_constraints + monotonicity_constraints

    mat_A = np.zeros((total_constraints, intensity_range), dtype=np.float64)
    mat_b = np.zeros((total_constraints, 1), dtype=np.float64)

    k = 0
    for i in range(num_samples):
        for j in range(num_images):
            z_ij = intensity_samples[i, j]
            w_ij = weighting_function(z_ij, np.max(Zmax_precomputed[j]))
            
            z_ij_scalar = int(z_ij)
            w_ij_scalar = float(w_ij) if np.isscalar(w_ij) else float(np.mean(w_ij))
            
            mat_A[k, z_ij_scalar - z_min] = w_ij_scalar
            mat_b[k, 0] = w_ij_scalar * log_exposures[i, j]
            k += 1

    for z_k in range(z_min + 1, int(np.max(Zmax_precomputed))):
        w_k = weighting_function(z_k, np.max(Zmax_precomputed[-1]))
        w_k_scalar = float(w_k) if np.isscalar(w_k) else float(np.mean(w_k))
        mat_A[k, z_k - z_min - 1:z_k - z_min + 2] = w_k_scalar * smoothing_lambda * np.array([-1, 2, -1])
        k += 1

    for z_k in range(z_min, int(np.max(Zmax_precomputed)) - 1):
        if k < total_constraints:
            mat_A[k, z_k - z_min] = -1
            mat_A[k, z_k - z_min + 1] = 1
            mat_b[k, 0] = 0.001
            k += 1
        else:
            break

    x = np.linalg.lstsq(mat_A, mat_b, rcond=None)[0]
    return x.flatten()

def save_radiance_map(radiance_map, directory, experiment_title, base_data_folder="data"):
    """Save the unscaled radiance map."""
    experiment_folder = os.path.basename(os.path.normpath(directory))
    final_data_folder = os.path.join(base_data_folder, experiment_folder, "final_data")
    os.makedirs(final_data_folder, exist_ok=True)

    radiance_file = os.path.join(final_data_folder, f"{experiment_title}_radiance_map.npy")
    np.save(radiance_file, radiance_map)
    print(f"Radiance map saved to: {radiance_file}")

def process_hdr_images(directory, experiment_title, base_data_folder="data", smoothing_lambda=1000):
    experiment_folder = os.path.basename(os.path.normpath(directory))
    data_folder = os.path.join(base_data_folder, experiment_folder)
    final_data_folder = os.path.join(data_folder, "final_data")
    
    # Load clipped and denoised data
    data_dict = load_clipped_denoised_data(directory, experiment_title, base_data_folder)
    
    # Load Slinear, Sd, and bias arrays
    Slinear = np.load(os.path.join(base_data_folder, 'Slinear.npy'))
    Sd = np.load(os.path.join(base_data_folder, 'Sd.npy'))
    bias = np.load(os.path.join(base_data_folder, 'b.npy'))
    
    processed_data = []
    
    for key, data in data_dict.items():
        images = data['image']
        exposure_times = data['exposure_time']
        
        print(f"Processing {key}")
        print(f"Images shape: {images.shape}")
        print(f"Exposure times shape: {exposure_times.shape}")
        print(f"Slinear shape: {Slinear.shape}")
        print(f"Sd shape: {Sd.shape}")
        print(f"bias shape: {bias.shape}")
        
        # Use return_all=True to get all the computed data
        Zmax_precomputed = precompute_zmax(Slinear, Sd, bias, exposure_times)
        print(f"Zmax_precomputed shape: {Zmax_precomputed.shape}")
        
        radiance_map, response_curve, z_min, z_max, intensity_samples, log_exposures = computeRadianceMap(
            images, exposure_times, Zmax_precomputed, smoothing_lambda=smoothing_lambda, return_all=True
        )
        
        radiance_map_filename = f"{key}_radiance_map.npy"
        np.save(os.path.join(final_data_folder, radiance_map_filename), radiance_map)
        logger.info(f"Radiance map saved to: {os.path.join(final_data_folder, radiance_map_filename)}")
        
        processed_data.append({
            'key': key,
            'radiance_map': radiance_map,
            'response_curve': response_curve,
            'z_min': z_min,
            'z_max': z_max,
            'intensity_samples': intensity_samples,
            'log_exposures': log_exposures
        })
        
        logger.info(f"Processed {key}")

    logger.info("Finished processing all files.")
    return processed_data

def plot_weighting_function(weighting_function, z_min, z_max, ax=None):
    if ax is None:
        ax = plt.gca()
    try:
        pixel_values = np.arange(z_min, z_max + 1)
        weights = weighting_function(pixel_values, z_max)
        ax.plot(pixel_values, weights)
        ax.set_xlabel("Pixel Intensity")
        ax.set_ylabel("Weight")
        ax.grid(True)
    except Exception as e:
        print(f"Error in plot_weighting_function: {str(e)}")
        ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', transform=ax.transAxes)

def plot_response_curve(intensity_samples, log_exposures, response_curve, z_min, z_max, ax=None):
    if ax is None:
        ax = plt.gca()
    try:
        print(f"Debug: intensity_samples shape: {intensity_samples.shape}")
        print(f"Debug: log_exposures shape: {log_exposures.shape}")
        print(f"Debug: response_curve shape: {response_curve.shape}")
        print(f"Debug: z_min: {z_min}, z_max: {z_max}")

        for j in range(intensity_samples.shape[1]):
            ax.scatter(intensity_samples[:, j], log_exposures[:, j], alpha=0.1, s=1, c='blue')
        
        pixel_values = np.arange(z_min, z_max + 1)
        
        if len(response_curve) != len(pixel_values):
            print(f"Warning: response_curve length ({len(response_curve)}) doesn't match pixel_values length ({len(pixel_values)})")
            min_length = min(len(response_curve), len(pixel_values))
            response_curve = response_curve[:min_length]
            pixel_values = pixel_values[:min_length]
        
        ax.plot(pixel_values, response_curve, 'r-', linewidth=2, label='Fitted Response Curve')
        
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Log Exposure')
        ax.set_title('Camera Response Function')
        ax.legend()
        ax.grid(True)
    except Exception as e:
        print(f"Error in plot_response_curve: {str(e)}")
        ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', transform=ax.transAxes)

def plot_crf_residuals(intensity_samples, log_exposures, response_curve, z_min, ax=None):
    if ax is None:
        ax = plt.gca()
    try:
        for j in range(intensity_samples.shape[1]):
            residuals = log_exposures[:, j] - response_curve[np.clip(intensity_samples[:, j].astype(int) - z_min, 0, len(response_curve) - 1)]
            ax.scatter(intensity_samples[:, j], residuals, alpha=0.1, s=1, c='blue')
        
        ax.axhline(y=0, color='r', linestyle='-')
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Residuals')
        ax.set_title('CRF Residuals')
        ax.grid(True)
    except Exception as e:
        print(f"Error in plot_crf_residuals: {str(e)}")
        ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', transform=ax.transAxes)

def plot_log_log_crf(response_curve, z_min, z_max, ax=None):
    if ax is None:
        ax = plt.gca()
    try:
        pixel_values = np.arange(z_min, z_max + 1)
        
        min_length = min(len(response_curve), len(pixel_values))
        response_curve = response_curve[:min_length]
        pixel_values = pixel_values[:min_length]
        
        print(f"plot_log_log_crf - pixel_values shape: {pixel_values.shape}, response_curve shape: {response_curve.shape}")
        
        ax.plot(pixel_values, response_curve, 'r-', linewidth=2, label='Fitted Response Curve')
        
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Log Exposure')
        ax.set_title('Camera Response Function (Log-Log)')
        ax.set_xscale('log')
        ax.set_yscale('linear')
        ax.legend()
        ax.grid(True)
    except Exception as e:
        print(f"Error in plot_log_log_crf: {str(e)}")
        ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', transform=ax.transAxes)

def plot_radiance_map(radiance_map, ax=None):
    if ax is None:
        ax = plt.gca()
    try:
        im = ax.imshow(radiance_map, cmap='viridis')
        plt.colorbar(im, ax=ax, label='Radiance')
        ax.set_title('Radiance Map')
    except Exception as e:
        print(f"Error in plot_radiance_map: {str(e)}")
        ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', transform=ax.transAxes)

def plot_radiance_histogram(radiance_map, ax=None):
    if ax is None:
        ax = plt.gca()
    try:
        radiance_values = radiance_map.flatten()
        radiance_values = radiance_values[np.isfinite(radiance_values)]
        
        if len(radiance_values) > 0:
            ax.hist(radiance_values, bins=100, range=(np.percentile(radiance_values, 1), np.percentile(radiance_values, 99)), log=True)
            ax.set_xlabel('Radiance')
            ax.set_ylabel('Frequency')
            ax.set_title('Radiance Histogram')
        else:
            ax.text(0.5, 0.5, "No valid radiance values", ha='center', va='center', transform=ax.transAxes)
    except Exception as e:
        print(f"Error in plot_radiance_histogram: {str(e)}")
        ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', transform=ax.transAxes)
        
def capture_plots(data, adaptive_weight):
    # Create a figure with a grid layout
    fig = plt.figure(figsize=(15, 20))
    gs = GridSpec(4, 2, figure=fig, height_ratios=[1, 1, 1, 1])

    plot_functions = [
        (plot_weighting_function, (0, 0), 'Weighting Function'),
        (plot_response_curve, (0, 1), 'CRF Fitting'),
        (plot_log_log_crf, (1, 0), 'Log-Log CRF'),
        (plot_crf_residuals, (1, 1), 'CRF Residuals'),
        (plot_radiance_histogram, (2, 0), 'Radiance Histogram'),
        (plot_radiance_map, (2, 1), 'Log Radiance Map'),
        (plot_radiance_map, (3, 0), 'Linear Radiance Map')
    ]

    for plot_func, (row, col), title in plot_functions:
        try:
            ax = fig.add_subplot(gs[row, col])
            if plot_func == plot_weighting_function:
                plot_func(adaptive_weight, data['z_min'], data['z_max'], ax=ax)
            elif plot_func == plot_radiance_map and title == 'Log Radiance Map':
                plot_func(np.log(data['radiance_map']), ax=ax)
            else:
                plot_func(**{k: v for k, v in data.items() if k in plot_func.__code__.co_varnames}, ax=ax)
            ax.set_title(title, pad=10)
        except Exception as e:
            print(f"Error in {title}: {str(e)}")
            print(f"Data shapes: {', '.join([f'{k}: {v.shape if isinstance(v, np.ndarray) else v}' for k, v in data.items()])}")
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    
    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.6, wspace=0.3)

    # Save the montage to a BytesIO object
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)

    return buf

def generate_multi_page_report(processed_data, directory, experiment_title, adaptive_weight, base_data_folder="data"):
    experiment_folder = os.path.basename(os.path.normpath(directory))
    data_folder = os.path.join(base_data_folder, experiment_folder)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'hdr_report_{experiment_title}_{timestamp}.pdf'
    output_path = os.path.join(data_folder, output_file)
    
    page_width, page_height = letter
    margin = 0.5 * inch
    
    class MyDocTemplate(BaseDocTemplate):
        def __init__(self, filename, **kw):
            BaseDocTemplate.__init__(self, filename, **kw)
            frame = Frame(self.leftMargin, self.bottomMargin + 0.5*inch,
                          self.width, self.height - inch,
                          id='normal')
            template = PageTemplate(id='Later', frames=frame, onPage=self.add_page_number)
            self.addPageTemplates([template])

        def add_page_number(self, canvas, doc):
            canvas.saveState()
            canvas.setFont('Helvetica', 9)
            page_num = canvas.getPageNumber()
            text = f"Page {page_num}"
            canvas.drawRightString(self.width + self.rightMargin, self.bottomMargin, text)
            canvas.restoreState()

    doc = MyDocTemplate(output_path, pagesize=letter,
                        leftMargin=margin, rightMargin=margin,
                        topMargin=margin,
                        bottomMargin=margin)
    
    story = []
    
    for data in processed_data:
        montage = capture_plots(data, adaptive_weight)
    
        available_width = page_width - 4*margin
        available_height = page_height - 5*margin
        
        img_reader = ImageReader(montage)
        orig_width, orig_height = img_reader.getSize()
        
        width_ratio = available_width / orig_width
        height_ratio = available_height / orig_height
        scale_factor = min(width_ratio, height_ratio)
        
        new_width = orig_width * scale_factor
        new_height = orig_height * scale_factor
        
        story.append(FilenamePlaceholder(data['key']))
        story.append(Image(montage, width=new_width, height=new_height))
        story.append(PageBreak())
        
        logger.info(f"Added report page for {data['key']}")
    
    # Add log to the end of the PDF
    logger.info("Finished processing all files.")
    log_stream = io.StringIO()
    log_handler = logging.StreamHandler(log_stream)
    logger.addHandler(log_handler)
    
    story.append(Paragraph("Processing Log", getSampleStyleSheet()['Heading1']))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(log_stream.getvalue(), getSampleStyleSheet()['BodyText']))
    
    # Build the PDF
    doc.build(story)
    
    logger.info(f"Report saved as {output_path}")

def save_crf_data(processed_data, directory, experiment_title, base_data_folder="data"):
    experiment_folder = os.path.basename(os.path.normpath(directory))
    data_folder = os.path.join(base_data_folder, experiment_folder, "final_data")
    os.makedirs(data_folder, exist_ok=True)

    crf_file = os.path.join(data_folder, f"{experiment_title}_crf_data.npz")
    
    print(f"Number of processed data items: {len(processed_data)}")

    save_dict = {}
    for i, data in enumerate(processed_data):
        print(f"\nProcessing data item {i+1}:")
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                print(f"  {key} shape: {value.shape}")
                save_dict[f'{key}_{i}'] = value
            else:
                print(f"  {key}: {value}")
                save_dict[f'{key}_{i}'] = value

    print("\nSaving CRF data...")
    try:
        np.savez(crf_file, **save_dict)
        print(f"CRF data saved to: {crf_file}")
        
        # Verify the saved data
        verify_data = np.load(crf_file, allow_pickle=True)
        print("Verified saved data. Keys in the file:")
        for key in verify_data.keys():
            print(f"  {key}")
    except Exception as e:
        print(f"Error saving CRF data: {str(e)}")

    return crf_file

def load_crf_data(crf_file):
    """Load the saved CRF data."""
    print(f"Loading CRF data from: {crf_file}")
    loaded_data = np.load(crf_file, allow_pickle=True)
    
    print("Keys in the loaded file:")
    for key in loaded_data.keys():
        print(f"  {key}")
    
    processed_data = []
    i = 0
    while f'key_{i}' in loaded_data:
        print(f"Processing data item {i}")
        data_item = {}
        for key in ['key', 'radiance_map', 'response_curve', 'z_min', 'z_max', 'intensity_samples', 'log_exposures']:
            full_key = f'{key}_{i}'
            if full_key in loaded_data:
                data_item[key] = loaded_data[full_key]
                if isinstance(data_item[key], np.ndarray):
                    print(f"  Loaded {full_key}: ndarray with shape {data_item[key].shape}, dtype: {data_item[key].dtype}")
                else:
                    print(f"  Loaded {full_key}: {type(data_item[key])}")
            else:
                print(f"  Warning: {full_key} not found in loaded data")
        if data_item:
            processed_data.append(data_item)
        else:
            print(f"  Warning: No data loaded for item {i}")
        i += 1
    
    print(f"Loaded {len(processed_data)} processed data items")
    
    return processed_data