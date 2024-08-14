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

def load_processed_data(directory, experiment_title, base_data_folder="data"):
    # Extract the last part of the directory path
    experiment_folder = os.path.basename(os.path.normpath(directory))
    
    # Create the full path for the data folder
    data_folder = os.path.join(base_data_folder, experiment_folder)
    
    # Check if the data folder exists
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"Data folder not found: {data_folder}")

    # Load darkcount-subtracted (denoised) images
    denoised_images = {}
    for file in os.listdir(data_folder):
        if file.endswith("_denoised.npy"):
            key = file.split('_')[0]  # Assuming the key is the first part of the filename
            denoised_file = os.path.join(data_folder, file)
            denoised_images[key] = np.load(denoised_file)
            print(f"Loaded denoised images for {key} from: {denoised_file}")

    # Load exposure times
    exposure_times_file = os.path.join(data_folder, f"{experiment_title}_exposure_times.npy")
    exposure_times = np.load(exposure_times_file)
    print(f"Loaded exposure times from: {exposure_times_file}")

    return {
        'denoised_images': denoised_images,
        'exposure_times': exposure_times
    }


def linearWeight(pixel_value, z_min, z_max):
    """Linear weighting function based on pixel intensity."""
    pixel_value = np.asarray(pixel_value)
    mid = (z_min + z_max) / 2
    weight = np.where(pixel_value <= mid, 
                      pixel_value - z_min, 
                      z_max - pixel_value)
    return weight.astype(np.float32)

def estimate_radiance(images, exposure_times):
    """Estimate the relative radiance for each pixel."""
    num_images, height, width = images.shape
    radiance = np.zeros((height, width))
    weight_sum = np.zeros((height, width))
    
    for i in range(num_images):
        weights = 1 - 2 * np.abs(images[i].astype(float) / np.max(images[i]) - 0.5)
        radiance += weights * images[i].astype(float) / exposure_times[i]
        weight_sum += weights
    
    return radiance / np.maximum(weight_sum, 1e-6)

def sampleIntensities(images, exposure_times, num_samples=50000):
    """Sample pixel intensities from the exposure stack."""
    num_images, height, width = images.shape
    z_min, z_max = int(np.min(images)), int(np.max(images))
    print(f"z_max: {z_max}; z_min: {z_min}")

    radiance = estimate_radiance(images, exposure_times)

    # Logarithmic binning   
    num_bins = min(num_samples // num_images, z_max - z_min + 1)
    bins = np.logspace(np.log10(z_min + 1), np.log10(z_max + 1), num_bins + 1) - 1
    bins = np.unique(bins.astype(int))

    sampled_pixels = []
    for img in images:
        for i in range(len(bins) - 1):
            bin_mask = (img >= bins[i]) & (img < bins[i+1])
            pixels_in_bin = np.where(bin_mask)
            if len(pixels_in_bin[0]) > 0:
                num_to_sample = min(len(pixels_in_bin[0]), num_samples // (num_images * num_bins))
                sampled_indices = np.random.choice(len(pixels_in_bin[0]), num_to_sample, replace=False)
                sampled_pixels.extend(list(zip(pixels_in_bin[0][sampled_indices], pixels_in_bin[1][sampled_indices])))

    intensity_samples = np.zeros((len(sampled_pixels), num_images), dtype=np.uint16)
    log_exposures = np.zeros((len(sampled_pixels), num_images))

    for i, (row, col) in enumerate(sampled_pixels):
        for j in range(num_images):
            intensity_samples[i, j] = images[j, row, col]
            log_exposures[i, j] = np.log(radiance[row, col] * exposure_times[j] + 1e-10)

    
    # Relaxed validity criteria
    valid_samples = np.sum((intensity_samples > 0) & (intensity_samples < z_max), axis=1) >= num_images // 2
    intensity_samples = intensity_samples[valid_samples]
    log_exposures = log_exposures[valid_samples]

    print(f"Number of valid samples: {intensity_samples.shape[0]}")

    return intensity_samples, log_exposures, z_min, z_max

def computeResponseCurve(intensity_samples, log_exposures, smoothing_lambda, weighting_function, z_min, z_max):
    """Find the camera response curve for a single color channel."""
    num_samples, num_images = intensity_samples.shape
    intensity_range = z_max - z_min + 1

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
            w_ij = weighting_function(z_ij, z_min, z_max)
            mat_A[k, z_ij - z_min] = w_ij
            mat_b[k, 0] = w_ij * log_exposures[i, j]
            k += 1

    for z_k in range(z_min + 1, z_max):
        w_k = weighting_function(z_k, z_min, z_max)
        mat_A[k, z_k - z_min - 1:z_k - z_min + 2] = w_k * smoothing_lambda * np.array([-1, 2, -1])
        k += 1

    for z_k in range(z_min, z_max - 1):
        if k < total_constraints:
            mat_A[k, z_k - z_min] = -1
            mat_A[k, z_k - z_min + 1] = 1
            mat_b[k, 0] = 0.001
            k += 1
        else:
            break

    x = np.linalg.lstsq(mat_A, mat_b, rcond=None)[0]
    return x.flatten()

def computeRadianceMap(images, exposure_times, smoothing_lambda=1000, return_all=False):
    """Calculate an unscaled radiance map for each pixel from the response curve."""
    intensity_samples, log_exposures, z_min, z_max = sampleIntensities(images, exposure_times)
    response_curve = computeResponseCurve(intensity_samples, log_exposures, smoothing_lambda, linearWeight, z_min, z_max)
    response_curve = savgol_filter(response_curve, window_length=51, polyorder=3)

    num_images, height, width = images.shape
    radiance_map = np.zeros((height, width), dtype=np.float32)
    sum_weights = np.zeros((height, width), dtype=np.float32)

    for i in range(num_images):
        w = linearWeight(images[i], z_min, z_max)
        radiance_map += w * (response_curve[np.clip(images[i] - z_min, 0, len(response_curve) - 1)] - np.log(exposure_times[i]))
        sum_weights += w

    sum_weights[sum_weights == 0] = 1e-6
    radiance_map /= sum_weights

    if return_all:
        return radiance_map, response_curve, z_min, z_max, intensity_samples, log_exposures
    else:
        return radiance_map

def save_radiance_map(radiance_map, directory, experiment_title, base_data_folder="data"):
    """Save the unscaled radiance map."""
    experiment_folder = os.path.basename(os.path.normpath(directory))
    data_folder = os.path.join(base_data_folder, experiment_folder)
    os.makedirs(data_folder, exist_ok=True)

    radiance_file = os.path.join(data_folder, f"{experiment_title}_radiance_map.npy")
    np.save(radiance_file, radiance_map)
    print(f"Radiance map saved to: {radiance_file}")
    


# Main processing function
def process_hdr_images(directory, experiment_title, base_data_folder="data", smoothing_lambda=1000):
    experiment_folder = os.path.basename(os.path.normpath(directory))
    data_folder = os.path.join(base_data_folder, experiment_folder)
    denoised_files = [f for f in os.listdir(data_folder) if f.endswith('_denoised.npy')]
    exposure_times_file = os.path.join(data_folder, f"{experiment_title}_exposure_times.npy")
    exposure_times = np.load(exposure_times_file)
    
    processed_data = []
    
    for npy_filename in denoised_files:
        images = np.load(os.path.join(data_folder, npy_filename))
        
        if images.dtype != np.uint16:
            logger.warning(f"Image data in {npy_filename} is not uint16. Consider updating the save process.")
        
        # Use return_all=True to get all the computed data
        radiance_map, response_curve, z_min, z_max, intensity_samples, log_exposures = computeRadianceMap(
            images, exposure_times, smoothing_lambda=smoothing_lambda, return_all=True
        )
        
        radiance_map_filename = f"{npy_filename[:-12]}_radiance_map.npy"
        np.save(os.path.join(data_folder, radiance_map_filename), radiance_map)
        logger.info(f"Radiance map saved to: {os.path.join(data_folder, radiance_map_filename)}")
        
        processed_data.append({
            'filename': npy_filename,
            'radiance_map': radiance_map,
            'response_curve': response_curve,
            'z_min': z_min,
            'z_max': z_max,
            'intensity_samples': intensity_samples,
            'log_exposures': log_exposures
        })
        
        logger.info(f"Processed {npy_filename}")

    logger.info("Finished processing all files.")
    return processed_data


def plot_weighting_function(weighting_function, z_min, z_max, ax=None):
    """Plot the weighting function against pixel intensity."""
    if ax is None:
        ax = plt.gca()
    pixel_values = np.arange(z_min, z_max + 1)
    weights = [weighting_function(z, z_min, z_max) for z in pixel_values]
    ax.plot(pixel_values, weights)
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Weight")
    ax.grid(True)

def plot_response_curve(intensity_samples, log_exposures, response_curve, z_min, z_max, ax=None):
    """Plot the camera response function."""
    if ax is None:
        ax = plt.gca()
    
    for j in range(intensity_samples.shape[1]):
        ax.scatter(intensity_samples[:, j], log_exposures[:, j], alpha=0.1, s=1, c='blue')
    
    pixel_values = np.arange(z_min, z_max + 1)
    ax.plot(pixel_values, response_curve, 'r-', linewidth=2, label='Fitted Response Curve')
    
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Log Exposure')
    ax.set_title('Camera Response Function')
    ax.legend()
    ax.grid(True)

def plot_crf_residuals(intensity_samples, log_exposures, response_curve, z_min, ax=None):
    """Plot the residuals of the camera response function."""
    if ax is None:
        ax = plt.gca()
    
    for j in range(intensity_samples.shape[1]):
        residuals = log_exposures[:, j] - response_curve[intensity_samples[:, j] - z_min]
        ax.scatter(intensity_samples[:, j], residuals, alpha=0.1, s=1, c='blue')
    
    ax.axhline(y=0, color='r', linestyle='-')
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Residuals')
    ax.set_title('CRF Residuals')
    ax.grid(True)

def plot_log_log_crf(response_curve, z_min, z_max, ax=None):
    """Plot the camera response function in log-log scale."""
    if ax is None:
        ax = plt.gca()
    
    pixel_values = np.arange(z_min, z_max + 1)
    ax.plot(pixel_values, response_curve, 'r-', linewidth=2, label='Fitted Response Curve')
    
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Log Exposure')
    ax.set_title('Camera Response Function (Log-Log)')
    ax.set_xscale('log')
    ax.set_yscale('linear')
    ax.legend()
    ax.grid(True)

def plot_radiance_map(radiance_map, ax=None):
    """Plot the radiance map."""
    if ax is None:
        ax = plt.gca()
    
    im = ax.imshow(radiance_map, cmap='viridis')
    plt.colorbar(im, ax=ax, label='Radiance')
    ax.set_title('Radiance Map')

def plot_radiance_histogram(radiance_map, ax=None):
    if ax is None:
        ax = plt.gca()
    
    # Flatten the radiance map and remove any infinite or NaN values
    radiance_values = radiance_map.flatten()
    radiance_values = radiance_values[np.isfinite(radiance_values)]
    
    # Plot histogram
    ax.hist(radiance_values, bins=100, range=(np.percentile(radiance_values, 1), np.percentile(radiance_values, 99)), log=True)
    ax.set_xlabel('Radiance')
    ax.set_ylabel('Frequency')
    ax.set_title('Radiance Histogram')


def capture_plots(data):
    # Create a figure with a grid layout
    fig = plt.figure(figsize=(15, 20))
    gs = GridSpec(4, 2, figure=fig, height_ratios=[1, 1, 1, 1])

    # Weighting function
    ax1 = fig.add_subplot(gs[0, 0])
    plot_weighting_function(linearWeight, data['z_min'], data['z_max'], ax=ax1)
    ax1.set_title('Weighting Function', pad=10)

    # CRF fitting
    ax2 = fig.add_subplot(gs[0, 1])
    plot_response_curve(data['intensity_samples'], data['log_exposures'], data['response_curve'], data['z_min'], data['z_max'], ax=ax2)
    ax2.set_title('CRF Fitting', pad=10)

    # Log-log CRF
    ax3 = fig.add_subplot(gs[1, 0])
    plot_log_log_crf(data['response_curve'], data['z_min'], data['z_max'], ax=ax3)
    ax3.set_title('Log-Log CRF', pad=10)

    # CRF residuals
    ax4 = fig.add_subplot(gs[1, 1])
    plot_crf_residuals(data['intensity_samples'], data['log_exposures'], data['response_curve'], data['z_min'], ax=ax4)
    ax4.set_title('CRF Residuals', pad=10)

    # Radiance histogram
    ax5 = fig.add_subplot(gs[2, 0])
    plot_radiance_histogram(data['radiance_map'], ax=ax5)
    ax5.set_title('Radiance Histogram', pad=10)

    # Log radiance map
    ax6 = fig.add_subplot(gs[2, 1])
    plot_radiance_map(np.log(data['radiance_map']), ax=ax6)
    ax6.set_title('Log Radiance Map', pad=10)

    # Linear radiance map
    ax7 = fig.add_subplot(gs[3, :])
    plot_radiance_map(data['radiance_map'], ax=ax7)
    ax7.set_title('Linear Radiance Map', pad=10)

    plt.tight_layout()
    
    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.6, wspace=0.3)

    # Save the montage to a BytesIO object
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)

    return buf


def generate_multi_page_report(processed_data, directory, experiment_title, base_data_folder="data"):
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
        montage = capture_plots(data)
    
        available_width = page_width - 4*margin
        available_height = page_height - 5*margin
        
        img_reader = ImageReader(montage)
        orig_width, orig_height = img_reader.getSize()
        
        width_ratio = available_width / orig_width
        height_ratio = available_height / orig_height
        scale_factor = min(width_ratio, height_ratio)
        
        new_width = orig_width * scale_factor
        new_height = orig_height * scale_factor
        
        story.append(FilenamePlaceholder(data['filename']))
        story.append(Image(montage, width=new_width, height=new_height))
        story.append(PageBreak())
        
        logger.info(f"Added report page for {data['filename']}")
    
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