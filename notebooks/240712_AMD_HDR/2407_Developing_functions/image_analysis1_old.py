import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import cv2
import io
import os
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Image, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from matplotlib.gridspec import GridSpec



def linearWeight(pixel_value, z_min, z_max):
    
    """
    Linear weighting function based on pixel intensity that reduces the
    weight of pixel values that are near saturation.

    Parameters
    ----------
    pixel_value : int or numpy.ndarray
        A pixel intensity value or array of values
    z_min : int
        Minimum possible pixel value
    z_max : int
        Maximum possible pixel value

    Returns
    -------
    weight : float or numpy.ndarray
        The weight(s) corresponding to the input pixel intensity(ies)
    """
    pixel_value = np.asarray(pixel_value)
    mid = (z_min + z_max) / 2
    weight = np.where(pixel_value <= mid, 
                    pixel_value - z_min, 
                    z_max - pixel_value)
    return weight.astype(np.float32)

def plot_weighting_function(weighting_function, z_min, z_max, ax=None):
    """
    Plot the weighting function against pixel intensity.

    Parameters:
    weighting_function : callable
        The weighting function to be plotted
    z_min : int
        Minimum pixel value
    z_max : int
        Maximum pixel value
    """
    if ax is None:
        ax = plt.gca()
    pixel_values = np.arange(z_min, z_max + 1)
    weights = [weighting_function(z, z_min, z_max) for z in pixel_values]
    ax.plot(pixel_values, weights)
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Weight")
    ax.grid(True)

def estimate_radiance(images, exposure_times):
    """
    Estimate the relative radiance for each pixel.
    
    Parameters:
    images : numpy.ndarray
        Stack of images (shape: num_images x height x width)
    exposure_times : numpy.ndarray
        Array of exposure times for each image
    
    Returns:
    numpy.ndarray
        Estimated radiance map
    """
    num_images, height, width = images.shape
    radiance = np.zeros((height, width))
    weight_sum = np.zeros((height, width))
    
    for i in range(num_images):
        # Simple weighting function: give more weight to mid-range pixel values
        weights = 1 - 2 * np.abs(images[i].astype(float) / np.max(images[i]) - 0.5)
        radiance += weights * images[i].astype(float) / exposure_times[i]
        weight_sum += weights
    
    return radiance / np.maximum(weight_sum, 1e-6)  # Avoid division by zero

def sampleIntensities(images, exposure_times, num_samples=150000):
    """
    Sample pixel intensities from the exposure stack, ensuring full intensity range coverage.

    Parameters
    ----------
    images : numpy.ndarray
        A 3D array containing a stack of single-channel images
        Shape: (num_images, height, width)
    exposure_times : numpy.ndarray
        Array of exposure times for each image
    num_samples : int, optional
        Target number of pixel locations to sample

    Returns
    -------
    intensity_samples : numpy.array
        An array containing sampled intensity values from each
        exposure layer (shape = num_samples x num_images)
    log_exposures : numpy.array
        An array containing log exposure values for each sample
    z_min : int
        Minimum intensity value in the images
    z_max : int
        Maximum intensity value in the images
    """
    if not isinstance(images, np.ndarray) or images.ndim != 3:
        raise ValueError("images must be a 3D numpy array")

    num_images, height, width = images.shape
    z_min = int(np.min(images))
    z_max = int(np.max(images))
    print(f"z_max: {z_max}; z_min: {z_min}")

    # Estimate radiance
    radiance = estimate_radiance(images, exposure_times)

    # Find the image with the widest intensity distribution
    intensity_ranges = [np.ptp(img) for img in images]
    widest_range_index = np.argmax(intensity_ranges)
    widest_range_image = images[widest_range_index]

    # Create bins across the full intensity range
    num_bins = min(num_samples, z_max - z_min + 1)
    bins = np.linspace(z_min, z_max, num_bins + 1, dtype=int)

    # Sample pixels from each bin
    sampled_pixels = []
    for i in range(len(bins) - 1):
        bin_mask = (widest_range_image >= bins[i]) & (widest_range_image < bins[i+1])
        pixels_in_bin = np.where(bin_mask)
        if len(pixels_in_bin[0]) > 0:
            # Randomly select one pixel from this bin
            idx = np.random.randint(len(pixels_in_bin[0]))
            sampled_pixels.append((pixels_in_bin[0][idx], pixels_in_bin[1][idx]))

    # Initialize the intensity values and log exposure arrays
    intensity_samples = np.zeros((len(sampled_pixels), num_images), dtype=np.uint16)
    log_exposures = np.zeros((len(sampled_pixels), num_images))

    # Sample the selected pixels across all exposures
    for i, (row, col) in enumerate(sampled_pixels):
        for j in range(num_images):
            intensity_samples[i, j] = images[j, row, col]
            log_exposures[i, j] = np.log(radiance[row, col] * exposure_times[j])

    # Remove any rows where all values are 0 or saturated
    valid_samples = np.all((intensity_samples > 0) & (intensity_samples < z_max), axis=1)
    intensity_samples = intensity_samples[valid_samples]
    log_exposures = log_exposures[valid_samples]

    print(f"Number of valid samples: {intensity_samples.shape[0]}")

    return intensity_samples, log_exposures, z_min, z_max

def computeResponseCurve(intensity_samples, log_exposures, smoothing_lambda, weighting_function, z_min, z_max):
    """
    Find the camera response curve for a single color channel

    Parameters
    ----------
    intensity_samples : numpy.ndarray
        Sampled intensity values (shape = num_samples x num_images)
    log_exposures : numpy.ndarray
        Log exposure values (shape = num_samples x num_images)
    smoothing_lambda : float
        Constant for scale correction between data and smoothing terms
    weighting_function : callable
        Function that computes a weight from a pixel intensity
    z_min : int
        Minimum intensity value
    z_max : int
        Maximum intensity value

    Returns
    -------
    numpy.ndarray
        Vector g(z) where g[i] is the log exposure of intensity value z_min + i
    """
    if not isinstance(intensity_samples, np.ndarray) or intensity_samples.ndim != 2:
        raise ValueError("intensity_samples must be a 2D numpy array")
    if not isinstance(log_exposures, np.ndarray) or log_exposures.ndim != 2:
        raise ValueError("log_exposures must be a 2D numpy array")
    if intensity_samples.shape != log_exposures.shape:
        raise ValueError("intensity_samples and log_exposures must have the same shape")

    num_samples, num_images = intensity_samples.shape
    intensity_range = z_max - z_min + 1

    # Calculate the total number of constraints
    data_constraints = num_samples * num_images
    smoothness_constraints = intensity_range - 2
    monotonicity_constraints = intensity_range - 1
    total_constraints = data_constraints + smoothness_constraints + monotonicity_constraints

    mat_A = np.zeros((total_constraints, intensity_range), dtype=np.float64)
    mat_b = np.zeros((total_constraints, 1), dtype=np.float64)

    k = 0
    # Data-fitting constraints
    for i in range(num_samples):
        for j in range(num_images):
            z_ij = intensity_samples[i, j]
            w_ij = weighting_function(z_ij, z_min, z_max)
            mat_A[k, z_ij - z_min] = w_ij
            mat_b[k, 0] = w_ij * log_exposures[i, j]
            k += 1

    # Smoothness constraints
    for z_k in range(z_min + 1, z_max):
        w_k = weighting_function(z_k, z_min, z_max)
        mat_A[k, z_k - z_min - 1:z_k - z_min + 2] = w_k * smoothing_lambda * np.array([-1, 2, -1])
        k += 1

    # Monotonicity constraints
    for z_k in range(z_min, z_max - 1):
        if k < total_constraints:
            mat_A[k, z_k - z_min] = -1
            mat_A[k, z_k - z_min + 1] = 1
            mat_b[k, 0] = 0.001  # Small positive value to ensure strict monotonicity
            k += 1
        else:
            break

    # Solve the system
    x = np.linalg.lstsq(mat_A, mat_b, rcond=None)[0]

    return x.flatten()

# def computeRadianceMap(images, exposure_times, response_curve, weighting_function, z_min, z_max):
    """
    Calculate a radiance map for each pixel from the response curve.

    Parameters
    ----------
    images : numpy.ndarray
        3D array containing single-channel images (num_images, height, width)
    exposure_times : numpy.ndarray
        Array containing the exposure times for each image
    response_curve : numpy.ndarray
        Least-squares fitted log exposure of each pixel value z
    weighting_function : callable
        Function that computes the weights
    z_min : int
        Minimum intensity value
    z_max : int
        Maximum intensity value

    Returns
    -------
    numpy.ndarray
        The image radiance map (in log space)
    """
    if not isinstance(images, np.ndarray) or images.ndim != 3:
        raise ValueError("images must be a 3D numpy array")
    if not isinstance(exposure_times, np.ndarray) or exposure_times.ndim != 1:
        raise ValueError("exposure_times must be a 1D numpy array")
    if images.shape[0] != exposure_times.shape[0]:
        raise ValueError("Number of images and exposure times must match")

    num_images, height, width = images.shape
    img_rad_map = np.zeros((height, width), dtype=np.float32)
    sum_weights = np.zeros((height, width), dtype=np.float32)

    for i in range(num_images):
        w = weighting_function(images[i], z_min, z_max)
        img_rad_map += w * (response_curve[np.clip(images[i] - z_min, 0, len(response_curve) - 1)] - np.log(exposure_times[i]))
        sum_weights += w

    # Avoid division by zero
    sum_weights[sum_weights == 0] = 1e-6
    img_rad_map /= sum_weights

    return img_rad_map

def computeRadianceMap(images, log_exposure_times, response_curve, weighting_function, z_min, z_max):
    """
    Calculate a radiance map for each pixel from the response curve.

    Parameters
    ----------
    images : numpy.ndarray
        3D array containing single-channel images (num_images, height, width)
    log_exposure_times : numpy.ndarray
        Array containing the log exposure times for each image
    response_curve : numpy.ndarray
        Least-squares fitted log exposure of each pixel value z
    weighting_function : callable
        Function that computes the weights
    z_min : int
        Minimum intensity value
    z_max : int
        Maximum intensity value

    Returns
    -------
    numpy.ndarray
        The image radiance map (in log space)
    """
    if not isinstance(images, np.ndarray) or images.ndim != 3:
        raise ValueError("images must be a 3D numpy array")
    if not isinstance(log_exposure_times, np.ndarray) or log_exposure_times.ndim != 1:
        raise ValueError("log_exposure_times must be a 1D numpy array")
    if images.shape[0] != log_exposure_times.shape[0]:
        raise ValueError("Number of images and exposure times must match")

    num_images, height, width = images.shape
    img_rad_map = np.zeros((height, width), dtype=np.float32)
    sum_weights = np.zeros((height, width), dtype=np.float32)

    for i in range(num_images):
        w = weighting_function(images[i], z_min, z_max)
        img_rad_map += w * (response_curve[np.clip(images[i] - z_min, 0, len(response_curve) - 1)] - log_exposure_times[i])
        sum_weights += w

    # Avoid division by zero
    sum_weights[sum_weights == 0] = 1e-6
    img_rad_map /= sum_weights

    return img_rad_map

def plot_response_curve(intensity_samples, log_exposures, response_curve, z_min, z_max, ax=None):
    if ax is None:
        ax = plt.gca()
    
    # Plot individual data points
    for j in range(intensity_samples.shape[1]):
        ax.scatter(intensity_samples[:, j], log_exposures[:, j], alpha=0.1, s=1, c='blue')
    
    # Plot the fitted response curve
    pixel_values = np.arange(z_min, z_max + 1)
    ax.plot(pixel_values, response_curve, 'r-', linewidth=2, label='Fitted Response Curve')
    
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Log Exposure')
    ax.set_title('Camera Response Function')
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    ax.legend()
    ax.grid(True)

def plot_crf_residuals(intensity_samples, log_exposures, response_curve, z_min, ax=None):
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
    if ax is None:
        ax = plt.gca()
    
    pixel_values = np.arange(z_min, z_max + 1)
    ax.plot(pixel_values, response_curve, 'r-', linewidth=2, label='Fitted Response Curve')
    
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Log Exposure')
    ax.set_title('Camera Response Function (Log-Log)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True)

def plot_radiance_map(img_rad_map, ax=None):
    if ax is None:
        ax = plt.gca()
    
    im = ax.imshow(img_rad_map, cmap='viridis')
    plt.colorbar(im, ax=ax, label='Radiance')
    ax.set_title('Radiance Map')

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import io

def capture_plots(images, exposure_times, smoothing_lambda=1000., gamma=0.6):
    # Create a figure with a grid layout
    fig = plt.figure(figsize=(12, 16))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1.2])

    # Compute HDR and get all necessary data
    hdr_image, response_curve, z_min, z_max, img_rad_map, intensity_samples, log_exposures = computeHDR(images, exposure_times, smoothing_lambda, gamma)

    # Weighting function plot
    ax1 = fig.add_subplot(gs[0, 0])
    plot_weighting_function(linearWeight, z_min, z_max, ax=ax1)
    ax1.set_title('Weighting Function', pad=20)

    # CRF fitting plot
    ax2 = fig.add_subplot(gs[0, 1:])
    plot_response_curve(intensity_samples, log_exposures, response_curve, z_min, z_max, ax=ax2)
    ax2.set_title('CRF Fitting', pad=20)

    # Log-log CRF plot
    ax3 = fig.add_subplot(gs[1, 0:2])
    plot_log_log_crf(response_curve, z_min, z_max, ax=ax3)
    ax3.set_title('Log-Log CRF', pad=20)

    # CRF residuals plot
    ax4 = fig.add_subplot(gs[1, 2])
    plot_crf_residuals(intensity_samples, log_exposures, response_curve, z_min, ax=ax4)
    ax4.set_title('CRF Residuals', pad=20)

    # Radiance map plot
    ax5 = fig.add_subplot(gs[2, 0])
    plot_radiance_map(img_rad_map, ax=ax5)
    ax5.set_title('Radiance Map', pad=20)

    # Linear radiance map plot
    ax6 = fig.add_subplot(gs[2, 1])
    plot_radiance_map(np.exp(img_rad_map), ax=ax6)
    ax6.set_title('Linear Radiance Map', pad=20)

    # HDR image plot
    ax7 = fig.add_subplot(gs[2, 2])
    im = ax7.imshow(hdr_image, cmap='gray')
    ax7.set_title('Final HDR Image', pad=20)
    plt.colorbar(im, ax=ax7, orientation='vertical', pad=0.08, aspect=25)

    plt.tight_layout()
    
    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # Save the montage to a BytesIO object
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)

    return buf, hdr_image, response_curve

from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate, Image, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER

def generate_report(images, exposure_times, directory, experiment_title, output_file='hdr_report.pdf'):
    montage, hdr_image, response_curve = capture_plots(images, exposure_times)
    
    # Create the full path for the output file
    output_path = os.path.join(directory, output_file)
    
    # Use landscape orientation
    page_width, page_height = landscape(letter)

    # Set margins
    margin = 0.5 * inch
    
    doc = SimpleDocTemplate(output_path, pagesize=landscape(letter),
                            leftMargin=margin, rightMargin=margin,
                            topMargin=margin, bottomMargin=margin)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('TitleStyle', parent=styles['Title'], alignment=TA_CENTER)
    story = []
    
    # Title
    story.append(Paragraph(f"HDR Image Generation Report: {experiment_title}", title_style))
    story.append(Spacer(1, 0.1*inch))
    
    # Add the montage to the report
    img = ImageReader(montage)
    img_width, img_height = img.getSize()
    aspect = img_width / float(img_height)
    
    # Set a maximum width and height for the image
    max_width = page_width - 2*margin
    max_height = page_height - 3*margin  # Leave some space for the title
    
    # Scale image to fit within the content area while maintaining aspect ratio
    if max_width / aspect <= max_height:
        img_width = max_width
        img_height = max_width / aspect
    else:
        img_height = max_height
        img_width = max_height * aspect

    story.append(Image(montage, width=img_width, height=img_height))
    
    # Build the PDF
    doc.build(story)
    print(f"Report saved as {output_path}")

def computeHDR(images, exposure_times, smoothing_lambda=1000., gamma=0.6):
    """
    Computational pipeline to produce the HDR images

    Parameters
    ----------
    images : numpy.ndarray
        A 3D array containing an exposure stack of single-channel images
        Shape: (num_images, height, width)
    exposure_times : numpy.ndarray
        The exposure times for each image in the exposure stack
    smoothing_lambda : float, optional
        A constant value to correct for scale differences between
        data and smoothing terms in the constraint matrix
    gamma : float, optional
        Gamma value for tone mapping

    Returns
    -------
    tuple
        (hdr_image, response_curve)
        hdr_image : numpy.ndarray
            The resulting HDR image with intensities scaled to fit uint8 range
        response_curve : numpy.ndarray
            The computed camera response function
    """
    if not isinstance(images, np.ndarray) or images.ndim != 3:
        raise ValueError("images must be a 3D numpy array")
    if not np.issubdtype(images.dtype, np.integer):
        raise ValueError("images must be an integer type")
    if not isinstance(exposure_times, np.ndarray) or exposure_times.ndim != 1:
        raise ValueError("exposure_times must be a 1D numpy array")
    if images.shape[0] != exposure_times.shape[0]:
        raise ValueError("Number of images and exposure times must match")

    

    print(images.dtype)
    # Sample image intensities
    intensity_samples, log_exposures, z_min, z_max = sampleIntensities(images, exposure_times)

    # Compute Response Curve
    response_curve = computeResponseCurve(intensity_samples, log_exposures, smoothing_lambda, linearWeight, z_min, z_max)

    # Apply Savitzky-Golay filter to smooth the response curve
    response_curve = savgol_filter(response_curve, window_length=51, polyorder=3)

    # Build radiance map
    img_rad_map = computeRadianceMap(images, exposure_times, response_curve, linearWeight, z_min, z_max)


    # Normalize the radiance map to [0, 1] range
    img_rad_map = (img_rad_map - np.min(img_rad_map)) / (np.max(img_rad_map) - np.min(img_rad_map))
    print(f"3. Normalized radiance map min: {np.min(img_rad_map)}, max: {np.max(img_rad_map)}")
    plot_radiance_map(img_rad_map)

    # Global tone mapping 
    def adaptive_log_tone_mapping(x, a=0.5):
        return (np.log(1 + a * x) / np.log(1 + a)) / (np.log(1 + a * np.max(x)) / np.log(1 + a))

    image_mapped = adaptive_log_tone_mapping(img_rad_map)
    print(f"4. After tone mapping min: {np.min(image_mapped)}, max: {np.max(image_mapped)}")
    plt.figure(figsize=(10, 8))
    plt.imshow(image_mapped, cmap='gray')
    plt.title('After Tone Mapping')
    plt.colorbar()
    plt.show()

    # Adjust image intensity based on the middle image from image stack
    template = images[len(images) // 2]
    scale_factor = np.mean(template) / np.mean(image_mapped)
    image_tuned = image_mapped * scale_factor
    print(f"5. After intensity adjustment min: {np.min(image_tuned)}, max: {np.max(image_tuned)}")

    # Normalize to [0, 1] range again after adjustment
    image_tuned = (image_tuned - np.min(image_tuned)) / (np.max(image_tuned) - np.min(image_tuned))

    # Convert to 8-bit image
    hdr_image = (image_tuned * 255).astype(np.uint8)
    print(f"6. Final HDR image min: {np.min(hdr_image)}, max: {np.max(hdr_image)}")

    plt.figure(figsize=(10, 8))
    plt.imshow(hdr_image, cmap='gray')
    plt.title('Final HDR Image')
    plt.colorbar()
    plt.show()

    return hdr_image, response_curve, z_min, z_max, img_rad_map, intensity_samples, log_exposures

def globalToneMapping(image, gamma):
    """
    Global tone mapping using gamma correction

    Parameters
    ----------
    image : numpy.ndarray
        Image needed to be corrected
    gamma : float
        The number for gamma correction. Higher value for brighter result; lower for darker

    Returns
    -------
    numpy.ndarray
        The resulting image after gamma correction
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("image must be a numpy array")
    if not isinstance(gamma, (int, float)) or gamma <= 0:
        raise ValueError("gamma must be a positive number")

    # Ensure all values are non-negative
    image = np.maximum(image, 0)
    
    # Avoid division by zero
    max_val = np.max(image)
    if max_val == 0:
        return np.zeros_like(image)
    
    return cv2.pow(image / max_val, 1.0 / gamma)

def intensityAdjustment(image, template):
    """
    Tune image intensity based on template

    Parameters
    ----------
    image : numpy.ndarray
        2D array of image to be adjusted
    template : numpy.ndarray
        2D array of template image (typically middle image from stack)

    Returns
    -------
    numpy.ndarray
        The resulting image after intensity adjustment
    """
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        raise ValueError("image must be a 2D numpy array")
    if not isinstance(template, np.ndarray) or template.ndim != 2:
        raise ValueError("template must be a 2D numpy array")
    if image.shape != template.shape:
        raise ValueError("image and template must have the same shape")

    image_avg = np.average(image)
    template_avg = np.average(template)
    return image * (template_avg / image_avg)


