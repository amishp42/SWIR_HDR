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
from reportlab.platypus import SimpleDocTemplate, Image, Paragraph, PageBreak, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from matplotlib.gridspec import GridSpec
from reportlab.platypus import Flowable, BaseDocTemplate, PageTemplate, Frame, PageBreak, Image, Paragraph, Spacer

import os
import numpy as np
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
and HDR image generation.
"""
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Image, Paragraph, PageBreak, Spacer, Flowable, BaseDocTemplate, PageTemplate, Frame
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
import io
import logging
from datetime import datetime

class FilenamePlaceholder(Flowable):
    def __init__(self, filename):
        Flowable.__init__(self)
        self.filename = filename
    def draw(self):
        self.canv.setFont('Helvetica', 9)
        self.canv.drawString(0, 0, f"File: {self.filename}")

def generate_multi_page_report(denoised_files, exposure_times, directory, experiment_title, output_file):
    """Generate a multi-page HDR report with a page for each npy file and a log at the end."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'hdr_report_{experiment_title}_{timestamp}.pdf'
    output_path = os.path.join(directory, output_file)
    
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
    
    for npy_filename in denoised_files:
        images = np.load(os.path.join(directory, npy_filename))
        
        if images.dtype != np.uint16:
            logger.warning(f"Image data in {npy_filename} is not uint16. Consider updating the save process.")
        
        montage, hdr_image, response_curve = capture_plots(images, exposure_times)
        
        available_width = page_width - 4*margin
        available_height = page_height - 5*margin
        
        img_reader = ImageReader(montage)
        orig_width, orig_height = img_reader.getSize()
        
        width_ratio = available_width / orig_width
        height_ratio = available_height / orig_height
        scale_factor = min(width_ratio, height_ratio)
        
        new_width = orig_width * scale_factor
        new_height = orig_height * scale_factor
        
        story.append(FilenamePlaceholder(npy_filename))
        story.append(Image(montage, width=new_width, height=new_height))
        story.append(PageBreak())
        
        logger.info(f"Processed {npy_filename}")
    
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


def linearWeight(pixel_value, z_min, z_max):
    """
    Linear weighting function based on pixel intensity that reduces the
    weight of pixel values that are near saturation.
    """
    pixel_value = np.asarray(pixel_value)
    mid = (z_min + z_max) / 2
    weight = np.where(pixel_value <= mid, 
                      pixel_value - z_min, 
                      z_max - pixel_value)
    return weight.astype(np.float32)

def estimate_radiance(images, exposure_times):
    """
    Estimate the relative radiance for each pixel.
    """
    num_images, height, width = images.shape
    radiance = np.zeros((height, width))
    weight_sum = np.zeros((height, width))
    
    for i in range(num_images):
        weights = 1 - 2 * np.abs(images[i].astype(float) / np.max(images[i]) - 0.5)
        radiance += weights * images[i].astype(float) / exposure_times[i]
        weight_sum += weights
    
    return radiance / np.maximum(weight_sum, 1e-6)

def sampleIntensities(images, exposure_times, num_samples=150000):
    """
    Sample pixel intensities from the exposure stack, ensuring full intensity range coverage.
    """
    num_images, height, width = images.shape
    z_min, z_max = int(np.min(images)), int(np.max(images))
    print(f"z_max: {z_max}; z_min: {z_min}")

    radiance = estimate_radiance(images, exposure_times)
    widest_range_image = images[np.argmax([np.ptp(img) for img in images])]

    num_bins = min(num_samples, z_max - z_min + 1)
    bins = np.linspace(z_min, z_max, num_bins + 1, dtype=int)

    sampled_pixels = []
    for i in range(len(bins) - 1):
        bin_mask = (widest_range_image >= bins[i]) & (widest_range_image < bins[i+1])
        pixels_in_bin = np.where(bin_mask)
        if len(pixels_in_bin[0]) > 0:
            idx = np.random.randint(len(pixels_in_bin[0]))
            sampled_pixels.append((pixels_in_bin[0][idx], pixels_in_bin[1][idx]))

    intensity_samples = np.zeros((len(sampled_pixels), num_images), dtype=np.uint16)
    log_exposures = np.zeros((len(sampled_pixels), num_images))

    for i, (row, col) in enumerate(sampled_pixels):
        for j in range(num_images):
            intensity_samples[i, j] = images[j, row, col]
            log_exposures[i, j] = np.log(radiance[row, col] * exposure_times[j] + 1e-10)

    valid_samples = np.all((intensity_samples > 0) & (intensity_samples < z_max), axis=1)
    intensity_samples = intensity_samples[valid_samples]
    log_exposures = log_exposures[valid_samples]

    print(f"Number of valid samples: {intensity_samples.shape[0]}")

    return intensity_samples, log_exposures, z_min, z_max

def computeResponseCurve(intensity_samples, log_exposures, smoothing_lambda, weighting_function, z_min, z_max):
    """
    Find the camera response curve for a single color channel
    """
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

def computeRadianceMap(images, exposure_times, response_curve, weighting_function, z_min, z_max):
    """
    Calculate a radiance map for each pixel from the response curve.
    """
    num_images, height, width = images.shape
    img_rad_map = np.zeros((height, width), dtype=np.float32)
    sum_weights = np.zeros((height, width), dtype=np.float32)

    for i in range(num_images):
        w = weighting_function(images[i], z_min, z_max)
        img_rad_map += w * (response_curve[np.clip(images[i] - z_min, 0, len(response_curve) - 1)] - np.log(exposure_times[i]))
        sum_weights += w

    sum_weights[sum_weights == 0] = 1e-6
    img_rad_map /= sum_weights

    return img_rad_map

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

def plot_radiance_map(img_rad_map, ax=None):
    """Plot the radiance map."""
    if ax is None:
        ax = plt.gca()
    
    im = ax.imshow(img_rad_map, cmap='viridis')
    plt.colorbar(im, ax=ax, label='Radiance')
    ax.set_title('Radiance Map')

def capture_plots(images, exposure_times, smoothing_lambda=1000., gamma=0.6):
    """Capture all plots for the HDR report."""
    fig = plt.figure(figsize=(8.5, 11))  # Adjusted to match letter size in inches
    gs = GridSpec(4, 2, figure=fig, height_ratios=[1, 1, 1, 1.2])

    hdr_image, response_curve, z_min, z_max, img_rad_map, intensity_samples, log_exposures = computeHDR(images, exposure_times, smoothing_lambda, gamma)

    ax1 = fig.add_subplot(gs[0, 0])
    plot_weighting_function(linearWeight, z_min, z_max, ax=ax1)
    ax1.set_title('Weighting Function', pad=10)

    ax2 = fig.add_subplot(gs[0, 1])
    plot_response_curve(intensity_samples, log_exposures, response_curve, z_min, z_max, ax=ax2)
    ax2.set_title('CRF Fitting', pad=10)

    ax3 = fig.add_subplot(gs[1, 0])
    plot_log_log_crf(response_curve, z_min, z_max, ax=ax3)
    ax3.set_title('Log-Log CRF', pad=10)

    ax4 = fig.add_subplot(gs[1, 1])
    plot_crf_residuals(intensity_samples, log_exposures, response_curve, z_min, ax=ax4)
    ax4.set_title('CRF Residuals', pad=10)

    ax5 = fig.add_subplot(gs[2, 0])
    plot_radiance_map(img_rad_map, ax=ax5)
    ax5.set_title('Radiance Map', pad=10)

    ax6 = fig.add_subplot(gs[2, 1])
    plot_radiance_map(np.exp(img_rad_map), ax=ax6)
    ax6.set_title('Linear Radiance Map', pad=10)

    ax7 = fig.add_subplot(gs[3, :])
    im = ax7.imshow(hdr_image, cmap='gray')
    ax7.set_title('Final HDR Image', pad=10)
    plt.colorbar(im, ax=ax7, orientation='vertical', pad=0.08, aspect=25)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5, wspace=0.3)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)

    return buf, hdr_image, response_curve


def generate_report(images, exposure_times, directory, experiment_title, npy_filename, output_file):
    """Generate the HDR report with a header containing the npy filename."""
    montage, hdr_image, response_curve = capture_plots(images, exposure_times)
    
    # Add date and timestamp to the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'{output_file[:-4]}_{timestamp}.pdf'  # Remove '.pdf' and add timestamp
    output_path = os.path.join(directory, output_file)
    
    # Use portrait orientation
    page_width, page_height = letter
    margin = 0.5 * inch
    
    def add_header(canvas, doc):
        canvas.saveState()
        header_style = ParagraphStyle('Header', fontSize=10, alignment=TA_CENTER)
        header_text = f"HDR Image Generation Report: {npy_filename}"
        header = Paragraph(header_text, header_style)
        w, h = header.wrap(doc.width, doc.topMargin)
        header.drawOn(canvas, doc.leftMargin, doc.height + doc.topMargin - h)
        canvas.restoreState()
    
    doc = SimpleDocTemplate(output_path, pagesize=letter,
                            leftMargin=margin, rightMargin=margin,
                            topMargin=margin + 0.5*inch,  # Extra space for header
                            bottomMargin=margin)
    
    # Create the story for the document
    story = []
    
    # Calculate available space
    available_width = page_width - 2*margin
    available_height = page_height - 4*margin  # Leave some space for the header
    
    # Read the image and get its original size
    img_reader = ImageReader(montage)
    orig_width, orig_height = img_reader.getSize()
    
    # Calculate scaling factor
    width_ratio = available_width / orig_width
    height_ratio = available_height / orig_height
    scale_factor = min(width_ratio, height_ratio)
    
    # Calculate new dimensions
    new_width = orig_width * scale_factor
    new_height = orig_height * scale_factor
    
    # Add the scaled image to the story
    story.append(Image(montage, width=new_width, height=new_height))
    
    # Build the PDF
    doc.build(story, onFirstPage=add_header, onLaterPages=add_header)
    print(f"Report saved as {output_path}")


def computeHDR(images, exposure_times, smoothing_lambda=1000., gamma=0.6):
    """Compute the HDR image."""
    intensity_samples, log_exposures, z_min, z_max = sampleIntensities(images, exposure_times)
    response_curve = computeResponseCurve(intensity_samples, log_exposures, smoothing_lambda, linearWeight, z_min, z_max)
    response_curve = savgol_filter(response_curve, window_length=51, polyorder=3)
    img_rad_map = computeRadianceMap(images, exposure_times, response_curve, linearWeight, z_min, z_max)

    img_rad_map = (img_rad_map - np.min(img_rad_map)) / (np.max(img_rad_map) - np.min(img_rad_map))

    def adaptive_log_tone_mapping(x, a=0.5):
        return (np.log(1 + a * x) / np.log(1 + a)) / (np.log(1 + a * np.max(x)) / np.log(1 + a))

    image_mapped = adaptive_log_tone_mapping(img_rad_map)
    
    template = images[len(images) // 2]
    scale_factor = np.mean(template) / np.mean(image_mapped)
    image_tuned = image_mapped * scale_factor

    image_tuned = (image_tuned - np.min(image_tuned)) / (np.max(image_tuned) - np.min(image_tuned))
    hdr_image = (image_tuned * 255).astype(np.uint8)

    return hdr_image, response_curve, z_min, z_max, img_rad_map, intensity_samples, log_exposures

def globalToneMapping(image, gamma):
    """
    Global tone mapping using gamma correction
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("image must be a numpy array")
    if not isinstance(gamma, (int, float)) or gamma <= 0:
        raise ValueError("gamma must be a positive number")

    image = np.maximum(image, 0)
    max_val = np.max(image)
    if max_val == 0:
        return np.zeros_like(image)
    
    return cv2.pow(image / max_val, 1.0 / gamma)

def intensityAdjustment(image, template):
    """
    Tune image intensity based on template
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