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

def asymptotic_fit(t, Smax, smoothness, slope, intercept):
    linear_term = slope * t + intercept
    return Smax - (Smax - linear_term) / (1 + np.exp(smoothness * (linear_term - Smax)))

def model_dark_current(darkcount_array, exposure_times, linear_range, global_popt):
    """Model dark current for each pixel using a two-step fitting process."""
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
    
    global_fit_count = 0
    global_Smax = global_popt[2]
    fit_failure_reasons = []

    for i in range(num_pixels):
        pixel_data = darkcount_data_reshaped[:, i]
        linear_pixel_data = pixel_data[linear_mask]
        
        # Step 1: Linear fit in the linear range
        try:
            popt_linear, _ = curve_fit(linear_fit, linear_exposure_times, linear_pixel_data)
            Sd[i], b[i] = popt_linear
        except:
            Sd[i], b[i], _, _, _ = stats.linregress(linear_exposure_times, linear_pixel_data)
        
        # Step 2: Asymptotic fit using fixed Sd and b
        try:
            max_pixel_value = np.max(pixel_data)
            initial_Smax_guess = max(global_Smax, max_pixel_value * 1.1)
            
            popt_asymptotic, _ = curve_fit(
                lambda t, Smax, smoothness: asymptotic_fit(t, Smax, smoothness, Sd[i], b[i]),
                exposure_times, pixel_data,
                p0=[initial_Smax_guess, 0.1],
                bounds=([max_pixel_value, 1e-6], [global_Smax * 2, 1000]),
                method='trf',  # Trust Region Reflective algorithm, which can handle bounds
                max_nfev=10000  # Increase max number of function evaluations
            )
            Smax[i], _ = popt_asymptotic
            
            # Relaxed sanity check on Smax
            if Smax[i] > global_Smax * 2 or Smax[i] < max_pixel_value:
                raise ValueError(f"Unreasonable Smax: {Smax[i]}")
        except Exception as e:
            Smax[i] = global_Smax
            global_fit_count += 1
            if len(fit_failure_reasons) < 10:
                fit_failure_reasons.append(f"Pixel {i}: Asymptotic fitting failed - {str(e)}")

    Sd = Sd.reshape(darkcount_array.shape[1], darkcount_array.shape[2])
    b = b.reshape(darkcount_array.shape[1], darkcount_array.shape[2])
    Smax = Smax.reshape(darkcount_array.shape[1], darkcount_array.shape[2])
    
    return Sd, b, Smax, global_fit_count, global_Smax, fit_failure_reasons

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
    Sd, b, Smax, global_fit_count, global_Smax, fit_failure_reasons = model_dark_current(darkcount_array, exposure_times, linear_range, popt)
    
    
    # Save model parameters
    save_model_parameters(Sd, b, Smax, output_dir, experiment_name)


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

    plot_buf = plot_model_parameters(Sd, b, Smax, global_fit_count, global_Smax, fit_failure_reasons)
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

# def model_dark_current(darkcount_array, exposure_times, linear_range, global_popt):
#     """Model dark current for each pixel, including individual Smax calculations."""
#     linear_mask = (exposure_times >= linear_range[0]) & (exposure_times <= linear_range[1])
    
#     if not np.any(linear_mask):
#         print("Warning: No exposure times fall within the calculated linear range.")
#         print(f"Linear range: {linear_range}")
#         print(f"Exposure times: {exposure_times}")
        
#         # Use all data points as a fallback
#         linear_mask = np.ones_like(exposure_times, dtype=bool)
    
#     linear_exposure_times = exposure_times[linear_mask]
#     linear_darkcount_data = darkcount_array[linear_mask]
    
#     if linear_darkcount_data.size == 0:
#         raise ValueError("No data points in the linear range. Check your linear_range and exposure_times.")
    
#     num_pixels = darkcount_array.shape[1] * darkcount_array.shape[2]
#     darkcount_data_reshaped = darkcount_array.reshape(darkcount_array.shape[0], -1)
    
#     Sd = np.zeros(num_pixels)
#     b = np.zeros(num_pixels)
#     Smax = np.zeros(num_pixels)
    
#     def fit_pixel(times, values):
#         try:
#             # Initial guess
#             slope_guess, intercept_guess, _, _, _ = stats.linregress(times, values)
#             max_value = np.max(values)
#             p0 = [slope_guess, intercept_guess, max_value * 1.1, 0.1]  # Increased initial Smax guess
            
#             # Relaxed bounds
#             bounds = ([0, 0, max_value, 0], [np.inf, np.inf, max_value * 10, 100])
            
#             # Fit the curve with increased max iterations
#             popt, _ = curve_fit(linear_to_asymptote, times, values, p0=p0, 
#                                 bounds=bounds, maxfev=5000)
#             return popt
#         except Exception as e:
#             print(f"Fitting failed: {str(e)}")
#             return None

#     global_fit_count = 0  # Counter for pixels using global fit
#     global_Smax = global_popt[2]  # Get the global Smax from the input parameters
#     fit_failure_reasons = []

#     for i in range(num_pixels):
#         pixel_data = darkcount_data_reshaped[:, i]
#         fit_result = fit_pixel(exposure_times, pixel_data)
        
#         if fit_result is not None:
#             Sd[i], b[i], Smax[i], _ = fit_result
#             if Smax[i] > global_Smax * 10 or Smax[i] < 0:  # Sanity check on Smax
#                 fit_result = None
        
#         if fit_result is None:
#             # If fitting fails, use linear regression for Sd and b, and global Smax
#             slope, intercept, _, _, _ = stats.linregress(linear_exposure_times, pixel_data[linear_mask])
#             Sd[i] = slope
#             b[i] = intercept
#             Smax[i] = global_Smax  # Use global Smax
#             global_fit_count += 1
#             if len(fit_failure_reasons) < 10:  # Limit the number of stored reasons
#                 fit_failure_reasons.append(f"Pixel {i}: Fitting failed or gave unreasonable Smax")

#     Sd = Sd.reshape(darkcount_array.shape[1], darkcount_array.shape[2])
#     b = b.reshape(darkcount_array.shape[1], darkcount_array.shape[2])
#     Smax = Smax.reshape(darkcount_array.shape[1], darkcount_array.shape[2])
    
#     return Sd, b, Smax, global_fit_count, global_Smax, fit_failure_reasons

def plot_model_parameters(Sd, b, Smax, global_fit_count, global_Smax, fit_failure_reasons):
    """Plot histograms of the model parameters and include summary statistics."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.ravel()  # Flatten the 2x2 array to make indexing easier
    
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
    
    # Use the fourth quadrant for summary statistics
    axs[3].axis('off')
    individual_fit_count = Smax.size - global_fit_count
    summary_text = (
        f"Summary Statistics:\n\n"
        f"Sd: {np.mean(Sd):.2f} ± {np.std(Sd):.2f}\n"
        f"b: {np.mean(b):.2f} ± {np.std(b):.2f}\n"
        f"Smax (all): {np.mean(Smax):.2f} ± {np.std(Smax):.2f}\n"
        f"Smax (individual fits): {np.mean(Smax[Smax != global_Smax]):.2f} ± {np.std(Smax[Smax != global_Smax]):.2f}\n"
        f"Global Smax: {global_Smax:.2f}\n\n"
        f"Pixels with individual Smax fit: {individual_fit_count} out of {Smax.size}\n"
        f"Pixels using global Smax: {global_fit_count} out of {Smax.size}\n\n"
        f"Sample fit failure reasons:\n" + "\n".join(fit_failure_reasons[:5])
    )
    axs[3].text(0.05, 0.95, summary_text, transform=axs[3].transAxes, 
                verticalalignment='top', fontsize=8)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf

def save_model_parameters(Sd, b, Smax, output_dir, experiment_name):
    """Save Sd and b as NPY files."""
    np.save(os.path.join(output_dir, f'{experiment_name}_Sd.npy'), Sd)
    np.save(os.path.join(output_dir, f'{experiment_name}_b.npy'), b)
    np.save(os.path.join(output_dir, f'{experiment_name}_Smax.npy'), Smax)