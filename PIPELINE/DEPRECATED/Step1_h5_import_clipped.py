import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from datetime import datetime
from reportlab.lib.pagesizes import letter, landscape
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
import io

"""
This module contains functions for importing, processing, and visualizing image data from H5 files,
as well as generating reports and saving processed data.

Functions:
1. import_h5(directory, experiment_title):
   Imports H5 files from a specified directory, processes the image data including darkcount subtraction based 
   on a model of exposure time-dependent dark counts,
   and returns a dictionary containing original images, exposure times, darkcount images, and denoised images.

2. plot_image_array(image_set, exposure_times, max_images=8):
   Creates plots of image arrays for each imaging condition, handling cases with many exposure times
   by selecting a subset if necessary. Returns plot data and intensity ranges.

3. save_processed_data(extracted_images, experiment_title, data_folder="data"):
   Saves the processed (denoised) image data and exposure times to specified files in a data folder.

4. generate_report(extracted_images, output_dir="reports"):
   Generates a comprehensive PDF report containing tables of exposure times and intensity ranges,
   as well as visualizations of both original and denoised images.

These functions work together to provide a complete workflow for importing, processing,
visualizing, and reporting on image data from H5 files, with a focus on handling multiple
exposure times and darkcount subtraction.
"""


import os
import numpy as np
import h5py
import gc
import tempfile

def determine_upper_clip(Sd, b, Smax, smoothness, linearity_threshold=0.01):
    t = np.linspace(0, np.maximum((Smax - b) / Sd, 0), 1000)
    response = Smax - (Smax - (Sd * t + b)) / (1 + np.exp(smoothness * ((Sd * t + b) - Smax)))
    linear_response = Sd * t + b
    relative_error = np.abs(response - linear_response) / np.maximum(linear_response, 1e-10)
    upper_limit_index = np.argmax(relative_error > linearity_threshold, axis=0)
    upper_limit_t = t[upper_limit_index]
    
    upper_clip = Smax - (Smax - (Sd * upper_limit_t + b)) / (1 + np.exp(smoothness * ((Sd * upper_limit_t + b) - Smax)))
    
    return upper_clip

def import_h5(directory, experiment_title):
    laser_wavelengths = {'1': '670', '2': '760', '3': '808'}
    emission_filters = {'12': 'BP1150', '13': 'BP1200', '14': 'BP1250', '15': 'BP1300', '16': 'BP1350', '17': 'BP1575'}

    # data location
    directory1 = f'/Users/allisondennis/Spectral_demixing/notebooks/PIPELINE/data/'
    bias_file = 'b.npy'
    slope_file = 'Sd.npy'
    saturation_file = 'Smax.npy'
    smooth_file = 'smooth.npy'

    # Load arrays
    b = np.load(os.path.join(directory1, bias_file))
    Sd = np.load(os.path.join(directory1, slope_file))
    Smax = np.load(os.path.join(directory1, saturation_file))
    smoothness = np.load(os.path.join(directory1, smooth_file))

    # Calculate upper clip (if needed)
    upper_clip = determine_upper_clip(Sd, b, Smax, smoothness)

    # Initialize dictionaries
    image_files = {}
    image_arrays = {}
    image_denoised_arrays = {}
    exposure_times = []

    # Get the list of image files and process them
    for laser_key, laser_value in laser_wavelengths.items():
        for filter_key, filter_value in emission_filters.items():
            key = f"{experiment_title}_{laser_value}_{filter_value}"
            image_files[key] = [f for f in os.listdir(directory) if f.startswith(f"{experiment_title}_{laser_key}_{filter_key}")]
            image_files[key].sort(key=lambda x: float(x.split('_')[-1][:-3]))

            image_data = []
            denoised_data = []
            
            for i in range(0, len(image_files[key]), batch_size):
                batch = image_files[key][i:i+batch_size]
                
                batch_images = []
                batch_exposure_times = []
                
                for file in batch:
                    file_path = os.path.join(directory, file)
                    with h5py.File(file_path, 'r') as h5f:
                        image = h5f['Cube']['Images'][()]
                        exposure_time = h5f['Cube']['TimeExposure'][()].item()
                        batch_images.append(image)
                        batch_exposure_times.append(exposure_time)

                batch_images = np.array(batch_images)
                
                # Process the batch
                clipped_batch = np.minimum(batch_images, upper_clip)
                darkcount_batch = Sd * np.array(batch_exposure_times)[:, np.newaxis, np.newaxis] + b
                denoised_batch = np.maximum(clipped_batch - darkcount_batch, 0)
                
                image_data.extend(batch_images)
                denoised_data.extend(denoised_batch)
                exposure_times.extend(batch_exposure_times)
                
                # Clear memory
                del batch_images, clipped_batch, darkcount_batch, denoised_batch
                gc.collect()

            image_arrays[key] = np.array(image_data)
            image_denoised_arrays[key] = np.array(denoised_data)

        exposure_times = np.unique(exposure_times)

    # Print information
    print(exposure_times.shape)
    print(exposure_times)
    for key in image_arrays.keys():
        print(f"Image array shape for {key}:", image_denoised_arrays[key].shape)

    return {
        "images": image_arrays,
        "exposure_times": exposure_times,
        "denoised_images": image_denoised_arrays,
        "upper_clip": upper_clip
    }


def plot_image_array(image_set, exposure_times, max_images=8):
    """
    Plot a tight arrangement of the full series of exposure times for each imaging condition.
    """
    if not isinstance(image_set, dict):
        raise ValueError("image_set must be a dictionary.")
    if not isinstance(exposure_times, np.ndarray):
        raise ValueError("exposure_times must be a numpy array.")

    min_max_dict = {key: (np.min(array), np.max(array)) for key, array in image_set.items()}
    plot_data = []

    for key, array in image_set.items():
        num_images = array.shape[0]
        
        # If there are more than max_images, select evenly spaced indices
        if num_images > max_images:
            indices = np.linspace(0, num_images - 1, max_images, dtype=int)
            selected_arrays = array[indices]
            selected_times = exposure_times[indices]
        else:
            selected_arrays = array
            selected_times = exposure_times

        num_selected = len(selected_arrays)
        
        # Calculate figure size based on image aspect ratio
        image_height, image_width = selected_arrays[0].shape
        aspect_ratio = image_width / image_height
        fig_width = min(20, max(15, num_selected * 2))  # Adjust width based on number of images
        fig_height = fig_width / (num_selected * aspect_ratio)
        
        fig, axes = plt.subplots(1, num_selected, figsize=(fig_width, fig_height))
        fig.suptitle(key)
    
        min_val, max_val = min_max_dict[key]
    
        for i, (img, time) in enumerate(zip(selected_arrays, selected_times)):
            ax = axes[i] if num_selected > 1 else axes
            im = ax.imshow(img, cmap='gray', vmin=min_val, vmax=max_val, aspect='equal')
            ax.set_title(f'{time:.4g} s')  # Use 4g format to show up to 4 significant digits
            ax.axis('off')
    
        plt.tight_layout()
        
        # Save plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plot_data.append((key, buf))
        plt.close(fig)

    return plot_data, min_max_dict




def save_processed_data(extracted_images, directory, experiment_title, base_data_folder="data"):
    # Extract the last part of the directory path
    experiment_folder = os.path.basename(os.path.normpath(directory))
    
    # Create the full path for the data folder
    data_folder = os.path.join(base_data_folder, experiment_folder)
    
    # Ensure the data folder exists
    os.makedirs(data_folder, exist_ok=True)

    # Save darkcount-subtracted (denoised) images
    for key, value in extracted_images['denoised_images'].items():
        denoised_file = os.path.join(data_folder, f"{key}_clipped.npy")
        # Ensure non-negative values and convert to uint16
        np.save(denoised_file, np.clip(value, 0, np.iinfo(np.uint16).max).astype(np.uint16))
        print(f"Clipped images for {key} saved to: {denoised_file}")

    # Save exposure times
    exposure_times_file = os.path.join(data_folder, f"{experiment_title}_exposure_times.npy")
    np.save(exposure_times_file, extracted_images['exposure_times'])
    print(f"Exposure times saved to: {exposure_times_file}")
    
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
        if file.endswith("_clipped.npy"):
            key = file.split('_')[0]  # Assuming the key is the first part of the filename
            denoised_file = os.path.join(data_folder, file)
            denoised_images[key] = np.load(denoised_file)
            print(f"Loaded clipped images for {key} from: {denoised_file}")

    # Load exposure times
    exposure_times_file = os.path.join(data_folder, f"{experiment_title}_exposure_times.npy")
    exposure_times = np.load(exposure_times_file)
    print(f"Loaded exposure times from: {exposure_times_file}")

    return {
        'denoised_images': denoised_images,
        'exposure_times': exposure_times
    }

def generate_import_report(extracted_images, directory, experiment_title, Sd, b, Smax, smoothness, base_data_folder="data"):
    # Extract the last part of the directory path
    experiment_folder = os.path.basename(os.path.normpath(directory))
    
    # Create the full path for the data folder
    data_folder = os.path.join(base_data_folder, experiment_folder)
    
    # Ensure the data folder exists
    os.makedirs(data_folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"import_report_{experiment_title}_{timestamp}.pdf"
    filepath = os.path.join(data_folder, filename)
    
    c = canvas.Canvas(filepath, pagesize=landscape(letter))
    width, height = landscape(letter)

    # Add title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, f"Import Report: {experiment_title}")
    
    # Add timestamp
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 70, f"Generated on: {timestamp}")
    
    y_position = height - 100

    # Create table data
    table_data = [['Exposure Time (s)', 'Original Min', 'Original Max', 'Denoised Min', 'Denoised Max']]
    for key in extracted_images['images'].keys():
        for time, orig_img, denoised_img in zip(extracted_images['exposure_times'], 
                                                extracted_images['images'][key],
                                                extracted_images['denoised_images'][key]):
            orig_min, orig_max = np.min(orig_img), np.max(orig_img)
            denoised_min, denoised_max = np.min(denoised_img), np.max(denoised_img)
            table_data.append([f"{time:.4g}", f"{orig_min:.4g}", f"{orig_max:.4g}", 
                            f"{denoised_min:.4g}", f"{denoised_max:.4g}"])

    # Create table
    table = Table(table_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))

    # Draw the table
    table.wrapOn(c, width - 100, height)
    table.drawOn(c, 50, y_position - table._height)

    y_position -= (table._height + 50)

    # Plot and add original images
    plot_data, min_max_dict = plot_image_array(extracted_images['images'], extracted_images['exposure_times'])
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_position, "Original Images")
    y_position -= 20
    for key, buf in plot_data:
        if y_position - 300 < 50:  # Check if there's enough space on the page
            c.showPage()
            y_position = height - 50

        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y_position, f"Image set: {key}")
        y_position -= 20

        c.setFont("Helvetica", 10)
        min_val, max_val = min_max_dict[key]
        c.drawString(50, y_position, f"Intensity range: Min = {min_val:.4g}, Max = {max_val:.4g}")
        y_position -= 30

        img = ImageReader(buf)
        img_width = width - 100
        img_height = img_width * (img.getSize()[1] / img.getSize()[0])  # Maintain aspect ratio
        c.drawImage(img, 50, y_position - img_height, width=img_width, height=img_height)
        y_position -= (img_height + 50)

    # Plot and add denoised images
    c.showPage()
    y_position = height - 50
    plot_data_denoised, min_max_dict_denoised = plot_image_array(extracted_images['denoised_images'], extracted_images['exposure_times'])
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_position, "Denoised Images")
    y_position -= 20
    for key, buf in plot_data_denoised:
        if y_position - 300 < 50:  # Check if there's enough space on the page
            c.showPage()
            y_position = height - 50

        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y_position, f"Image set: {key}")
        y_position -= 20

        c.setFont("Helvetica", 10)
        min_val, max_val = min_max_dict_denoised[key]
        c.drawString(50, y_position, f"Intensity range: Min = {min_val:.4g}, Max = {max_val:.4g}")
        y_position -= 30

        img = ImageReader(buf)
        img_width = width - 100
        img_height = img_width * (img.getSize()[1] / img.getSize()[0])  # Maintain aspect ratio
        c.drawImage(img, 50, y_position - img_height, width=img_width, height=img_height)
        y_position -= (img_height + 50)

    upper_clip = extracted_images["upper_clip"]
    c.drawString(50, height - 90, "Data clipped on a per-pixel basis")
    c.drawString(50, height - 110, f"Upper clip range: [{np.min(upper_clip):.2f}, {np.max(upper_clip):.2f}]")
    
    # Calculate dark current ranges for the first and last exposure time as examples
    first_time = extracted_images["exposure_times"][0]
    last_time = extracted_images["exposure_times"][-1]
    
    dc_first = Sd * first_time + b
    dc_last = Sd * last_time + b
    
    c.drawString(50, height - 130, f"Dark current range for {first_time}s exposure:")
    c.drawString(70, height - 150, f"[{np.min(dc_first):.2f}, {np.max(dc_first):.2f}]")
    
    c.drawString(50, height - 170, f"Dark current range for {last_time}s exposure:")
    c.drawString(70, height - 190, f"[{np.min(dc_last):.2f}, {np.max(dc_last):.2f}]")


    c.save()
    print(f"Import report saved: {filepath}")

# Usage example:
# # Specify data location and extract pixel intensities and specific image acquisition information
# directory = '/path/to/your/data/240330_frozen_RO_supine_macro/'
# experiment_title = 'frozen_RO_supine_macro'
# extracted_images = import_h5(directory, experiment_title)
#    
# # Saving data
# save_processed_data(extracted_images, directory, experiment_title)
# 
# # Loading data
# loaded_data = load_processed_data(directory, experiment_title)
#
# Generate import report showing thumbnails of selected images
# generate_import_report(extracted_images, directory, experiment_title)
    
if __name__ == '__main__':
    # Enter the directory path and experiment title
    directory = '/Users/allisondennis/Library/CloudStorage/OneDrive-NortheasternUniversity/Shared Documents - Dennis Lab/XZ/Data/IR VIVO data/240515 Animal/240515_2hr p.i. RO right side re'
    experiment_title = '2hr p.i. RO right side re'
    extracted_images = import_h5(directory, experiment_title)
