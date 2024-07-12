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
   Imports H5 files from a specified directory, processes the image data including darkcount subtraction,
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


def import_h5(directory, experiment_title):

    laser_wavelengths = {'1': '670', '2': '760', '3': '808'}
    emission_filters = {'12': 'BP1150', '13': 'BP1200', '14': 'BP1250', '15': 'BP1300', '16': 'BP1350', '17': 'BP1575'}

    # Get the list of darkcount files
    darkcount_files = [f for f in os.listdir(directory) if f.startswith('darkcounts')]
    darkcount_files.sort(key=lambda x: float(x[10:-3]))

    # Initialize a dictionary to store the image file lists
    image_files = {}

    # Get the list of image files for each parameter combination
    for laser_key, laser_value in laser_wavelengths.items():
        for filter_key, filter_value in emission_filters.items():
            key = f"{experiment_title}_{laser_value}_{filter_value}"
            image_files[key] = [f for f in os.listdir(directory) if f.startswith(f"{experiment_title}_{laser_key}_{filter_key}")]
            image_files[key].sort(key=lambda x: float(x.split('_')[-1][:-3]))

    # Read the darkcount files and store the data
    darkcount_data = []
    exposure_times = []
    for file in darkcount_files:
        file_path = os.path.join(directory, file)
        with h5py.File(file_path, 'r') as h5f:
            darkcount = h5f['Cube']['Images'][()]
            exposure_time = h5f['Cube']['TimeExposure'][()].item()
            darkcount_data.append(darkcount)
            exposure_times.append(exposure_time)

    # Concatenate the darkcount data into an array with dimensions (num_exposure_times, height, width)
    darkcount_array = np.squeeze(np.array(darkcount_data))

    # Convert exposure_times to a NumPy array
    exposure_times = np.array(exposure_times)

    # Read the image files and store the data for each parameter combination
    image_arrays = {}
    for key, files in image_files.items():
        image_data = []
        for file in files:
            file_path = os.path.join(directory, file)
            with h5py.File(file_path, 'r') as h5f:
                image = h5f['Cube']['Images'][()]
                image_data.append(image)
        image_arrays[key] = np.squeeze(np.array(image_data))

    # remove noise/darkcounts
    # Calculate the mean and standard deviation of pixel intensities for each exposure time in the darkcount cube
    darkcount_mean = np.mean(darkcount_array[:, :, :], axis=(1, 2))
    darkcount_std = np.std(darkcount_array[:, :, :], axis=(1, 2))
    # print("Average darkcount value:", darkcount_mean)
    # print("Darkcount standard deviation:", darkcount_std)

    # Define the threshold multiplier (e.g., 2 for mean + 2*std)
    threshold_multiplier = 0

    # Create a dictionary to store the denoised image arrays
    image_denoised_arrays = {}

    # Zero out pixels below the threshold in each image cube
    for key in image_arrays.keys():
        threshold = darkcount_mean + threshold_multiplier * darkcount_std
        denoised_array = np.where(image_arrays[key] > threshold[:, np.newaxis, np.newaxis], image_arrays[key] - threshold[:, np.newaxis, np.newaxis], 0)
        image_denoised_arrays[key] = denoised_array

    # print("Threshold:", threshold)
    # print("Denoised arrays:", image_denoised_arrays[key])

    # Print the shapes of the arrays
    # print("Darkcount array shape:", darkcount_array.shape)
    # print("Exposure times array shape:", exposure_times.shape)
    # print(exposure_times)
    # for key in image_arrays.keys():
    #     print(f"Image array shape for {key}:", image_arrays[key].shape)

    return {
        "images": image_arrays,
        "exposure_times": exposure_times,
        "darkcount_images": darkcount_array,
        "denoised_images": image_denoised_arrays
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
        denoised_file = os.path.join(data_folder, f"{experiment_title}_{key}_denoised.npy")
        # Ensure non-negative values and convert to uint16
        np.save(denoised_file, np.clip(value, 0, np.iinfo(np.uint16).max).astype(np.uint16))
        print(f"Denoised images for {key} saved to: {denoised_file}")

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
        if file.startswith(f"{experiment_title}_") and file.endswith("_denoised.npy"):
            key = file.split('_')[-2]  # Assuming the key is the second-to-last part of the filename
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

def generate_import_report(extracted_images, directory, experiment_title, base_data_folder="data"):
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
    for time, orig_img, denoised_img in zip(extracted_images['exposure_times'], 
                                            extracted_images['images'].values(),
                                            extracted_images['denoised_images'].values()):
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