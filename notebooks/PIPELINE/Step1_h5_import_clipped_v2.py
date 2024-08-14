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


def import_and_save_raw(directory, experiment_title, base_data_folder="data"):
    laser_wavelengths = {'1': '670', '2': '760', '3': '808'}
    emission_filters = {'12': 'BP1150', '13': 'BP1200', '14': 'BP1250', '15': 'BP1300', '16': 'BP1350', '17': 'BP1575'}

    experiment_folder = os.path.basename(os.path.normpath(directory))
    data_folder = os.path.join(base_data_folder, experiment_folder, "raw_data")
    os.makedirs(data_folder, exist_ok=True)

    for laser_key, laser_value in laser_wavelengths.items():
        for filter_key, filter_value in emission_filters.items():
            key = f"{experiment_title}_{laser_value}_{filter_value}"
            image_files = [f for f in os.listdir(directory) if f.startswith(f"{experiment_title}_{laser_key}_{filter_key}")]
            image_files.sort(key=lambda x: float(x.split('_')[-1][:-3]))

            image_data = []
            exposure_times = []

            for file in image_files:
                file_path = os.path.join(directory, file)
                with h5py.File(file_path, 'r') as h5f:
                    image = h5f['Cube']['Images'][()]
                    exposure_time = h5f['Cube']['TimeExposure'][()].item()
                    
                    # Ensure image is 2D
                    if image.ndim == 3 and image.shape[0] == 1:
                        image = image.squeeze(0)
                    
                    image_data.append(image)
                    exposure_times.append(exposure_time)

            # Convert lists to numpy arrays
            image_array = np.array(image_data)
            exposure_times = np.array(exposure_times)

            # Create a structured array with exposure times as the first dimension
            structured_data = np.zeros(len(exposure_times), dtype=[('exposure_time', float), ('image', float, (640, 512))])
            structured_data['exposure_time'] = exposure_times
            structured_data['image'] = image_array

            print(f"Final shape of image array for {key}: {structured_data.shape}, dtype: {structured_data.dtype}")

            np.save(os.path.join(data_folder, f"{key}_raw.npy"), structured_data)

    print(f"Raw data saved in: {data_folder}")
    return data_folder

def clip_and_save(directory, Slinear, base_data_folder="data"):
    experiment_folder = os.path.basename(os.path.normpath(directory))
    raw_data_folder = os.path.join(base_data_folder, experiment_folder, "raw_data")
    clipped_data_folder = os.path.join(base_data_folder, experiment_folder, "processed_data")
    os.makedirs(clipped_data_folder, exist_ok=True)

    for file in os.listdir(raw_data_folder):
        if file.endswith("_raw.npy"):
            key = file[:-8]  # Remove '_raw.npy'
            raw_data_file = os.path.join(raw_data_folder, file)
            
            # Load raw data
            raw_data = np.load(raw_data_file)
            
            # Initialize an array to store clipped data
            clipped_data = np.zeros(raw_data.shape, dtype=raw_data.dtype)
            
            # Ensure Slinear has the correct shape
            if Slinear.shape != raw_data['image'][0].shape:
                Slinear = np.broadcast_to(Slinear, raw_data['image'][0].shape)
            
            # Process all images at once
            clipped_images = np.minimum(raw_data['image'], Slinear)

            # Store the clipped data
            clipped_data['exposure_time'] = raw_data['exposure_time']
            clipped_data['image'] = clipped_images

            # Save the clipped data
            np.save(os.path.join(clipped_data_folder, f"{key}_clipped.npy"), clipped_data)

    print(f"Clipped data saved in: {clipped_data_folder}")
    return clipped_data_folder

def denoise_and_save(directory, experiment_title, Sd, b, base_data_folder="data"):
    experiment_folder = os.path.basename(os.path.normpath(directory))
    raw_data_folder = os.path.join(base_data_folder, experiment_folder, "raw_data")
    denoised_data_folder = os.path.join(base_data_folder, experiment_folder, "processed_data")
    os.makedirs(denoised_data_folder, exist_ok=True)

    for file in os.listdir(raw_data_folder):
        if file.endswith("_raw.npy"):
            key = file[:-8]  # Remove '_raw.npy'
            raw_data = np.load(os.path.join(raw_data_folder, file))

            denoised_data = np.zeros(raw_data.shape, dtype=raw_data.dtype)
            exposure_times = raw_data['exposure_time']
            denoised_images = np.maximum(raw_data['image'] - (Sd * exposure_times[:, np.newaxis, np.newaxis] + b), 0)
            
            denoised_data['exposure_time'] = exposure_times
            denoised_data['image'] = denoised_images

            np.save(os.path.join(denoised_data_folder, f"{key}_denoised.npy"), denoised_data)

    print(f"Denoised data saved in: {denoised_data_folder}")

def clip_denoise_and_save(directory, experiment_title, Sd, b, base_data_folder="data"):
    experiment_folder = os.path.basename(os.path.normpath(directory))
    clipped_data_folder = os.path.join(base_data_folder, experiment_folder, "processed_data")
    clipped_denoised_data_folder = os.path.join(base_data_folder, experiment_folder, "final_data")
    os.makedirs(clipped_denoised_data_folder, exist_ok=True)

    for file in os.listdir(clipped_data_folder):
        if file.endswith("_clipped.npy"):
            key = file[:-12]  # Remove '_clipped.npy'
            clipped_data = np.load(os.path.join(clipped_data_folder, file))

            clipped_denoised_data = np.zeros(clipped_data.shape, dtype=clipped_data.dtype)
            exposure_times = clipped_data['exposure_time']
            clipped_denoised_images = np.maximum(clipped_data['image'] - (Sd * exposure_times[:, np.newaxis, np.newaxis] + b), 0)
            
            clipped_denoised_data['exposure_time'] = exposure_times
            clipped_denoised_data['image'] = clipped_denoised_images

            np.save(os.path.join(clipped_denoised_data_folder, f"{key}_clipped_denoised.npy"), clipped_denoised_data)

    print(f"Clipped and denoised data saved in: {clipped_denoised_data_folder}")

def calculate_Slinear_adjusted(directory, experiment_title, Sd, b, Slinear, base_data_folder="data"):
    experiment_folder = os.path.basename(os.path.normpath(directory))
    clipped_denoised_data_folder = os.path.join(base_data_folder, experiment_folder, "final_data")
    
    Slinear_adjusted = np.zeros((640, 512))

    for file in os.listdir(clipped_denoised_data_folder):
        if file.endswith("_clipped_denoised.npy"):
            clipped_denoised_data = np.load(os.path.join(clipped_denoised_data_folder, file))
            
            # Access all exposure times at once
            exposure_times = clipped_denoised_data['exposure_time']
            
            for t in exposure_times:
                Slinear_adjusted = np.maximum(Slinear_adjusted, Slinear - Sd * t - b)

    np.save(os.path.join(clipped_denoised_data_folder, f"{experiment_title}_Slinear_adjusted.npy"), Slinear_adjusted)
    print(f"Slinear_adjusted saved in: {clipped_denoised_data_folder}")

def generate_import_report(directory, experiment_title, Sd, b, Slinear, base_data_folder="data"):
    experiment_folder = os.path.basename(os.path.normpath(directory))
    raw_data_folder = os.path.join(base_data_folder, experiment_folder, "raw_data")
    final_data_folder = os.path.join(base_data_folder, experiment_folder, "final_data")
    report_folder = os.path.join(base_data_folder, experiment_folder, "reports")
    os.makedirs(report_folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"import_report_{experiment_title}_{timestamp}.pdf"
    filepath = os.path.join(report_folder, filename)
    
    c = canvas.Canvas(filepath, pagesize=landscape(letter))
    width, height = landscape(letter)

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, f"Import Report: {experiment_title}")
    
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 70, f"Generated on: {timestamp}")
    
    y_position = height - 100

    # Table data
    table_data = [['Image Set', 'Exposure Time (s)', 'Raw Min', 'Raw Max', 'Clipped Denoised Min', 'Clipped Denoised Max']]

    for file in os.listdir(raw_data_folder):
        if file.endswith("_raw.npy"):
            key = file[:-8]  # Remove '_raw.npy'
            raw_data = np.load(os.path.join(raw_data_folder, file))
            clipped_denoised_data = np.load(os.path.join(final_data_folder, f"{key}_clipped_denoised.npy"))

            for i in range(len(raw_data)):
                table_data.append([
                    key,
                    f"{raw_data['exposure_time'][i]:.4g}",
                    f"{np.min(raw_data['image'][i]):.4g}",
                    f"{np.max(raw_data['image'][i]):.4g}",
                    f"{np.min(clipped_denoised_data['image'][i]):.4g}",
                    f"{np.max(clipped_denoised_data['image'][i]):.4g}"
                ])

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

    table.wrapOn(c, width - 100, height)
    table.drawOn(c, 50, y_position - table._height)

    y_position -= (table._height + 50)

    # Find the middle exposure time
    exposure_times = raw_data['exposure_time']
    middle_index = len(exposure_times) // 2
    middle_time = exposure_times[middle_index]

    # Calculate dark current for the middle exposure time
    dc_middle = Sd * middle_time + b

    c.drawString(50, y_position, f"Dark current for middle exposure time ({middle_time:.2f}s):")
    y_position -= 20
    c.drawString(70, y_position, f"[{np.min(dc_middle):.2f}, {np.max(dc_middle):.2f}]")
    y_position -= 40



    # Add images
    for file in os.listdir(raw_data_folder):
        if file.endswith("_raw.npy"):
            key = file[:-8]  # Remove '_raw.npy'
            raw_data = np.load(os.path.join(raw_data_folder, file))
            clipped_denoised_data = np.load(os.path.join(final_data_folder, f"{key}_clipped_denoised.npy"))

            if y_position - 300 < 50:
                c.showPage()
                y_position = height - 50

            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y_position, f"Image set: {key}")
            y_position -= 20

            # Plot raw image (middle exposure time)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            im1 = ax1.imshow(raw_data['image'][middle_index], cmap='gray')
            ax1.set_title(f"Raw Image (t={middle_time:.2f}s)")
            plt.colorbar(im1, ax=ax1)

            # Plot clipped and denoised image (middle exposure time)
            im2 = ax2.imshow(clipped_denoised_data['image'][middle_index], cmap='gray')
            ax2.set_title(f"Clipped and Denoised Image (t={middle_time:.2f}s)")
            plt.colorbar(im2, ax=ax2)

            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            img = ImageReader(buf)
            img_width = width - 100
            img_height = img_width * (img.getSize()[1] / img.getSize()[0])
            c.drawImage(img, 50, y_position - img_height, width=img_width, height=img_height)

            y_position -= (img_height + 50)
            plt.close(fig)


    c.drawString(50, y_position, "Data clipping and denoising information:")
    y_position -= 20
    c.drawString(50, y_position, f"Upper clip range: [{np.min(Slinear):.2f}, {np.max(Slinear):.2f}]")
    y_position -= 20
    
    # Calculate dark current ranges for the first and last exposure time as examples
    first_time = raw_data['exposure_time'][5]  # Get the first exposure time
    last_time = raw_data['exposure_time'][-1]  # Get the last exposure time
    
    dc_first = Sd * first_time + b
    dc_last = Sd * last_time + b
    
    c.drawString(50, y_position, f"Dark current range for {first_time:.2f}s exposure:")
    y_position -= 20
    c.drawString(70, y_position, f"[{np.min(dc_first):.2f}, {np.max(dc_first):.2f}]")
    y_position -= 20
    
    c.drawString(50, y_position, f"Dark current range for {last_time:.2f}s exposure:")
    y_position -= 20
    c.drawString(70, y_position, f"[{np.min(dc_last):.2f}, {np.max(dc_last):.2f}]")

    c.save()
    print(f"Import report saved: {filepath}")

