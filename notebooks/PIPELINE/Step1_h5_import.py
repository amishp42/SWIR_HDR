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


def load_parameters(param_directory):
    """Load processing parameters from files in specified directory.
    Parameters are to be used with process_and_save function."""
    param_files = {
        'b': 'b.npy',
        'Sd': 'Sd.npy',
        'Smax': 'Smax.npy',
        'smooth': 'smooth.npy',
        'Slinear': 'Slinear.npy'
    }
    
    params = {}
    for param_name, filename in param_files.items():
        filepath = os.path.join(param_directory, filename)
        try:
            params[param_name] = np.load(filepath)
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
            
    return params


def process_and_save(directory, experiment_title, base_data_folder, operations=None, params=None):
    """
    Process image data with configurable operations.
    """
    operations = operations or []
    params = params or {}
    
    raw_folder = import_and_save_raw(directory, experiment_title, base_data_folder)
    
    if not operations:
        return raw_folder
        
    # Changed to use processed_data folder
    output_folder = os.path.join(directory, base_data_folder, "processed_data")
    os.makedirs(output_folder, exist_ok=True)
    
    for file in os.listdir(raw_folder):
        if file.endswith("_raw.npy"):
            key = file[:-8]
            data = np.load(os.path.join(raw_folder, file))
            processed_data = np.zeros(data.shape, dtype=data.dtype)
            processed_data['exposure_time'] = data['exposure_time']
            images = data['image']
            
            if 'clip' in operations:
                if 'Slinear' not in params:
                    raise ValueError("Slinear required for clipping")
                images = np.minimum(images, params['Slinear'])
                
            if 'denoise' in operations:
                if not all(k in params for k in ['Sd', 'b']):
                    raise ValueError("Sd and b required for denoising")
                images = np.maximum(
                    images - (params['Sd'] * data['exposure_time'][:, np.newaxis, np.newaxis] + params['b']), 
                    0
                )
            
            processed_data['image'] = images
            op_names = '_'.join(operations)
            np.save(os.path.join(output_folder, f"{key}_{op_names}.npy"), processed_data)
    
    print(f"Processed data saved in: {output_folder}")
    return output_folder

def import_and_save_raw(directory, experiment_title, base_data_folder, date_cutoff_1='2024/02/13', date_cutoff_2='2024/11/07'):
    """
    Import raw data from h5 files and save as numpy arrays.
    
    Args:
        directory: Base directory containing the data
        experiment_title: Title of the experiment
        base_data_folder: Name of the folder to store raw data
        date_cutoff_1: First cutoff date string in format 'YYYY/MM/DD' 
        date_cutoff_2: Second cutoff date string in format 'YYYY/MM/DD'
    """
    from datetime import datetime
    
    laser_wavelengths = {'1': '670', '2': '760', '3': '808'}
    
    # Define three sets of emission filters
    early_emission_filters = {
        '12': 'BP1100', '13': 'BP1150', '14': 'BP1300',
        '15': 'BP1350', '16': 'BP1500', '17': 'BP1555'
    }
    
    middle_emission_filters = {
        '12': 'BP1150', '13': 'BP1200', '14': 'BP1250',
        '15': 'BP1300', '16': 'BP1350', '17': 'BP1575'
    }
    
    latest_emission_filters = {
        '12': 'BP1150', '13': 'BP1200', '14': 'BP1300',
        '15': 'BP1350', '16': 'BP1500', '17': 'BP1550'
    }
    
    data_folder = os.path.join(directory, base_data_folder, "raw_data")
    os.makedirs(data_folder, exist_ok=True)
    
    cutoff_datetime_1 = datetime.strptime(date_cutoff_1, '%Y/%m/%d')
    cutoff_datetime_2 = datetime.strptime(date_cutoff_2, '%Y/%m/%d')
    
    for laser_key, laser_value in laser_wavelengths.items():
        for file in os.listdir(directory):
            if not file.startswith(f"{experiment_title}_{laser_key}_"):
                continue
                
            try:
                with h5py.File(os.path.join(directory, file), 'r') as h5f:
                    # Get timestamp and seconds from file
                    timestamp_str = h5f['Cube']['Timestamp'][()].decode('utf-8')
                    file_date = datetime.strptime(timestamp_str, '%Y/%m/%d')
                    
                    # Determine which filter set to use based on date and seconds
                    if file_date < cutoff_datetime_1:
                        emission_filters = early_emission_filters
                    elif file_date < cutoff_datetime_2:
                        emission_filters = middle_emission_filters
                    elif file_date >= cutoff_datetime_2:
                        emission_filters = latest_emission_filters
                    
                    # Extract filter key from filename
                    parts = file.split('_')
                    filter_key = next((part for part in parts if any(fk in part for fk in emission_filters.keys())), None)
                    
                    if filter_key is None:
                        print(f"Could not find valid filter key in filename: {file}")
                        continue
                        
                    filter_key = next(fk for fk in emission_filters.keys() if fk in filter_key)
                    filter_value = emission_filters[filter_key]
                    
                    key = f"{experiment_title}_{laser_value}_{filter_value}"
                    
                    # Rest of the processing remains the same as before
                    image_files = [f for f in os.listdir(directory) 
                                 if f.startswith(f"{experiment_title}_{laser_key}_{filter_key}")]
                    
                    if not image_files:
                        print(f"No files found for combination: {key}")
                        continue
                    
                    image_files.sort(key=lambda x: float(x.split('_')[-1][:-3]))
                    image_data = []
                    exposure_times = []
                    
                    for img_file in image_files:
                        file_path = os.path.join(directory, img_file)
                        try:
                            with h5py.File(file_path, 'r') as img_h5f:
                                image = img_h5f['Cube']['Images'][()]
                                exposure_time = img_h5f['Cube']['TimeExposure'][()].item()
                                
                                if image.ndim == 3 and image.shape[0] == 1:
                                    image = image.squeeze(0)
                                
                                image_data.append(image)
                                exposure_times.append(exposure_time)
                        except Exception as e:
                            print(f"Error processing file {img_file}: {str(e)}")
                            continue
                    
                    if not image_data:
                        print(f"No valid data collected for combination: {key}")
                        continue
                    
                    image_array = np.array(image_data)
                    exposure_times = np.array(exposure_times)
                    
                    print(f"Processing {key} - Shape: {image_array.shape}, Exposure times: {exposure_times}")
                    print(f"Used filter set based on date {timestamp_str} and seconds {seconds}")
                    
                    structured_data = np.zeros(len(exposure_times),
                                            dtype=[('exposure_time', float),
                                                  ('image', float, (640, 512))])
                    structured_data['exposure_time'] = exposure_times
                    for i, image in enumerate(image_array):
                        structured_data['image'][i] = np.squeeze(image_array[i])
                    
                    print(f"Final shape of image array for {key}: {structured_data.shape}, "
                          f"dtype: {structured_data.dtype}")
                    np.save(os.path.join(data_folder, f"{key}_raw.npy"), structured_data)
                    
            except Exception as e:
                print(f"Error processing file {file}: {str(e)}")
                continue
    
    print(f"Raw data saved in: {data_folder}")
    return data_folder

def clip_and_save(directory, Slinear, base_data_folder):
    raw_data_folder = os.path.join(directory, base_data_folder, "raw_data")
    clipped_data_folder = os.path.join(directory, base_data_folder, "processed_data")
    os.makedirs(clipped_data_folder, exist_ok=True)

    for file in os.listdir(raw_data_folder):
        if file.endswith("_raw.npy"):
            key = file[:-8]  # Remove '_raw.npy'
            raw_data_file = os.path.join(raw_data_folder, file)
            
            # Load raw data
            raw_data = np.load(raw_data_file)
            print(raw_data.shape)
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

def denoise_and_save(directory, experiment_title, Sd, b, base_data_folder):
    raw_data_folder = os.path.join(directory, base_data_folder, "raw_data")
    denoised_data_folder = os.path.join(directory, base_data_folder, "processed_data")
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

def clip_denoise_and_save(directory, experiment_title, Sd, b, base_data_folder):
    clipped_data_folder = os.path.join(directory, base_data_folder, "processed_data")
    clipped_denoised_data_folder = os.path.join(directory,base_data_folder, "final_data")
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

def calculate_Slinear_adjusted(directory, experiment_title, Sd, b, Slinear, base_data_folder):
    clipped_denoised_data_folder = os.path.join(directory, base_data_folder, "final_data")
    
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

def generate_import_report(directory, experiment_title, params, base_data_folder, operations):
    """Generate report based on specified processing operations."""
    #experiment_folder = os.path.basename(os.path.normpath(directory))
    raw_data_folder = os.path.join(directory, base_data_folder, "raw_data")
    op_names = '_'.join(operations) if operations else 'raw'
    processed_folder = os.path.join(directory, base_data_folder, "processed_data")
    report_folder = os.path.join(directory, base_data_folder, "reports")
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
    c.drawString(50, height - 90, f"Operations performed: {op_names}")
    
    y_position = height - 120

    # Dynamic table headers
    headers = ['Image Set', 'Exposure Time (s)', 'Raw Min', 'Raw Max']
    if operations:
        headers.extend([f'{op_names} Min', f'{op_names} Max'])
    table_data = [headers]

    for file in os.listdir(raw_data_folder):
        if file.endswith("_raw.npy"):
            key = file[:-8]
            raw_data = np.load(os.path.join(raw_data_folder, file))
            
            if operations:
                processed_data = np.load(os.path.join(processed_folder, f"{key}_{op_names}.npy"))

            for i in range(len(raw_data)):
                row = [
                    key,
                    f"{raw_data['exposure_time'][i]:.4g}",
                    f"{np.min(raw_data['image'][i]):.4g}",
                    f"{np.max(raw_data['image'][i]):.4g}"
                ]
                if operations:
                    row.extend([
                        f"{np.min(processed_data['image'][i]):.4g}",
                        f"{np.max(processed_data['image'][i]):.4g}"
                    ])
                table_data.append(row)

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
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))

    table.wrapOn(c, width - 100, height)
    table.drawOn(c, 50, y_position - table._height)
    y_position -= (table._height + 50)

    # Processing parameters section
    exposure_times = raw_data['exposure_time']
    middle_index = len(exposure_times) // 2
    middle_time = exposure_times[middle_index]

    if 'denoise' in operations:
        Sd, b = params['Sd'], params['b']
        dc_middle = Sd * middle_time + b
        c.drawString(50, y_position, f"Dark current for middle exposure time ({middle_time:.2f}s):")
        y_position -= 20
        c.drawString(70, y_position, f"[{np.min(dc_middle):.2f}, {np.max(dc_middle):.2f}]")
        y_position -= 40

    # Image visualization
    for file in os.listdir(raw_data_folder):
        if file.endswith("_raw.npy"):
            key = file[:-8]
            raw_data = np.load(os.path.join(raw_data_folder, file))
            
            if operations:
                processed_data = np.load(os.path.join(processed_folder, f"{key}_{op_names}.npy"))

            if y_position - 300 < 50:
                c.showPage()
                y_position = height - 50

            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y_position, f"Image set: {key}")
            y_position -= 20

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            im1 = ax1.imshow(raw_data['image'][middle_index], cmap='gray')
            ax1.set_title(f"Raw Image (t={middle_time:.2f}s)")
            plt.colorbar(im1, ax=ax1)

            if operations:
                im2 = ax2.imshow(processed_data['image'][middle_index], cmap='gray')
                ax2.set_title(f"Processed Image (t={middle_time:.2f}s)")
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

    # Processing parameters summary
    if operations:
        if 'clip' in operations:
            Slinear = params['Slinear']
            c.drawString(50, y_position, f"Upper clip range: [{np.min(Slinear):.2f}, {np.max(Slinear):.2f}]")
            y_position -= 20

        if 'denoise' in operations:
            first_time = exposure_times[0]
            last_time = exposure_times[-1]
            
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