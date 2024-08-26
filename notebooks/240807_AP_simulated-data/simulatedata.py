import numpy as np
import math
import matplotlib.pyplot as plt
import h5py
import os

def create_circle_mask(width=640, height=512, num_circles=12, min_avg_intensity=1e-6, power=2, 
                       mean_saturation_level=4500, std_saturation_level=500, 
                       initial_condition=None, exposure_time=1.0, seed=42):
    # Set a fixed random seed for reproducibility
    np.random.seed(seed)
    
    # Generate a saturation level map
    saturation_map = np.random.normal(mean_saturation_level, std_saturation_level, (height, width))
    saturation_map = np.clip(saturation_map, 0, None)  # Ensure non-negative values
    
    # Create a new array with a background of 0 or use the initial condition
    if initial_condition is None:
        mask = np.zeros((height, width), dtype=np.float64)
    else:
        mask = initial_condition.copy().astype(np.float64)
    
    # Calculate the radius and spacing
    max_dimension = min(width, height)
    radius = int(max_dimension / (num_circles + 1))  
    
    # Calculate the number of rows and columns for even distribution
    num_rows = int(np.sqrt(num_circles))
    num_cols = math.ceil(num_circles / num_rows)
    
    # Calculate spacing
    x_spacing = width / (num_cols + 1)
    y_spacing = height / (num_rows + 1)
    
    # Create a meshgrid for the entire image
    y, x = np.ogrid[:height, :width]
    
    # Create circles with power scaling
    intensity_step = (np.mean(saturation_map) - min_avg_intensity) / (num_circles - 1)
    circle_positions = []
    for i in range(num_circles):
        row = i // num_cols
        col = i % num_cols
        cx = int((col + 1) * x_spacing)
        cy = int((row + 1) * y_spacing)
        circle_positions.append((cx, cy, i))
    
    # Sort circle positions by intensity (which is determined by the index i)
    circle_positions.sort(key=lambda x: x[2])
    
    pix_val_store = np.zeros(len(circle_positions))
    for cx, cy, i in circle_positions:
        # Create the circle mask for the entire image
        circle_mask = ((x - cx)**2 + (y - cy)**2 <= radius**2)
        
        # Apply power scaling
        pixel_values = ((i ** power) * intensity_step + min_avg_intensity)
        
        # Apply exposure time
        pixel_values *= exposure_time
        
        # Add the circle values to the mask
        mask += circle_mask * pixel_values
        pix_val_store[i] = pixel_values
    
    # Apply pixel-specific saturation to the entire mask after all circles have been added
    mask = np.where(
        mask < saturation_map,
        mask,
        saturation_map + (mask - saturation_map) / (1 + (mask - saturation_map) *(3.243/196.3845))  #ratio of typical intensity scaling to linear range once saturation is reached
    )
    
    return mask, pix_val_store

# The display_mask function remains unchanged
def display_mask(mask, exposure_time):
    plt.figure(figsize=(8, 6))
    plt.imshow(mask, cmap='gray')
    plt.title(f'Exposure Time: {exposure_time:.2f} seconds')
    plt.colorbar()
    plt.axis('off')
    plt.show()

def display_hist(mask, exposure_time):
    plt.figure(figsize=(8, 6))
    plt.hist(mask.ravel(), bins=256)
    plt.title(f'Exposure Time: {exposure_time:.2f} seconds')
    plt.xlabel('Pixel Value')
    plt.ylabel('Count')
    plt.show()

def generateWellData(exposuretimes):
    """
    Generates simulated well data based on the provided exposure times.

	    Parameters:
	    exposuretimes (list): A list of exposure times.

	    Returns:
	    circle_mask (numpy array): A 3D numpy array representing the simulated well data.
        pix_val_store (numpy array): A 1D numpy array containing the true pixel values for each circle.
    """
    
    #Start with blank 640x512 image
    shape = [640, 512]
    simulatedNoise = np.empty([shape[0],shape[1],len(exposuretimes)], dtype=np.uint16)
    for ind in range (0,len(exposuretimes)):
        
        x = exposuretimes[ind]
        mean = 196.3845 * x + 1195.5938
        std_dev = 3.0389 * x + 13.9957
        
        simulatedNoise[:,:,ind] = np.random.normal(loc=mean, scale=std_dev, size=shape).astype(np.uint16)
    
    #import darkcount information from dark count analysis
    #provide exposure times between the ranges 0.01 and 10 seconds (linear range of noise)
    #Rough equations for linear relation between dark counts and exposure times
    #y = 196.3845x + 1195.5938
    #y = 3.0389x + 13.9957
    #Correlation coefficient: 0.9994
    #Correlation coefficient: 0.9975


    #generate simulated signals in a well plate within image frame
    #make 12 evenly spaced circles within image frame
    # Create the mask with power of 3 scaling and saturation
    circle_mask = np.zeros_like(simulatedNoise)
    pixel_values = np.zeros([len(exposuretimes),16])
    
    for ind in range (len(exposuretimes)):
        
        circle_mask[:,:,ind], pixel_values[ind,:] = create_circle_mask(width=512, height=640, num_circles=16, min_avg_intensity=10, power=1.5, 
                       mean_saturation_level=4414, std_saturation_level=266.7, 
                       initial_condition=simulatedNoise[:,:,ind], exposure_time=exposuretimes[ind], seed=42)
    
    #Plot each mask
    for ind in range (len(exposuretimes)):
        display_mask(circle_mask[:,:,ind], exposure_time=exposuretimes[ind])

    #Plot histogram
    for ind in range (len(exposuretimes)):
        display_hist(circle_mask[:,:,ind], exposure_time=exposuretimes[ind])


    return circle_mask.astype(np.uint16), pixel_values


def save_h5 (fileinput, exposuretimes):

    # Load the .npy file
    image_stack = np.load(fileinput)

    # generate metadata for each image with exposure time and excitation wavelength
    metadata = []
    for i in range(len(exposuretimes)):
        print(i)
        metadata.append({'TimeExposure': exposuretimes[i]})

    # Create a directory to store the .h5 files
    output_dir = 'output_h5_files'
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through the image stack
    for i in range(0,len(exposuretimes)):
        # Create a filename for the .h5 file
        filename = f'well_image_1_12_{i:01d}.h5'  #Adding tags for excitation wavelength and emission filter to ensure compatibility with image processing pipeline
        filepath = os.path.join(output_dir, filename)
    
        # Create the .h5 file and save the image data and metadata
        with h5py.File(filepath, 'w') as f:
            # Save the image data
            f.create_dataset('image', data=image_stack[:,:,i])
        
            # Save the metadata
            for key, value in metadata[i].items():
                f.attrs[key] = value

    print(f"Saved {len(exposuretimes)} images as .h5 files in the '{output_dir}' directory.")

