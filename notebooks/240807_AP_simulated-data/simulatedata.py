import numpy as np
import math
import matplotlib.pyplot as plt

def create_circle_mask(width=512, height=640, num_circles=16, min_avg_intensity=1e-6, power=1, saturation_level=4000, initial_condition=None, exposure_time=1.0):
    # Create a new array with a background of 0 or use the initial condition
    if initial_condition is None:
        mask = np.zeros((height, width), dtype=np.float64)
    else:
        mask = initial_condition.copy().astype(np.float64)

    # Calculate the radius and spacing
    max_dimension = min(width, height)
    radius = int(max_dimension / (num_circles + 2))  # Smaller radius for better distribution

    # Calculate the number of rows and columns for even distribution
    num_rows = int(np.sqrt(num_circles))
    num_cols = math.ceil(num_circles / num_rows)

    # Calculate spacing
    x_spacing = width / (num_cols + 1)
    y_spacing = height / (num_rows + 1)

    # Create a meshgrid for the entire image
    y, x = np.ogrid[:height, :width]

    # Create circles with power scaling and saturation
    intensity_step = (saturation_level - min_avg_intensity) / (num_circles - 1)
    circle_positions = []

    for i in range(num_circles):
        row = i // num_cols
        col = i % num_cols
        cx = int((col + 1) * x_spacing)
        cy = int((row + 1) * y_spacing)
        circle_positions.append((cx, cy, i))

    # Sort circle positions by intensity (which is determined by the index i)
    circle_positions.sort(key=lambda x: x[2])

    for cx, cy, i in circle_positions:
        # Create the circle mask for the entire image
        circle_mask = ((x - cx)**2 + (y - cy)**2 <= radius**2)
        
        # Apply power scaling
        pixel_values = ((i ** power) * intensity_step + min_avg_intensity)
        
        # Apply exposure time
        pixel_values *= exposure_time
        
        # Apply saturation
        pixel_values = np.where(
            pixel_values < saturation_level,
            pixel_values,
            saturation_level + (pixel_values - saturation_level) / (1 + (pixel_values - saturation_level) / 100)
        )
        
        # Add the circle values to the mask
        mask += circle_mask * pixel_values

    return mask

def display_mask(mask):
    # Display the mask using Matplotlib
    plt.figure(figsize=(8, 6))
    plt.imshow(mask, cmap='gray')
    plt.colorbar()
    plt.axis('off')
    plt.show()


def generateWellData(exposuretimes):
    """
    Generates simulated well data based on the provided exposure times.

	    Parameters:
	    exposuretimes (list): A list of exposure times.

	    Returns:
	    circle_mask (numpy array): A 3D numpy array representing the simulated well data.
    """

    #Start with blank 640x512 image
    shape = [640, 512]
    simulatedNoise = np.empty([shape[0],shape[1],len(exposuretimes)], dtype=np.uint16)
    for ind in range (len(exposuretimes)):
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
    for ind in range (len(exposuretimes)):
        
        circle_mask[:,:,ind] = create_circle_mask(width=512, height=640, num_circles=16, min_avg_intensity=10, power=1.5, saturation_level=4000, initial_condition=simulatedNoise[:,:,ind], exposure_time=exposuretimes[ind])
    
    #Plot each mask
    for ind in range (len(exposuretimes)):
        display_mask(circle_mask[:,:,ind])


    return circle_mask

