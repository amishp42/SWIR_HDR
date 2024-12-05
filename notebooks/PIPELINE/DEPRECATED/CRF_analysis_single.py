import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.cluster.vq import kmeans, vq
from sklearn.metrics import silhouette_score
from Step2_image_radiance_maps_from_denoised import adaptive_weight

# def load_crf_data(filename):
#     """Load the saved CRF data."""
#     data = np.load(filename)
#     return data['intensity_samples'], data['log_exposures'], data['response_curves']

def load_crf_data(filename):
    """Load the saved CRF data."""
    data = np.load(filename, allow_pickle=True)
    print("Available keys in the data file:")
    for key in data.keys():
        print(f"- {key}")
    
    intensity_samples = data['intensity_samples_0']
    log_exposures = data['log_exposures_0']
    response_curve = data['response_curve_0']
    
    print(f"Shape of intensity_samples: {intensity_samples.shape}")
    print(f"Shape of log_exposures: {log_exposures.shape}")
    print(f"Shape of response_curve: {response_curve.shape}")
    
    return intensity_samples, log_exposures, response_curve

def analyze_crf(intensity_samples, log_exposures, response_curve):
    """Analyze a single CRF."""
    # Compute slope and intercept for the CRF
    slope, intercept, _, _, _ = linregress(np.arange(len(response_curve)), response_curve)
    
    # Plot the CRF
    plt.figure(figsize=(10, 6))
    plt.plot(response_curve, label='Camera Response Function')
    plt.plot(np.arange(len(response_curve)), slope * np.arange(len(response_curve)) + intercept, 
             label=f'Linear fit (slope={slope:.4f}, intercept={intercept:.4f})')
    plt.xlabel('Pixel Value')
    plt.ylabel('Log Exposure')
    plt.title('Camera Response Function Analysis')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot intensity samples vs log exposures
    plt.figure(figsize=(10, 6))
    for i in range(intensity_samples.shape[1]):
        plt.scatter(intensity_samples[:, i], log_exposures[:, i], alpha=0.1, s=1)
    plt.plot(np.arange(len(response_curve)), response_curve, 'r-', linewidth=2, label='CRF')
    plt.xlabel('Pixel Value')
    plt.ylabel('Log Exposure')
    plt.title('Intensity Samples vs Log Exposures')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return slope, intercept

def generate_general_crf(response_curves, labels):
    """Generate a general CRF for each identified cluster."""
    general_crfs = []
    for cluster in np.unique(labels):
        cluster_curves = response_curves[labels == cluster]
        general_crf = np.mean(cluster_curves, axis=0)
        general_crfs.append(general_crf)
    return general_crfs

def plot_adaptive_weighting(Smax):
    """Visualize the adaptive weighting function."""
    pixel_values = np.linspace(0, Smax, 1000)
    weights = adaptive_weight(pixel_values, Smax)
    
    plt.figure(figsize=(10, 6))
    plt.plot(pixel_values, weights)
    plt.title('Adaptive Weighting Function')
    plt.xlabel('Pixel Value')
    plt.ylabel('Weight')
    plt.grid(True)
    plt.show()

def plot_multiple_crfs(response_curves, labels):
    """Visualize and compare multiple CRFs."""
    plt.figure(figsize=(12, 8))
    for i, curve in enumerate(response_curves):
        plt.plot(curve, alpha=0.5, label=f'Cluster {labels[i]}')
    plt.title('Comparison of Multiple CRFs')
    plt.xlabel('Pixel Value')
    plt.ylabel('Log Exposure')
    plt.legend()
    plt.grid(True)
    plt.show()

