import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.cluster.vq import kmeans, vq
from sklearn.metrics import silhouette_score
from Step2_image_radiance_dev import adaptive_weight

def load_crf_data(filename):
    """Load the saved CRF data."""
    data = np.load(filename)
    return data['intensity_samples'], data['log_exposures'], data['response_curves']

def analyze_crfs(intensity_samples, log_exposures, response_curves, num_clusters=3):
    """Analyze CRFs to determine if different ones are needed for different wavelength regimes."""
    # Compute slope and intercept for each CRF
    slopes = []
    intercepts = []
    for curve in response_curves:
        slope, intercept, _, _, _ = linregress(np.arange(len(curve)), curve)
        slopes.append(slope)
        intercepts.append(intercept)
    
    # Perform k-means clustering
    features = np.column_stack((slopes, intercepts))
    centroids, _ = kmeans(features, num_clusters)
    labels, _ = vq(features, centroids)
    
    # Compute silhouette score to evaluate clustering quality
    silhouette_avg = silhouette_score(features, labels)
    
    return labels, silhouette_avg

def generate_general_crf(response_curves, labels):
    """Generate a general CRF for each identified cluster."""
    general_crfs = []
    for cluster in np.unique(labels):
        cluster_curves = response_curves[labels == cluster]
        general_crf = np.mean(cluster_curves, axis=0)
        general_crfs.append(general_crf)
    return general_crfs

def plot_adaptive_weighting(Smax, DC, exposure_time):
    """Visualize the adaptive weighting function."""
    pixel_values = np.linspace(0, Smax, 1000)
    weights = adaptive_weight(pixel_values, Smax, DC, exposure_time)
    
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

