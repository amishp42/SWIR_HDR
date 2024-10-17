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
    
    intensity_samples = []
    log_exposures = []
    response_curves = []
    
    for i in range(len(data)):
        if f'intensity_samples_{i}' in data:
            intensity_samples.append(data[f'intensity_samples_{i}'])
            log_exposures.append(data[f'log_exposures_{i}'])
            response_curves.append(data[f'response_curve_{i}'])
    
    print(f"Loaded {len(response_curves)} CRFs")
    print(f"Shape of first intensity_samples: {intensity_samples[0].shape}")
    print(f"Shape of first log_exposures: {log_exposures[0].shape}")
    print(f"Length of first response_curve: {len(response_curves[0])}")
    
    return intensity_samples, log_exposures, response_curves

def analyze_crfs(intensity_samples, log_exposures, response_curves, num_clusters=3):
    """Analyze CRFs to determine if different ones are needed for different wavelength regimes."""
    # Compute slope and intercept for each CRF
    slopes = []
    intercepts = []
    for curve in response_curves:
        x = np.arange(len(curve))
        slope, intercept, _, _, _ = linregress(x, curve)
        slopes.append(slope)
        intercepts.append(intercept)
    
    # Perform k-means clustering
    features = np.column_stack((slopes, intercepts))
    
    # Check if we have enough samples for clustering
    if len(features) < num_clusters:
        print(f"Warning: Not enough CRFs ({len(features)}) for {num_clusters} clusters. Setting num_clusters to {len(features)}.")
        num_clusters = len(features)
    
    if num_clusters > 1:
        centroids, _ = kmeans(features, num_clusters)
        labels, _ = vq(features, centroids)
        
        # Compute silhouette score to evaluate clustering quality
        silhouette_avg = silhouette_score(features, labels) if len(features) > num_clusters else 0
    else:
        labels = np.zeros(len(features), dtype=int)
        silhouette_avg = 0
    
    return labels, silhouette_avg

def generate_general_crf(response_curves, labels):
    """Generate a general CRF for each identified cluster."""
    general_crfs = []
    for cluster in np.unique(labels):
        cluster_curves = [curve for i, curve in enumerate(response_curves) if labels[i] == cluster]
        max_length = max(len(curve) for curve in cluster_curves)
        padded_curves = [np.pad(curve, (0, max_length - len(curve)), 'constant', constant_values=np.nan) for curve in cluster_curves]
        general_crf = np.nanmean(padded_curves, axis=0)
        general_crfs.append(general_crf)
    return general_crfs

def plot_adaptive_weighting(Zmax):
    """Visualize the adaptive weighting function."""
    pixel_values = np.linspace(0, Zmax, 1000)
    weights = adaptive_weight(pixel_values, Zmax)
    
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
        plt.plot(np.arange(len(curve)), curve, alpha=0.5, label=f'Cluster {labels[i]}')
    plt.title('Comparison of Multiple CRFs')
    plt.xlabel('Pixel Value')
    plt.ylabel('Log Exposure')
    plt.legend()
    plt.grid(True)
    plt.show()
