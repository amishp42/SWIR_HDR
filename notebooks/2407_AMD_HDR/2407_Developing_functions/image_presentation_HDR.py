




def computeHDR(images, exposure_times, smoothing_lambda=1000., gamma=0.6):
    """Compute the HDR image."""
    intensity_samples, log_exposures, z_min, z_max = sampleIntensities(images, exposure_times)
    response_curve = computeResponseCurve(intensity_samples, log_exposures, smoothing_lambda, linearWeight, z_min, z_max)
    response_curve = savgol_filter(response_curve, window_length=51, polyorder=3)
    radiance_map = computeRadianceMap(images, exposure_times, response_curve, linearWeight, z_min, z_max)

    radiance_map = (radiance_map - np.min(radiance_map)) / (np.max(radiance_map) - np.min(radiance_map))

    def adaptive_log_tone_mapping(x, a=0.5):
        return (np.log(1 + a * x) / np.log(1 + a)) / (np.log(1 + a * np.max(x)) / np.log(1 + a))

    image_mapped = adaptive_log_tone_mapping(radiance_map)
    
    template = images[len(images) // 2]
    scale_factor = np.mean(template) / np.mean(image_mapped)
    image_tuned = image_mapped * scale_factor

    image_tuned = (image_tuned - np.min(image_tuned)) / (np.max(image_tuned) - np.min(image_tuned))
    hdr_image = (image_tuned * 255).astype(np.uint8)

    return hdr_image, response_curve, z_min, z_max, radiance_map, intensity_samples, log_exposures








def globalToneMapping(image, gamma):
    """
    Global tone mapping using gamma correction
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("image must be a numpy array")
    if not isinstance(gamma, (int, float)) or gamma <= 0:
        raise ValueError("gamma must be a positive number")

    image = np.maximum(image, 0)
    max_val = np.max(image)
    if max_val == 0:
        return np.zeros_like(image)
    
    return cv2.pow(image / max_val, 1.0 / gamma)

def intensityAdjustment(image, template):
    """
    Tune image intensity based on template
    """
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        raise ValueError("image must be a 2D numpy array")
    if not isinstance(template, np.ndarray) or template.ndim != 2:
        raise ValueError("template must be a 2D numpy array")
    if image.shape != template.shape:
        raise ValueError("image and template must have the same shape")

    image_avg = np.average(image)
    template_avg = np.average(template)
    return image * (template_avg / image_avg)


def determine_effective_range(radiance_map, low_percentile=1, high_percentile=99):
    """Determine the effective range of the radiance map."""
    low_val = np.percentile(radiance_map, low_percentile)
    high_val = np.percentile(radiance_map, high_percentile)
    return low_val, high_val

def reinhard_tone_mapping(radiance_map, low_val, high_val, key=0.18):
    """Apply Reinhard's photographic tone mapping."""
    # Normalize
    normalized = (radiance_map - low_val) / (high_val - low_val)
    
    # Apply Reinhard's formula
    L_w = key * normalized
    L_d = L_w / (1 + L_w)
    
    return L_d

def log_histogram_equalization(image):
    """Apply log-scale histogram equalization."""
    # Convert to log scale
    log_image = np.log1p(image)
    
    # Normalize to 0-255
    log_image = ((log_image - log_image.min()) / (log_image.max() - log_image.min()) * 255).astype(np.uint8)
    
    # Apply histogram equalization
    return cv2.equalizeHist(log_image)

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8,8)):
    """Apply CLAHE."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)

def process_hdr_to_8bit(radiance_map):
    # Determine effective range
    low_val, high_val = determine_effective_range(radiance_map)
    
    # Apply tone mapping
    tone_mapped = reinhard_tone_mapping(radiance_map, low_val, high_val)
    
    # Convert to 8-bit
    image_8bit = (tone_mapped * 255).astype(np.uint8)
    
    # Apply log-scale histogram equalization
    log_he = log_histogram_equalization(image_8bit)
    
    # Apply CLAHE
    clahe = apply_clahe(log_he)
    
    return image_8bit, log_he, clahe

