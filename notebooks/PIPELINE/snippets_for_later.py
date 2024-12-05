import numpy as np
# when ready with pre-computed CRF, the calculation of the radiance maps will go much faster....


def compute_radiance_map_fixed_crf(images, exposure_times, crf, weighting_function):
    num_images, height, width = images.shape
    radiance_map = np.zeros((height, width), dtype=np.float32)
    weights_sum = np.zeros((height, width), dtype=np.float32)

    for i, image in enumerate(images):
        # Apply inverse CRF
        linearized = crf[image]
        
        # Compute weights
        weights = weighting_function(image)
        
        # Accumulate weighted radiance
        radiance_map += weights * (linearized / exposure_times[i])
        weights_sum += weights

    # Normalize
    radiance_map /= np.maximum(weights_sum, 1e-6)
    
    return radiance_map

# Usage
crf = load_precomputed_crf()  # Load your pre-computed CRF
radiance_map = compute_radiance_map_fixed_crf(images, exposure_times, crf, weighting_function)



# until fixed CRFs exist, it could at least help to use the last fit to prime the next fit (and fit the CRF faster)

def computeResponseCurve(intensity_samples, log_exposures, exposure_times, smoothing_lambda, weighting_function, Zmin_precomputed, Zmax_precomputed, initial_curve=None):
    num_samples, num_images = intensity_samples.shape
    z_min = int(np.min(Zmin_precomputed))
    z_max = int(np.max(Zmax_precomputed))
    intensity_range = int(z_max - z_min + 1)
    z_mid = int((z_min + z_max) // 2)

    data_constraints = num_samples * num_images
    smoothness_constraints = intensity_range - 2
    monotonicity_constraints = intensity_range - 1
    z_mid_constraint = 1
    total_constraints = data_constraints + smoothness_constraints + monotonicity_constraints + z_mid_constraint

    mat_A = np.zeros((total_constraints, intensity_range), dtype=np.float64)
    mat_b = np.zeros((total_constraints, 1), dtype=np.float64)

    k = 0
    for i in range(num_samples):
        for j in range(num_images):
            z_ij = intensity_samples[i, j]
            # Use the correct Zmin and Zmax for this specific sample and exposure
            w_ij = weighting_function(z_ij, Zmin_precomputed[j], Zmax_precomputed[j])
            
            z_ij_scalar = int(z_ij)
            # Use the mean weight if w_ij is an array
            w_ij_scalar = np.mean(w_ij) if isinstance(w_ij, np.ndarray) else float(w_ij)
            
            if z_ij_scalar - z_min < intensity_range:
                mat_A[k, z_ij_scalar - z_min] = w_ij_scalar
                mat_b[k, 0] = w_ij_scalar * log_exposures[i, j]
                k += 1

    for z_k in range(z_min + 1, z_max):
        w_k = weighting_function(z_k, z_min, z_max)
        w_k_scalar = np.mean(w_k) if isinstance(w_k, np.ndarray) else float(w_k)
        if z_k - z_min + 1 < intensity_range:
            mat_A[k, z_k - z_min - 1:z_k - z_min + 2] = w_k_scalar * smoothing_lambda * np.array([-1, 2, -1])
            k += 1

    for z_k in range(z_min, z_max - 1):
        if k < total_constraints - 1 and z_k - z_min + 1 < intensity_range:
            mat_A[k, z_k - z_min] = -1
            mat_A[k, z_k - z_min + 1] = 1
            mat_b[k, 0] = 0.001
            k += 1
        else:
            break

    # Add the constraint: g(Z_mid) = 0
    if z_mid - z_min < intensity_range:
        mat_A[k, z_mid - z_min] = 1
        mat_b[k, 0] = 0

    # If we have an initial curve, use it as the starting point
    if initial_curve is not None and len(initial_curve) == intensity_range:
        x0 = initial_curve
    else:
        x0 = None

    # Use the initial guess in the least squares solver
    x, residuals, rank, s = np.linalg.lstsq(mat_A, mat_b, rcond=None)
    
    if x0 is not None:
        # If we have an initial guess, use it as a starting point for an iterative solver
        from scipy.optimize import lsq_linear
        result = lsq_linear(mat_A, mat_b.ravel(), x0=x0, method='trf')
        response_curve = result.x
    else:
        response_curve = x.flatten()

    return response_curve
