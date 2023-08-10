import numpy as np
from scipy.ndimage import gaussian_filter, laplace
from scipy.signal import medfilt2d


def enhance_peaks(img, clipping_value, gaussian_sigma=2.5, plane_max = None):
    type_in = img.dtype
    filtered_img = medfilt2d(img.astype(np.float64))
    filtered_img = gaussian_filter(filtered_img, gaussian_sigma)
    filtered_img = laplace(filtered_img)
    filtered_img *= -1
    
    filtered_img -= filtered_img.min()
    filtered_img = np.nan_to_num(filtered_img)
    
    if isinstance(plane_max, type(None)):
        filtered_img /= filtered_img.max()
    else:
        filtered_img /= plane_max

    # To leave room to label in the 3d detection.
    filtered_img *= clipping_value
    return filtered_img.astype(type_in)
