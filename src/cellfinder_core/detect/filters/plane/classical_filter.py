import numpy as np
from pywt import wavedecn, waverecn
from scipy.ndimage import gaussian_filter, laplace
from scipy.signal import medfilt2d

def wavelet_based_BG_subtraction(image, num_levels, noise_lvl):

  coeffs = wavedecn(image, 'db1', level=None) #decomposition
  coeffs2 = coeffs.copy()
  
  for BGlvl in range(1, num_levels):
      coeffs[-BGlvl] = {k: np.zeros_like(v) for k, v in coeffs[-BGlvl].items()} #set lvl 1 details  to zero
  
  Background = waverecn(coeffs, 'db1') #reconstruction
  del coeffs
  BG_unfiltered = Background
  Background = gaussian_filter(Background, sigma=2**num_levels) #gaussian filter sigma = 2^#lvls 
  
  coeffs2[0] = np.ones_like(coeffs2[0]) #set approx to one (constant)
  for lvl in range(1, len(coeffs2)-noise_lvl):
      coeffs2[lvl] = {k: np.zeros_like(v) for k, v in coeffs2[lvl].items()} #keep first detail lvl only
  Noise = waverecn(coeffs2, 'db1') #reconstruction
  del coeffs2
  
  return Background, Noise, BG_unfiltered


def enhance_peaks(img, clipping_value, gaussian_sigma=2.5, plane_max = None):
    type_in = img.dtype
    img = img.astype(np.float64)
    bkg, noise, _ = wavelet_based_BG_subtraction(img, 4, 1)
        
    img = img - bkg
    img = img - noise
    img[img<0] = 0
    
    filtered_img = medfilt2d(img.astype(np.float64))
    filtered_img = gaussian_filter(filtered_img, gaussian_sigma)
    filtered_img = laplace(filtered_img)
    filtered_img *= -1
    
    filtered_img -= filtered_img.min()
    filtered_img = np.nan_to_num(filtered_img)
    
    if isinstance(plane_max, type(None)) and filtered_img.max() != 0:
        filtered_img /= filtered_img.max()
    elif filtered_img.max() != 0:
        filtered_img /= plane_max

    # To leave room to label in the 3d detection.
    filtered_img *= clipping_value
    return filtered_img.astype(type_in)
