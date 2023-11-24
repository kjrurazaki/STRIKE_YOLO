import numpy as np

from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import img_as_float

def denoise_image(z):
    # Convert to float as the function supports float images
    z_float = img_as_float(z)
    # Estimation of the noise standard deviation.
    # Use this if you don't know the noise standard deviation of the image.
    sigma_est = np.mean(estimate_sigma(z_float, multichannel=False))
    denoised_z = denoise_nl_means(z_float, h = 1.15 * sigma_est, fast_mode=True,
                                  patch_size=5, # patch size
                                  patch_distance=3,  # patch distance
                                  multichannel=False)
    return denoised_z
