import numpy as np
from scipy.ndimage import laplace

from helpers.common_utils import timer_decorator

def compute_gradient(dx, dy, z):
    """
    Compute gradient of meshgrid x, y with z
    """
    # Compute the numerical derivative of z with respect to x and y
    dz_dx = np.gradient(z, dx, axis=1)
    dz_dy = np.gradient(z, dy, axis=0)

    # Compute the magnitude of the gradient vector
    gradient_magnitude = np.sqrt(dz_dx**2 + dz_dy**2)

    # Compute the direction of the gradient vector
    gradient_direction = np.arctan2(dz_dy, dz_dx)

    return dz_dx, dz_dy, gradient_magnitude, gradient_direction

@timer_decorator
def gradient_transform(dx, dy, z):
  """
  Return the transforms of gradients
  """
  # First derivatives
  dz_dx, dz_dy, gradient_magnitude, gradient_direction = compute_gradient(dx, dy, z)
  # Second derivatives in x
  dz_dx_dx, dz_dx_dy, gradient_magnitude, gradient_direction = compute_gradient(dx, dy, dz_dx)
  # Second derivatives in y
  dz_dy_dx, dz_dy_dy, gradient_magnitude, gradient_direction = compute_gradient(dx, dy, dz_dy)
  return dz_dx, dz_dy, dz_dx_dx, dz_dx_dy, dz_dy_dx, dz_dy_dy

def laplace_treshold(x, y, z):
  """
  Returns the z values with the laplacian filter applied and the x and y values
  when the threshold is applied.
  ---
  : Notes
  :: The threshold for laplacian grater or equal than zero is removed (background)
  """
  # Find local maxima coordinates
  z_laplace = laplace(z)
  filter_out = np.nonzero(z_laplace >= 0)
  z_laplace[filter_out] = 0
  x_laplace = x.copy()
  x_laplace[filter_out] = 0
  y_laplace = y.copy()
  y_laplace[filter_out] = 0
  # Get the non-zero indices in the x_laplace and y_laplace (meaning center of gaussians)
  nonzero_indices = np.nonzero(x_laplace * y_laplace)

  # Get the non-zero pairs in the x_laplace and y_laplace (meaning center of gaussians)
  nonzero_pairs = np.column_stack((x_laplace[nonzero_indices],
                                  y_laplace[nonzero_indices]))
  return nonzero_pairs, z_laplace
