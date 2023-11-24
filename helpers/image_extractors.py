import numpy as np
from scipy.signal import argrelextrema

from helpers.common_utils import timer_decorator
from helpers.auxiliary_functions import (find_closest_indices)

def center_extraction(x, y, z,
                      expected_x, expected_y,
                      expected_xdeflection, expected_ydeflection,
                      square, threshold, version, return_one = False):
  """
  Finds in the crop the expected center of the gaussian by the maximum amplitude in the most probable sub-region
  The expected values comes from the built of the grids and the previous experiments / conditions the SPIDER is operating
  The threshold is need so no noise / really weak signal are considered a Gaussian - Reducing False Positives
  Mental note: The estimations might be easier the more a priori info about the Gaussians are known.
  "Hope" that the deflection in one column / line is replicated to all beams, not just one
  ---
  Versions:
  v0: center as the maximum value in z
  v1: center as the local maximum value of z with "compressed" values in the x/y axis
  ---
  : returns
  :: x, y center for each gaussian
  :: return_one - if just one center per image
  """
  xmin = expected_x - expected_xdeflection - square/2
  xmax = expected_x + expected_xdeflection + square/2
  ymin = expected_y - expected_ydeflection - square/2
  ymax = expected_y + expected_ydeflection + square/2

  # Mask for square region
  mask = ((x > xmin) & (x < xmax) & (y > ymin) & (y < ymax))

  # Apply the mask to x, y, and z
  x_masked = np.extract(mask, x)
  y_masked = np.extract(mask, y)

  if version == 'v0':
    z_masked = np.extract(mask, z)
    # Find the coordinates of maximum z
    filtered_indices = np.where(z_masked >= threshold)[0]
    if len(filtered_indices) > 0:
      max_index = filtered_indices[np.argmax(z_masked[filtered_indices])]
      center_x = x_masked[max_index]
      center_y = y_masked[max_index]
    else:
      center_x = np.nan
      center_y = np.nan
  elif version == 'v1':
    z_masked = np.zeros_like(z)
    z_masked[mask] = z[mask]
    # apply threshold
    indices = np.nonzero(z_masked < threshold)
    z_masked[indices] = 0

    # Find the local maximums in the 1D summed profiled
    max_indices_x = argrelextrema(z_masked.sum(axis = 0), np.greater)[0]
    max_indices_y = argrelextrema(z_masked.sum(axis = 1), np.greater)[0]

    if (len(max_indices_x) != 0) & (len(max_indices_y) != 0):
      if return_one == True:
        # Match the most centered one
        closest_index_x = find_closest_indices(z_masked.shape, max_indices_x, axis = 1)
        closest_index_y = find_closest_indices(z_masked.shape, max_indices_y, axis = 0)

        # Now, map these indices back to the original x and y values
        center_x = x[0, closest_index_x]
        center_y = y[closest_index_y, 0]

      else:
        # Now, map these indices back to the original x and y values
        center_x = x[0, np.array(max_indices_x).ravel()]
        center_y = y[np.array(max_indices_y).ravel(), 0]

    else:
        center_x = np.nan
        center_y = np.nan
  return center_x, center_y

@timer_decorator
def find_centers(x, y, z, params, version = 'v0', return_one = False):
  """
  Function to iterate over all expected centers and call the function
  center_extraction
  """
  # Find centers
  x_centers = []
  y_centers = []
  for expected_x in params['expected_xs']:
    for expected_y in params['expected_ys']:
      x_center, y_center = center_extraction(x, y, z,
                                            expected_x, expected_y,
                                            params['expected_xdeflection'], params['expected_ydeflection'],
                                            params['square'], params['threshold'], version = version,
                                            return_one = return_one)
      x_centers.append(x_center)
      y_centers.append(y_center)
  return  x_centers, y_centers
