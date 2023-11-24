import numpy as np
import scipy.stats as stats
from sklearn.mixture import GaussianMixture
from skimage.feature import peak_local_max

from helpers.common_utils import timer_decorator
from helpers.transforms import laplace_treshold

# Function to generate a Gaussian
def generate_gaussian(x, y, mean, cov, amplitude):
    normalizing_constant = np.sqrt((2 * np.pi) ** 2 * np.linalg.det(cov))
    gauss = amplitude * stats.multivariate_normal(mean, cov).pdf(np.dstack((x, y))) * normalizing_constant
    return gauss

# Function to sum up all the Gaussians
def sum_gaussians(x, y, gaussians):
    # initialize an empty grid
    total_gauss = np.zeros((x.shape[0], x.shape[1]))
    for mean, cov, amplitude in gaussians:
        gauss = generate_gaussian(x, y, mean, cov, amplitude)
        total_gauss += gauss
    return total_gauss

# Function to compute MSE
def compute_mse(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    return mse

# Function to estimate Gaussian parameters
@timer_decorator
def estimate_parameters_gmm(x,
                            y,
                            n_gaussians,
                            means_init,
                            em_iterations,
                            covariance_gaussians_type):
    """
    Estimation of the gaussian parameters using gmm
    """
    points = np.array([x.flatten(), y.flatten()]).T
    gmm = GaussianMixture(n_components = n_gaussians,
                          max_iter = em_iterations,
                          means_init = means_init,
                          covariance_type = covariance_gaussians_type,
                          random_state = 13)
    gmm.fit(points)
    return gmm, gmm.means_, gmm.covariances_

def estimate_gaussians_centers(image,
                               peak_method = 'laplace'):
  """
  Returns the estimated centers of the gaussians
  ---
  : params
  :: image: The image as intensities (the ones generate, not saved ones)
  :: peak_method: which method to find the maximum peaks. 'image' uses the image
  whithout any treatment, 'laplace' uses the transformed laplace image
  """
  nonzero_pairs, z_laplace = laplace_treshold(image[0], image[1], image[2])
  if peak_method == 'laplace':
    coordinates = peak_local_max(-1 * z_laplace, min_distance = 1)
  elif peak_method == 'image':
    coordinates = peak_local_max(image[2], min_distance = 1)
  return nonzero_pairs, coordinates

def estimate_gaussians_amplitudes(means,
                                  image,
                                  image_ranges = [(0, 0.143), (0, 0.377)],
                                  image_size = [143, 377],
                                  center_in_pixel = False):
  """
  Returns the amplitude given the estimated centers and the image
  ---
  : params
  :: image: (x_coordinates, y_coordinates, image_intensities)
  :: means: The center of the image as reference coordinates (x, y)
  :: center_in_pixel: If True, the means has to be passed using pixel reference
  in the image (#Pixel in y, #Pixel in x)
  """
  if center_in_pixel:
    center_location_pixel = [(x, y) for y, x in means] # Check params
  else:
    # Estimation of amplitude
    xs_center = means[:, 0]/image_ranges[0][-1] * image_size[0] # In Pixels
    ys_center = means[:, 1]/image_ranges[1][-1] * image_size[1] # In Pixels
    center_location_pixel = [(int(x), int(y)) for
                            x,y in zip(xs_center, ys_center)]
  # Amplitude estimation - in the intensity matrix, y is line and x column
  amplitudes = [image[2][y, x] for
                x, y in center_location_pixel]
  return amplitudes

def extract_center_dispersion(tensor_box,
                              img_ranges,
                              img_size):
  """
  NOT BEING USED
  YOLO are labelled using a box that has 3 times the sigma in x/y direction
  ---
  Example call:
  # class 0 boxes
  # tensor_boxes = results[0].boxes.xyxy

  # # Extract centers and dispersion from the prediction
  # centers_dispersions = [extract_center_dispersion(tensor_boxes[i],
  #                         image_ranges,
  #                         image_size) for i in range(len(tensor_boxes))]

  # # Centers
  # means = [centers_dispersions[i][0] for i in range(len(centers_dispersions))]

  # # Covariances
  # covariances = [centers_dispersions[i][1] for i in range(len(centers_dispersions))]
  # covariances = np.array([np.diag(mat) for mat in covariances])

  """
  # Center and dispersion computation
  x_min = float(tensor_box[0] * img_ranges[0][-1]/img_size[0])
  y_min = float(tensor_box[1] * img_ranges[1][-1]/img_size[1])
  width = float((tensor_box[2] - tensor_box[0]) * img_ranges[0][-1]/img_size[0])
  length = float((tensor_box[3] - tensor_box[1]) * img_ranges[1][-1]/img_size[1])
  return (x_min + width/2, y_min + length/2), (width/2/3, length/2/3)

def correct_coordinates(center,
                        image_shape,
                        step_size):
  """
  If the center is in the edge of the image size, there is the problem of the
  new computed point with the step_size to be outside of the image as well.
  This moves the center so when summing the step size the point that lies outside
  is moved in the edge
  """
  # Extract the x and y from center tuple
  y, x = center

  # Extract the image size in y (height) and x (width)
  image_height, image_width = image_shape

  # Correct x
  if x + step_size >= image_width:
      x = image_width - step_size - 1
  elif x - step_size < 0:
      x = step_size
  else:
      x

  # Correct y
  if y + step_size >= image_height:
      y = image_height - step_size - 1
  elif y - step_size < 0:
      y = step_size
  else:
      y

  return (y, x)

def inverse_px(step,
                alpha):
  """
  :: alpha: Relation between the amplitude in the point stepped in and in the center.
  :: step: Size of the step in the reference coordinates
  ---
  :: returns the computed sigma * sqrt(2) from the probability density distribution
  """
  # Flagging as negative when alpha is greater than 1 (probable overlapping)
  if np.log(1/alpha) < 0:
    return -1 * np.sqrt(step ** 2 / -np.log(1/alpha))
  else:
    return np.sqrt(step ** 2 / np.log(1/alpha))

def center_and_neighbors(coord,
                         center,
                         step_size,
                         direction):
  """
  returns the reference coordinates given center and step_size
  direction indicates if the step is in x or y
  """
  if direction == 'x':
    return (coord[center[0]][center[1]],
            coord[center[0]][center[1] + step_size],
            coord[center[0]][center[1] - step_size])
  elif direction == 'y':
    return (coord[center[0]][center[1]],
            coord[center[0] + step_size][center[1]],
            coord[center[0] - step_size][center[1]])

def construct_covariance_and_logical(n_gaussian,
                                     wxs,
                                     wys):
    """
    Construct the covariances matrices and return the logicals of up and down
    given the gaussian number and the inverse values computed
    ---
    : returns
    :: Covariance: the sign is to keep track of negative relation returned caused
    by the amplitude in the points used in the estimation to be higher than the center
    :: logical: Used to remove the negative values of covariance from the final
    averaged covariance
    """
    covariance = np.array([[np.sign(wxs[n_gaussian]) * wxs[n_gaussian] ** 2, 0],
                           [0, np.sign(wys[n_gaussian]) * wys[n_gaussian] ** 2]])

    logical = np.array([[(wxs[n_gaussian] >= 0).astype(int), 0],
                        [0, (wys[n_gaussian] >= 0).astype(int)]])

    return covariance, logical

def estimate_parameters_PX(x,
                           y,
                           image,
                           means_init,
                           step_size
                           ):
  """
  For each gaussians computes the values for FWHM. It works for Gaussians
  aligned in xy coordinates, if rotated there will need to be correction
  ---
  : params
  :: x, y: x and y grid points for mesh
  :: step_size: the number of points to use as step size to perform the estimation,
  note this step_size differs from step
  :: means_init: the estimated centers of the gaussians as the tuple (x, y)
  :: image: the values of intensities / image itself
  ---
  : returns
  :: covariances: the squared values of FWHM. Up, down and the mean between the two
  """
  list_covariances_up = []
  list_covariances_down = []
  list_covariances = []

  list_wxups = []
  list_wxdowns = []
  list_wyups = []
  list_wydowns = []

  for center in means_init:
    center = correct_coordinates(center,
                                 image.shape,
                                 step_size)
    # Reference coordinates
    x_center, x_up, x_down = center_and_neighbors(x,
                                                  center,
                                                  step_size,
                                                  'x')
    # y
    y_center, y_up, y_down = center_and_neighbors(y,
                                                  center,
                                                  step_size,
                                                  'y')

    # Amplitudes (intensities)
    (amplitude_center,
     amplitude_x_up,
     amplitude_x_down) = center_and_neighbors(image,
                                              center,
                                              step_size,
                                              'x')
    (amplitude_center,
     amplitude_y_up,
     amplitude_y_down) = center_and_neighbors(image,
                                              center,
                                              step_size,
                                              'y')

    # FWHM
    list_wxups.append(inverse_px(x_up - x_center,
                                 amplitude_x_up/amplitude_center))
    list_wxdowns.append(inverse_px(x_down - x_center,
                                   amplitude_x_down/amplitude_center))
    list_wyups.append(inverse_px(y_up - y_center,
                                 amplitude_y_up/amplitude_center))
    list_wydowns.append(inverse_px(y_down - y_center,
                                   amplitude_y_down/amplitude_center))


  # Don't use negative values
  for n_gaussian in range(len(list_wxups)):
    # Covariances
    covariance_up, logical_up = construct_covariance_and_logical(n_gaussian,
                                                                 list_wxups,
                                                                 list_wyups)
    list_covariances_up.append(covariance_up)

    (covariance_down,
     logical_down) = construct_covariance_and_logical(n_gaussian,
                                                      list_wxdowns,
                                                      list_wydowns)
    list_covariances_down.append(covariance_down)

    # Logical - eliminates the negative values to compute final covariance
    covariance = (np.multiply(covariance_up, logical_up) +
                  np.multiply(covariance_down, logical_down))
    covariance[0][0] = covariance[0][0]/(logical_up[0][0] + logical_down[0][0])
    covariance[1][1] = covariance[1][1]/(logical_up[1][1] + logical_down[1][1])

    # Treat exception: both covariances up and down are negative
    if logical_up[0][0] + logical_down[0][0] == 0:
      covariance[0][0] = -1 * max(covariance_up[0][0], covariance_down[0][0])

    if logical_up[1][1] + logical_down[1][1] == 0:
      covariance[1][1] = -1 * max(covariance_up[1][1], covariance_down[1][1])

    list_covariances.append(covariance)

  return list_covariances_up, list_covariances_down, list_covariances

def refine_centers(image,
                   center_coordinates,
                   step_trial_x,
                   step_trial_y,
                   warning_boundary = True):
  """
  Using the previous center_coordinates provided update center looking
  for the maximum around a trial space centered in the center_coordinate
  ---
  : params
  :: step_trial_x: the number of points in x to analyse (center_x -/+ step)
  :: step_trial_y: the number of points in y to analyse (center_y -/+ step)
  """
  new_center_coordinates = []
  for i, center in enumerate(center_coordinates):
    center = center.astype(int)
    # Get region
    max_region = image[center[0] - step_trial_y :
                          center[0] + step_trial_y + 1,
                          center[1] - step_trial_x :
                          center[1] + step_trial_x + 1]

    indices_max = np.unravel_index(np.argmax(max_region),
                                   max_region.shape)
    desloc_center = (indices_max[0] - step_trial_y,
                     indices_max[1] - step_trial_x)
    new_center = center + desloc_center
    new_center_coordinates.append(new_center)

    if warning_boundary:
      if ((desloc_center[0] == step_trial_y) |
          (desloc_center[1] == step_trial_x)):
        print(f'[Warning] Center {i}: Max in the boundary of region, might be outside.')
  return new_center_coordinates

def remove_outliers_using_iqr(data):
    """
    Remove outliers using the IQR method.

    Parameters:
    - data: A list or numpy array of data points

    Returns:
    - A numpy array with outliers removed
    """

    # Calculate Q1, Q2 (median) and Q3
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)

    # Compute the IQR
    IQR = Q3 - Q1

    # Define bounds for the outliers
    lower_bound = Q1 - 1 * IQR
    upper_bound = Q3 + 1 * IQR

    # Filter out the outliers
    return [(i >= lower_bound) & (i <= upper_bound) for i in data]

def covariances_index_replacemenvalue(**kwargs):
  # Indexes to analyse and replace
  replace_index = [idx for idx in
                   np.multiply(np.array([not idx for idx in
                                         kwargs['diffs_onmargin']]),
                   range(len(kwargs['new_covariances_up']))) if idx != 0]
  # Select the minimum convariances if the difference is an oulier
  up_values = np.array(kwargs['new_covariances_up'])[replace_index]
  down_values = np.array(kwargs['new_covariances_down'])[replace_index]
  replace_result = [min(val for val in
                                [i[kwargs['direction']][kwargs['direction']],
                                j[kwargs['direction']][kwargs['direction']]] if
                                val >= 0)
                    for (i, j) in zip(up_values, down_values) if
                        any(val >= 0 for
                             val in [i[kwargs['direction']][kwargs['direction']],
                                     j[kwargs['direction']][kwargs['direction']]])
                   ]
  return replace_index, replace_result

def replace_covariances_outlier(new_covariances,
                                new_covariances_up,
                                new_covariances_down):
  """
  Replaces the value of covariance when UP and DOWN differs too much
  This replacement chooses the covariance with lowest value
  NOTE: IF Gaussians are rotated, algorithm needs to be analysed
  """
  # Difference of covariance up and down
  diffs_cov_x = [i[0][0] - j[0][0] for (i, j) in zip(new_covariances_up,
                                                    new_covariances_down)]
  diffs_cov_y = [i[1][1] - j[1][1] for (i, j) in zip(new_covariances_up,
                                                     new_covariances_down)]
  # Compute outliers (IQR method)
  diffs_x_onmargin = remove_outliers_using_iqr(diffs_cov_x)
  diffs_y_onmargin = remove_outliers_using_iqr(diffs_cov_y)

  # Return indexes and values to replace
  (replace_x_index,
   replace_x_result) = covariances_index_replacemenvalue(new_covariances_up = new_covariances_up,
                                                         new_covariances_down = new_covariances_down,
                                                         diffs_onmargin = diffs_x_onmargin,
                                                         direction = 0)
  # Return indexes and values to replace
  (replace_y_index,
   replace_y_result) = covariances_index_replacemenvalue(new_covariances_up = new_covariances_up,
                                                         new_covariances_down = new_covariances_down,
                                                         diffs_onmargin = diffs_y_onmargin,
                                                         direction = 1)

  for idx, replacement_val in zip(replace_x_index, replace_x_result):
    new_covariances[idx][0][0] = replacement_val

  for idx, replacement_val in zip(replace_y_index, replace_y_result):
    new_covariances[idx][1][1] = replacement_val

  return new_covariances

def ensemble_gaussians_estimation(all_gaussians_modelone,
                                  all_gaussians_modeltwo,
                                  error_gaussians_modelone,
                                  error_gaussians_modeltwo):
  """
  Combine model estimations so final result might be improved. Comparions is
  performed two models each time and is based in the rmse pixel error
  for each gaussian
  ---
  : params
  :: error_gaussians_models : The errors for the estimated Gaussians - make sure
  the errors are ordered in a correct way so the Gaussian at position 0, for example
  is the same estimated Gaussian in the two vectors
  ---
  : returns
  :: all_gaussians_ensembled: Final combinated estimation for best model for
  each estimated Gaussian
  """
  # Compare error of model two and check if it is lower than model one
  compare_errors = [x < y for x, y in zip(error_gaussians_modeltwo,
                                          error_gaussians_modelone)]
  # Id better gaussians in modeltwo
  id_gaussians_bettertwo = [idx for idx, condition in enumerate(compare_errors)
                           if condition]

  # Final ensembled result
  def replace_values_at_positions(original_list, replace_list, positions_to_replace):
    for position in positions_to_replace:
        original_list[position] = replace_list[position]
    return original_list

  # Replace better gaussians for the second in the gaussians for the first one
  all_gaussians_ensembled = replace_values_at_positions(all_gaussians_modelone.copy(),
                                                        all_gaussians_modeltwo,
                                                        id_gaussians_bettertwo)

  return all_gaussians_ensembled, id_gaussians_bettertwo
