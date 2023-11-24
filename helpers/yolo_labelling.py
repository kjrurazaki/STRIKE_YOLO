import numpy as np
from scipy import interpolate
from skimage.draw import polygon

from pathlib import Path
from skimage import measure

from helpers.gaussian_generator import generate_distribution

def create_gaussian_mask(gaussian_params,
                         img_ranges,
                         img_size,
                         n_contour_points = None,
                         octagon_pol = False):
    """
    Generates a binary mask and normalized contours (segmentation mask)
    for a single 2D Gaussian specified.

    The mask threshold is set for Half of the Amplitude
    Adapted from https://towardsdatascience.com/trian-yolov8-instance-segmentation-on-your-data-6ffa04b2debd

    ---
    :: Params
    : n_contour_points: sets the maximum number of points in the contour.
    If None, the points are not uniform for the gaussians and are the direct
    output from find_contours
    : octagon_pol: sets the polygon of the contour as a octagon. If None the
    measure.find_contours is used
    """
    # Gaussian parameters
    x_centroid, y_centroid, sigma_x, sigma_y, amplitude, theta = gaussian_params

    x, y, z = generate_distribution(img_ranges[0], 
                                    img_ranges[1], 
                                    img_size[0], 
                                    img_size[1], 
                                    [gaussian_params])

    # Threshold at half maximum to create binary mask
    mask = z > (amplitude / 2)

    if octagon_pol == False:
      # Find contours of the mask
      contours = measure.find_contours(mask, 0.8)

      # Normalize the contours
      norm_contours = [[p[1] / img_size[0], p[0] / img_size[1]] for contour in contours for p in contour]

      if n_contour_points is not None:
        norm_contours = []
        for contour in contours:
          # Ensure that the contour is a closed loop
          if np.any(contour[0] != contour[-1]):
              contour = np.vstack([contour, contour[0]])

          # Parametric representation of the contour
          t = np.arange(contour.shape[0])

          # Interpolate x(t) and y(t) separately
          fx = interpolate.interp1d(t, contour[:, 1], kind='cubic')
          fy = interpolate.interp1d(t, contour[:, 0], kind='cubic')

          # Generate the new parameter values
          tt = np.linspace(0, t.max(), n_contour_points)

          # Generate the new contour points
          resampled_contour = np.column_stack([fx(tt), fy(tt)])

          # Normalize the resampled contour points and add to the list
          norm_contours.append([[p[1] / img_size[1], p[0] / img_size[0]] for p in resampled_contour])

    elif octagon_pol == True:
      # Octagon construction
      radius_x = 2 * np.sqrt(2 * np.ln(2)) * sigma_x # FWHM
      radius_y = 2 * np.sqrt(2 * np.ln(2)) * sigma_y # FWHM

      # Generate octagon points
      theta = np.linspace(0, 2 * np.pi, 
                          n_contour_points, 
                          endpoint = False)  # angles for each point
      x_points = radius_x * np.cos(theta) + x_centroid
      y_points = radius_y * np.sin(theta) + y_centroid

      # Ensure points are within image boundary
      x_points = np.clip(x_points, 0, img_size[1]-1)
      y_points = np.clip(y_points, 0, img_size[0]-1)

      # Get pixels inside polygon (mask)
      rr, cc = polygon(y_points, x_points)
      mask = np.zeros((img_size[0], img_size[1]), dtype=bool)
      mask[rr, cc] = 1

      # Normalize points
      norm_points = [[p[0] / img_size[1], p[1] / img_size[0]] for p in zip(x_points, y_points)]

    return mask, norm_contours, x, y, z

def create_labels(gaussian_params_list,
                  img_ranges,
                  img_size,
                  label_dir,
                  label_name,
                  remove_out_of_image = True,
                  task = "segment",
                  n_contour_points = None,
                  octagon_pol = False):
  """
  Creates label files for a list of Gaussians, where each label file contains
  the bounding box and segmentation mask for one Gaussian.
  Bounding box and mask values are normalized by the image size to be in the
  range [0, 1].
  The bounding box is specified in YOLO format (class, x_center, y_center,
  width, height), and the segmentation mask is specified as a list of polygon
  points.

  :params
  ::remove_out_of_image: Returns the labels only for gaussian with the center
  contained inside the image

  :adapted from
  https://towardsdatascience.com/trian-yolov8-instance-segmentation-on-your-data-6ffa04b2debd
  """
  label_dir = Path(label_dir)
  label_dir.mkdir(parents = True, exist_ok = True)

  # labels
  labels = ''
  indices_out_of_image = []
  for i, gaussian_params in enumerate(gaussian_params_list):
    mask, norm_contours, *_ = create_gaussian_mask(gaussian_params,
                                                  img_ranges,
                                                  img_size,
                                                  n_contour_points = None,
                                                  octagon_pol = False)

    # Mark Gaussians that has its center out of image
    if out_of_image(gaussian_params, img_ranges):
      indices_out_of_image.append(i)

    # Bounding box (class, x_center, y_center, width, height)
    x_center, y_center = (gaussian_params[0] / img_ranges[0][-1],
                          gaussian_params[1] / img_ranges[1][-1])
    # Theoretically the gaussian is almost fully contained in a box with 3 times sigma
    width, height = (3 * gaussian_params[2] / img_ranges[0][-1],
                      3 * gaussian_params[3] / img_ranges[1][-1])

    if task == "segment":
      # Label line - Class + Bounding box center, width and height + contour of figure
      label_line = ('0 ' + f"{x_center:.5f}".rstrip('0') + ' ' +
                    f"{y_center:.5f}" + ' ' +
                    f"{width:.5f}".rstrip('0') + ' ' +
                    f"{height:.5f}".rstrip('0') + ' ' +
                    ' '.join([f"{f'{cord[0]:.4f}'.rstrip('0')} {f'{cord[1]:.4f}'.rstrip('0')}" for cord in norm_contours]))

    elif task == "detect":
      # Label line - Class + Bounding box center, width and height
      label_line = ('0 ' + f"{x_center:.5f}".rstrip('0') + ' ' +
                    f"{y_center:.5f}" + ' ' +
                    f"{width:.5f}".rstrip('0') + ' ' +
                    f"{height:.5f}".rstrip('0'))

    # each labelled object should be in a line in the text
    if (remove_out_of_image == True) & out_of_image(gaussian_params, img_ranges):
      continue
    else:
      labels = labels + '\n' + label_line

  # Save label
  label_path = label_dir / f'{label_name}.txt'
  with label_path.open('w') as f:
    f.write(labels[1:])

  return labels[1:], indices_out_of_image

def out_of_image(gaussian_params, img_ranges):
  """
  Checks if the center of the Gaussian is out of the image
  """
  return ((gaussian_params[0] <= img_ranges[0][0]) |
        (gaussian_params[0] >= img_ranges[0][1]) |
        (gaussian_params[1] <= img_ranges[1][0]) |
        (gaussian_params[1] >= img_ranges[1][1]))
