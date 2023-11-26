import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import mean_squared_error

from helpers.gaussian_estimation import compute_mse, sum_gaussians
from helpers.yolo_labelling import out_of_image


def eval_identification(
    n_image, means, xs, ys, estimation_method, image_ranges=[(0, 0.143), (0, 0.377)]
):
    """
    : params
    :: means: tuple (x, y) centers of the gaussians
    :: xs, ys: x and y centers used to generate the gaussians
    :: estimation_method: Method used to estimate the gaussians parameters
    """

    # Count gaussians
    number_estimated = len(means)

    # For now identifies only the Gaussians with center out of image
    indices_gaussians_out = []
    for i in range(len(xs)):
        if out_of_image([xs[i], ys[i]], image_ranges):
            indices_gaussians_out.append(i)

    number_gaussians_identifiable = len(xs) - len(indices_gaussians_out)

    return {
        "Image": [n_image],
        f"Number identified - {estimation_method}": [number_estimated],
        f"Number identifiable - {estimation_method}": [number_gaussians_identifiable],
        f"Percentage - {estimation_method}": [
            round(100 * number_estimated / number_gaussians_identifiable, 2)
        ],
    }


def calculate_center_errors(
    predicted_centers,
    true_centers,
    match_row_ind,
    match_col_ind,
    return_percentage=False,
):

    matched_predicted_centers = predicted_centers[match_row_ind]
    matched_true_centers = true_centers[match_col_ind]

    error_x = rmse_computation(
        matched_true_centers[:, 0],
        matched_predicted_centers[:, 0],
        return_percentage=return_percentage,
    )

    error_y = rmse_computation(
        matched_true_centers[:, 1],
        matched_predicted_centers[:, 1],
        return_percentage=return_percentage,
    )

    total_error = np.sqrt(error_x**2 + error_y**2)

    return error_x, error_y, total_error


def eval_centers(
    n_image,
    predicted_centers,
    true_centers,
    estimation_method,
    match_row_ind,
    match_col_ind,
    return_percentage=False,
):
    """
    Estimates the prediction of the center x, y error
    : params
    :: means: tuple (x, y) - estimated centers of the gaussians
    :: xs, ys: x and y centers used to generate the gaussians
    :: estimation_method: Method used to estimate the gaussians parameters
    """

    # Error in x, y and total
    (error_x, error_y, total_error) = calculate_center_errors(
        predicted_centers, true_centers, match_row_ind, match_col_ind, return_percentage
    )

    return {
        "Image": [n_image],
        f"RMSE_cx - {estimation_method}": [error_x],
        f"RMSE_cy - {estimation_method}": [error_y],
        f"RMSE_c - {estimation_method}": [total_error],
    }


def calculate_dispersion_errors(
    match_row_ind,
    match_col_ind,
    predicted_covariance,
    true_covariance,
    return_percentage=False,
):
    """
    :params
    :: predicted_covariance - 2D numpy arrays
    """
    matched_predicted_covariance = predicted_covariance[match_row_ind]
    matched_true_covariance = true_covariance[match_col_ind]

    print(f"covariances {matched_predicted_covariance[0:1]}")
    print(f"covariances true {matched_true_covariance[0:1]}")

    error_x = rmse_computation(
        matched_true_covariance[:, 0, 0],
        matched_predicted_covariance[:, 0, 0],
        return_percentage=return_percentage,
    )

    error_y = rmse_computation(
        matched_true_covariance[:, 1, 1],
        matched_predicted_covariance[:, 1, 1],
        return_percentage=return_percentage,
    )

    total_error = np.sqrt(error_x**2 + error_y**2)

    # Max relative errors
    max_x_error = (
        np.max(
            np.abs(
                (
                    matched_true_covariance[:, 0, 0]
                    - matched_predicted_covariance[:, 0, 0]
                )
                / matched_true_covariance[:, 0, 0]
            )
        )
        * 100
    )

    max_y_error = (
        np.max(
            np.abs(
                (
                    matched_true_covariance[:, 1, 1]
                    - matched_predicted_covariance[:, 1, 1]
                )
                / matched_true_covariance[:, 1, 1]
            )
        )
        * 100
    )

    total_max_error = np.sqrt(max_x_error**2 + max_y_error**2)

    return error_x, error_y, total_error, max_x_error, max_y_error, total_max_error


def eval_dispersion(
    n_image,
    covariances,
    dispersion_x,
    dispersion_y,
    estimation_method,
    match_row_ind,
    match_col_ind,
    return_percentage=False,
):

    # Convert to the way the curves are fitted
    # Covariances estimated from GMM and the ones present in the mat file
    covariances_wxy = np.sqrt(np.abs(covariances))

    # Covariances used in the generation
    true_covariances = np.array(
        [np.diag(mat) for mat in list(zip(dispersion_x, dispersion_y))]
    )

    # Error estimation
    (
        error_x,
        error_y,
        total_error,
        max_error_x,
        max_error_y,
        total_max_error,
    ) = calculate_dispersion_errors(
        match_row_ind,
        match_col_ind,
        covariances_wxy,
        true_covariances,
        return_percentage=return_percentage,
    )

    return {
        "Image": [n_image],
        f"RMSE_sigmax - {estimation_method}": [error_x],
        f"RMSE_sigmay - {estimation_method}": [error_y],
        f"RMSE_sigma - {estimation_method}": [total_error],
        f"MaxE_sigmax - {estimation_method}": [max_error_x],
        f"MaxE_sigmay - {estimation_method}": [max_error_y],
        f"MaxE_sigma - {estimation_method}": [total_max_error],
    }


def eval_amplitude(
    n_image,
    amplitude,
    estimated_amplitude,
    estimation_method,
    match_row_ind,
    match_col_ind,
    return_percentage=False,
):
    # Error estimation
    error = rmse_computation(
        np.array(amplitude)[match_col_ind],
        np.array(estimated_amplitude)[match_row_ind],
        return_percentage=return_percentage,
    )
    return {"Image": [n_image], f"RMSE_amplitude - {estimation_method}": [error]}


def match_gaussians(predicted_centers, true_centers):
    """
    Matches the predict centers with the true ones via optmization
    Objective: Reduce the total cost_matrix (error)
    ---
    : returns
    : row_ind: the indentification for the predicted
    : col_ind: the indentification for the true one
    """
    cost_matrix = np.zeros((len(predicted_centers), len(true_centers)))
    for i, pc in enumerate(predicted_centers):
        for j, tc in enumerate(true_centers):
            cost_matrix[i, j] = np.sqrt((pc[0] - tc[0]) ** 2 + (pc[1] - tc[1]) ** 2)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind


def rmse_computation(true_values, predicted_values, return_percentage=False):
    """
    Returns the error evaluated
    : params
    :: true_values: Vector of true values
    :: predict_values: Vector of the predicted values
    :: return_percentage - Indicates if returns percentage or absolute
    """
    error_abs = np.sqrt(mean_squared_error(true_values, predicted_values))
    if return_percentage:
        # Calculate percentage error for each sample
        error_percent = (
            np.mean(np.abs((true_values - predicted_values) / true_values)) * 100
        )
        return error_percent
    else:
        return error_abs


def extract_region(x, y, image, center, covariance, return_cropped=True):
    """
    Returns the filtered region
    ---
    : returns
    :: cropped_region : 2D array with only the region
    :: sub_region: 2D array with original shape, nan values for outside region
    """
    # Get the region around the center
    x_min = center[0] - np.sqrt(covariance[0, 0])
    x_max = center[0] + np.sqrt(covariance[0, 0])
    y_min = center[1] - np.sqrt(covariance[1, 1])
    y_max = center[1] + np.sqrt(covariance[1, 1])

    # Creating mask for 2D arrays
    mask_x = np.logical_and(x >= x_min, x <= x_max)
    mask_y = np.logical_and(y >= y_min, y <= y_max)
    mask = np.logical_and(mask_x, mask_y)

    sub_region = np.where(mask, image, np.nan)

    if return_cropped:
        # Find the cropping box coordinates (non-nan values)
        rows = np.any(~np.isnan(sub_region), axis=1)
        cols = np.any(~np.isnan(sub_region), axis=0)
        rowmin, rowmax = np.where(rows)[0][[0, -1]]
        colmin, colmax = np.where(cols)[0][[0, -1]]

        # Extract the 2D sub-array
        cropped_region = sub_region[rowmin : rowmax + 1, colmin : colmax + 1]

        return sub_region, cropped_region, [(rowmin, rowmax), (colmin, colmax)]
    else:
        return sub_region


def rmsep_each_gaussian(generated_image, x, y, image, image_ranges, all_gaussians):
    """
    Computes the Root Mean Square Error by pixel between the generated image and
    the true one.
    ---
    : params
    :: generated_image: 2D array of intensities computed using the estimated
    parameters
    :: image: true image
    :: image_ranges: x and y ranges of the images
    :: all_gaussians: All Gaussians paramters organized in (means, covariances,
    amplitudes)
    ---
    : returns
    :: error_gaussians: vector of the RMSEP for each method
    """
    error_gaussians = []
    for n_gaussian in range(len(all_gaussians)):
        center = all_gaussians[n_gaussian][0]
        # Check if center is inside the image
        if (
            (center[0] < image_ranges[0][0])
            | (center[0] > image_ranges[0][1])
            | (center[1] < image_ranges[1][0])
            | (center[1] > image_ranges[1][1])
        ):
            print(f"Gaussian {n_gaussian} center is out of bounds")
            # Append nan just to keep location of Gaussians
            error_gaussians.append(np.nan)
        else:
            covariance = all_gaussians[n_gaussian][1]
            # Extract region - reconstructed with true one
            sub_region, cropped_region, _ = extract_region(
                x, y, generated_image, center, covariance
            )
            # Extract region - true one
            sub_region_truth, cropped_region_truth, _ = extract_region(
                x, y, image, center, covariance
            )
            # Append the error of the estimated Gaussian (RMSEP)
            error_gaussians.append(
                np.sqrt(compute_mse(cropped_region, cropped_region_truth))
            )
    return error_gaussians


def compute_error_per_gaussian(
    image, means, covariances, amplitudes, image_ranges=[(0, 0.143), (0, 0.377)]
):
    """
    Computes the pixel error per estimated Gaussian comparing to the true image
    : params
    :: image: (x, y, intensities) vector
    :: means: list of (x, y) esimated centers
    :: covariances: list of 2 x 2 estimated covariances (should have sigma value)
    as the sum_gaussians generations considers the sigma in the gaussian formula
    :: amplitudes: list of estimated amplitudes
    : returns
    :: all_gaussians: Array with all estimated parameters
    :: generated_z: Image generated with the estimated parameters
    :: rmsep: rmean rmse of the pixel error
    :: vec_rmsep: vector of the rmse of each gassian error
    """
    # Unifies in a unique vector
    all_gaussians = list(zip(means, list(covariances), amplitudes))

    # Generates gaussian
    generated_z = sum_gaussians(image[0], image[1], all_gaussians)  # x  # y

    # Error
    vec_rmsep = rmsep_each_gaussian(
        generated_z,
        image[0],  # x
        image[1],  # y
        image[2],
        image_ranges=image_ranges,
        all_gaussians=all_gaussians,
    )
    # Mean of all gaussians
    rmsep = np.nanmean(vec_rmsep)
    return all_gaussians, generated_z, rmsep, vec_rmsep


def eval_error_per_gaussian(
    n_image,
    estimation_method,
    image,
    means,
    covariances,
    amplitudes,
    image_ranges=[(0, 0.143), (0, 0.377)],
):
    (all_gaussians, generated_z, rmsep, vec_rmsep) = compute_error_per_gaussian(
        image, means, covariances, amplitudes, image_ranges=image_ranges
    )
    return {"Image": [n_image], f"RMSEPixel - {estimation_method}": [rmsep]}
