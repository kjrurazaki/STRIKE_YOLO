import numpy as np
import os
import random
import shutil
import cv2

from matplotlib import pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

from helpers.image_extractors import find_centers
from helpers.common_utils import timer_decorator
from helpers.auxiliary_functions import (
    combine_centers,
    parameters_extract,
    load_image_and_params,
)
from helpers.aux_plotter import plot_withcenters, plot_grayscale
from helpers.gaussian_estimation import (
    estimate_gaussians_centers,
    estimate_parameters_gmm,
    estimate_gaussians_amplitudes,
    estimate_parameters_PX,
    replace_covariances_outlier,
    refine_centers,
    ensemble_gaussians_estimation,
)
from helpers.yolo_labelling import create_labels
from helpers.error_estimation import (
    compute_error_per_gaussian,
    eval_identification,
    eval_centers,
    eval_dispersion,
    eval_amplitude,
    eval_error_per_gaussian,
    match_gaussians,
)


@timer_decorator
def ppl_lines_squares(
    image, params, plot=True, fig_size=(10, 8), version="v0", return_one=False
):
    """
    This pipeline are the one tested to identify the Gaussian centers using lines
    Each line in x and y should pass through the Gaussian;
    The process needs to be repeated 80 times.
    The centers are found by the maximum amplitude in the cropped imaged found
    ---
    : params
    :: image: The 377 x 143 pixel image of the gaussians
    :: params: include all parameters needed in the functions
    :: return_one: to return just one center by analysed block or more
    ---
    : functions
    :: find_centers is from image_extractors modulus
    ---
    : versions
    v0: 01/07/2023
    v1: 04/07/2023 - ALL NEXT CONSIDERATIONS MIGHT BE SOLVED MY THE SCIPY LOCAL
                     ARGMAX (AND SUMUP THE AXIS BEFORE APPLYING in 1D)
    TODO v2: Different implementations (stopped to develop the other ppls):
      - Identify if image has two centers (when computing max - use Lagrangian).
        If it has it needs to correct the square
      - Laplacian applied to the extraction so if the gaussian was declining and
        started increasing, crops this point inwards (meaning where the current
        Gaussians is mixing with other Gaussian?)
      - Use a transformation for the image (for example, laplacian or gradient)
    """
    # Getting the centers
    x, y, z = image[0], image[1], image[2]
    if version == "v0":
        x_centers, y_centers = find_centers(x, y, z, params, version="v0")
    if version == "v1":
        # Returns multiple centers for each image
        if return_one == False:
            x_centers, y_centers = find_centers(
                x, y, z, params, version="v1", return_one=False
            )
            x_centers, y_centers = combine_centers(x_centers, y_centers)
        else:
            x_centers, y_centers = find_centers(
                x, y, z, params, version="v1", return_one=True
            )
    if plot == True:
        fig, axs = plt.subplots(1, 2, figsize=fig_size)
        plot_withcenters(x, y, z, [x_centers, y_centers], axs)
    else:
        return x_centers, y_centers


@timer_decorator
def ppl_fit_gaussians(
    image,
    means_init=None,
    peak_method="laplace",
    em_iterations=100,
    covariance_gaussians_type="diag",
    n_gaussians=None,
    image_ranges=[(0, 0.143), (0, 0.377)],
    image_size=[143, 377],
    fig_size=(10, 8),
    plot=True,
    plot_points=False,
    beta_apriori=1,
):
    """
    Just fit the Mixture of Gaussians (GMM) to find means, covariance and amplitude.
    The need to correct the sigma by betas comes from the fact that the gaussians
    are built filtering the points from a image, which bias towards overestimating or
    underestimating the standard deviation
    ---
    : params
    :: x, y - Points that the gaussians are fitted (it doesn't consider z values)
    :: center_coordinates - Initial estimated centers for the Gaussians
    :: covariance type - Use "diag" when gaussians are not rotated
    ---
    : returns
    :: covariances - Covariances as the squared FWHM corrected by beta
                     (found in training)
    ---
    Notes : The x, y points were select in the project as the values of laplacian
    greater or equal than 0, restricting the coordinates to the points belonging to
    gaussians
    """
    # Initialize estimated centers
    if means_init == "laplace":
        nonzero_pairs, center_coordinates = estimate_gaussians_centers(
            image, peak_method="laplace"
        )
        cxs = image[0][0, [center_coordinates[:, 1]]].reshape(-1, 1)
        cys = image[1][center_coordinates[:, 0], 0].reshape(-1, 1)
        means_init = np.column_stack((cxs, cys))
        amplitudes_apriori = image[2][
            center_coordinates[:, 0], center_coordinates[:, 1]
        ]

    if means_init is not None:
        n_gaussians = len(means_init)

    # Plot points that are fitted
    if plot_points == True:
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.scatter(nonzero_pairs[:, 0], nonzero_pairs[:, 1], s=1)
        aspect_ratio = image_size[1] / image_size[0]
        ax.set_aspect(aspect_ratio)

    # Estimate means and covariances fitting GMM model
    (gmm, means, covariances) = estimate_parameters_gmm(
        nonzero_pairs[:, 0],
        nonzero_pairs[:, 1],
        n_gaussians=n_gaussians,
        means_init=means_init,
        em_iterations=em_iterations,
        covariance_gaussians_type=covariance_gaussians_type,
    )
    if covariance_gaussians_type == "diag":
        diagonal_matrices = [np.diag(cov) for cov in covariances]
        covariances = diagonal_matrices

    # Convert sigmas to wxys to be comparable (sigma times sqrt(2))
    covariances = np.array(covariances) * (np.sqrt(2)) ** 2 * beta_apriori
    angles = []
    ellipses = []
    for mean, covariance in zip(means, covariances):
        # Generate points on an ellipse around the mean
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        angles.append(angle)
        width, height = 2 * np.sqrt(eigenvalues)  # Eigenvalues is FWHM
        ellipses.append(
            patches.Ellipse(
                xy=mean,
                width=width,
                height=height,
                angle=angle,
                edgecolor="r",
                facecolor="none",
            )
        )

    if plot == True:
        # 2D plot in grayscale - selected image for initial development
        fig, ax = plt.subplots(figsize=fig_size)
        plot_grayscale(image[0], image[1], image[2], ax)
        ax.scatter(means[:, 0], means[:, 1], s=5)
        for ellipse in ellipses:
            ax.add_patch(ellipse)

    # Estimation of amplitude
    amplitudes = estimate_gaussians_amplitudes(
        means, image, image_ranges=image_ranges, image_size=image_size
    )

    return means, covariances, angles, amplitudes


@timer_decorator
def ppl_2D_xy_profile_v0(image, params, plot=True):
    """
    Not developed. scikit-image focus
    """
    return None


def ppl_label_YOLOv0(
    images,
    images_features,
    label_dir,
    remove_out_of_image=True,
    task="segment",
    n_contour_points=None,
    octagon_pol=False,
    image_ranges=[(0, 0.143), (0, 0.377)],
    image_size=[143, 377],
):
    """
    Pipeline to create the labels of the Gaussians from the CNR data
    Labels are using the FWMH divided by np.sqrt(2)
    ---
    : params
    :: images: The array of images with shape (377, 143, number of images)
    :: images_features: Features that were used to generate the images with shape
    (number of gaussians per image * number of parameters), number of images)
    ---
    : versions
    :: v0: Excludes Gaussians that the center are out of the image
    """
    label_dir = Path(label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)

    # Image
    for n_image in range(images.shape[-1]):
        # Parameters extraction
        (amplitude, xs, ys, dispersion_x, dispersion_y) = parameters_extract(
            images_features, n_image
        )

        # Organize the tuples
        params = list(
            zip(
                xs,
                ys,
                dispersion_x / np.sqrt(2),
                dispersion_y / np.sqrt(2),
                amplitude,
                len(amplitude) * [0],
            )
        )

        labels, indices_out_of_image = create_labels(
            params,
            image_ranges,
            image_size,
            label_dir,
            f"gaussian_{n_image}",
            remove_out_of_image,
            task,
            n_contour_points,
            octagon_pol,
        )


def ppl_distribute_images_yolo(
    images_folder,
    labels_folder,
    yolo_folder,
    range_images,
    proportions,
    task="segment",
):
    """
    Distributes images into 'train', 'val', and 'test' folders based on given
    proportions.
    This step is needed as the structure required to train the YOLO model
    ----
    : params:
    ::image_folder: Path to the folder containing all the images.
    ::yolo_folder: Path to the folder that the model will use
    ::proportions: A dictionary specifying the proportion of images for each
    category. Example: {'train': 0.7, 'val': 0.2, 'test': 0.1}
    """
    # Create 'train', 'val', and 'test' directories if they don't exist
    for folder in ["train", "val", "test"]:
        folder_path = os.path.join(yolo_folder, folder)
        if not os.path.exists(folder_path):
            os.makedirs(os.path.join(folder_path, "images"))
            os.makedirs(os.path.join(folder_path, "labels"))

    # Calculate the number of images for each category based on proportions
    num_images = len(range_images)
    num_train = int(num_images * proportions["train"])
    num_val = int(num_images * proportions["val"])
    num_test = num_images - num_train - num_val

    # Shuffle the indices of images to distribute randomly
    random.shuffle(range_images)

    # Distribute images into 'train', 'val', and 'test' folders
    for i, n_image in enumerate(range_images):
        if i < num_train:
            destination_folder = "train"
        elif i < num_train + num_val:
            destination_folder = "val"
        else:
            destination_folder = "test"

        # Copy image
        source_path = os.path.join(images_folder, f"gaussian_{n_image}.png")
        destination_path = os.path.join(
            yolo_folder, destination_folder, "images", f"gaussian_{n_image}.png"
        )
        shutil.copy(source_path, destination_path)

        # Copy label
        source_path = os.path.join(labels_folder, f"gaussian_{n_image}.txt")
        destination_path = os.path.join(
            yolo_folder, destination_folder, "labels", f"gaussian_{n_image}.txt"
        )
        shutil.copy(source_path, destination_path)


def ppl_yolo_predict_gaussians(
    mat_data,
    n_image,
    model,
    image_path,
    image_ranges=[(0, 0.143), (0, 0.377)],
    image_size=[143, 377],
):
    """
    Function using detection task of Yolo.
    Predict bounding boxes of image in "image_path" using "model"
    ---
    : params
    :: model: YOLO model object
    :: image_path: Image location ("~path/image.png")
    :: image_ranges: Ranges of image ([x range, y range])
    :: image_size : Size of image ([x_shape, y_shape])
    ---
    : returns
    :: covariances: As the squared FWHM
    """
    # Make the prediction YOLO
    img = cv2.imread(image_path)
    results = model.predict(img)

    # class 0 boxes
    tensor_boxes = results[0].boxes.xywhn
    # Centers
    means = [
        (float(x * image_ranges[0][-1]), float(y * image_ranges[1][-1]))
        for x, y in zip(tensor_boxes[:, 0], tensor_boxes[:, 1])
    ]

    # Covariances
    covariances = [
        (float(wx * image_ranges[0][-1] / 3), float(wy * image_ranges[1][-1] / 3))
        for wx, wy in zip(tensor_boxes[:, 2], tensor_boxes[:, 3])
    ]
    covariances = np.array([np.diag(mat) for mat in covariances])

    # Back to FWHM - labelling divdes it by np.sqrt(2)
    covariances = covariances * np.sqrt(2)

    # "standard deviation" to "variance"
    covariances = np.square(covariances)

    # Amplitude - needs the intensity from the generated images
    (image, amplitude, xs, ys, dispersion_x, dispersion_y) = load_image_and_params(
        n_image,
        data_images=mat_data["F_data"],
        data_parameters=mat_data["Fit_flux"],
        image_ranges=image_ranges,
        image_size=image_size,
    )
    # Defining using the same strategy from GMM
    amplitudes = estimate_gaussians_amplitudes(
        np.array(means), image, image_ranges=image_ranges, image_size=image_size
    )
    return means, covariances, amplitudes


def ppl_yolo_px_predict_gaussians(
    mat_data,
    n_image,
    model,
    image_path,
    image_ranges=[(0, 0.143), (0, 0.377)],
    image_size=[143, 377],
    px_step_size=3,
    infer_overlapping=False,
):
    """
    Yolo model prediction of the Gaussians with posterior refinement of the centers
    and estimation of the standard deviation using the probability density distribution
    formula.
    : params
    :: infer_overlapping: If this is true, the computed covariances for the points
    up and down are compared to check if there is some kind of overlapping. By
    simplification, the covariance with lower value is used.
    """
    # Load image and params
    (image, amplitude, xs, ys, dispersion_x, dispersion_y) = load_image_and_params(
        n_image,
        data_images=mat_data["F_data"],
        data_parameters=mat_data["Fit_flux"],
        image_ranges=image_ranges,
        image_size=image_size,
    )

    # Prediction with Yolo
    (means, covariances, amplitudes_yolo) = ppl_yolo_predict_gaussians(
        mat_data,
        n_image,
        model,
        image_path,
        image_ranges=image_ranges,
        image_size=image_size,
    )

    # GRID coordinates
    x, y = image[0], image[1]

    # center coordinates in pixels
    y_pixel = np.array(means)[:, 1] / image_ranges[1][1] * image_size[1]
    x_pixel = np.array(means)[:, 0] / image_ranges[0][1] * image_size[0]
    center_coordinates = np.array(list(zip(np.round(y_pixel), np.round(x_pixel))))

    # Compute new means refining the center (Center = Maximum amplitude)
    new_means = refine_centers(
        image[2],
        center_coordinates,
        step_trial_x=1,
        step_trial_y=1,
        warning_boundary=False,
    )

    # Convert to reference coordinates and organized in the ideal format of x, y
    # Notice that the indices ranges from 0 to image_size[1] - 1
    converted_new_means = new_means.copy()
    for n in range(len(new_means)):
        converted_new_means[n] = converted_new_means[n].astype(float)
        converted_new_means[n][1] = (
            new_means[n][0] / (image_size[1] - 1) * image_ranges[1][-1]
        )
        converted_new_means[n][0] = (
            new_means[n][1] / (image_size[0] - 1) * image_ranges[0][-1]
        )

    # Convert to tuple to be equal the other models
    converted_new_means = [tuple(arr) for arr in converted_new_means]

    # Compute covariances
    (
        new_covariances_up,
        new_covariances_down,
        new_covariances,
    ) = estimate_parameters_PX(x, y, image[2], new_means, step_size=px_step_size)
    # To array
    new_covariances = np.array(new_covariances)

    # Infer overlapping
    if infer_overlapping == True:
        new_covariances = replace_covariances_outlier(
            new_covariances, new_covariances_up, new_covariances_down
        )

    # Compute new estimated amplitudes
    amplitudes_px = estimate_gaussians_amplitudes(
        new_means,
        image,
        image_ranges=image_ranges,
        image_size=image_size,
        center_in_pixel=True,
    )
    return (
        converted_new_means,
        new_covariances,
        amplitudes_px,
        new_means,
        new_covariances_up,
        new_covariances_down,
    )


def ppl_ensemble_predict_gaussians(
    n_image,
    mat_data,
    estimation_method,
    peak_method=None,
    image_path=None,
    yolo_model=None,
    covariance_gaussians_types="diag",
    return_percentage=False,
    image_ranges=[(0, 0.143), (0, 0.377)],
    image_size=[143, 377],
    print_bettersecond=False,
):
    """
    Make ensembled prediction using the list of estimation methods.
    All prediction methods should return the same number of Gaussians
    For now, can only compare and compute using YOLO and YOLO_px.
    For now, can only ensemble 2 methods.
    """
    assert isinstance(estimation_method, list) & (
        len(estimation_method) > 1
    ), "Unique method."
    assert "gmm" not in estimation_method, "gmm is still not supported."
    assert len(estimation_method) <= 2, "More than 2 methods not supported."
    # Load image and params
    (image, amplitude, xs, ys, dispersion_x, dispersion_y) = load_image_and_params(
        n_image,
        data_images=mat_data["F_data"],
        data_parameters=mat_data["Fit_flux"],
        image_ranges=image_ranges,
        image_size=image_size,
    )
    list_all_gaussians = []
    list_all_errors_gaussians = []
    # Predict ensemble
    for estimation in estimation_method:
        (
            means,
            covariances,
            angles,
            estimated_amplitude,
        ) = ppl_select_estimation_method(
            image=image,
            n_image=n_image,
            mat_data=mat_data,
            estimation_method=estimation,
            peak_method=peak_method,
            image_path=image_path,
            yolo_model=yolo_model,
            covariance_gaussians_types=covariance_gaussians_types,
            return_percentage=return_percentage,
            image_ranges=image_ranges,
            image_size=image_size,
        )
        # Unifies in a unique vector - covariances/2 (methods returns sigma/sqrt(2))
        (
            all_gaussians,
            generated_z,
            rmsep,
            error_gaussians,
        ) = compute_error_per_gaussian(
            image,
            means,
            list(np.array(covariances / 2)),
            estimated_amplitude,
            image_ranges=image_ranges,
        )
        # Keep all gaussians
        list_all_gaussians.append(all_gaussians)
        list_all_errors_gaussians.append(error_gaussians)

    # Ensemble first and second method
    (all_gaussians_ensembled, id_gaussians_bettertwo) = ensemble_gaussians_estimation(
        list_all_gaussians[0],
        list_all_gaussians[1],
        list_all_errors_gaussians[0],
        list_all_errors_gaussians[1],
    )

    if print_bettersecond:
        print(id_gaussians_bettertwo)
    # Converting back "covariances" to (sigma * sqrt(2))**2
    all_gaussians_ensembled = [
        (item[0], item[1] * 2, item[2]) for item in all_gaussians_ensembled
    ]
    return all_gaussians_ensembled


def ppl_select_estimation_method(**kwargs):
    """
    Selects estimation method
    """
    if kwargs["estimation_method"] == "gmm":
        # Predict using GMM - EM
        (means, covariances, angles, estimated_amplitude) = ppl_fit_gaussians(
            kwargs["image"],
            means_init="laplace",
            peak_method="laplace",
            em_iterations=1000,
            covariance_gaussians_type=kwargs["covariance_gaussians_types"],
            n_gaussians=None,
            image_ranges=kwargs["image_ranges"],
            image_size=kwargs["image_size"],
            fig_size=(10, 8),
            plot=False,
            plot_points=False,
        )
        return means, covariances, angles, estimated_amplitude

    elif kwargs["estimation_method"] == "YOLO":
        # Check if path is already total
        if kwargs["image_path"].split(".")[-1] == "png":
            image_path = kwargs["image_path"]
        else:
            image_path = f"{kwargs['image_path']}gaussian_{kwargs['n_image']}.png"
        # Predict using YOLO
        (means, covariances, estimated_amplitude) = ppl_yolo_predict_gaussians(
            kwargs["mat_data"],
            kwargs["n_image"],
            kwargs["yolo_model"],
            image_path,
            kwargs["image_ranges"],
            kwargs["image_size"],
        )
        return means, covariances, np.nan, estimated_amplitude

    elif kwargs["estimation_method"] == "YOLO_px":
        # Check if path is already total
        if kwargs["image_path"].split(".")[-1] == "png":
            image_path = kwargs["image_path"]
        else:
            image_path = f"{kwargs['image_path']}gaussian_{kwargs['n_image']}.png"
        # Predict using YOLO refined by center max and standard deviation px
        (means, covariances, estimated_amplitude, *_) = ppl_yolo_px_predict_gaussians(
            kwargs["mat_data"],
            kwargs["n_image"],
            kwargs["yolo_model"],
            image_path,
            kwargs["image_ranges"],
            kwargs["image_size"],
        )

        return means, covariances, np.nan, estimated_amplitude

    elif kwargs["estimation_method"] == "YOLO_px_infer":
        # Check if path is already total
        if kwargs["image_path"].split(".")[-1] == "png":
            image_path = kwargs["image_path"]
        else:
            image_path = f"{kwargs['image_path']}gaussian_{kwargs['n_image']}.png"
        # Predict using YOLO refined by center max and standard deviation px
        (means, covariances, estimated_amplitude, *_) = ppl_yolo_px_predict_gaussians(
            kwargs["mat_data"],
            kwargs["n_image"],
            kwargs["yolo_model"],
            image_path,
            kwargs["image_ranges"],
            kwargs["image_size"],
            infer_overlapping=True,
        )

        return means, covariances, np.nan, estimated_amplitude


def ppl_error_estimation(
    n_image,
    mat_data,
    estimation_method,
    peak_method=None,
    image_path=None,
    yolo_model=None,
    covariance_gaussians_types="diag",
    return_percentage=False,
    image_ranges=[(0, 0.143), (0, 0.377)],
    image_size=[143, 377],
):
    """
    Pipeline to compute errors of the methods. Computes:
    - Number of gaussian identified / in the image
    - Error in the centers estimation
    - Error in the dispersion estimation
    - Erro in the estimated amplitudes
    - Mean Pixel/intensity error for the gaussians
    ---
    : params
    :: n_image - Image identification

    :: peak_method - When using GMM method it identifies how the centers
    estimations are found
    :: estimation_method - Method to estimate the paramters of the Gaussians.
    Available: GMM and YOLO
    :: mat_data: Contains 1) data_images - The 3D array of images (mat['F_data']).
    Format: (shape_x, shape_y, number of images); 2) data_parameters - Parameter used
    to generate the gaussians (mat['Fit_flux']) Format: (number of parameters,
    number of gaussians * number of images)
    :: image_path: the path for the gaussians. Required when prediction
    using YOLO
    :: yolo_model: Pass YOLO model object to do the prediction. Required when
    using YOLO
    :: covariance_gaussians_types: types of the covariances of the gaussians
    default: 'diag' = each component has its own diagonal covariance matrix
    :: return_percentage: Return the relative errors (percentage)
    """
    # Load image and params
    (image, amplitude, xs, ys, dispersion_x, dispersion_y) = load_image_and_params(
        n_image,
        data_images=mat_data["F_data"],
        data_parameters=mat_data["Fit_flux"],
        image_ranges=image_ranges,
        image_size=image_size,
    )

    # Estimate Gaussians
    if isinstance(estimation_method, list):
        # Call ppl ensembling
        all_gaussians_ensembled = ppl_ensemble_predict_gaussians(
            n_image,
            mat_data,
            estimation_method,
            peak_method=peak_method,
            image_path=image_path,
            yolo_model=yolo_model,
            covariance_gaussians_types=covariance_gaussians_types,
            return_percentage=return_percentage,
            image_ranges=image_ranges,
            image_size=image_size,
        )
        # Split in each prediction
        means = [i[0] for i in all_gaussians_ensembled]
        covariances = [i[1] for i in all_gaussians_ensembled]
        estimated_amplitude = [i[2] for i in all_gaussians_ensembled]
    else:
        (
            means,
            covariances,
            angles,
            estimated_amplitude,
        ) = ppl_select_estimation_method(
            image=image,
            n_image=n_image,
            mat_data=mat_data,
            estimation_method=estimation_method,
            peak_method=peak_method,
            image_path=image_path,
            yolo_model=yolo_model,
            covariance_gaussians_types=covariance_gaussians_types,
            return_percentage=return_percentage,
            image_ranges=image_ranges,
            image_size=image_size,
        )

    # Predicted Centers
    predicted_centers = np.array([(i[0], i[1]) for i in means])

    # True Centers used in the generation of gaussians
    true_centers = np.array(list(zip(xs, ys)))

    # Matches Gaussians
    row_ind, col_ind = match_gaussians(predicted_centers, true_centers)

    if isinstance(estimation_method, list):
        estimation_method = "ensembled"  # changes the name to the eval dictionaries

    # Count number of gaussians identified
    results = eval_identification(
        n_image, means, xs, ys, estimation_method, image_ranges
    )

    # Center error estimation
    results.update(
        eval_centers(
            n_image,
            predicted_centers,
            true_centers,
            estimation_method,
            row_ind,
            col_ind,
            return_percentage,
        )
    )

    # Dispersion error estimation
    results.update(
        eval_dispersion(
            n_image,
            covariances,
            dispersion_x,
            dispersion_y,
            estimation_method,
            row_ind,
            col_ind,
            return_percentage,
        )
    )

    # Amplitude error estimation
    results.update(
        eval_amplitude(
            n_image,
            amplitude,
            estimated_amplitude,
            estimation_method,
            row_ind,
            col_ind,
            return_percentage,
        )
    )

    # RMSEP mean pixel error for estimated gaussians
    results.update(
        eval_error_per_gaussian(
            n_image,
            estimation_method,
            image,
            means,
            list(np.array(covariances) / 2),  # correct to sigma
            estimated_amplitude,
            image_ranges=image_ranges,
        )
    )
    return results
