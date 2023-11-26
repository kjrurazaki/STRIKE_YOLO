import time
import numpy as np
from pathlib import Path

from datetime import datetime

from helpers.gaussian_generator import generate_grid

from PIL import Image
import cv2
from matplotlib import pyplot as plt


def parameters_extract(df, n_image):
    """
    Extract parameters of each image in the generation from the matlab file
    """
    # Select the image column
    image_column = df[:, n_image]

    # Extracting amplitude and next amplitude
    amplitude = image_column[0::5]
    next_amplitude = image_column[5::5]

    # Extracting xs, ys, dispersion_x, dispersion_y
    xs = image_column[1::5]
    ys = image_column[2::5]
    dispersion_x = image_column[3::5]
    dispersion_y = image_column[4::5]

    return amplitude, xs, ys, dispersion_x, dispersion_y


def combine_centers(centers_x, centers_y):
    # Ensure all data is iterable, removing None values
    centers_x = [[i] if isinstance(i, float) else i for i in centers_x if i is not None]
    centers_y = [[i] if isinstance(i, float) else i for i in centers_y if i is not None]

    # Pairs of centers
    pairs = [
        (i, j)
        for sublist1, sublist2 in zip(centers_x, centers_y)
        for i in sublist1
        for j in sublist2
    ]

    # return centers
    centers_x, centers_y = zip(*pairs)

    return list(centers_x), list(centers_y)


def find_closest_indices(shape, indices, axis):
    """
    Find closest indices to center of image
    ---
    : params
    :: shape: Shape of image
    :: indices to match
    :: axis (1 - x axis or 0 - y axis)
    """
    center_index = shape[axis] / 2
    closest_index = min(indices, key=lambda index: abs(index - center_index))
    return closest_index


def save_all_gaussians(mat, folder):
    """
    Save all gaussians while maintaning the pixel dimensions
    Extracted from: https://stackoverflow.com/questions/13714454/specifying-and-saving-a-figure-with-exact-size-in-pixels
    """
    for save_id in range(0, mat["F_data"].shape[-1]):
        plt.imsave(
            fname=f"{folder}/gaussian_{save_id}.png",
            arr=mat["F_data"][:, :, save_id],
            cmap="gray_r",
            format="png",
        )


def load_label(label_path):
    """
    Used in the labelling of Yolo model
    """
    with open(label_path, "r") as file:
        labels = file.read()
    return labels


def load_image(image_path):
    """
    Used in the labelling of Yolo model
    """
    with open(image_path, "r") as file:
        image = Image.open(image_path)
    z = np.array(image)
    return z


def load_image_and_params(
    n_image,
    data_images,
    data_parameters,
    image_ranges=[(0, 0.143), (0, 0.377)],
    image_size=[143, 377],
):
    """
    Function used in the error estimation pipeline
    ---
    : params
    :: n_image - Image identification
    :: data_images - The 3D array of images (mat['F_data']).
    format: (shape_x, shape_y, number of images)
    :: data_parameters - Parameter used to generate the gaussians
    (mat['Fit_flux'])
    format: (number of parameters, number of gaussians * number of images)
    """
    # Generate GRID
    x, y = generate_grid(image_ranges[0], image_ranges[1], image_size[0], image_size[1])

    # Image - z intensities
    image = [x, y, data_images[:, :, n_image]]

    # Parameters extraction
    amplitude, xs, ys, dispersion_x, dispersion_y = parameters_extract(
        data_parameters, n_image
    )

    return image, amplitude, xs, ys, dispersion_x, dispersion_y


def convert_pixel_intensity(mat, n_image, image_path):
    """
    Converts the image from the pixel greyscale (reverse) to the intensities
    Important when working with YOLo model
    ---
    : params
    :: n_image - the number of identification of the image (used to load the intensities)
    :: image_path - final path to the image
    """
    (image, amplitude, xs, ys, dispersion_x, dispersion_y) = load_image_and_params(
        n_image, data_images=mat["F_data"], data_parameters=mat["Fit_flux"]
    )

    img_png = cv2.imread(image_path)
    # To greyscale
    img_png = cv2.cvtColor(img_png, cv2.COLOR_BGR2GRAY)
    # AS THE IMAGES ARE INVERTED GREYSCALE
    img_png = 255 - img_png
    # Range of conversion
    min_intensity = image[-1].min()
    max_intensity = image[-1].max()
    # Normalizing to 0 - 1
    normalized_img = img_png / 255.0
    # Then scale and shift to original range
    img_png = (normalized_img * (max_intensity - min_intensity)) + min_intensity

    print(f"Max error in convertion: {np.max(np.abs(image[-1] - img_png))}")
    print(f"Max intensity: {np.max(np.abs(image[-1]))}")
    return img_png


def checkpoint_error(snapshot_folder_path, error_dataframe, annotation, name):
    """
    Save errors and annotate
    """
    path = Path(snapshot_folder_path)
    if not path.exists():
        # If it doesn't exist, create the folder
        path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    error_dataframe.to_csv(f"{snapshot_folder_path}results_{timestamp}_{name}.csv")
    text = f"""Annotation: {annotation}"""

    file_name = f"{snapshot_folder_path}annotations_{timestamp}_{name}.txt"
    with open(file_name, "w") as file:
        file.write(text)
