import numpy as np
from .auxiliary_functions import load_label, load_image
from .gaussian_generator import generate_grid

from matplotlib import pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_grayscale(x, y, z, ax=None, colorbar=True):
    """
    2D plot in grayscale
    """
    assert ax != None, "Please pass the axis object."
    im = ax.imshow(
        z, origin="lower", cmap="gray_r", extent=(x.min(), x.max(), y.min(), y.max())
    )

    if colorbar == True:
        # Create divider for the existing axes instance
        divider = make_axes_locatable(ax)
        # Append axes to the right of ax, with 5% width of ax
        cax = divider.append_axes("right", size="5%", pad=0.05)
        # Color bar in the appended axis
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label("Z value")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")


def plot_withcenters(x, y, z, centers, axs):
    """
    2D plot in grayscale with estimated centers
    """
    assert len(axs) > 1, "Please pass the two axis objects."
    plot_grayscale(x, y, z, axs[0])
    plot_grayscale(x, y, z, axs[1])
    axs[1].scatter(np.array(centers[0]), np.array(centers[1]), s=5)


def plot_data_sav(line_z, smoothed, first_derivative, second_derivative, fig_size):
    # Calculate inflection points (zeros of second derivative)
    inflection_points = np.where(np.diff(np.sign(second_derivative)))[0]

    fig, axs = plt.subplots(3, figsize=fig_size)

    # Original scatter points with curve fitted
    axs[0].scatter(
        range(len(line_z)), line_z, label="Original data", color="blue", s=10
    )
    axs[0].plot(smoothed, label="Smoothed data", color="red")
    axs[0].plot(
        inflection_points, smoothed[inflection_points], "go", label="Inflection points"
    )
    axs[0].legend()
    axs[0].set_title("Original and Smoothed Data")

    # First derivative
    axs[1].plot(first_derivative, label="First derivative", color="green")
    axs[1].legend()
    axs[1].set_title("First Derivative")

    # Second derivative
    axs[2].plot(second_derivative, label="Second derivative", color="purple")
    axs[2].legend()
    axs[2].set_title("Second Derivative")

    plt.tight_layout()
    plt.show()


def gaussians_labelled(
    image_path,
    label_path,
    ax,
    task="segment",
    colorbar=True,
    edge_color="r",
    plot_centers=False,
    grid_width=0.143,
    grid_length=0.377,
    img_width=143,
    img_length=377,
):
    """
    Plot the gaussians and their labels
      - Bounding box
      - Contour
    ---
    :params
    :: image_path: Path of the image location (png)
    example: f"~/Tesi/Data/Images/gaussian_{n_image}.png"
    :: label_path: Path of the labels of the image (txt)
    example: f"~/Tesi/Data/Labelsv0/gaussian_{n_image}.txt"
    :: ax: ax object from plot
    """
    # Generate GRID
    x, y = generate_grid((0, grid_width), (0, grid_length), img_width, img_length)

    labels = load_label(label_path)
    z = load_image(image_path)

    # Plot ground truth - labels
    plot_grayscale(x, y, z, ax=ax, colorbar=colorbar)

    # Extract structure from the labels
    labels = labels.split("\n")
    centers = []
    for i in range(len(labels)):
        if task == "segment":
            countours = labels[i].split(" ")[5:]
            countours = [
                [float(x), float(y)] for x, y in zip(countours[::2], countours[1::2])
            ]
            countours = np.array(countours)
            ax.plot(
                countours[:, 0] * grid_width, countours[:, 1] * grid_length, linewidth=1
            )

        x_center = float(labels[i].split(" ")[1])
        y_center = float(labels[i].split(" ")[2])
        centers.append((x_center, y_center))

        width = float(labels[i].split(" ")[3])
        height = float(labels[i].split(" ")[4])

        # Calculate bounding box coordinates
        x_min = (x_center - width / 2) * grid_width
        y_min = (y_center - height / 2) * grid_length
        box_width = width * grid_width
        box_height = height * grid_length

        # Plot the bounding box
        rect = patches.Rectangle(
            (x_min, y_min),
            box_width,
            box_height,
            linewidth=1,
            edgecolor=edge_color,
            facecolor="none",
        )
        ax.add_patch(rect)
    # Plot the center points using scatter plot
    if plot_centers == True:
        ax.scatter(
            [c[0] * grid_width for c in centers],
            [c[1] * grid_length for c in centers],
            s=1,
        )

    print(f"Number of indentifiable Gaussians (center in the image): {len(labels)}")
    return x, y, z


def plot_boxes(tensor, ax, edge_color="r"):
    """
    Plots bounding boxes from the runs of YOLO detection
    """
    for box in tensor:
        # Create a Rectangle patch
        rect = patches.Rectangle(
            (box[0], box[1]),
            (box[2] - box[0]),
            (box[3] - box[1]),
            linewidth=1,
            edgecolor=edge_color,
            facecolor="none",
        )

        # Add the patch to the Axes
        ax.add_patch(rect)


def plot_boxes_gmm(means, covariances, ax, beta=1, edge_color="r"):
    """
    Plots bounding boxes from the runs of YOLO detection
    : params
    :: proportionality_constant: parameter related to the relation of real sigma
    and the value returned in the GMM fitting
    """
    for center, box in list(
        zip(
            means,
            [
                (sigmax, sigmay)
                for sigmax, sigmay in zip(
                    np.array(covariances)[:, 0, 0], np.array(covariances)[:, 1, 1]
                )
            ],
        )
    ):
        # Create a Rectangle patch
        length = np.sqrt(np.abs(box[1])) * 1000 * beta
        width = np.sqrt(np.abs(box[0])) * 1000 * beta
        rect = patches.Rectangle(
            (center[0] * 1000 - width / 2, center[1] * 1000 - length / 2),
            width,
            length,
            linewidth=1,
            edgecolor=edge_color,
            facecolor="none",
        )

        # Add the patch to the Axes
        ax.add_patch(rect)


def plot_box_labels(
    image,
    means,
    covariances,
    ax,
    box_edge_color="r",
    colorbar=False,
    center_color="b",
    annotate_gaussians=False,
    text_color="black",
):

    """
    Plot true labels of gaussians with the boxes and centers
    Notice the boxes sizes are the FWHM multiplied by sqrt(2)
    ---
    : params
    :: image : The original image as [x, y, intensities]
    :: means, covariances: parameters of the Gaussians
    :: annotate_gaussians: if true plots the number of the gaussian
    (indice) in the image
    """
    plot_grayscale(image[0], image[1], image[2], colorbar=colorbar, ax=ax)

    for box in [(mean, covariance) for mean, covariance in zip(means, covariances)]:
        box_width = box[1][0] * np.sqrt(2)
        box_length = box[1][1] * np.sqrt(2)
        # Create a Rectangle patch
        rect = patches.Rectangle(
            (box[0][0] - box_width / 2, box[0][1] - box_length / 2),
            box_width,
            box_length,
            linewidth=1,
            edgecolor=box_edge_color,
            facecolor="none",
        )

        # Add the patch to the Axes
        ax.add_patch(rect)

    ax.scatter(np.array(means)[:, 0], np.array(means)[:, 1], s=3, c=center_color)

    if annotate_gaussians:
        texts = []
        for i, (x, y) in enumerate(means):
            texts.append(
                ax.text(x, y, str(i), ha="center", va="bottom", color=text_color)
            )

        return texts


def plot_3d_region(
    image,
    x_center,
    y_center,
    width,
    length,
    show_surface=True,
    colorscale="Viridis",
    show_points=False,
    annotate=True,
):
    """
    Plot an interactive 3D graph around the given x and y centers.

    : params
    :: image: tuple containing (x_values, y_values, z_values)
    :: x_center, y_center: The center around which you want to plot.
    :: width, length: The width and length of the region you want to plot
    around the center.
    :: colorscale: Color map to use for the surface plot. If set to None,
    uses a uniform color.
    :: show_points: Boolean flag to show points on the surface plot.
    :: annotate: Boolean flag to annotate points with their indices.
    """
    import plotly.graph_objects as go

    x_values, y_values, z_values = image

    # Get the region around the center
    x_min, x_max = x_center - width / 2, x_center + width / 2
    y_min, y_max = y_center - length / 2, y_center + length / 2

    # Creating mask for 2D arrays
    mask_x = np.logical_and(x_values > x_min, x_values < x_max)
    mask_y = np.logical_and(y_values > y_min, y_values < y_max)
    mask = np.logical_and(mask_x, mask_y)

    z_region = np.where(mask, z_values, np.nan)  # Use NaN for values outside the region

    # Create the 3D plot
    fig = go.Figure()

    if show_surface:
        if colorscale is None:
            fig.add_trace(
                go.Surface(
                    z=z_region,
                    x=x_values,
                    y=y_values,
                    colorscale=[(0, "white"), (1, "white")],
                    showscale=False,
                )
            )
        else:
            fig.add_trace(
                go.Surface(z=z_region, x=x_values, y=y_values, colorscale=colorscale)
            )

    if show_points:
        # Create annotations
        hovertexts = None
        if annotate:
            hovertexts = [
                f"({i}, {j})"
                for i, row in enumerate(mask)
                for j, val in enumerate(row)
                if val
            ]

        fig.add_trace(
            go.Scatter3d(
                x=x_values[mask],
                y=y_values[mask],
                z=z_region[mask],
                mode="markers",
                marker=dict(size=2, color="red"),
                hovertext=hovertexts,
            )
        )

    fig.update_layout(
        title="3D Plot",
        autosize=False,
        width=800,
        height=800,
        margin=dict(t=65, b=40, l=60, r=60),
    )
    return fig


def plot_2d_region(
    image,
    x_center,
    y_center,
    width,
    length,
    colorscale="Viridis",
    show_points=False,
    annotate=False,
    extra_points=None,
):
    """
    Plot an interactive 2D heatmap around the given x and y centers.

    Parameters:
    - image: tuple containing (x_values, y_values, z_values)
    - x_center, y_center: The center around which you want to plot.
    - width, length: The width and length of the region you want to plot around the center.
    - colorscale: Color map to use for the heatmap.
    - show_points: Boolean flag to show points on the heatmap.
    - annotate: Boolean flag to annotate points with their indices on hover.
    """
    import plotly.graph_objects as go

    x_values, y_values, z_values = image

    # Get the region around the center
    x_min, x_max = x_center - width / 2, x_center + width / 2
    y_min, y_max = y_center - length / 2, y_center + length / 2

    # Creating mask for 2D arrays
    mask_x = np.logical_and(x_values > x_min, x_values < x_max)
    mask_y = np.logical_and(y_values > y_min, y_values < y_max)
    mask = np.logical_and(mask_x, mask_y)

    z_region = np.where(mask, z_values, np.nan)  # Use NaN for values outside the region

    # Create the 2D heatmap
    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            x=x_values[0, :], y=y_values[:, 0], z=z_region, colorscale=colorscale
        )
    )

    if show_points:
        hovertexts = None
        if annotate:
            hovertexts = [
                f"({i}, {j})"
                for i, row in enumerate(mask)
                for j, val in enumerate(row)
                if val
            ]

        fig.add_trace(
            go.Scatter(
                x=x_values[mask],
                y=y_values[mask],
                mode="markers",
                marker=dict(size=2, color="red"),
                hovertext=hovertexts,
                hoverinfo="text",
            )
        )

    if extra_points is not None:
        for point in extra_points:
            fig.add_trace(
                go.Scatter(
                    x=[point["x"]],
                    y=[point["y"]],
                    mode="markers",
                    marker=dict(size=5, color=point["color"]),
                )
            )
    fig.update_layout(
        title="2D Heatmap",
        autosize=False,
        width=800,
        height=800,
        margin=dict(t=65, b=40, l=60, r=60),
        xaxis=dict(range=[x_min - width / 8, x_max + width / 8]),
        yaxis=dict(range=[y_min - length / 8, y_max + length / 8]),
    )
    return fig
