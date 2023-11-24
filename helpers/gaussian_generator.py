import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from helpers.gaussian_estimation import generate_gaussian

# Function to generate the 2D grid
def generate_grid(x_range, y_range, num_subdivisions_x, num_subdivisions_y):
    x = np.linspace(x_range[0], x_range[1], num_subdivisions_x)
    y = np.linspace(y_range[0], y_range[1], num_subdivisions_y)
    return np.meshgrid(x, y)

# Function to generate a 2D Gaussian
def generate_gaussian_formula(x, y, x_centroid, y_centroid, sigma_x, sigma_y, amplitude):
    return amplitude * np.exp( - ( ((x - x_centroid)**2 / (2 * sigma_x**2)) + ((y - y_centroid)**2 / (2 * sigma_y**2)) ) )

def generate_rotated_gaussian(x, y, x_centroid, y_centroid, sigma_x, sigma_y, amplitude, theta):
    theta = np.deg2rad(theta)

    # Rotation matrix
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    # Shift coordinates by the centroid
    x_shifted = x - x_centroid
    y_shifted = y - y_centroid

    # Apply the rotation matrix to the shifted coordinates
    coordinates = np.stack([x_shifted, y_shifted], axis=-1)
    rotated_coordinates = coordinates @ rotation_matrix.T

    # Separate the rotated coordinates into x and y
    x_rotated, y_rotated = rotated_coordinates[..., 0], rotated_coordinates[..., 1]

    # Calculate the Gaussian
    return amplitude * np.exp( - ( ((x_rotated)**2 / (2 * sigma_x**2)) + ((y_rotated)**2 / (2 * sigma_y**2)) ) )

# Function to generate the summed distribution of many 2D Gaussians passed as "Params"
def generate_distribution(x_range,
                          y_range,
                          num_subdivisions_x,
                          num_subdivisions_y,
                          params,
                          rotation = True):
    x, y = generate_grid(x_range, y_range, num_subdivisions_x, num_subdivisions_y)
    z_sum = np.zeros(x.shape)

    if rotation:
      for p in params:
          z_sum += generate_rotated_gaussian(x, y, *p)
    else:
      for p in params:
          z_sum += generate_gaussian(x, y, *p)

    return x, y, z_sum

def TfromXX(nx, ny, XX):
    npar = 5
    nbeamlets = len(XX) // npar
    x = np.linspace(0.0, nx*10**-3, nx)
    y = np.linspace(0.0, ny*10**-3, ny)
    x, y = np.meshgrid(x, y)

    A = XX[::npar]
    x0 = XX[1::npar]
    y0 = XX[2::npar]
    wx = XX[3::npar]
    wy = XX[4::npar]

    h = np.zeros((ny, nx))
    for j in range(nbeamlets):
      h += A[j] * np.exp(-(((x - x0[j])/wx[j])**2 + ((y - y0[j])/wy[j])**2))

    return x, y, h

# Function to interpolate z values along a line
def interpolate_line(start, end, steps, x, y, z, method='linear'):
    # Create array of points along the line
    line_x = np.linspace(start[0], end[0], steps)
    line_y = np.linspace(start[1], end[1], steps)

    # Interpolate z values at these points
    grid_z = griddata((x.flatten(), y.flatten()), z.flatten(), (line_x, line_y), method=method)

    return line_x, line_y, grid_z

# Function to calculate the true z values along a line
def calculate_true_z(start, end, steps, params):
    # Create array of points along the line
    line_x = np.linspace(start[0], end[0], steps)
    line_y = np.linspace(start[1], end[1], steps)

    z_true = np.zeros(line_x.shape)
    for p in params:
        z_true += generate_rotated_gaussian(line_x, line_y, *p)

    return line_x, line_y, z_true

# Noise generator
def generate_noise(z, percentage):
    max_amplitude = np.max(z)
    return np.random.uniform(-max_amplitude * percentage, max_amplitude * percentage, size=z.shape)

# Add noise to z
def add_noise(z, percentage):
    return z + generate_noise(z, percentage)

# Function to plot the interpolated line
def plot_path(x, y, z, line_x, line_y, line_z):
    plt.figure()
    # Plotting the contours
    contours = plt.contour(x, y, z, levels=10)
    plt.clabel(contours, inline=True, fontsize=8)
    # Plotting the line
    plt.plot(line_x, line_y, 'r-')
    plt.colorbar()
    plt.show()

    # Plotting the path topology
    plt.figure()
    plt.plot(np.sqrt((line_x-line_x[0])**2 + (line_y-line_y[0])**2), line_z, 'r-')
    plt.xlabel('Distance')
    plt.ylabel('Z')
    plt.show()

# Function to compare interpolated and true z values
def compare_z_values(x, y, z, line_x, line_y, line_z, z_true):
    # Plotting the path topology
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(np.sqrt((line_x-line_x[0])**2 + (line_y-line_y[0])**2), line_z, 'r-', label='Interpolated')
    plt.xlabel('Distance')
    plt.ylabel('Z')
    plt.legend()

    plt.subplot(1, 2, 1)
    plt.plot(np.sqrt((line_x-line_x[0])**2 + (line_y-line_y[0])**2), z_true, 'b-', label='True')
    plt.xlabel('Distance')
    plt.ylabel('Z')
    plt.legend()

    plt.show()
