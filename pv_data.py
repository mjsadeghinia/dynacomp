# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

from pathlib import Path
import pymatreader
from structlog import get_logger

logger = get_logger()


# %%
def load_pv_data(pv_data_dir, p_channel=1, v_channel=2, recording_num=2):
    # Check if directory exist
    if not pv_data_dir.is_dir():
        logger.error("the folder does not exist")

    # Ensure there is exactly one .mat file
    mat_files = list(pv_data_dir.glob("*.mat"))
    if len(mat_files) != 1:
        logger.error("Data folder must contain exactly 1 .mat file.")

    mat_file = mat_files[0]
    logger.info(f"{mat_file.name} is loading.")

    data = pymatreader.read_mat(mat_file)
    pressures_fname = "data__chan_" + str(p_channel) + "_rec_" + str(recording_num)
    pressures = data[pressures_fname]
    volumes_fname = "data__chan_" + str(v_channel) + "_rec_" + str(recording_num)
    volumes = data[volumes_fname]

    return pressures, volumes


def divide_pv_data(pres, vols, t_interval=175):
    # Dividing the data into different curves
    pres_divided = []
    vols_divided = []
    peaks, _ = find_peaks(pres, distance=150)

    num_cycles = int(len(peaks))
    for i in range(num_cycles - 1):
        pres_divided.append(pres[peaks[i] : peaks[i + 1]])
        vols_divided.append(vols[peaks[i] : peaks[i + 1]])

    return pres_divided, vols_divided


def slice_divided_data(pres_divided, vols_divided, offset=50):
    # If there was any overlapping we slice the data
    # we do so by finding the closest point to the initial data point, except the first few points specified by offset
    vols_divided_sliced = []
    pres_divided_sliced = []
    for n, (vols, pres) in enumerate(zip(vols_divided, pres_divided)):
        pres = pres / np.max(pres)
        vols = vols / np.max(vols)
        dpres = pres - pres[0]
        dvols = vols - vols[0]
        dist = np.sqrt((dpres[offset:]) ** 2 + (dvols[offset:]) ** 2)
        ind = np.where(dist == np.min(dist))[0][0]
        ind += offset
        vols_divided_sliced.append(vols_divided[n][:ind])
        pres_divided_sliced.append(pres_divided[n][:ind])

    return pres_divided_sliced, vols_divided_sliced


def average_array(arrays):
    # Determine the length of the longest array
    max_length = max(len(array) for array in arrays)

    # Define common x values
    average_x = np.linspace(0, max_length - 1, num=100)

    # Interpolate each array to the common x-axis
    interpolated_arrays = []
    for array in arrays:
        x = np.linspace(0, len(array) - 1, num=len(array))
        f = interp1d(x, array, kind="linear", fill_value="extrapolate")
        interpolated_y = f(average_x)
        interpolated_arrays.append(interpolated_y)

    # Convert list of arrays to a 2D NumPy array for averaging
    interpolated_arrays = np.array(interpolated_arrays)

    # Calculate the average along the common x-axis
    average_y = np.mean(interpolated_arrays, axis=0)
    return average_y


# %%
directory_path = Path("00_data/AS/3week/156_1/")
pv_data_dir = directory_path / "PV data"
t_interval = 175
pres, vols = load_pv_data(pv_data_dir)
pres_divided, vols_divided = divide_pv_data(pres, vols, t_interval=t_interval)
p_average = average_array(pres_divided)
v_average = average_array(vols_divided)
p_average_sliced, v_average_sliced = slice_divided_data(
    [p_average], [v_average], offset=50
)
# closing the average data
# v_average_sliced = np.append(v_average_sliced[0], v_average_sliced[0][0])
# p_average_sliced = np.append(p_average_sliced[0], p_average_sliced[0][0])
v_average_sliced = v_average_sliced[0]
p_average_sliced = p_average_sliced[0]
# %%
average_x = np.linspace(0, t_interval, len(p_average))
fig, ax = plt.subplots(figsize=(8, 6))
for i in range(len(pres_divided)):
    ax.plot(pres_divided[i], "k", linewidth=0.01)
ax.plot(average_x, p_average)
plt.show

fig, ax = plt.subplots(figsize=(8, 6))
for i in range(len(vols_divided)):
    ax.plot(vols_divided[i], "k", linewidth=0.01)
ax.plot(average_x, v_average)
plt.show

average_x = np.linspace(0, t_interval, len(p_average_sliced))
fig, ax = plt.subplots(figsize=(8, 6))
for i in range(len(vols_divided)):
    ax.plot(vols_divided[i], pres_divided[i], "k", linewidth=0.01)
ax.plot(v_average_sliced, p_average_sliced)
plt.show

# %%
PV_data = np.vstack((p_average_sliced, v_average_sliced))
ind_ED = np.where(PV_data[1] == np.max(PV_data[1]))[0][0]
PV_data = np.hstack((PV_data[:, ind_ED:], PV_data[:, :ind_ED]))
# %%
average_x = np.linspace(0, t_interval, PV_data.shape[1])
fig, ax = plt.subplots(figsize=(8, 6))
for i in range(len(vols_divided)):
    ax.plot(vols_divided[i], pres_divided[i], "k", linewidth=0.01)
ax.scatter(PV_data[1, :], PV_data[0, :])
ax.scatter(PV_data[1, 0], PV_data[0, 0])
plt.show


# %%
from scipy.interpolate import splprep, splev

vols = PV_data[1, :]
pres = PV_data[0, :]
tck, _ = splprep([vols, pres], s=.1, per=True)
# Evaluate the B-spline
unew = np.linspace(0, 1.0, 200)
data = splev(unew, tck)
fig, ax = plt.subplots(figsize=(8, 6))
for i in range(len(vols_divided)):
    ax.plot(vols_divided[i], pres_divided[i], "k", linewidth=0.01)
ax.plot(vols, pres, label='Smoothed B-Spline')
PV_data_sampled = np.vstack((data[1], data[0]))
ax.scatter(PV_data_sampled[1, :], PV_data_sampled[0, :])

# %%
# %%
fname = pv_data_dir / "PV_data.csv"
np.savetxt(fname, PV_data_sampled, delimiter=",")

#%%

import numpy as np
from scipy.interpolate import splprep, splev
from scipy.integrate import quad

vols = PV_data[1, :]
pres = PV_data[0, :]

# Create the spline parameterization
tck, _ = splprep([vols, pres], s=0.1, per=True)

# Define the number of points to sample
n = 20

# Function to calculate the derivative of the spline
def derivative(u, tck):
    dxdt, dydt = splev(u, tck, der=1)
    return np.sqrt(dxdt**2 + dydt**2)

# Calculate the arc length from 0 to 1 using numerical integration
arc_length, _ = quad(derivative, 0, 1, args=(tck,))

# Generate equally spaced arc length values
target_arc_lengths = np.linspace(0, arc_length, n)

# Inverse function to find the parameter value for a given arc length
def find_u_for_arc_length(target_length, tck):
    def objective(u):
        length, _ = quad(derivative, 0, u, args=(tck,))
        return length - target_length
    from scipy.optimize import brentq
    return brentq(objective, 0, 1)

# Find the parameter values that correspond to the equally spaced arc lengths
u_values = [find_u_for_arc_length(target_length, tck) for target_length in target_arc_lengths]
# Evaluate the spline at these parameter values
sampled_points = splev(u_values, tck)

# Extract the sampled volumes and pressures
sampled_vols = sampled_points[0]
sampled_pres = sampled_points[1]

fig, ax = plt.subplots(figsize=(8, 6))
for i in range(len(vols_divided)):
    ax.plot(vols_divided[i], pres_divided[i], "k", linewidth=0.01)
ax.plot(vols, pres, label='Smoothed B-Spline')
ax.scatter(sampled_vols, sampled_pres)

# %%
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.integrate import quad
import matplotlib.pyplot as plt

vols = PV_data[1, :]
pres = PV_data[0, :]

# Create the spline parameterization
tck, _ = splprep([vols, pres], s=0.1, per=True)

# Define the number of points to sample
n = 20

# Function to calculate the derivative of the spline
def derivative(u, tck):
    dxdt, dydt = splev(u, tck, der=1)
    return np.sqrt(dxdt**2 + dydt**2)

# Calculate the arc length from 0 to 1 using numerical integration
arc_length, _ = quad(derivative, 0, 1, args=(tck,))

# Generate equally spaced arc length values
target_arc_lengths = np.linspace(0, arc_length, n)

# Inverse function to find the parameter value for a given arc length
def find_u_for_arc_length(target_length, tck):
    def objective(u):
        length, _ = quad(derivative, 0, u, args=(tck,))
        return length - target_length
    from scipy.optimize import brentq
    return brentq(objective, 0, 1)

# Find the parameter values that correspond to the equally spaced arc lengths
u_values = [find_u_for_arc_length(target_length, tck) for target_length in target_arc_lengths]

# Evaluate the spline at these parameter values
sampled_points = splev(u_values, tck)

# Extract the sampled volumes and pressures
sampled_vols = sampled_points[0]
sampled_pres = sampled_points[1]

# Plot the original data, smoothed B-spline, and sampled points
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the original data points
# ax.plot(vols, pres, 'ro', label='Original Data')

# Plot the smoothed B-spline
spline_points = splev(np.linspace(0, 1, 1000), tck)
ax.plot(spline_points[0], spline_points[1], label='Smoothed B-Spline')

# Scatter plot of the equally spaced sampled points
ax.scatter(sampled_vols, sampled_pres, color='blue', label='Equally Spaced Points')

ax.legend()
ax.set_xlabel('Volumes')
ax.set_ylabel('Pressures')
ax.set_title('Equally Spaced Points on Smoothed B-Spline')
plt.show()

# %%
