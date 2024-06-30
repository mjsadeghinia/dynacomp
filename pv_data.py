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
directory_path = Path("00_data/AS/3week/156_1/")
pv_data_dir = directory_path / "PV data"


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

def divide_pv_data(pres,vols, t_interval = 175):
    # Dividing the data into different curves
    pres_divided = []
    vols_divided = []
    peaks, _ = find_peaks(pres,distance=150)

    num_cycles = int(len(peaks))
    for i in range(num_cycles-1):
        pres_divided.append(pres[peaks[i]:peaks[i+1]])
        vols_divided.append(vols[peaks[i]:peaks[i+1]])
    
    return pres_divided, vols_divided

def slice_divided_data(pres_divided, vols_divided, offset = 50):
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
        f = interp1d(x, array, kind='linear', fill_value="extrapolate")
        interpolated_y = f(average_x)
        interpolated_arrays.append(interpolated_y)

    # Convert list of arrays to a 2D NumPy array for averaging
    interpolated_arrays = np.array(interpolated_arrays)

    # Calculate the average along the common x-axis
    average_y = np.mean(interpolated_arrays, axis=0)
    return average_y, average_x

# %%
pres, vols = load_pv_data(pv_data_dir)
pres_divided, vols_divided = divide_pv_data(pres, vols)

#%%

# Find common x-axis range
p_average, average_x = average_array(pres_divided)
fig, ax = plt.subplots(figsize=(8, 6))
for i in range(len(pres_divided)):
    ax.plot(pres_divided[i],'k', linewidth = 0.01)
ax.plot(average_x,p_average)
plt.show

v_average, average_x = average_array(vols_divided)
fig, ax = plt.subplots(figsize=(8, 6))
for i in range(len(vols_divided)):
    ax.plot(vols_divided[i],'k', linewidth = 0.01)
ax.plot(average_x,v_average)
plt.show

fig, ax = plt.subplots(figsize=(8, 6))
for i in range(len(vols_divided)):
    ax.plot(vols_divided[i],pres_divided[i],'k', linewidth = 0.01)
ax.plot(v_average,p_average)
plt.show

#%%