# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev

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

def smooth_data(pres_divided_sliced,vols_divided_sliced, smooth_level = 1):
    pres_smooth = []
    vols_smooth = []
    for vols, pres in zip(vols_divided_sliced, pres_divided_sliced):
        vols = np.append(vols, vols[0])
        pres = np.append(pres, pres[0])
        tck, _ = splprep([vols, pres], s=smooth_level, per=True)
        # Evaluate the B-spline
        unew = np.linspace(0, 1.0, 1000)
        data = splev(unew, tck)
        vols_smooth.append(data[0])
        pres_smooth.append(data[1])
    return pres_smooth, vols_smooth

def plot_data(pres, vols):
    for i in range(len(pres)):
        plt.figure()
        plt.plot(vols[i],pres[i])
# %%P
pres, vols = load_pv_data(pv_data_dir)
pres_divided, vols_divided = divide_pv_data(pres, vols)
pres_divided_sliced, vols_divided_sliced = slice_divided_data(pres_divided, vols_divided)
res_smooth, vols_smooth = smooth_data(pres_divided_sliced,vols_divided_sliced)



#%%
k = 29
plt.figure()
plt.plot(vols_divided_sliced[k],pres_divided_sliced[k])
plt.plot(vols_divided[k],pres_divided[k])

n=29
pres = pres_divided[n]
vols = vols_divided[n]
#%%
pres = pres[:1000]
plt.plot(pres)
plt.plot(peaks, pres[peaks], "x")
plt.plot(np.zeros_like(pres), "--", color="gray")
plt.show()



pres_divided, vols_divided = divide_pv_data(pres, vols)
pres_divided_sliced, vols_divided_sliced = slice_divided_data(pres_divided, vols_divided)

k =29
plt.plot(vols_divided[k],pres_divided[k])
k= 0
plt.plot(vols_divided_sliced[k],pres_divided_sliced[k])
#%%



for _ in range(len(pres_divided)):
    plt.figure()
    plt.title(str(_))
    plt.plot(vols_divided[_],pres_divided[_])


fig, ax = plt.subplots(figsize=(8, 6))

n=0
for vols, pres in zip(vols_divided, pres_divided):
    n+=1
    vols = np.append(vols, vols[0])
    pres = np.append(pres, pres[0])
    tck, _ = splprep([vols, pres], s=1, per=True)
    # Evaluate the B-spline
    unew = np.linspace(0, 1.0, 1000)
    data = splev(unew, tck)
    ax.plot(data[0], data[1], label='Smoothed B-Spline')
    
ax.set_xlabel('Volume')
ax.set_ylabel('Pressure')
ax.legend()
plt.title('Pressure-Volume Curves with B-Spline Fitting')
plt.show()


#%%
# # pres_divided = []
# vols_divided = []
# n = 0
# ind = 0
# for _ in range(100):
#     offset = 50
#     initial_slice_length = 170
#     dpres = pres[n:] - pres[n]
#     dvols = vols[n:] - vols[n]
#     ind_i = n + offset
#     ind_f = n + offset + initial_slice_length
#     if ind_f > len(pres):
#         logger.info
#         break
#     dist = np.sqrt((dpres[ind_i:ind_f]) ** 2 + (dvols[ind_i:ind_f]) ** 2)

#     ind = np.where(dist == np.min(dist))[0][0]
#     ind += offset
#     ind += n
#     pres_divided.append(pres[n:ind])
#     vols_divided.append(vols[n:ind])
#     n = ind
#     plt.figure()
#     plt.plot(vols_divided[_], pres_divided[_])
#     plt.scatter(vols[ind], pres[ind])
    
# %%

    
# %%
