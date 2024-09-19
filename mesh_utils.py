# %%
import numpy as np
import cv2 as cv
import h5py

from pathlib import Path
import pymatreader
from structlog import get_logger

logger = get_logger()

def compile_h5(directory_path, scan_type, overwrite=False, is_inverted = False ):
    if scan_type == 'CINE':
        h5_file_address = compile_h5_CINE(directory_path, overwrite=overwrite, is_inverted=is_inverted)
    elif scan_type == 'TPM':
        h5_file_address = compile_h5_TPM(directory_path, overwrite=overwrite)
    else:
        logger.error(f'The settings for scan_type is invalid ({scan_type}), it should be either CINE or TPM')
    return h5_file_address

def compile_h5_CINE(directory_path, overwrite, is_inverted):
    directory_path = Path(directory_path)

    # Check if directory exist
    if not directory_path.is_dir():
        logger.error("the folder does not exist")

    # Check for existing .h5 files
    h5_files = list(directory_path.glob("*.h5"))
    if len(h5_files) > 1:
        logger.error("There are multiple h5 files!")
        return

    if h5_files and not overwrite:
        logger.warning("H5 file already exists. Set overwrite=True to overwrite it.")
        return h5_files[0].as_posix()

    # Ensure there is exactly one .mat file
    mat_files = list(directory_path.glob("*.mat"))
    if len(mat_files) != 1:
        logger.error("Data folder must contain exactly 1 .mat file.")
        return

    mat_file = mat_files[0]
    logger.info(f"{mat_file.name} is loading.")
    try:
        # Load .mat file
        data = pymatreader.read_mat(mat_file)
        data = dict(sorted(data["setstruct"].items()))

        num_acquisition = len(data["IM"])
        for i in range(num_acquisition):
            if len(data["IM"][i].shape) == 4:
                i_shax = i
                break
        
        x_coords_epi = data["EpiX"][i_shax]
        y_coords_epi = data["EpiY"][i_shax]
        x_coords_endo = data["EndoX"][i_shax]
        y_coords_endo = data["EndoY"][i_shax]
        coords_dataset = prepare_coords_dataset(x_coords_epi,y_coords_epi,x_coords_endo,y_coords_endo, is_inverted)

        # Prepare attributes
        K = len(coords_dataset["coords_epi"])               # Number of slices
        I = int(data["XSize"][i_shax])                      # Image matrix size
        time_res = data["TIncr"][i_shax]                    # temporal resolution
        slice_thickness = data["SliceThickness"][i_shax]    # slice thickness in mm
        resolution = data["ResolutionX"][i_shax]
        attrs = {
            "image_matrix_size": I,
            "number_of_slices": K,
            "temporal_resolution": time_res,
            "slice_thickness": slice_thickness,
            "resolution": resolution,
            }
        
        # exporting the dataset and attributes to the h5 file
        h5_file_address = mat_file.with_suffix(".h5").as_posix()
        save_to_h5(h5_file_address, coords_dataset, attrs)
        logger.info(f"{mat_file.with_suffix('.h5').name} is created.")
        return h5_file_address

    except Exception as e:
        logger.error(f"Failed processing due to {e}")

def compile_h5_TPM(directory_path, overwrite):
    """
    Compiles .mat files from OUS datasets into a structured .h5 file.

    Handles linear time interpolation between end systole and end diastole to correctly align stacks.

    Parameters:
        directory_path (str): Path to the directory containing the .mat file.
        overwrite (bool): Whether to overwrite an existing .h5 file.
    """
    directory_path = Path(directory_path)

    # Check if directory exist
    if not directory_path.is_dir():
        logger.error("the folder does not exist")

    # Check for existing .h5 files
    h5_files = list(directory_path.glob("*.h5"))
    if len(h5_files) > 1:
        logger.error("There are multiple h5 files!")
        return

    if h5_files and not overwrite:
        logger.warning("H5 file already exists. Set overwrite=True to overwrite it.")
        return h5_files[0].as_posix()

    # Ensure there is exactly one .mat file
    mat_files = list(directory_path.glob("*.mat"))
    if len(mat_files) != 1:
        logger.error("Data folder must contain exactly 1 .mat file.")
        return

    mat_file = mat_files[0]
    logger.info(f"{mat_file.name} is loading.")

    try:
        # Load .mat file
        data = pymatreader.read_mat(mat_file)
        data = dict(sorted(data["ComboData"].items()))

        # Sort from base to apex
        label = "pss0" if "pss0" in data else "PVM_SPackArrSliceOffset"
        slice_loc = [(data[label][k]) for k in range(len(data[label]))]
        has_positive = any(x > 0 for x in slice_loc)
        has_negative = any(x < 0 for x in slice_loc)
        sorted_indexes = np.argsort(
                [(data[label][k]) for k in range(len(data[label]))]
            )
        if has_negative and not has_positive:
            sorted_indexes = sorted_indexes[::-1]
        data = {key: [data[key][i] for i in sorted_indexes] for key in data.keys()}

        # Compute necessary arrays and matrices
        I = int(data["I"][0])  # Image matrix size
        K = len(data[label])  # Number of slices
        T_end = np.min([int(val) for val in data["TimePointEndAcquisition"]])
        TED = [int(val) for val in data["TimePointEndDiastole"]]
        TES = [int(val) for val in data["TimePointEndSystole"]]
        S = len(data["WallThickness"][0][0])  # Number of segments

        # Interpolate T_array
        T_array = interpolate_T_array(TES, TED, K)

        # Generate and populate datasets
        datasets = prepare_datasets(K, I, S, T_array, data)
        # Prepare attributes
        # NOTE: Slice thickness is in cm which should be converted to mm.
        attrs = {
            "image_matrix_size": I,
            "number_of_slices": K,
            "temporal_resolution": data["TR"][0],
            "slice_thickness": data["SliceThickness"][0],
            "resolution": data["Resolution"][0] * 10,
            "TED": list(TED),
            "TES": list(TES),
            "T_end_acquisition": T_end,
            "T_end": len(T_array),
            "S": S,
        }
        h5_file_address = mat_file.with_suffix(".h5").as_posix()
        save_to_h5(h5_file_address, datasets, attrs)
        logger.info(f"{mat_file.with_suffix('.h5').name} is created.")
        return h5_file_address

    except Exception as e:
        logger.error(f"Failed processing due to {e}")


def interpolate_T_array(TES, TED, K):
    """
    Generates the T_array used for interpolation.
    """
    nn = TED[0] - TES[0]
    steps = (np.array(TED) - np.array(TES)) / (nn + 1)
    steps = (np.array(TED) - np.array(TES)) / (nn + 1)
    T_array = []
    for i in range(1, nn + 1):
        array = TES + np.round(steps * i).astype(int)
        T_array.append(array.tolist())
    return np.array(T_array)


def get_first_timestep_from_coords_data(*coords):
    sliced_coords = []
    for coord in coords:
        sliced_coords.append(coord[:,0,:])
    return sliced_coords

def invert_coords(*coords):
    inverted_coords = []
    for coord in coords:
        inverted_coords.append(coord[:, ::-1])
    return inverted_coords

def remove_nan_coords_data(*coords):
    # removing the nan datatype in the coords
    nonan_coords = []
    epi_x_coord = coords[0]
    valid_coords = ~np.all(np.isnan(epi_x_coord), axis=0)
    for coord in coords:
        nonan_coord = coord[:, valid_coords]
        nonan_coords.append(nonan_coord)
    return nonan_coords

def reorder_coords(x_coords, y_coords):
    reorder_coords = []
    k = x_coords.shape[1]
    for i in range(k):
        reorder_coords.append(np.column_stack((x_coords[:,i],y_coords[:,i])))
    return reorder_coords
            
def remove_incomplete_coords(*coords):
    # the epis may be incomplete, i.e., not corresponding to a circle.
    removed_incomplete_coords = []
    K = coords[0].shape[1]
    valid_coords = np.zeros(K, dtype=bool)
    for k in range(K):
        epi_coord_k = np.column_stack((coords[0][:,k],coords[1][:,k]))
        area = calculate_enclosed_area(epi_coord_k)
        radius_ave, radius_std = calculate_avg_std_radius(epi_coord_k)
        ave_area_circle = np.pi * radius_ave**2
        if area/ave_area_circle>0.75:
            valid_coords[k] = True
    for coord in coords:
        removed_incomplete_coord = coord[:,valid_coords]
        removed_incomplete_coords.append(removed_incomplete_coord)
        
    # Log a warning for the indices being removed
    removed_indices = np.where(valid_coords == False)[0]
    if removed_indices.size > 0:
        for idx in removed_indices:
            logger.warning(f"Slice no. {idx} is removed due to incomplete circular shape, possibly above basal plane.")

    return removed_incomplete_coords
    
def calculate_enclosed_area(coords):
    x = coords[:,0]
    y = coords[:,1]
    # Use the shoelace formula to calculate the area
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return area

def calculate_avg_std_radius(coords):
    center_x = np.mean(coords[:, 0])
    center_y = np.mean(coords[:, 1])
    
    radius = np.sqrt((coords[:, 0] - center_x)**2 + (coords[:, 1] - center_y)**2)
    
    radius_ave = np.mean(radius)
    radius_std = np.std(radius)
    
    return radius_ave, radius_std

def prepare_coords_dataset(x_epi,y_epi,x_endo,y_endo, is_inverted):
    t0_coords = get_first_timestep_from_coords_data(x_epi, y_epi, x_endo, y_endo)
    if is_inverted: 
        corrected_coords = invert_coords(*t0_coords)
    else:
        corrected_coords = t0_coords
    nonan_coords = remove_nan_coords_data(*corrected_coords)
    removed_incomplete_coords = remove_incomplete_coords(*nonan_coords)
    
    x_epi, y_epi, x_endo, y_endo = removed_incomplete_coords
    coords_epi = reorder_coords(x_epi, y_epi)
    coords_endo = reorder_coords(x_endo, y_endo)
    return{
        "coords_epi": coords_epi,
        "coords_endo": coords_endo,
    }

# import matplotlib.pyplot as plt

# def plot_epi_endo(corrected_coords, name = 'test'):
#     x_epi, y_epi, x_endo, y_endo = corrected_coords  # Unpack the corrected coordinates
#     k = x_endo.shape[1]  # Number of slices
#     nrows = 5
#     ncols = 3

#     fig, axs = plt.subplots(nrows, ncols, figsize=(15, 15))  # Create 5x3 subplots
    
#     for i in range(k):
#         row = i // ncols
#         col = i % ncols
#         # Plot the endocardial curve for slice i
#         axs[row, col].plot(x_endo[:, i], y_endo[:, i], label='Endo', color='blue')
#         # Plot the epicardial curve for slice i
#         axs[row, col].plot(x_epi[:, i], y_epi[:, i], label='Epi', color='red')
        
#         # Setting titles and labels for clarity
#         axs[row, col].set_title(f'Slice {i+1}')
#         axs[row, col].set_xlabel('X')
#         axs[row, col].set_ylabel('Y')
#         axs[row, col].legend()
    
#     # Hide any unused subplots if k < 15
#     for j in range(k, nrows * ncols):
#         fig.delaxes(axs.flatten()[j])

#     # Adjust layout for better spacing
#     plt.tight_layout()
#     plt.savefig(name)

def transform_to_img_cs_for_all_slices(coords, resolution, I):
    transformed_coords = []
    k = len(coords)
    for coord in coords:
        transformed_coord = transform_to_img_cs(coord, resolution, I)
        # Remove duplicates
        transformed_coord_unique = remove_duplicates(transformed_coord)
        transformed_coords.append(transformed_coord_unique)
    return transformed_coords
    
def transform_to_img_cs(coord, resolution, I):
    img_coord = np.zeros((len(coord[:,0]),2))
    img_coord[:, 0] = coord[:,1] * resolution
    img_coord[:, 1] = coord[:,0] * resolution
    img_coord[:, 1] = I * resolution - img_coord[:, 1]
    return img_coord

# Function to remove duplicates while preserving the order
def remove_duplicates(points):
    seen = set()
    unique_points = []
    for point in points:
        # Convert point to tuple to make it hashable for the set
        point_tuple = tuple(point)
        if point_tuple not in seen:
            seen.add(point_tuple)
            unique_points.append(point)
    return np.array(unique_points)

def prepare_datasets(K, I, S, T_array, data):
    """
    Prepares datasets for saving to H5.
    """
    LVmask = np.zeros((K, I, I, len(T_array)), dtype=bool)
    WallThickness = np.zeros((K, S, len(T_array)), dtype=float)

    for i, t in enumerate(T_array):
        for k in range(K):
            LVmask[k, :, :, i] = data["Mask"][k][:, :, t[k] - 1]
            WallThickness[k, :, i] = data["WallThickness"][k][t[k] - 1, :]

    return {
        "LVmask": LVmask,
        # "SeptumSector_position": data["SeptumSector_position"],
        # "RVSector": data["RVSector"],
        # "WallThickness": WallThickness,
        # "KE_SumS": data["KE_SumS"],
        "T": T_array,
    }


def save_to_h5(h5_path, datasets, attrs):
    """
    Saves datasets to an H5 file.
    """
    with h5py.File(h5_path, "w") as f:
        for name, dataset in datasets.items():
            f.create_dataset(name, data=dataset)
        # Save attributes to the H5 file
        for attr, value in attrs.items():
            f.attrs[attr] = value


def load_from_h5(h5_path):
    datasets = {}
    attrs = {}
    with h5py.File(h5_path, "r") as f:
        for name in f.keys():
            datasets[name] = f[name][:]
        for attr in f.attrs.keys():
            attrs[attr] = f.attrs[attr]
    return datasets, attrs


def update_h5_file(h5_path, datasets={}, attrs={}):
    existing_datasets, existing_attrs = load_from_h5(h5_path)

    # Update existing datasets and attributes with new ones
    updated_datasets = {**existing_datasets, **datasets}
    updated_attrs = {**existing_attrs, **attrs}

    save_to_h5(h5_path, updated_datasets, updated_attrs)
    return updated_datasets, updated_attrs


# %%
def pre_process_mask(
    h5_path,
    save_flag=False,
    results_folder: str = "00_Results",
    settings: list = {1,1,1,1,1,1,1,1,1,1},
):
    datasets, attrs = load_from_h5(h5_path)
    K, I, T_end = attrs["number_of_slices"], attrs["image_matrix_size"], attrs["T_end"]
    mask = datasets["LVmask"]
    if save_flag:
        output_dir = results_folder / "01_GapClosed"
        output_dir.mkdir(exist_ok=True)

    mask_closed = np.empty((K, I, I, T_end))

    for t in range(T_end):
        for k in range(K):
            mask_t = mask[k, :, :, t]
            mask_closed[k, :, :, t] = close_gaps(
                    mask_t, settings[k]
                )
            if save_flag:
                image_comparison = show_image(mask_t, mask_closed[k, :, :, t])
                img_path = output_dir / f"{t+1}_{k+1}.tiff"
                cv.imwrite(img_path.as_posix(), image_comparison)

    updated_datasets = {"LVmask": mask_closed}
    update_h5_file(h5_path, datasets=updated_datasets)
    return h5_path


def calculate_center_binary_image(binary_image):
    """
    Calculates the center (centroid) of the True values in a binary image.
    
    Parameters:
    - binary_image: numpy array of shape (H, W) with boolean values
    
    Returns:
    - center: tuple (y_center, x_center) representing the coordinates of the center
    """
    # Get the indices of the True values
    true_indices = np.argwhere(binary_image)
    
    # If there are no True values, return None or an appropriate value
    if true_indices.size == 0:
        return None
    
    # Calculate the mean of the indices to find the center
    y_center, x_center = np.mean(true_indices, axis=0)
    
    return (y_center, x_center)


def shift_binary_image(binary_image, x_shift, y_shift):
    """
    Shifts all the True values in a binary image by x_shift and y_shift.
    
    Parameters:
    - binary_image: numpy array of shape (H, W) with boolean values
    - x_shift: integer, shift along the x-axis (columns)
    - y_shift: integer, shift along the y-axis (rows)
    
    Returns:
    - shifted_image: numpy array of shape (H, W) with boolean values
    """
    # Get the shape of the input binary image
    H, W = binary_image.shape
    
    # Create an empty image with the same shape
    shifted_image = np.zeros_like(binary_image, dtype=int)
    
    # Compute the new positions of the True values
    for y in range(H):
        for x in range(W):
            if binary_image[y, x]:
                new_x = x + x_shift
                new_y = y + y_shift
                # Check if the new positions are within bounds
                if 0 <= new_x < W and 0 <= new_y < H:
                    shifted_image[new_y, new_x] = True
    
    return shifted_image

def shift_slice_mask(
    h5_file,
    slice_num,
    slice_num_ref,
    save_flag = False,
    results_folder: str = "00_Results",
):
    datasets, attrs = load_from_h5(h5_file)
    K, I, T_end = attrs["number_of_slices"], attrs["image_matrix_size"], attrs["T_end"]
    mask = datasets["LVmask"]

    if save_flag:
        output_dir = results_folder / "01_GapClosed"
        output_dir.mkdir(exist_ok=True)
        
    mask_shifted = np.empty((K, I, I, T_end))

    for t in range(T_end):
        for k in range(K):
            mask_kt = mask[k, :, :, t]
            if k == slice_num:
                (y_center, x_center) = calculate_center_binary_image(mask[slice_num_ref,:,:,t])
                (y_center_slice, x_center_slice) = calculate_center_binary_image(mask[slice_num,:,:,t])
                y_shift = int(y_center - y_center_slice)
                x_shift = int(x_center - x_center_slice)
                mask_kt_shifted = shift_binary_image(mask_kt, x_shift, y_shift)
                mask_shifted[k,:,:,t] = mask_kt_shifted
                if save_flag:
                    img_array = np.uint8(mask_shifted[k,:,:,t] * 255)
                    img_path = output_dir / f"{t+1}_{k+1}.tiff"
                    cv.imwrite(img_path.as_posix(),img_array)
            else:
                mask_shifted[k,:,:,t] = mask_kt
            
    logger.info(f"Masks are shifted for all slice number {slice_num}")
    updated_datasets = {"LVmask": mask_shifted}
    update_h5_file(h5_file, datasets=updated_datasets)
    return h5_file

def show_image(img_array, img_dilated_eroded):
    new_image = np.zeros((img_array.shape[0], img_array.shape[1], 3), dtype=np.uint8)
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            if img_array[i, j] != 0 and img_dilated_eroded[i, j] != 0:
                # Both have value - set to white
                new_image[i, j] = [255, 255, 255]
            elif img_array[i, j] != 0:
                # Only img_array has value - set to red
                new_image[i, j] = [0, 0, 255]
            elif img_dilated_eroded[i, j] != 0:
                # Only img_dilated_eroded has value - set to blue
                new_image[i, j] = [255, 0, 0]
            # Else: remain black (as initialized)
    return new_image


def close_gaps(mask_t, itr):
    img_array = np.uint8(mask_t * 255)
    kernel = np.ones((3, 3), np.uint8)
    img_dilated = cv.dilate(img_array, kernel, iterations=itr)
    img_dilated_eroded = cv.erode(img_dilated, kernel, iterations=itr)
    return img_dilated_eroded

def close_apex(
    h5_file,
    itr = 2,
    itr_dilation = 3,
    save_flag = False,
    results_folder: str = "00_Results",
):
    datasets, attrs = load_from_h5(h5_file)
    K, I, T_end = attrs["number_of_slices"], attrs["image_matrix_size"], attrs["T_end"]
    mask = datasets["LVmask"]

    if save_flag:
        output_dir = results_folder / "01_GapClosed"
        output_dir.mkdir(exist_ok=True)
        
    mask_closed_apex = np.zeros((K+1,I,I,T_end))
    kernel = np.ones((3, 3), np.uint8)

    for t in range(T_end):
        mask_closed_apex[:-1,:,:,t] = mask[:,:,:,t]
        mask_kt = np.uint8(mask[K-1,:,:,t] * 255)
        mask_kt_closed = cv.dilate(mask_kt, kernel, iterations=itr_dilation)
        mask_kt_closed_eroded = cv.erode(mask_kt_closed, kernel, iterations=itr+itr_dilation)
        mask_closed_apex[-1,:,:,t] = mask_kt_closed_eroded
        if save_flag:
            new_image = np.zeros((mask_kt_closed_eroded.shape[0], mask_kt_closed_eroded.shape[1], 3), dtype=np.uint8)
            for i in range(new_image.shape[0]):
                for j in range(new_image.shape[1]):
                    if mask_kt_closed_eroded[i, j] != 0 :
                        # Both have value - set to Blue
                        new_image[i, j] = [255, 0, 0]
            img_path = output_dir / f"{t+1}_{K+1}.tiff"
            cv.imwrite(img_path.as_posix(),new_image)
            
    
    logger.info(f"An additional closed mask added for apex closure")
    updated_datasets = {"LVmask": mask_closed_apex}
    updated_attr = {'number_of_slices' : K+1}
    update_h5_file(h5_file, datasets=updated_datasets, attrs=updated_attr)
    return h5_file

def repair_slice(
    h5_file,
    slice_num = 0,
    erosion_flag = False,
    save_flag = False,
    results_folder: str = "00_Results",
):
    datasets, attrs = load_from_h5(h5_file)
    K, I, T_end = attrs["number_of_slices"], attrs["image_matrix_size"], attrs["T_end"]
    mask = datasets["LVmask"]

    if save_flag:
        output_dir = results_folder / "01_GapClosed"
        output_dir.mkdir(exist_ok=True)
        
    mask_repaired = np.zeros((K,I,I,T_end))
    kernel = np.ones((3, 3), np.uint8)
    
    for t in range(T_end):
        mask_repaired[:,:,:,t] = mask[:,:,:,t]
        mask_kt = np.uint8(mask_repaired[slice_num,:,:,t] * 255)
        mask_kt_plus = np.uint8(mask_repaired[slice_num+1,:,:,t] * 255)
        if erosion_flag:
            mask_kt_plus = cv.erode(mask_kt_plus, kernel, iterations=1)
        # Superimpose the two masks
        superimposed_mask = mask_kt + mask_kt_plus
        # Clip values to stay within 0-255
        superimposed_mask = np.clip(superimposed_mask, 0, 1)  

        mask_repaired[slice_num,:,:,t] = superimposed_mask
    
        if save_flag:
            new_image = np.zeros((mask_kt.shape[0], mask_kt.shape[1], 3), dtype=np.uint8)
            for i in range(mask_kt.shape[0]):
                for j in range(mask_kt.shape[1]):
                    if mask_kt[i, j] != 0 and superimposed_mask[i, j] != 0:
                        # Both have value - set to white
                        new_image[i, j] = [255, 255, 255]
                    elif superimposed_mask[i, j] != 0:
                        # Only superimposed_mask has value - set to blue
                        new_image[i, j] = [255, 0, 0]
                    img_path = output_dir / f"{t+1}_{slice_num+1}.tiff"
            cv.imwrite(img_path.as_posix(),new_image)
    
    logger.info(f"Slice no. {slice_num} is superimposed with the next slice")
    updated_datasets = {"LVmask": mask_repaired}
    update_h5_file(h5_file, datasets=updated_datasets)
    return h5_file

def remove_slice(
    h5_file,
    slice_num = 0,
    save_flag = True,
    results_folder: str = "00_Results",
):
    datasets, attrs = load_from_h5(h5_file)
    K, I, T_end = attrs["number_of_slices"], attrs["image_matrix_size"], attrs["T_end"]
    mask = datasets["LVmask"]

    if save_flag:
        output_dir = results_folder / "01_GapClosed"
        output_dir.mkdir(exist_ok=True)
    
    mask_removed = np.zeros((K-1,I,I,T_end))
    
    for t in range(T_end):
        kk = 0
        for k in range(K):
            if k == slice_num:
                mask_kt = np.uint8(mask[k,:,:,t] * 255)
                if save_flag:
                    new_image = np.zeros((I,I, 3), dtype=np.uint8)
                    for i in range(I):
                        for j in range(I):
                            if mask_kt[i,j] != 0:
                                # Both have value - set to white
                                new_image[i, j] = [0, 0, 255]
                            img_path = output_dir / f"{t+1}_{slice_num+1}.tiff"
                    cv.imwrite(img_path.as_posix(),new_image)
            else:
                mask_removed[kk,:,:,t] = mask[k,:,:,t]
                kk += 1
    
    
    logger.info(f"Slice no. {slice_num} has been removed")
    updated_datasets = {"LVmask": mask_removed}
    updated_attr = {'number_of_slices' : K-1}
    update_h5_file(h5_file, datasets=updated_datasets, attrs=updated_attr)

    return h5_file

def read_data_h5_TPM(file_dir):
    with h5py.File(file_dir, "r") as f:
        # Read datasets
        LVmask = f["LVmask"][:]
        T = f["T"][:]
        metadata = {key: value for key, value in f.attrs.items()}
        slice_thickness=metadata['slice_thickness'] # in mm
        resolution=metadata['resolution']           # to convert to mm/pixel
        I = metadata["image_matrix_size"]
    return LVmask,T,slice_thickness,resolution, I  

def read_data_h5_CINE(file_dir):
    with h5py.File(file_dir, "r") as f:
        # Read datasets
        coords_endo = f["coords_endo"][:]
        coords_epi = f["coords_epi"][:]
        metadata = {key: value for key, value in f.attrs.items()}
        slice_thickness=metadata['slice_thickness'] # in mm
        resolution=metadata['resolution']           # to convert to mm/pixel
        I = metadata["image_matrix_size"]
    return coords_endo,coords_epi,slice_thickness,resolution, I 