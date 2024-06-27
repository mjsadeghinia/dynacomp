# %%
import numpy as np
import cv2 as cv
import h5py

from pathlib import Path
import pymatreader
from structlog import get_logger

logger = get_logger()


def compile_h5(directory_path, overwrite=False):
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
        sorted_indexes = np.argsort(
            [np.abs(data[label][k]) for k in range(len(data[label]))]
        )
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
            "slice_thickness": data["SliceThickness"][0] / 10,
            "resolution": data["Resolution"][0],
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
        "SeptumSector_position": data["SeptumSector_position"],
        "RVSector": data["RVSector"],
        "WallThickness": WallThickness,
        "KE_SumS": data["KE_SumS"],
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
    settings: dict = dict(slice_number=6, num_itr_slice_1=0, num_itr_slice_2=1),
):
    datasets, attrs = load_from_h5(h5_path)
    K, I, T_end = attrs["number_of_slices"], attrs["image_matrix_size"], attrs["T_end"]
    mask = datasets["LVmask"]

    if save_flag:
        output_dir = Path(h5_path).parent / "01_GapClosed"
        output_dir.mkdir(exist_ok=True)

    mask_closed = np.empty((K, I, I, T_end))

    for t in range(T_end):
        for k in range(K):
            mask_t = mask[k, :, :, t]
            if k < K - settings["slice_number"]:
                mask_closed[k, :, :, t] = close_gaps(
                    mask_t, settings["num_itr_slice_1"]
                )
            else:
                mask_closed[k, :, :, t] = close_gaps(
                    mask_t, settings["num_itr_slice_2"]
                )
            if save_flag:
                image_comparison = show_image(mask_t, mask_closed[k, :, :, t])
                img_path = output_dir / f"{t+1}_{k+1}.tiff"
                cv.imwrite(img_path.as_posix(), image_comparison)

    updated_datasets = {"LVmask": mask_closed}
    update_h5_file(h5_path, datasets=updated_datasets)
    return h5_path


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
