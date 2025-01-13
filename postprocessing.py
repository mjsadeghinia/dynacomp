# %%
import numpy as np
from pathlib import Path
from structlog import get_logger
import arg_parser
import json
import argparse


logger = get_logger()


#%%# UNITS:
# [kg]   [mm]    [s]    [mN]     [kPa]       [mN-mm]	    g = 9.806e+03
def initialize_results_dict(group_list, time_list, diameter_list):
    ids = dict()
    times = dict()
    volumes = dict()
    pressures = dict()
    for g in group_list:
        ids.setdefault(g, {})
        times.setdefault(g, {})
        pressures.setdefault(g, {})
        volumes.setdefault(g, {})
        for t in time_list:
            if g == "SHAM":
                ids[g].setdefault(t, [])
                times[g].setdefault(t, [])
                pressures[g].setdefault(t, [])
                volumes[g].setdefault(t, [])
            else:
                ids[g].setdefault(t, {})
                times[g].setdefault(t, {})
                pressures[g].setdefault(t, {})
                volumes[g].setdefault(t, {})
                for d in diameter_list:
                    ids[g][t].setdefault(d, [])
                    times[g][t].setdefault(d, [])
                    pressures[g][t].setdefault(d, [])
                    volumes[g][t].setdefault(d, [])
    return ids, times, volumes, pressures

#%%
def main(args=None) -> int:
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--settings_dir",
        default="/home/shared/dynacomp/settings",
        type=Path,
        help="The settings directory where json files are stored.",
    )
    
    parser.add_argument(
        "-o",
        "--output_folder",
        default= "t3",
        type=str,
        help="The result folder name tha would be created in the directory of the sample.",
    )

    args = parser.parse_args(args)
    
    # Initialize the results dicts
    group_list = ["SHAM", "AS"]
    time_list = [6,12,20]
    diameter_list = [107,130,150]
    ids, times, volumes, pressures = initialize_results_dict(group_list, time_list, diameter_list)
    
    setting_dir = Path(args.settings_dir)
    output_folder = args.output_folder
    for settings_fname in sorted(setting_dir.iterdir()):
        if not settings_fname.suffix == '.json':
            continue
        with open(settings_fname, "r") as file:
            settings = json.load(file)
            
        sample_name = settings_fname.stem
        data_dir = Path(settings["path"]) / output_folder / "00_Modeling"
        result_path = data_dir / "results_data.csv"
        if result_path.exists():
            sample_data = np.loadtxt(result_path, delimiter=',', skiprows=1)  
        else:
            logger.warning(f"Results does not exist for {sample_name}")
        group = settings["group"]
        time = settings["time"]
        diameter = settings["ring_diameter"]
        
        if diameter is None:
            ids[group][time].append(sample_name)
            times[group][time].append(sample_data[:,0])
            pressures[group][time].append(sample_data[:,1])
            volumes[group][time].append(sample_data[:,2])
        else:
            ids[group][time][diameter].append(sample_name)
            times[group][time][diameter].append(sample_data[:,0])
            pressures[group][time][diameter].append(sample_data[:,1])
            volumes[group][time][diameter].append(sample_data[:,2])
        
if __name__ == "__main__":
    main()
# %%
