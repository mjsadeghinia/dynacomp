import numpy as np
import subprocess
from structlog import get_logger

logger = get_logger()
a_min = 0.25  # Minimum value of a
a_max = 10.5  # Maximum value of a
af_min = 0.25  # Minimum value of a_f
af_max = 10.5  # Maximum value of a_f
step = 0.25  # Step size for a and a_f
# Create a range of a and a_f values
a_values = np.arange(a_min, a_max + step, step)  # Values of a
a_f_values = np.arange(af_min, af_max + step, step)  # Values of a_f
# for a in a_values:
#     for a_f in a_f_values:
#         # Run the preprocessing command with the current values of a and a_f
#         logger.info(f"Running preprocessing with a={a}, a_f={a_f}")
#         preprocess_cmd = f'mpirun -n 8 python3 dynacomp/inflator.py -n 37 --a_matparam {a} --af_matparam {a_f} -nc --pressure_multiplier {round((a+a_f)/4,2)} --pressure_steps 3'
#         subprocess.run(preprocess_cmd, shell=True, check=True)
#         logger.info("---------------------------")

# Now call the contour‐plot script with the same sweep settings
subprocess.run(
    f"python3 dynacomp/create_matparam_sweep_contour.py "
    f"-n 37 "
    f"--a_min {a_min} --a_max {a_max} --af_min {af_min} --af_max {af_max} "
    f"--grid_size {len(a_values)}",
    shell=True, check=True
)