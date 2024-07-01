from pathlib import Path
from typing import Protocol
import matplotlib.pyplot as plt
from structlog import get_logger
import csv
import numpy as np

logger = get_logger()


class Problem(Protocol):
    def save(self, t: float, outdir: Path) -> None: ...


class DataCollector:
    def __init__(self, outdir: Path, problem: Problem) -> None:
        self.times = []
        self.activations = []
        self.volumes = []
        self.target_volumes = []
        self.pressures = []
        self.problem = problem
        outdir.mkdir(exist_ok=True, parents=True)
        self.outdir = outdir
        if hasattr(problem, "comm"):
            self.comm = problem.comm
        else:
            from dolfin import MPI

            self.comm = MPI.comm_world

    def collect(
        self,
        time: float,
        activation: float,
        volume: float,
        target_volume: float,
        pressure: float,
    ) -> None:
        if self.comm.rank == 0:
            logger.info(
                "Collecting data",
                time=time,
                activation=activation,
                volume=volume,
                target_volume=target_volume,
                pressure=pressure,
            )
        # print('start collecting from ', self.comm.rank)
        self.times.append(time)
        self.activations.append(activation)
        self.volumes.append(volume)
        self.target_volumes.append(target_volume)
        self.pressures.append(pressure)
        self.save(time)

    @property
    def csv_file(self):
        return Path(self.outdir) / "results_data.csv"

    @property
    def figure(self):
        return Path(self.outdir) / "results.png"

    def _save_csv(self):
        with open(self.csv_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "Time [ms]",
                    "Activation [kPa]",
                    "Volume [microL]",
                    "Target Volume [microL]" "LV Pressure [kPa]",
                ]
            )
            for time, activation, vol, target_vol, pres_val in zip(
                self.times,
                self.activations,
                self.volumes,
                self.target_volumes,
                self.pressures,
            ):
                writer.writerow([time, activation, vol, target_vol, pres_val])

    def _plot(self):
        fig, axs = plt.subplots(
            1, 2, figsize=(15, 5)
        )  # Create a figure and two subplots
        axs[1].plot(self.volumes, self.pressures, 'k-', label="Simulation")
        axs[1].scatter(self.target_volumes, self.pressures, s=10, c='r', label="Experiment")
        axs[1].set_ylabel("Pressure (kPa)")
        axs[1].set_xlabel("Volume (micro l)")
        axs[1].legend()
        ax2 = axs[1].twinx()
        pressures_mmHg = np.array(self.pressures) * 7.50062  # Convert to mmHg
        # Plotting the same data but converted on the second y-axis
        ax2.plot(
            self.volumes, pressures_mmHg, "r-", alpha=0
        )  # invisible plot just for axis
        ax2.set_ylabel("Pressure (mmHg)")

        lns1 = axs[0].plot(self.times, self.activations, "k-", label="Fiber Activation")
        axs[0].set_ylabel("Fiber Activation (kPa)")
        axs[0].set_xlabel("Time (-)")
        ax2 = axs[0].twinx()
        lns2 = ax2.plot(self.times, self.pressures, "k--", label="LV Pressure")
        ax2.set_ylabel("LV Pressure (kPa)")
        axs[0].legend()
        lns = lns1+lns2
        labs = [l.get_label() for l in lns]
        axs[0].legend(lns, labs, loc=0)
        fig.savefig(self.figure)
        plt.close(fig)

    def save(self, t: float) -> None:
        self.problem.save(t, self.outdir)
        if self.comm.rank == 0:
            self._plot()
            self._save_csv()

    def read_csv(self):
        data = {
            "time": [],
            "activation": [],
            "volume": [],
            "lv_pressure": [],
            "aortic_pressure": [],
            "outflow": [],
        }
        with open(self.csv_file, mode="r") as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                data["time"].append(float(row["Time [ms]"]))
                data["activation"].append(float(row["Activation [kPa]"]))
                data["volume"].append(float(row["Volume [ml]"]))
                data["lv_pressure"].append(float(row["LV Pressure [kPa]"]))
                data["aortic_pressure"].append(float(row["Aortic Pressure [kPa]"]))
                data["outflow"].append(float(row["Outflow [ml/ms]"]))
        return data
