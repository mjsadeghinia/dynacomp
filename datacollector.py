from pathlib import Path
from typing import Protocol
import matplotlib.pyplot as plt
from structlog import get_logger
import csv
import numpy as np
import scipy.stats

logger = get_logger()


class Problem(Protocol):
    def save(self, t: float, outdir: Path) -> None: ...


class DataCollector:
    def __init__(self, outdir: Path, model: Problem, save_all: bool) -> None:
        self.times = []
        self.activations = []
        self.volumes = []
        self.target_volumes = []
        self.pressures = []
        self.model = model
        self.save_all = save_all
        outdir.mkdir(exist_ok=True, parents=True)
        self.outdir = outdir
        if hasattr(model, "comm"):
            self.comm = model.comm
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
                    "Target Volume [microL]",
                    "LV Pressure [kPa]",
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
        self.model.save(t, self.outdir, all=self.save_all)
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

class DataCollectorInflator:
    def __init__(
        self,
        outdir: Path,
        model,
        pv_vols: np.ndarray = None,
        pv_pres: np.ndarray = None,
        edpvr_vols: np.ndarray = None,
        edpvr_pres: np.ndarray = None,
        live_plot: bool = False
    ) -> None:
        self.times = []
        self.volumes = []
        self.pressures = []
        self.model = model
        outdir.mkdir(exist_ok=True, parents=True)
        self.outdir = outdir
        self.comm = getattr(model, 'comm', None) or __import__('dolfin').MPI.comm_world

        # Reference data
        self.pv_vols = pv_vols
        self.pv_pres = pv_pres
        self.edpvr_vols = edpvr_vols
        self.edpvr_pres = edpvr_pres
        self.live_plot = live_plot and self.comm.rank == 0

        # Pre-compute regression and annotation
        res = scipy.stats.linregress(self.edpvr_vols, self.edpvr_pres)
        self.slope = res.slope
        self.intercept = res.intercept
        self.stderr = res.stderr
        tinv = lambda p, df: abs(scipy.stats.t.ppf(p/2, df))
        self.ts = tinv(0.05, len(self.edpvr_vols) - 2)
        self.v0 = -self.intercept / self.slope if self.slope != 0 else float('nan')
        self.v0_est = self.model.compute_volume(activation_value=0, pressure_value=0)

        if self.live_plot:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(8, 6))
            # Plot static PV/EDPVR
            self.ax.plot(self.pv_vols, self.pv_pres, 'k', linewidth=1)
            self.ax.scatter(self.pv_vols, self.pv_pres, s=15, c='k', label='PV Data')
            self.ax.scatter(self.edpvr_vols, self.edpvr_pres, s=8, c='r', label='EDPVR')
            # Regression line
            self.ax.plot(self.edpvr_vols, self.intercept + self.slope * self.edpvr_vols, 'b')
            self.ax.axhline(0, color='gray', linestyle='--')
            # Simulation placeholders
            self.sim_line, = self.ax.plot([], [], 'g-', linewidth=1, label='Simulation')
            self.sim_scatter = self.ax.scatter([], [], s=8, c='g')
            # Annotate slope and intercept
            textstr = (
                f'slope (95%): {self.slope:.3f} ± {self.ts * self.stderr:.3f}\n'
                f'v0 (P=0): {self.v0:.2f}\n'
                f'v0 est (sim): {self.v0_est:.2f}'
            )
            self.ax.text(
                0.05,
                0.95,
                textstr,
                transform=self.ax.transAxes,
                fontsize=10,
                verticalalignment='top'
            )
            self.ax.set_xlabel('Volume [microL]')
            self.ax.set_ylabel('LV Pressure [kPa]')
            self.ax.legend(loc='lower left')
            plt.show()

    def collect(self, time: float, volume: float, pressure: float) -> None:
        if self.comm.rank == 0:
            logger.info(f"Inflation step {time}: ", pressure=round(pressure,3), volume=round(volume,3))
        self.times.append(time)
        self.volumes.append(volume)
        self.pressures.append(pressure)
        self.save(time)

        if self.live_plot:
            # Update simulation curve
            self.sim_line.set_data(self.volumes, self.pressures)
            self.sim_scatter.set_offsets(np.column_stack([self.volumes, self.pressures]))
            # Rescale axes
            self.ax.relim()
            self.ax.autoscale_view()
            self.fig.canvas.draw()
            # Add a second y-axis for LV Pressure in kPa
            ax2 = self.ax.twinx()
            kPa_to_mmHg = 1/0.133322
            ymin, ymax = self.ax.get_ylim()
            ax2.set_ylim(ymin * kPa_to_mmHg, ymax * kPa_to_mmHg)
            ax2.set_ylabel("LV Pressure [mmHg]")
            plt.pause(0.01)
            # Overwrite same figure
            self.fig.savefig(self.figure, dpi=300)

    @property
    def csv_file(self) -> Path:
        return self.outdir / 'results_data.csv'

    @property
    def figure(self) -> Path:
        return self.outdir / 'inflation_results.png'

    def _save_csv(self) -> None:
        with open(self.csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Time [ms]', 'Volume [microL]', 'LV Pressure [kPa]'])
            for t, v, p in zip(self.times, self.volumes, self.pressures):
                writer.writerow([t, v, p])

    def save(self, t: float) -> None:
        self.model.save(t, self.outdir, all=False)
        if self.comm.rank == 0:
            self._save_csv()

    def finalize_plot(self) -> None:
        # Save final static plot (already updated live)
        plt.ioff()
        plt.close(self.fig)

    def read_csv(self) -> dict:
        data = {'time': [], 'volume': [], 'lv_pressure': []}
        with open(self.csv_file, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data['time'].append(float(row['Time [ms]']))
                data['volume'].append(float(row['Volume [microL]']))
                data['lv_pressure'].append(float(row['LV Pressure [kPa]']))
        return data
