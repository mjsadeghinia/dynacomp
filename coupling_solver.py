from typing import Protocol
import numpy as np
from pathlib import Path

from structlog import get_logger

from circ.datacollector import DataCollector
import dolfin

logger = get_logger()


class HeartModel(Protocol):
    activation: float 
    
    def compute_volume(self, activation: float, pressure: float) -> float: ...

    def dVda(self, activation: float, pressure: float) -> float: ...

    def get_pressure(self) -> float: ...

    def get_volume(self) -> float: ...

    def save(self, t: float, outdir: Path) -> None: ...


class CirculationModel(Protocol):
    aortic_pressure: float
    aortic_pressure_derivation: float
    valve_open: bool

    def Q(self, pressure_current: float, pressure_old: float, dt: float) -> float: ...

    def dQdp(
        self, pressure_current: float, pressure_old: float, dt: float
    ) -> float: ...

    def update_aortic_pressure(self) -> float: ...


def newton_solver(
    heart_model: HeartModel,
    pres: np.ndarray,
    vols: np.ndarray,
    collector: DataCollector | None = None,
    start_time: int = 0,
    comm=None,
):
    """
    Solves a heart_model for a specific PV loop to find activation .

    Parameters:
    heart_model (HeartModel): Instance of HeartModel for cardiac dynamics.
    pres (np.array): pressure values 
    vols (np.array): volume values
    collector (DataCollector): For collecting the data and saving
    start_time (int): it is possible the model was initialized so the time is actually numbered from 1 or 2 instead 0, default = 0
    
    """
    if comm is None:
        comm = dolfin.MPI.comm_world

    if collector is None:
        collector = DataCollector(outdir=Path("results"), problem=heart_model)

    for i, p in enumerate(pres):
        # Getting state variable pressure and volume
        a_old = float(heart_model.activation)
        # Current activation level

        # initial guess for the current pressure pressure
        if i == 0 or i == 1:
            a_current = a_old if a_old>0 else 10
        else:
            da = collector.activations[-1] - collector.activations[-2]
            da_sign = np.sign(da)
            da_value = min(np.abs(da), 2.0)
            a_current = collector.activations[-1] + da_sign * da_value        
        p_current = p
        if comm.rank == 0:
            logger.info("Updated parameters", a_current=a_current, p_current = p_current)
        tol = 0.1
        iter = 0
        v_diff = 1.0
        while abs(v_diff) > tol and iter < 20:
            v_current = heart_model.compute_volume(a_current, p_current)
            v_diff = v_current - vols[i]
            if comm.rank == 0:
                logger.info(
                    "Iteration",
                    v_diff=v_diff,
                    p_current=p_current,
                    v_target=vols[i],
                    v_current=v_current,
                    num_iter=iter,
                )

            # Updataing p_current based on relative error using newton method
            if abs(v_diff) > tol:
                J = heart_model.dVda(a_current, p_current)
                a_current = a_current - v_diff / J
                iter += 1

        a_current = float(heart_model.activation)
        p_current = heart_model.get_pressure()
        v_current = heart_model.get_volume()
        collector.collect( 
            i + start_time,
            a_current,
            v_current,
            vols[i],
            p_current,
        )
    return collector
