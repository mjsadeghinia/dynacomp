# %%
# import dolfin.function
import numpy as np
from pathlib import Path
import structlog 
import logging

import pulse
import dolfin


# Configure structlog to use ConsoleRenderer with custom formatting and colors
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="ISO", utc=True),  # Timestamp formatting
        structlog.processors.add_log_level,  # Add log level to output
        structlog.dev.ConsoleRenderer(colors=True)  # Enable colors in the console
    ],
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


# %%
class HeartModelDynaComp:
    def __init__(
        self,
        geo: pulse.HeartGeometry,
        geo_refinement: int = None,
        bc_params: dict = None,
        matparams: dict = None,
        comm=None,
    ):
        """
        Initializes the heart model with given geometrical parameters and folder for geometrical data.

        Parameters:
        geo_params (dict, optional): Dictionary of geometric parameters.
        geo_folder (Path): Path object indicating the folder where geometry data is stored.
        """
        logging.getLogger("pulse").setLevel(logging.WARNING)

        if comm is None:
            comm = dolfin.MPI.comm_world
        self.comm = comm

        self.geometry = geo
        if geo_refinement is not None:
            geo_refined = self.refine_geo(self.geometry, geo_refinement)
            self.geometry = geo_refined

        self.lv_pressure = dolfin.Constant(0.0, name="LV Pressure")
        self.activation = dolfin.Constant(0.0, name="Activation")

        self.F0 = dolfin.Identity(self.geometry.mesh.geometric_dimension())
        self.E_ff = []
        self.myocardial_work = []

        self.material = self.get_material_model(matparams)
        self._get_bc_params(bc_params)
        self.bcs = self.apply_bcs()
        self.problem = pulse.MechanicsProblem(self.geometry, self.material, self.bcs)
        # # TODO: make it compatible with MPI
        # Check if it is unloaded!
        # if self.comm.Get_rank()==0:
        #     U, _ = self.problem.state.split(deepcopy=True)
        #     F = pulse.kinematics.DeformationGradient(U)
        #     Wtotal = dolfin.assemble(self.material.strain_energy(F)*dolfin.dx)

        #     if Wtotal != 0:
        #         logger.critical(f'Initially, the Wtotal is  {Wtotal} and not zero')
        #     point = dolfin.Point(self.geometry.mesh.coordinates()[0])
        #     f_proj = dolfin.project(self.material.f0, dolfin.VectorFunctionSpace(self.geometry.mesh, "DG", 1))
        #     if np.linalg.norm(f_proj(point)) != 1:
        #         f_proj = dolfin.project(self.material.f0, dolfin.VectorFunctionSpace(self.geometry.mesh, "DG", 1))
        #         logger.info(f'The f0 at the point is {f_proj(point)} with a norm of {np.linalg.norm(f_proj(point))}')
        # # self.comm.Barrier()

        self.problem.solve()
        
        # def log_matrix(matrix, label = 'matrix'):
        #     matrix = matrix.reshape(3, 3)
        #     logger.info(f'The {label} matrix at the point is:\n' + '\n'.join([' '.join([f'{item: .6f}' for item in row]) for row in matrix]))

        # # Logger function to print a vector vertically
        # def log_vector(vector):
        #     logger.info('The vector at the point is:\n' + '\n'.join([f'{item: .6f}' for item in vector]))


        # logger.info(f"The inital pressure is {self.lv_pressure.values()}")
        # logger.info(f"The inital activation is {self.activation.values()}")
        # self.problem.solve()
        # u, p = self.problem.state.split(deepcopy=True)
        # point = dolfin.Point(self.geometry.mesh.coordinates()[0])

        # F = pulse.kinematics.DeformationGradient(u)
        # F_proj = dolfin.project(F, dolfin.TensorFunctionSpace(self.geometry.mesh, "DG", 0))
        # logger.info(f'The F matrix at the point is:')
        # log_matrix(F_proj(point),  label='matrix F')

        # C = pulse.kinematics.RightCauchyGreen(F)
        # C_proj = dolfin.project(C, dolfin.TensorFunctionSpace(self.geometry.mesh, "DG", 0))
        # logger.info(f'The C matrix at the point is:')
        # log_matrix(C_proj(point), label='matrix C')

        # f_proj = dolfin.project(self.material.f0, dolfin.VectorFunctionSpace(self.geometry.mesh, "DG", 1))
        # log_vector(f_proj(point))
        # logger.info(f'The norm of the vecotr is {np.linalg.norm(np.linalg.norm(self.geometry.f0.vector().get_local()[:3]))}')

        # I4 = pulse.kinematics.I4(F, self.material.f0, self.material.isochoric)
        # I4_proj = dolfin.project(I4, dolfin.FunctionSpace(self.geometry.mesh, "DG", 0))
        # logger.info(f'Initially, the I4 is  {I4_proj(point)} and not unity')
        # breakpoint()


    def compute_volume(self, activation_value: float, pressure_value: float) -> float:
        """
        Computes the volume of the heart model based on activation and pressure values.

        Parameters:
        activation_value (float): The activation value to be applied.
        pressure_value (float): The pressure value to be applied.

        Returns:
        float: The computed volume of the heart model.
        """
        pulse.iterate.iterate(self.problem, self.activation, activation_value)
        pulse.iterate.iterate(self.problem, self.lv_pressure, pressure_value)
        volume_current = self.problem.geometry.cavity_volume(
            u=self.problem.state.sub(0)
        )
        if self.comm.rank == 0:
            logger.info("Computed volume", volume_current=volume_current)
        return volume_current

    def dVda(
        self,
        activation_value: float,
        pressure_value: float,
        delta_a_percent: float = 0.01,
    ) -> float:
        """
        Computes dV/da, with V is the volume of the model and a is the activation.
        The derivation is computed as the change of volume due to a small change in the activation at a given activation.
        After computation the problem is reset to its initial state.

        NB! We use problem.solve() instead of pulse.iterate.iterate, as the pressure change is small and iterate may fail.

        Parameters:
        activation_value (float): The activation value to be applied.
        pressure_value (float): The pressure value to be applied.

        Returns:
        float: The computed dV/da .
        """
        if self.comm.rank == 0:
            logger.info(
                "Computing dV/da",
                activation_value=activation_value,
                pressure_value=pressure_value,
            )
        # Backing up the problem
        state_backup = self.problem.state.copy(deepcopy=True)
        pressure_backup = float(self.lv_pressure)
        activation_backup = float(self.activation)
        # Update the problem with the give activation and pressure and store the initial State of the problem
        self.assign_state_variables(activation_value, pressure_value)
        self.problem.solve()

        a_i = float(self.activation)
        v_i = self.get_volume()

        # small change in pressure and computing the volume
        a_f = a_i * (1 + delta_a_percent)
        # breakpoint()
        self.activation.assign(a_f)
        self.problem.solve()

        # dolfin.MPI.barrier(self.comm)

        v_f = self.get_volume()

        dV_da = (v_f - v_i) / (a_i * delta_a_percent)
        if self.comm.rank == 0:
            logger.info("Computed dV/da", dV_da=dV_da)

        # reset the problem to its initial state
        self.problem.state.assign(state_backup)
        self.assign_state_variables(activation_backup, pressure_backup)

        return dV_da

    def get_pressure(self) -> float:
        return float(self.lv_pressure)

    def get_volume(self) -> float:
        return self.problem.geometry.cavity_volume(u=self.problem.state.sub(0))

    def initial_loading(self, EDP):
        volume = self.compute_volume(activation_value=0, pressure_value=EDP)
        results_u, _ = self.problem.state.split(deepcopy=True)
        self.F0 = pulse.kinematics.DeformationGradient(results_u)
        return volume

    def save_tensor(self, tensor, fname, t, name="tensor"):
        mesh = self.problem.geometry.mesh
        tensor_element = dolfin.TensorElement("DG", mesh.ufl_cell(), 0)
        function_space = dolfin.FunctionSpace(mesh, tensor_element)
        tensor_proj = dolfin.project(tensor, function_space)
        tensor_proj.t = t + 1
        with dolfin.XDMFFile(self.comm, fname.as_posix()) as xdmf:
            xdmf.write_checkpoint(
                tensor_proj, name, float(t + 1), dolfin.XDMFFile.Encoding.HDF5, True
            )

    def save_scalar(self, scalar, fname, t, name="scalar"):
        mesh = self.problem.geometry.mesh
        tensor_element = dolfin.FiniteElement("DG", mesh.ufl_cell(), 0)
        function_space = dolfin.FunctionSpace(mesh, tensor_element)
        tensor_proj = dolfin.project(scalar, function_space)
        tensor_proj.t = t + 1
        with dolfin.XDMFFile(self.comm, fname.as_posix()) as xdmf:
            xdmf.write_checkpoint(
                tensor_proj, name, float(t + 1), dolfin.XDMFFile.Encoding.HDF5, True
            )

    def save(self, t: float, outdir: Path = Path("results")):
        """
        Saves the current state of the heart model at a given time to a specified file.

        Parameters:
        t (float): The time at which to save the model state.
        outname (Path): The file path to save the model state.
        """
        fname = outdir / "displacement.xdmf"

        results_u, _ = self.problem.state.split(deepcopy=True)
        results_u.t = t
        with dolfin.XDMFFile(self.comm, fname.as_posix()) as xdmf:
            xdmf.write_checkpoint(
                results_u,
                "Displacement",
                float(t + 1),
                dolfin.XDMFFile.Encoding.HDF5,
                True,
            )

        F = pulse.kinematics.DeformationGradient(results_u)
        E = pulse.kinematics.GreenLagrangeStrain(F)
        Cauchy = self.problem.material.CauchyStress(F)
        S = self.problem.material.SecondPiolaStress(F)
        MW = dolfin.inner(S, E)

        fname = outdir / "Deformation_Gradient.xdmf"
        self.save_tensor(F, fname, t, name="Deformation Gradiant")

        fname = outdir / "Cauchy_Stress.xdmf"
        self.save_tensor(Cauchy, fname, t, name="Cauchy Stress")

        fname = outdir / "Myocardial_Work.xdmf"
        self.save_scalar(MW, fname, t, name="Myocardium Work")

        fname = outdir / "Activation_results.xdmf"
        self.save_scalar(self.activation, fname, t, name="Activation")

        # fname = outdir / "Green_Lagrange_Strain.xdmf"
        # self.save_tensor(E, fname, t, name = 'Green Lagrange Strain')
        # fname = outdir / "Second_Piola_Stress.xdmf"
        # self.save_tensor(S, fname, t, name = 'Second Piola Stress')

    def get_deformed_mesh(self):
        results_u, _ = self.problem.state.split(deepcopy=True)
        element = self.problem.state_space.sub(0).ufl_element()
        FunSpace = dolfin.FunctionSpace(self.problem.geometry.mesh, element)
        results_u_interp = dolfin.interpolate(results_u, FunSpace)
        deformed_mesh = self.geometry.mesh
        dolfin.ALE.move(deformed_mesh, results_u_interp)
        return deformed_mesh

    def _compute_fiber_strain(self, u):
        F = pulse.kinematics.DeformationGradient(u) * dolfin.inv(self.F0)
        E = pulse.kinematics.GreenLagrangeStrain(F)
        # Green strain normal to fiber direction
        V = dolfin.FunctionSpace(self.geometry.mesh, "DG", 0)
        Eff = dolfin.project(dolfin.inner(E * self.geometry.f0, self.geometry.f0), V)
        E_ff_segment = []
        num_segments = len(set(self.problem.geometry.cfun.array()))
        for n in range(num_segments):
            indices = np.where(self.problem.geometry.cfun.array() == n + 1)[0]
            E_ff_segment.append(Eff.vector()[indices])
        self.E_ff.append(E_ff_segment)
        return Eff

    def _compute_myocardial_work(self, u):
        F = pulse.kinematics.DeformationGradient(u)
        E = pulse.kinematics.GreenLagrangeStrain(F * dolfin.inv(self.F0))
        Eff = dolfin.inner(
            E * self.material.f0, self.material.f0
        )  # Green Lagrange strain in fiber directions
        sigma = self.problem.material.CauchyStress(F)
        f_current = F * self.material.f0  # fiber directions in current configuration
        t = sigma * f_current
        tff = dolfin.inner(t, f_current)  # traction, forces, in fiber direction
        myocardial_work = tff * Eff

        V = dolfin.FunctionSpace(self.problem.geometry.mesh, "DG", 0)
        myocardial_work_values = dolfin.project(myocardial_work, V)
        myocardial_work_values_segment = []
        num_segments = len(set(self.problem.geometry.cfun.array()))
        for n in range(num_segments):
            indices = np.where(self.problem.geometry.cfun.array() == n + 1)[0]
            myocardial_work_values_segment.append(
                myocardial_work_values.vector()[indices]
            )
        self.myocardial_work.append(myocardial_work_values_segment)
        return myocardial_work

    def assign_state_variables(self, activation_value, pressure_value):
        self.lv_pressure.assign(pressure_value)
        self.activation.assign(activation_value)

    def create_geometry(self, geo, fiber_angles):
        import ldrb

        # Convert markers to correct format
        markers = {
            "base": geo.markers["BASE"][0],
            "lv": geo.markers["ENDO"][0],
            "epi": geo.markers["EPI"][0],
        }
        # Choose space for the fiber fields
        # This is a string on the form {family}_{degree}
        fiber_space = "P_1"

        # Compute the microstructure
        fiber, sheet, sheet_normal = ldrb.dolfin_ldrb(
            mesh=geo.mesh,
            fiber_space=fiber_space,
            ffun=geo.ffun,
            markers=markers,
            **fiber_angles,
        )
        if self.comm.Get_rank() == 1:
            logger.info("---------- Fibers regenerated ----------")

        microstructure = pulse.Microstructure(f0=fiber, s0=sheet, n0=sheet_normal)
        marker_functions = pulse.MarkerFunctions(ffun=geo.ffun)

        return pulse.HeartGeometry(
            mesh=geo.mesh,
            markers=geo.markers,
            microstructure=microstructure,
            marker_functions=marker_functions,
        )

    def get_fiber_angles(self, fiber_angles):
        # Use provided fiber_angles or default ones if not provided
        default_fiber_angles = self.get_default_fiber_angles()
        fiber_angles = (
            {
                key: fiber_angles.get(key, default_fiber_angles[key])
                for key in default_fiber_angles
            }
            if fiber_angles
            else default_fiber_angles
        )
        return fiber_angles

    @staticmethod
    def get_default_fiber_angles():
        """
        Default fiber angles parameter for the left ventricle
        """
        angles = dict(
            alpha_endo_lv=60,  # Fiber angle on the LV endocardium
            alpha_epi_lv=-60,  # Fiber angle on the LV epicardium
            beta_endo_lv=-15,  # Sheet angle on the LV endocardium
            beta_epi_lv=15,  # Sheet angle on the LV epicardium
        )
        return angles

    @staticmethod
    def refine_geo(geo, geo_refinement):
        mesh, cfun, ffun = geo.mesh, geo.cfun, geo.ffun
        dolfin.parameters["refinement_algorithm"] = "plaza_with_parent_facets"
        for _ in range(geo_refinement):
            mesh = dolfin.adapt(mesh)
            cfun = dolfin.adapt(cfun, mesh)
            ffun = dolfin.adapt(ffun, mesh)

        geo.f0.set_allow_extrapolation(True)
        geo.s0.set_allow_extrapolation(True)
        geo.n0.set_allow_extrapolation(True)

        V_refined = dolfin.FunctionSpace(mesh, geo.f0.function_space().ufl_element())

        f0_refined = dolfin.Function(V_refined)
        f0_refined.interpolate(geo.f0)
        s0_refined = dolfin.Function(V_refined)
        s0_refined.interpolate(geo.s0)
        n0_refined = dolfin.Function(V_refined)
        n0_refined.interpolate(geo.n0)

        marker_functions = pulse.MarkerFunctions(cfun=cfun, ffun=ffun)
        microstructure = pulse.Microstructure(
            f0=f0_refined, s0=s0_refined, n0=n0_refined
        )
        return pulse.HeartGeometry(
            mesh=mesh,
            markers=geo.markers,
            marker_functions=marker_functions,
            microstructure=microstructure,
        )

    def get_material_model(self, matparams):
        """
        Constructs the material model for the heart using default parameters.

        Returns:
        A material model object for use in a pulse.MechanicsProblem.
        """
        # Based on rat model of https://doi.org/10.1016/j.jmbbm.2021.104430
        
        matparams = self.get_matparams(matparams)
        
        return pulse.HolzapfelOgden(
            activation=self.activation,
            active_model="active_stress",
            parameters=matparams,
            f0=self.geometry.f0,
            s0=self.geometry.s0,
            n0=self.geometry.n0,
        )

    def get_matparams(self, matparams: dict = dict()):
        # Use provided fiber_angles or default ones if not provided
        default_matparams = self.get_default_matparams()
        matparams = (
            {
                key: matparams.get(key, default_matparams[key])
                for key in default_matparams
            }
            if matparams
            else default_matparams
        )
        return matparams

    @staticmethod
    def get_default_matparams():
        """
        Default material parameters for the left ventricle
        """
        return dict(
            a=10.726,
            a_f=7.048,
            b=2.118,
            b_f=0.001,
            a_s=0.0,
            b_s=0.0,
            a_fs=0.0,
            b_fs=0.0,
        )

    def apply_bcs(self):
        bcs = pulse.BoundaryConditions(
            dirichlet=(self._fixed_base_z,),
            neumann=self._neumann_bc(),
            robin=self._robin_bc(),
        )
        return bcs

    def _fixed_endoring(self, W):
        V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)

        # Fixing the endo ring in all directions to prevent rigid body motion
        endo_ring_points = self._get_endo_ring()
        endo_ring_points_x0 = np.mean(endo_ring_points[:, 0])
        endo_ring_points_radius = np.sqrt(
            np.min((endo_ring_points[:, 1] ** 2 + endo_ring_points[:, 2] ** 2))
        )

        class EndoRing_subDomain(dolfin.SubDomain):
            def __init__(self, x0, x2):
                super().__init__()
                self.x0 = x0
                self.x2 = x2
                print(x0)

            def inside(self, x, on_boundary):
                return dolfin.near(x[0], self.x0, 0.01) and dolfin.near(
                    pow(pow(x[1], 2) + pow(x[2], 2), 0.5), self.x2, 0.1
                )

        endo_ring_fixed = dolfin.DirichletBC(
            V,
            dolfin.Constant((0.0, 0.0, 0.0)),
            EndoRing_subDomain(endo_ring_points_x0, endo_ring_points_radius),
            method="pointwise",
        )
        return endo_ring_fixed

    def _fixed_base(self, W):
        V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)

        # Fixing the base in x[0] direction
        bc_fixed_based = dolfin.DirichletBC(
            V,
            dolfin.Constant((0.0, 0.0, 0.0)),
            self.geometry.ffun,
            self.geometry.markers["BASE"][0],
        )

        return bc_fixed_based

    def _fixed_base_z(self, W):
        V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)

        # Fixing the base in x[2] direction (z direction)
        bc_fixed_based = dolfin.DirichletBC(
            V.sub(2),
            dolfin.Constant((0.0)),
            self.geometry.ffun,
            self.geometry.markers["BASE"][0],
        )

        return bc_fixed_based

    def _neumann_bc(self):
        # LV Pressure
        lv_marker = self.geometry.markers["ENDO"][0]
        lv_pressure = pulse.NeumannBC(
            traction=self.lv_pressure, marker=lv_marker, name="lv"
        )
        neumann_bc = [lv_pressure]
        return neumann_bc

    def _robin_bc(self):
        if self.bc_params["pericardium_spring"] > 0.0:
            robin_bc = [
                pulse.RobinBC(
                    value=dolfin.Constant(self.bc_params["pericardium_spring"]),
                    marker=self.geometry.markers["EPI"][0],
                ),
            ]
        else:
            robin_bc = []
        if self.bc_params["base_spring"] > 0.0:
            robin_bc += [
                pulse.RobinBC(
                    value=dolfin.Constant(self.bc_params["base_spring"]),
                    marker=self.geometry.markers["BASE"][0],
                ),
            ]
        return robin_bc

    def _get_geo_params(self, geo_params):
        # Use provided geo_params or default ones if not provided
        default_geo_params = self.get_default_geo_params()
        self.geo_params = (
            {
                key: geo_params.get(key, default_geo_params[key])
                for key in default_geo_params
            }
            if geo_params
            else default_geo_params
        )

    def _get_bc_params(self, bc_params):
        # Use provided geo_params or default ones if not provided
        default_bc_params = self.get_default_bc_params()
        self.bc_params = (
            {
                key: bc_params.get(key, default_bc_params[key])
                for key in default_bc_params
            }
            if bc_params
            else default_bc_params
        )

    def _get_endo_ring(self):
        endo_ring_points = []
        for fc in dolfin.facets(self.geometry.mesh):
            if self.geometry.ffun[fc] == self.geometry.markers["BASE"][0]:
                for vertex in dolfin.vertices(fc):
                    endo_ring_points.append(vertex.point().array())
        endo_ring_points = np.array(endo_ring_points)
        return endo_ring_points

    @staticmethod
    def get_default_bc_params():
        """
        Default BC parameter for the left ventricle
        """
        return {
            "pericardium_spring": 0,
            "base_spring": 0,
        }
