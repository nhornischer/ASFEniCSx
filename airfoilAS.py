import os
import gmsh
from asfenicsx import FEniCSxSim
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
from ufl import FacetNormal, Measure, VectorElement, FiniteElement, FunctionSpace, MixedElement
from ufl import inner, grad, div, dx, split, derivative, TrialFunction, TestFunction, as_vector
from dolfinx.fem import FunctionSpace, Function
from dolfinx.fem import locate_dofs_topological, dirichletbc, form, apply_lifting, set_bc, assemble_scalar
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, create_matrix
from dolfinx.io import XDMFFile, gmshio

import math
import os
from asfenicsx import clustering, functional, ASFEniCSx, utils
import tqdm.autonotebook

class stationaryNavierStokes(FEniCSxSim):
    def __init__(self):
        super().__init__()
        self.domain = {"L": 6.0, "H": 2.0, "gdim": 2, "mesh_resolution": 0.1}
        self.domain_markers = {"inlet": 1, "outlet": 2, "wall": 3, "upper": 4, "lower": 5}
        self.physical_parameters = {"nu": 0.01, "rho": 1.0, "inflow_velocity": 4.0}

    def _create_obstacle(self, params : np.ndarray, conversion = True):
        """
        Creates the obstacle based on the parameters.
        
        Args:
            params (np.ndarray): Parameters of the obstacle.
            conversion (bool, optional): If True, the parameters are converted from the range (-1, 1) to the actual range. Defaults to True.
            
        Returns:
            x_u (callable): Function that represents a parametrization of the upper part of the obstacle in x-direction.
            x_l (callable): Function that represents a parametrization of the lower part of the obstacle in x-direction.
            y_u (callable): Function that represents a parametrization of the upper part of the obstacle in y-direction.
            y_l (callable): Function that represents a parametrization of the lower part of the obstacle in y-direction.
            (float): Maximal thickness of the obstacle.

        Raises:
            ValueError: If the number of parameters does not match the number of dimensions of the parameter space.

        """
        # Define the ranges of the parameters   
        ranges= [[0.010, 0.960], [0.030, 0.970], [-0.074, 0.247], [-0.102, 0.206],[0.2002, 0.4813], [0.0246,0.3227], [0.1750,1.4944], [0.1452,4.8724]]
        
        # Denormalise parameters from range (-1, 1) based on max, min for each value
        if conversion and len(params) == len(ranges):
            for i in range(len(params)):
                params[i] = (params[i] + 1) / 2 * (ranges[i][1] - ranges[i][0]) + ranges[i][0]
        elif len(params) != len(ranges):
            raise ValueError("Number of parameters does not match the dimension of the parameter space.")
        else:
            # Check if the parameters are in the correct range
            for i in range(len(params)):
                if params[i] < ranges[i][0] or params[i] > ranges[i][1]:
                    raise ValueError("Parameter {} is not in the correct range.".format(i))
        
        num_airfoil_refinement = 1000
        # Extract parameters
        c_1, c_2 = params[0], params[1]                     # Coefficients of camber-line-abscissa parameter equation
        c_3, c_4 = params[2], params[3]                     # Coefficients of camber-line-ordinate parameter equation
        X_T = params[4]                                     # Position of maximum thickness
        T = params[5]                                       # Maximum thickness of the airfoil
        beta_TE = params[6]*np.arctan(T/(1-X_T))            # Angle of the trailing edge
        rho_0 = params[7]*(T/X_T)**2                        # Leading edge radius

        # Calculate coefficients of thickness equation by solving a linear system of equations
        rhs = np.array([T, 0, -np.tan(beta_TE/2), np.sqrt(2*rho_0), 0])
        A = np.array([[X_T**0.5, X_T, X_T**2, X_T**3, X_T**4],
                    [0.5* X_T**(-0.5), 1, 2*X_T, 3*X_T**2, 4*X_T**3],
                    [0.25, 0.5, 1, 1.5, 2],
                    [1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1]])

        coeff_thick = np.linalg.solve(A, rhs)

        # Bézier curves with control parameter k \in [0,1]
        x_c = lambda k: 3 * c_1 * k * (1 - k)**2 + 3 * c_2 * (1-k) * k**2 + k**3    # Camber-line-abscissa
        y_c = lambda k: 3 * c_3 * k * (1 - k)**2 + 3 * c_4 * (1-k) * k**2           # Camber-line-abscissa

        # Thickness equation for a 6% thick airfoil
        thickness = lambda x: (coeff_thick[0] * np.sqrt(x) + coeff_thick[1] * x + coeff_thick[2] * x**2 + coeff_thick[3] * x**3 + coeff_thick[4] * x**4)

        # Position of the airfoil in the computational domain defined by the coordinates of the leading edge
        leading_edge_x = np.max([self.domain["L"]/8, 0.5])
        leading_edge_y = self.domain["H"]/2

        # Upper and lower surface of the airfoil
        x_u = x_l = lambda k: leading_edge_x + x_c(k)
        y_u = lambda k: leading_edge_y + y_c(k) + 0.5 * thickness(x_c(k))
        y_l = lambda k: leading_edge_y + y_c(k) - 0.5 * thickness(x_c(k))

        thickness_max = np.max([thickness(x_c(k)) for k in np.linspace(0,1,num_airfoil_refinement)])
        return x_u, x_l, y_u, y_l, thickness_max

    def create_mesh(self, params : np.ndarray):
        gmsh_model_rank = 0
        x_u, x_l, y_u, y_l, thickness_max = self._create_obstacle(params)
        num_airfoil_refinement = 1000
        lc = self.domain["mesh_resolution"]
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal",0)
        if self.comm.rank == gmsh_model_rank:
            rectangle = gmsh.model.occ.addRectangle(0,0,0, self.domain["L"], self.domain["H"], tag=3)
            # Define lower curve of the airfoil using the BSplines and given points
            points_lower_curve=[]
            for k in np.linspace(0,1,num_airfoil_refinement):
                points_lower_curve.append(gmsh.model.occ.addPoint(x_l(k), y_l(k), 0.0, lc))

            # Define upper curve of the airfoil using the BSplines and given points
            points_upper_curve=[points_lower_curve[0]]
            for k in np.linspace(0,1,num_airfoil_refinement)[1:-1]:
                points_upper_curve.append(gmsh.model.occ.addPoint(x_u(k), y_u(k), 0.0, lc))
            points_upper_curve.append(points_lower_curve[-1])

            C1 = gmsh.model.occ.addBSpline(points_lower_curve, degree=3)
            C2 = gmsh.model.occ.addBSpline(points_upper_curve, degree=3)

            # Create airfoil and cut out of computational domain
            W = gmsh.model.occ.addWire([C1,C2])
            obstacle=gmsh.model.occ.addPlaneSurface([W])

            # Remove points of the airfoil
            for i in list(dict.fromkeys(points_lower_curve + points_upper_curve)):
                gmsh.model.occ.remove([(0, i)])

            # Cut out airfoil from computational domain
            fluid = gmsh.model.occ.cut([(self.domain["gdim"], rectangle)], [(self.domain["gdim"], obstacle)])
            gmsh.model.occ.synchronize()

            # Create a distance field to the airfoil
            distance_field = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(distance_field, "CurvesList", [C1,C2])
            gmsh.model.mesh.field.setNumber(distance_field,"Sampling", num_airfoil_refinement*2)
            # Create refined mesh using a threshold field
            refinement= gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(refinement, "IField", distance_field)
            # Set the refinement levels (LcMin for the mesh size in the refined region, LcMax for the mesh size far from the refined region)
            gmsh.model.mesh.field.setNumber(refinement, "LcMin", lc/5)
            gmsh.model.mesh.field.setNumber(refinement, "LcMax", lc)
            # Set the threshold value where which refinement should be applied
            gmsh.model.mesh.field.setNumber(refinement, "DistMin", thickness_max)
            gmsh.model.mesh.field.setNumber(refinement, "DistMax", thickness_max*2)

            # Set the field as background mesh
            gmsh.model.mesh.field.setAsBackgroundMesh(refinement)

            # 8=Frontal-Delaunay for Quads
            gmsh.option.setNumber("Mesh.Algorithm", 8)
            # 2=simple full-quad
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
            # Apply recombination algorithm
            gmsh.option.setNumber("Mesh.RecombineAll", 1)
            # Mesh subdivision algorithm
            gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
            # Mesh generation
            gmsh.model.mesh.generate(self.domain["gdim"])
            # Mesh order
            gmsh.model.mesh.setOrder(1)
            # Mesh optimisation
            gmsh.model.mesh.optimize("Netgen")

        """
            Defining boundary markers for the mesh
        """

        fluid_marker = 1
        inflow, outflow, wall, upper_obstacle, lower_obstacle = [], [], [], [], []
        obstacles, centers_of_mass=[], []

        if self.comm.rank == gmsh_model_rank:
            # Get all surfaces of the mesh
            surfaces = gmsh.model.getEntities(dim=self.domain["gdim"])
            boundaries = gmsh.model.getBoundary(surfaces, oriented=False)

            gmsh.model.addPhysicalGroup(surfaces[0][0], [surfaces[0][1]], fluid_marker)
            gmsh.model.setPhysicalName(surfaces[0][0], fluid_marker, "Fluid")


            for boundary in boundaries:
                center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
                if np.allclose(center_of_mass, [0, self.domain["H"]/2, 0]):
                    inflow.append(boundary[1])
                elif np.allclose(center_of_mass, [self.domain["L"], self.domain["H"]/2, 0]):
                    outflow.append(boundary[1])
                elif np.allclose(center_of_mass, [self.domain["L"]/2, self.domain["H"], 0]):
                    wall.append(boundary[1])
                elif np.allclose(center_of_mass, [self.domain["L"]/2, 0, 0]):
                    wall.append(boundary[1])
                else:
                    obstacles.append(boundary)

            for obstacle in obstacles:
                centers_of_mass.append(gmsh.model.occ.getCenterOfMass(obstacle[0], obstacle[1]))

            # Assign obstacle with lower center of mass to lower_obstacle and vice versa
            if centers_of_mass[0][1] < centers_of_mass[1][1]:
                lower_obstacle.append(obstacles[0][1])
                upper_obstacle.append(obstacles[1][1])
            else:
                lower_obstacle.append(obstacles[1][1])
                upper_obstacle.append(obstacles[0][1])

            # Set physical markers for the boundaries
            gmsh.model.addPhysicalGroup(self.domain["gdim"]-1, wall, self.domain_markers["wall"])
            gmsh.model.setPhysicalName(self.domain["gdim"]-1, self.domain_markers["wall"] , "wall")
            gmsh.model.addPhysicalGroup(self.domain["gdim"]-1, inflow, self.domain_markers["inlet"])
            gmsh.model.setPhysicalName(self.domain["gdim"]-1, self.domain_markers["inlet"], "inlet")
            gmsh.model.addPhysicalGroup(self.domain["gdim"]-1, outflow, self.domain_markers["outlet"])
            gmsh.model.setPhysicalName(self.domain["gdim"]-1, self.domain_markers["outlet"], "outlet")
            gmsh.model.addPhysicalGroup(self.domain["gdim"]-1, lower_obstacle, self.domain_markers["lower"])
            gmsh.model.setPhysicalName(self.domain["gdim"]-1, self.domain_markers["lower"], "lower_obstacle")
            gmsh.model.addPhysicalGroup(self.domain["gdim"]-1, upper_obstacle, self.domain_markers["upper"])
            gmsh.model.setPhysicalName(self.domain["gdim"]-1, self.domain_markers["upper"], "upper_obstacle")

            gmsh.model.occ.remove(gmsh.model.getEntities(dim=self.domain["gdim"]-1))
            gmsh.model.occ.remove(gmsh.model.getEntities(dim=self.domain["gdim"]))
            gmsh.model.occ.remove(gmsh.model.getEntities(dim=0))

        
        self.mesh, self.cell_tags, self.facet_tags = gmshio.model_to_mesh(gmsh.model, self.comm, gmsh_model_rank, gdim=self.domain["gdim"])
        gmsh.finalize()

    def define_problem(self):
        V_element = VectorElement("Lagrange", self.mesh.ufl_cell(), 2)
        Q_element = FiniteElement("Lagrange", self.mesh.ufl_cell(), 1)
            
        # Function spaces
        W_element = MixedElement(V_element, Q_element)
        self.V = FunctionSpace(self.mesh, W_element)

        boundaries_inflow = self.facet_tags.find(self.domain_markers["inlet"])
        boundaries_outflow = self.facet_tags.find(self.domain_markers["outlet"])
        boundaries_wall = self.facet_tags.find(self.domain_markers["wall"])
        boundaries_lower_obstacle = self.facet_tags.find(self.domain_markers["lower"])
        boundaries_upper_obstacle = self.facet_tags.find(self.domain_markers["upper"])

        def u_in_eval(x):
            values = np.zeros((self.domain["gdim"], x.shape[1]),dtype=PETSc.ScalarType)
            # values[0] = 8 * 1.5 * x[1] * (self.domain["H"] - x[1])/(self.domain["H"]**2)
            values[0,:] = self.physical_parameters["inflow_velocity"]
            return values

        def u_wall_eval(x):
            """Return the zero velocity at the wall."""
            return np.zeros((2, x.shape[1]))
        
        # Boundary conditions
        u_in = Function(self.V.sub(0).collapse()[0])
        u_in.interpolate(u_in_eval)
        u_wall = Function(self.V.sub(0).collapse()[0])
        u_wall.interpolate(u_wall_eval)
        bdofs_V_1 = locate_dofs_topological(
            (self.V.sub(0), self.V.sub(0).collapse()[0]), self.mesh.topology.dim - 1, boundaries_inflow)
        bdofs_V_2 = locate_dofs_topological(
            (self.V.sub(0), self.V.sub(0).collapse()[0]), self.mesh.topology.dim - 1, boundaries_wall)
        bdofs_V_3 = locate_dofs_topological(
            (self.V.sub(0), self.V.sub(0).collapse()[0]), self.mesh.topology.dim - 1, boundaries_lower_obstacle)
        bdofs_V_4 = locate_dofs_topological(
            (self.V.sub(0), self.V.sub(0).collapse()[0]), self.mesh.topology.dim - 1, boundaries_upper_obstacle)
        inlet_bc = dirichletbc(u_in, bdofs_V_1, self.V.sub(0))
        wall_bc = dirichletbc(u_wall, bdofs_V_2, self.V.sub(0))
        lower_obstacle_bc = dirichletbc(u_wall, bdofs_V_3, self.V.sub(0))
        upper_obstacle_bc = dirichletbc(u_wall, bdofs_V_4, self.V.sub(0))
        self.bc = [inlet_bc, wall_bc, lower_obstacle_bc, upper_obstacle_bc]

        # Test and trial functions: monolithic
        vq = TestFunction(self.V)
        (v, q) = split(vq)
        dup = TrialFunction(self.V)
        up = Function(self.V)
        (u, p) = split(up)

        # Variational forms
        F = (self.physical_parameters["nu"] * inner(grad(u), grad(v)) * dx
                + self.physical_parameters["rho"] * inner(grad(u) * u, v) * dx
                - inner(p, div(v)) * dx
                + inner(div(u), q) * dx)
        J = derivative(F, up, dup)

        self.F_form = form(F)
        self.J_form = form(J)
        self.obj_vec = create_vector(self.F_form)
        self._solution = up
        self.P = None
    
    def _create_snes_solution(self) -> PETSc.Vec:  
        """
        Create a PETSc.Vec to be passed to PETSc.SNES.solve.

        The returned vector will be initialized with the initial guess provided in `_solution`.
        """
        x = self._solution.vector.copy()
        with x.localForm() as _x, self._solution.vector.localForm() as _solution:
            _x[:] = _solution
        return x

    def _update_solution(self, x: PETSc.Vec):  
        """Update `self._solution` with data in `x`."""
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        with x.localForm() as _x, self._solution.vector.localForm() as _solution:
            _solution[:] = _x

    def _obj(self, solver: PETSc.SNES, x: PETSc.Vec) -> np.float64:
        """Compute the norm of the residual."""
        self._F_function(solver, x, self.obj_vec)
        return self.obj_vec.norm()  

    def _F_function(self, solver: PETSc.SNES, x: PETSc.Vec, F_vec: PETSc.Vec):
        """Assemble the residual."""
        self._update_solution(x)
        with F_vec.localForm() as F_vec_local:
            F_vec_local.set(0.0)
        assemble_vector(F_vec, self.F_form)
        apply_lifting(F_vec, [self.J_form], [self.bc], x0=[x], scale=-1.0)
        F_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(F_vec, self.bc, x, -1.0)

    def _J_function(self, solver: PETSc.SNES, x: PETSc.Vec, J_mat: PETSc.Mat, P_mat: PETSc.Mat):
        """Assemble the jacobian."""
        J_mat.zeroEntries()
        assemble_matrix(  
            J_mat, self.J_form, self.bc, diagonal=1.0)  
        J_mat.assemble()
        if self.P is not None:
            P_mat.zeroEntries()
            assemble_matrix(  
                P_mat, self.P, self.bc, diagonal=1.0)  
            P_mat.assemble()

    def _solve(self):
        # Create problem
        F_vec = create_vector(self.F_form)
        J_mat = create_matrix(self.J_form)

        # Solve
        solver = PETSc.SNES().create(self.mesh.comm)
        solver.setTolerances(max_it=100)
        solver.getKSP().setType("preonly")
        solver.getKSP().getPC().setType("lu")
        solver.getKSP().getPC().setFactorSolverType("mumps")
        solver.setObjective(self._obj)
        solver.setFunction(self._F_function, F_vec)
        solver.setJacobian(self._J_function, J=J_mat, P=None)
        # solver.setMonitor(lambda _, it, residual: print(it, residual))

        up_copy = self._create_snes_solution()
        solver.solve(None, up_copy)
        self._update_solution(up_copy)
        up_copy.destroy()
        solver.destroy()

    def solution(self):
        if not hasattr(self, "_solution"):
            raise RuntimeError("Solver has not been run yet.")
        (u_, p_) = self._solution.split()
        u_.name = "Velocity"
        p_.name = "Pressure"
        return (u_, p_)
    
    def _update_problem(self, params):
        self.create_mesh(params)
        self.define_problem()
    
    def quantity_of_interest(self, params):
        self._update_problem(params)
        self._solve()

        (u_, p_) = self.solution()
        n = -FacetNormal(self.mesh)
        dObs = Measure("ds", domain=self.mesh, subdomain_data=self.facet_tags, subdomain_id=(self.domain_markers["lower"], self.domain_markers["upper"]))
        u_t = inner(as_vector((n[1], -n[0])), u_)
        drag = form(2 / 0.1 * (self.physical_parameters["nu"] / self.physical_parameters["rho"] * inner(grad(u_t), n) * n[1] - p_ * n[0]) * dObs)
        lift = form(-2 / 0.1 * (self.physical_parameters["nu"] / self.physical_parameters["rho"] * inner(grad(u_t), n) * n[0] + p_ * n[1]) * dObs)
        
        drag_coefficient = assemble_scalar(drag)
        lift_coefficient = assemble_scalar(lift)

        return lift_coefficient

if __name__ == "__main__":
    # Define parameter space
    m = 8               # Number of parameters
    alpha = 10           # Oversampling factor
    n = m + 1           # Largest dimension we can handle
    k = 5               # Number of clusters 

    M = int(alpha * n * math.log(m))        # Number of samples

    dir = os.path.dirname(__file__)
    file = os.path.join(dir,"airfoilNavierStokes/samples.json")

    # If sampling file exists, load it otherwise create it
    if os.path.isfile(file):
        samples = utils.load(file)
        M = samples.M
    else:
        samples = clustering(M, m, k)
        samples.detect()
        samples.save(file)
    simulation = stationaryNavierStokes()
    # Check if samples already include values
    if not hasattr(samples, "_values"):
        # Assing values to the samples in the parameter space
        progress = tqdm.autonotebook.tqdm(desc="Evaluating Simulation", total=M)
        for i in range(M):
            try:
                value = simulation.quantity_of_interest(samples.extract(i))
            except:
                Warning("Error in the simulation. Continuing with next sample.")
            samples.assign_value(i, value)
            progress.update(1)
            samples.save(os.path.join(dir,"airfoilNavierStokes/samples.json"))
        progress.close()

    # Define test problem
    sample = np.random.uniform(-1, 1, m)
    print(simulation.quantity_of_interest(sample))
    simulation.save_solution(os.path.join(dir,"airfoilNavierStokes/solution.xdmf"), overwrite=True)
    
    # Define the functional
    cost = functional(m, simulation.quantity_of_interest)

    cost.interpolation(samples, order= 1, interpolation_method = "LS", clustering = True)
    cost.get_gradient_method("I")

    asfenicsx = ASFEniCSx(2, cost, samples)
    U, S = asfenicsx.random_sampling_algorithm()
    asfenicsx.plot_eigenvalues()
    print(U, S)

