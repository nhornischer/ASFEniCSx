import os
import numpy as np
import gmsh
import tqdm.autonotebook

from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.fem import (Constant, Function, FunctionSpace, 
                         assemble_scalar, dirichletbc, form, locate_dofs_topological, set_bc)
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector, 
                               create_vector, create_matrix, set_bc)
from dolfinx.io import XDMFFile

from ufl import VectorElement, FiniteElement, FacetNormal, Measure
from ufl import TrialFunction, TestFunction, dot, inner, grad, rhs, lhs, div, nabla_grad, dx, as_vector

import dolfinx

# Import mesh in dolfinx
gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD

np.set_printoptions(formatter={'float_kind': "{:.3f}".format})

dir = os.path.dirname(__file__)
if not os.path.exists("airfoilNavierStokes"):
    os.mkdir("airfoilNavierStokes")

dir = os.path.join(dir, "airfoilNavierStokes")

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    DEFINE THE COMPUTATIONAL MESH
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

lc = 0.1
num_airfoil_refinement = 1000   

L = 8
H = 2
gdim = 2

"""
    Defining the shape of the airfoil using Bézier Curves
    Geometrical parameters of the airfoil according to Xiaoqiang et al 
"""
# Define parametes
m = 8
inflow_marker, outflow_marker, wall_marker, upper_obstacle_marker, lower_obstacle_marker = 1, 2, 3, 4, 5
ranges= [[0.010, 0.960], [0.030, 0.970], [-0.074, 0.247], [-0.102, 0.206],[0.2002, 0.4813], [0.0246,0.3227], [0.1750,1.4944], [0.1452,4.8724]]
def calculate_airfoil_parametrisation(params, conversion = True):

    # Denormalise parameters from range (-1, 1) based on max, min for each value
    if conversion:
        for i in range(len(params)):
            params[i] = (params[i] + 1) / 2 * (ranges[i][1] - ranges[i][0]) + ranges[i][0]
    

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

    c_1, c_2 = 0.5, 0.5             # Coefficients of camber-line-abscissa parameter equation
    c_3, c_4 = 0.1, 0.05             # Coefficients of camber-line.ordinate parameter equation
    coeff_thick = [0.2969,-0.126,-0.3516,0.2843,-0.1036]

    # Bézier curves with control parameter k \in [0,1]
    x_c = lambda k: 3 * c_1 * k * (1 - k)**2 + 3 * c_2 * (1-k) * k**2 + k**3    # Camber-line-abscissa
    y_c = lambda k: 3 * c_3 * k * (1 - k)**2 + 3 * c_4 * (1-k) * k**2           # Camber-line-abscissa

    # Thickness equation for a 6% thick airfoil
    thickness = lambda x: (coeff_thick[0] * np.sqrt(x) + coeff_thick[1] * x + coeff_thick[2] * x**2 + coeff_thick[3] * x**3 + coeff_thick[4] * x**4)

    # Position of the airfoil in the computational domain defined by the coordinates of the leading edge
    leading_edge_x = np.max([L/8, 0.5])
    leading_edge_y = H/2

    # Upper and lower surface of the airfoil
    x_u = x_l = lambda k: leading_edge_x + x_c(k)
    y_u = lambda k: leading_edge_y + y_c(k) + 0.5 * thickness(x_c(k))
    y_l = lambda k: leading_edge_y + y_c(k) - 0.5 * thickness(x_c(k))

    thickness_max = np.max([thickness(x_c(k)) for k in np.linspace(0,1,num_airfoil_refinement)])
    return (x_u, x_l, y_u, y_l, thickness_max)

"""
    Meshing the airfoil using gmsh
"""
def create_mesh(x_u, x_l, y_u, y_l, thickness_max):
    print("Creating mesh...")
    gmsh.initialize()
    if mesh_comm.rank == gmsh_model_rank:
        rectangle = gmsh.model.occ.addRectangle(0,0,0, L, H, tag=3)
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
        fluid = gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, obstacle)])
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
        gmsh.model.mesh.generate(gdim)
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

    if mesh_comm.rank == gmsh_model_rank:
        # Get all surfaces of the mesh
        surfaces = gmsh.model.getEntities(dim=gdim)
        boundaries = gmsh.model.getBoundary(surfaces, oriented=False)

        gmsh.model.addPhysicalGroup(surfaces[0][0], [surfaces[0][1]], fluid_marker)
        gmsh.model.setPhysicalName(surfaces[0][0], fluid_marker, "Fluid")


        for boundary in boundaries:
            center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
            if np.allclose(center_of_mass, [0, H/2, 0]):
                inflow.append(boundary[1])
            elif np.allclose(center_of_mass, [L, H/2, 0]):
                outflow.append(boundary[1])
            elif np.allclose(center_of_mass, [L/2, H, 0]):
                wall.append(boundary[1])
            elif np.allclose(center_of_mass, [L/2, 0, 0]):
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
        gmsh.model.addPhysicalGroup(gdim-1, wall, wall_marker)
        gmsh.model.setPhysicalName(gdim-1, wall_marker, "wall")
        gmsh.model.addPhysicalGroup(gdim-1, inflow, inflow_marker)
        gmsh.model.setPhysicalName(gdim-1, inflow_marker, "inflow")
        gmsh.model.addPhysicalGroup(gdim-1, outflow, outflow_marker)
        gmsh.model.setPhysicalName(gdim-1, outflow_marker, "outflow")
        gmsh.model.addPhysicalGroup(gdim-1, lower_obstacle, lower_obstacle_marker)
        gmsh.model.setPhysicalName(gdim-1, lower_obstacle_marker, "lower_obstacle")
        gmsh.model.addPhysicalGroup(gdim-1, upper_obstacle, upper_obstacle_marker)
        gmsh.model.setPhysicalName(gdim-1, upper_obstacle_marker, "upper_obstacle")

        gmsh.model.occ.remove(gmsh.model.getEntities(dim=gdim-1))
        gmsh.model.occ.remove(gmsh.model.getEntities(dim=gdim))
        gmsh.model.occ.remove(gmsh.model.getEntities(dim=0))

        gmsh.write(os.path.join(dir,"mesh.msh"))

    # gmsh.fltk.run()

    mesh, cell_tags, facet_tags = dolfinx.io.gmshio.read_from_msh(os.path.join(dir,"mesh.msh"), mesh_comm, gmsh_model_rank, gdim=gdim)

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, os.path.join(dir,"mesh.xdmf"),"w") as mesh_file_xdmf:
        mesh_file_xdmf.write_mesh(mesh)
        mesh_file_xdmf.write_meshtags(cell_tags)
        mesh_file_xdmf.write_meshtags(facet_tags)

    return mesh, cell_tags, facet_tags

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Define Navier Stokes problem
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
t_0 = 0
T = 6                     # Final time
dt = 1/200                # Time step size
num_steps = int(T/dt)
# Define boundary conditions
class InletVelocity():
    def __init__(self, t):
        self.t = t

    def __call__(self, x):
        values = np.zeros((gdim, x.shape[1]),dtype=PETSc.ScalarType)
        k = 2
        t_const = 2 
        time_scale_factor = 1 / (1 + np.exp(-2 * k * (self.t - t_const)))
        print(time_scale_factor)
        values[0] = 8 * 1.5 * time_scale_factor * x[1] * (H - x[1])/(H**2)
        return values

def solve_NavierStokes(mesh, cell_tags, facet_tags):
    print("Solving")
    # Define constants of the model problem (wraped as constants to reduce compilation time, because values can be changed)
    k = Constant(mesh, PETSc.ScalarType(dt))        
    mu = Constant(mesh, PETSc.ScalarType(0.001))  # Dynamic viscosity
    rho = Constant(mesh, PETSc.ScalarType(1))     # Density

    # Define ansatz spaces (piecewise quatratic for the velocity space V and piecewise linear for the pressure space Q, in order to increase stability)
    # For the velocity we use a vector valued function space to capture the velocity in each direction, for the pressure a scalar valued function space is used.
    v_cg2 = VectorElement("CG", mesh.ufl_cell(), 2)
    s_cg1 = FiniteElement("CG", mesh.ufl_cell(), 1)
    V = FunctionSpace(mesh, v_cg2)
    Q = FunctionSpace(mesh, s_cg1)

    fdim = mesh.topology.dim - 1

    # Inlet
    u_inlet = Function(V)
    inlet_velocity = InletVelocity(t_0)
    u_inlet.interpolate(inlet_velocity)
    bcu_inflow = dirichletbc(u_inlet, locate_dofs_topological(V, fdim, facet_tags.find(inflow_marker)))
    # Walls
    u_nonslip = np.array((0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
    bcu_walls = dirichletbc(u_nonslip, locate_dofs_topological(V, fdim, facet_tags.find(wall_marker)), V)
    # Obstacle
    bcu_lower_obstacle = dirichletbc(u_nonslip, locate_dofs_topological(V, fdim, facet_tags.find(lower_obstacle_marker)), V)
    bcu_upper_obstacle = dirichletbc(u_nonslip, locate_dofs_topological(V, fdim, facet_tags.find(upper_obstacle_marker)), V)
    bcu = [bcu_inflow, bcu_lower_obstacle, bcu_upper_obstacle, bcu_walls]
    # Outlet
    bcp_outlet = dirichletbc(PETSc.ScalarType(0), locate_dofs_topological(Q, fdim, facet_tags.find(outflow_marker)), Q)
    bcp = [bcp_outlet]

    # Define the ansatz and test functions with the text time-iterate as Function Object
    u = TrialFunction(V)
    v = TestFunction(V)
    u_ = Function(V)
    u_.name = "u"
    u_s = Function(V)
    u_n = Function(V)
    u_n1 = Function(V)
    p = TrialFunction(Q)
    q = TestFunction(Q)
    p_ = Function(Q)
    p_.name = "p"
    phi = Function(Q)

    # Define variational formulation for the splitting scheme
    f = Constant(mesh, PETSc.ScalarType((0,0)))
    F1 = rho / k * dot(u - u_n, v) * dx 
    F1 += inner(dot(1.5 * u_n - 0.5 * u_n1, 0.5 * nabla_grad(u + u_n)), v) * dx
    F1 += 0.5 * mu * inner(grad(u + u_n), grad(v))*dx - dot(p_, div(v))*dx
    F1 += dot(f, v) * dx
    a1 = form(lhs(F1))
    L1 = form(rhs(F1))
    A1 = create_matrix(a1)
    b1 = create_vector(L1)

    a2 = form(dot(grad(p), grad(q))*dx)
    L2 = form(-rho / k * dot(div(u_s), q) * dx)
    A2 = assemble_matrix(a2, bcs=bcp)
    A2.assemble()
    b2 = create_vector(L2)

    a3 = form(rho * dot(u, v)*dx)
    L3 = form(rho * dot(u_s, v)*dx - k * dot(nabla_grad(phi), v)*dx)
    A3 = assemble_matrix(a3)
    A3.assemble()
    b3 = create_vector(L3)

    # Solver for step 1
    # MUMPS
    solver1 = PETSc.KSP().create(mesh.comm)
    solver1.setOperators(A1)
    solver1.setType(PETSc.KSP.Type.BCGS)
    pc1 = solver1.getPC()
    pc1.setType(PETSc.PC.Type.JACOBI)

    # Solver for step 2
    solver2 = PETSc.KSP().create(mesh.comm)
    solver2.setOperators(A2)
    solver2.setType(PETSc.KSP.Type.MINRES)
    pc2 = solver2.getPC()
    pc2.setType(PETSc.PC.Type.HYPRE)
    pc2.setHYPREType("boomeramg")

    # Solver for step 3
    solver3 = PETSc.KSP().create(mesh.comm)
    solver3.setOperators(A3)
    solver3.setType(PETSc.KSP.Type.CG)
    pc3 = solver3.getPC()
    pc3.setType(PETSc.PC.Type.SOR)

    # Calculate lift and drag of the aerofoil
    n = -FacetNormal(mesh) # Normal pointing out of obstacle
    dObs = Measure("ds", domain=mesh, subdomain_data=facet_tags, subdomain_id=(lower_obstacle_marker, upper_obstacle_marker))
    u_t = inner(as_vector((n[1], -n[0])), u_)
    drag = form(2 / 0.1 * (mu / rho * inner(grad(u_t), n) * n[1] - p_ * n[0]) * dObs)
    lift = form(-2 / 0.1 * (mu / rho * inner(grad(u_t), n) * n[0] + p_ * n[1]) * dObs)
    if mesh.comm.rank == 0:
        C_D = np.zeros(num_steps, dtype=PETSc.ScalarType)
        C_L = np.zeros(num_steps, dtype=PETSc.ScalarType)
        t_u = np.zeros(num_steps, dtype=np.float64)
        v_max=np.zeros(num_steps, dtype=np.float64)
        v_inlet_max=np.zeros(num_steps, dtype=np.float64)
        p_max=np.zeros(num_steps, dtype=np.float64)

    # Time-stepping scheme
    # Solve the time-dependent Problem
    t = t_0
    storage_id = int(np.random.uniform(0,100000,1))
    xdmf = XDMFFile(mesh.comm, os.path.join(dir,f"airfoil_{storage_id}.xdmf"), "w")
    xdmf.write_mesh(mesh)
    xdmf.write_function(u_, t)
    xdmf.write_function(p_, t)
    progress = tqdm.autonotebook.tqdm(desc="Solving PDE", total=num_steps)
    export_indices=np.arange(0, num_steps+1,int(1/dt/50))
    
    for i in range(num_steps):
        # Update current time step
        t += dt
        # Update inlet velocity
        inlet_velocity.t = t
        u_inlet.interpolate(inlet_velocity)

        # Step 1: Tentative velocity step
        A1.zeroEntries()
        assemble_matrix(A1, a1, bcs=bcu)
        A1.assemble()
        with b1.localForm() as loc:
            loc.set(0)
        assemble_vector(b1, L1)
        apply_lifting(b1, [a1], [bcu])
        b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b1, bcu)
        solver1.solve(b1, u_s.vector)
        u_s.x.scatter_forward()

        # Step 2: Pressure corrrection step
        with b2.localForm() as loc:
            loc.set(0)
        assemble_vector(b2, L2)
        apply_lifting(b2, [a2], [bcp])
        b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b2, bcp)
        solver2.solve(b2, phi.vector)
        phi.x.scatter_forward()

        p_.vector.axpy(1, phi.vector)
        p_.x.scatter_forward()

        # Step 3: Velocity correction step
        with b3.localForm() as loc:
            loc.set(0)
        assemble_vector(b3, L3)
        b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        solver3.solve(b3, u_.vector)
        u_.x.scatter_forward()

        # Update variable with solution form this time step
        with u_.vector.localForm() as loc_, u_n.vector.localForm() as loc_n, u_n1.vector.localForm() as loc_n1:
            loc_n.copy(loc_n1)
            loc_.copy(loc_n)

        # Save solution to file (xdmf)
        if i in export_indices:
            xdmf.write_function(u_, t)
            xdmf.write_function(p_, t)

        # Compute physical quantities
        # For this to work in paralell, we gather contributions from all processors
        # to processor zero and sum the contributions. 
        drag_coeff = mesh.comm.gather(assemble_scalar(drag), root=0)
        lift_coeff = mesh.comm.gather(assemble_scalar(lift), root=0)
        v_max_entries = mesh.comm.gather(np.max(u_.vector.array), root=0)
        v_inlet_max_entries = mesh.comm.gather(np.max(u_inlet.vector.array), root=0)
        p_max_entries = mesh.comm.gather(np.max(p_.vector.array), root=0)
        if mesh.comm.rank == 0:
            t_u[i] = t
            C_D[i] = sum(drag_coeff)
            C_L[i] = sum(lift_coeff)
            v_max[i] = max(v_max_entries)
            v_inlet_max[i] = max(v_inlet_max_entries)
            p_max[i] = max(p_max_entries)
            if v_max[i] > 1e3:
                raise ValueError("Velocity is too high, simulation is unstable")
        progress.update(1)

    progress.close()

    if mesh.comm.rank == 0:
        import matplotlib.pyplot as plt
        if not os.path.exists(os.path.join(dir,"figures")):
            os.mkdir(os.path.join(dir,"figures"))
        fig,ax = plt.subplots()
        ax.plot(t_u,C_L, label="Lift")
        ax.plot(t_u,C_D, label="Drag")
        ax.set_xlabel("Time")
        ax.set_ylabel("Coefficient")
        ax.legend()
        ax2 = ax.twinx()
        ax2.plot(t_u,v_max,linestyle = ":", label="Max velocity")
        ax2.plot(t_u,v_inlet_max,linestyle = ":", label="Max inlet velocity")
        ax2.legend()
        ax2.set_ylabel("Max velocity (dotted)")
        
        fig.savefig(os.path.join(dir,f"./figures/Drag_Lift_coefficients_{storage_id}.pdf"))
        plt.figure()
        plt.plot(t_u,p_max)
        plt.xlabel("Time")
        plt.ylabel("Max pressure")  
        plt.savefig(os.path.join(dir,f"./figures/Max_pressure_{storage_id}.pdf"))
        return C_D[-1], C_L[-1]
    else: 
        return 0,0

def quantity_of_interest(x_params):
    x_u, x_l, y_u, y_l, thickness_max = calculate_airfoil_parametrisation(x_params)
    print("Done")
    mesh, cell_tags, facet_tags = create_mesh(x_u, x_l, y_u, y_l, thickness_max)
    drag, lift = solve_NavierStokes(mesh, cell_tags, facet_tags)
    if mesh.comm.rank == 0:
        print("Lift: ", lift)
        print("Drag: ", drag)
        print("Lift/Drag: ", lift/drag)
        return lift/drag


if __name__ == "__main__":
    x_params = [0.5, 0.5, 0.1, 0.05, 0.25, 0.2, 0.8, 0.3]
    ranges= [[0.2002, 0.4813], [0.0246,0.3227], [0.1750,1.4944], [0.1452,4.8724]]
    x_u, x_l, y_u, y_l, thickness_max = calculate_airfoil_parametrisation(x_params, conversion= False)

    # import matplotlib.pyplot  as plt
    # fig,ax = plt.subplots()
    # x_u_data = [x_u(k) for k in np.linspace(0,1,1000)]
    # x_l_data = [x_l(k) for k in np.linspace(0,1,1000)]
    # y_u_data = [y_u(k) for k in np.linspace(0,1,1000)]
    # y_l_data = [y_l(k) for k in np.linspace(0,1,1000)]
    # ax.plot(x_u_data,y_u_data, label="Upper")
    # ax.plot(x_l_data,y_l_data, label="Lower")
    # ax.legend()
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_ylim(0.5,1.5)
    # ax.set_xlim(0.5, 2.5)
    # ax.set_aspect('equal', adjustable='box')
    # plt.show()
    # exit()

    mesh, cell_tags, facet_tags = create_mesh(x_u, x_l, y_u, y_l, thickness_max)

    drag, lift = solve_NavierStokes(mesh, cell_tags, facet_tags)

    print("Lift: ", lift)
    print("Drag: ", drag)
