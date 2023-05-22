# We consider the following Poisson Equation with parametrized, spatially varying coefficients.
# On the computational unit domain \Omega = [0,1]^2, let u = u(s,x) be the solution to
#       - \nabla_s \cdot (a(s,x) \nabla_s u(s,x)) = 1 in \Omega                                     (1)
#                                          u(s,x) = 0 on \Gamma_D (Dirichlet boundary)              (2)
#                                \nabla u \cdot n = 0 on \Gamma_N (Neumann boundary)                (3)
# where s is the vector of the spatial coordinates and x are the parameters. \Gamma_N is the right
# boundary of the spatial domain and \Gamma_D = \partial \Omega \setminus \Gamma_N.

import os

import numpy as np
import math
import numpy.linalg as la
import matplotlib.pyplot as plt

import ufl
import gmsh
import tqdm.autonotebook

from mpi4py import MPI
from dolfinx import fem
from dolfinx.fem import FunctionSpace
from dolfinx.io import gmshio, XDMFFile
from petsc4py.PETSc import ScalarType
from scipy.sparse.linalg import eigsh
import asfenicsx

gdim = 2
dir = os.path.dirname(__file__)

# Check if directory parametrizedPoisson exists if not create it
if not os.path.exists(os.path.join(dir,"parametrizedPoisson")):
    os.makedirs(os.path.join(dir,"parametrizedPoisson"))

# Create mesh
gmsh.initialize()
boundary_Dirichlet, boundary_Neumann = [], []
marker_dirichlet, marker_neumann = 1,2
rectangle = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, 1.0, 1.0, tag = 1)
gmsh.model.occ.synchronize()
volumes = gmsh.model.getEntities(dim=gdim)
gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], 1)
boundaries = gmsh.model.getBoundary(volumes, oriented=False)
for boundary in boundaries:
    center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
    if np.isclose(center_of_mass[0], 0.0):
        boundary_Dirichlet.append(boundary[1])
    elif np.isclose(center_of_mass[0], 1.0):
        boundary_Neumann.append(boundary[1])
    elif np.isclose(center_of_mass[1], 0.0):
        boundary_Dirichlet.append(boundary[1])
    elif np.isclose(center_of_mass[1], 1.0):
        boundary_Dirichlet.append(boundary[1])
gmsh.model.addPhysicalGroup(gdim-1, boundary_Dirichlet, marker_dirichlet)
gmsh.model.addPhysicalGroup(gdim-1, boundary_Neumann, marker_neumann)
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.02)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.02)
gmsh.option.setNumber("Mesh.Algorithm", 8)
gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
gmsh.option.setNumber("Mesh.RecombineAll", 1)
gmsh.model.mesh.generate(2)
gmsh.model.mesh.setOrder(1)
gmsh.write(os.path.join(dir,"parametrizedPoisson/mesh.msh"))

gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD

mesh, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=gdim)

num_nodes = mesh.geometry.x.shape[0]
print("Number of nodes: ", num_nodes)

# Create function space with standard P1 linear Lagrangian elements
V = FunctionSpace(mesh, ("CG", 1))

# Define homogeneous Dirichlet boundary conditions  on \Gamma_D 
tdim = mesh.topology.dim
fdim = tdim - 1

# Use only the Dirichlet boundary
dirichlet_dofs = fem.locate_dofs_topological(V, fdim, facet_markers.find(marker_dirichlet))
bc_D = fem.dirichletbc(ScalarType(0), dirichlet_dofs, V)

# Define variational problem, multiplying equation (1) with a test funciton v \in H^1_0(\Omega)
# and integrating by parts over the domain \Omega yields
#      \int_\Omega a(s,x) \nabla_s u(s,x) \cdot \nabla_s v(s,x) ds = \int_\Omega v(s,x) ds         (4)
# for all v \in H^1_0(\Omega). Here the boundary terms vanish due to the homogeneous Neumann
# boundary (3) for \Gamma_N and due to the compact support of v for \Gamma_D.

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# According to Constantine et al (2014) the log of the coefficients a = a(s,x) of the differential
# operator are given by a truncated Karhunen-Loeve (KL) expansion
#     log(a(s,x)) = sum_{i=1}^m x_i \gamma_i \phi_i(s)                                             (5)
# where the x_i are the independent, identically distributed standard normal random variables and
# the {\phi_i(s), \gamma_i} are the eigenpairs of the correlation operator
#       C(s,t) = exp(\beta^{-1} ||s-t||_1)
# with \beta > 0.

def calculate_eigenpairs(mesh, beta):
    vertices = mesh.geometry.x

    def correlation_operator(s, t, beta):
        # C(s,t) = exp(-\beta^{-1} ||s-t||_1)
        return np.exp(-la.norm(s-t, ord=1)/beta)
    
    # Calculate the correlation matrix for the mesh grid
    corr_mat = np.zeros((num_nodes, num_nodes))
    progress = tqdm.autonotebook.tqdm(desc="Constructing Correlation Matrix", total=(num_nodes**2-num_nodes)//2)
    for i in range(num_nodes):
        s = vertices[i,:tdim]
        for j in range(i,num_nodes):
            t = vertices[j,:tdim]
            # Evaluate only upper triangular part of the correlation matrix, because the correlation matrix is symmetric
            corr_mat[i, j] = corr_mat[j, i] = correlation_operator(s, t, beta)
            progress.update(1)
    progress.close()
    # Calculate the eigenpairs of the correlation matrix
    eigenvalues, eigenvectors = eigsh(corr_mat,k=m)

    # Sort eigenpairs in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]

    norm = la.norm(eigenvectors, axis=0)
    eigenvectors = eigenvectors/norm
    
    # Plot eigenvalues on a log axis
    plt.figure()
    plt.plot(eigenvalues)
    plt.yscale('log')
    plt.xlabel("Eigenvalue Index")
    plt.ylabel("Eigenvalue")
    plt.title("Eigenvalues of the Correlation Matrix")
    plt.savefig(os.path.join(dir,"parametrizedPoisson/eigenvalues.png"))
    plt.close()
    return (eigenvalues, eigenvectors)

def kl_expansion(x):
    log_a =np.zeros(num_nodes)
    for i in range(m):
        log_a += x[i]*eigenvalues[i]*eigenvectors[:,i]
    return np.exp(log_a)

# Set source term to f(x) = 1
f = fem.Function(V)
f.interpolate(lambda x: 0*x[0]+1)

# Set diffusion field coefficients to a(x) = 1
a = fem.Function(V)
a.name="Diffusion Field"
a.interpolate(lambda x: 0*x[0]+1)

linear = f * v * ufl.dx
bilinear = ufl.dot(a*ufl.grad(u), ufl.grad(v))*ufl.dx
problem = fem.petsc.LinearProblem(bilinear, linear, bcs=[bc_D],petsc_options={"ksp_type": "preonly", "pc_type": "lu"})


def solve_problem(x):
    
    # KL expansion based on eigenpairs of the correlation operator
    if x is not None:
        a.vector[:] = kl_expansion(x)
#    problem = fem.petsc.LinearProblem(bilinear, linear, bcs=[bc_D],petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()
    return uh

def store_data(uh, filename="parametrizedPoisson/solution.xdmf"):
        uh.name = "u"
        with XDMFFile(mesh.comm,filename, "w") as xdmf:
            xdmf.write_mesh(mesh)
            xdmf.write_function(uh)

def test_solver():
    real = fem.Function(V)
    real.interpolate(lambda x: (x[0]-0.5*x[0]**2)*np.sin(np.pi*x[1]))

    f.interpolate(lambda x: (1.0-0.5*math.pi**2*(x[0]-2.0)*x[0])*np.sin(np.pi*x[1]))

    uh = solve_problem(None)
    
    store_data(uh, "parametrizedPoisson/tests/solution.xdmf")
    store_data(real, "parametrizedPoisson/tests/analytical.xdmf")

    error_L2 = np.sqrt(mesh.comm.allreduce(fem.assemble_scalar(fem.form((uh - real)**2 * ufl.dx)), op=MPI.SUM))

    if mesh.comm.rank == 0:
        print(f"L2-error: {error_L2:.2e}")

    # Compute values at mesh vertices
    error_max = mesh.comm.allreduce(np.max(np.abs(uh.x.array-real.x.array)), op=MPI.MAX)
    error_mean = mesh.comm.allreduce(np.mean(np.abs(uh.x.array-real.x.array)), op=MPI.SUM)/num_nodes
    if mesh.comm.rank == 0:
        print(f"Error_max: {error_max:.2e}")
        print(f"Error_mean: {error_mean:.2e}")

# The quantity of interest is the mean integral over the right boundary of the domain
#                f(x) = \frac{1}{|\Gamma_N|} \int_{\Gamma_N} u(s,x) ds                             (6)
# where |\Gamma_N| is the length of the right boundary of the domain \Omega.
# We approximate the linear functional using the symmetric mass matrix M
#      f(x) \approx c^T M u(x)                                                                     (7)
# and the components of c correspod to the mesh nodes on \Gamma_N that are equal to one with 
# the rest equal to zero.

c=fem.Function(V)
c.interpolate(lambda x: np.isclose(x[0],1.0).astype(int))

def calculate_qof(x):
    uh = solve_problem(x)
    return fem.assemble_scalar(fem.form(ufl.inner(c,uh)*ufl.ds))

if __name__ == "__main__":
    # Validation of the solver using an unparametrized poisson problem
    test_solver()

    # Dimensions of parameter space
    m = 100
    M = 300

    # Set the parameter values
    samples = asfenicsx.Sampling(M,m)

    (eigenvalues, eigenvectors)=calculate_eigenpairs(mesh, 1)

    xdmf = XDMFFile(mesh.comm, "parametrizedPoisson/solutions.xdmf", "w")
    xdmf.write_mesh(mesh)
    progress = tqdm.autonotebook.tqdm(desc="Solving Problem", total=M)
    for i in range(M):
        uh=solve_problem(samples.extract(i))
        uh.name = "u"
        xdmf.write_function(uh,i+1)
        xdmf.write_function(a,i+1)
        progress.update(1)
    xdmf.close()
    progress.close()

    cost = asfenicsx.Functional(m, calculate_qof)
    cost.get_gradient_method('FD')

    active_subspace = asfenicsx.ASFEniCSx(20, cost, samples)

    U, S = active_subspace.random_sampling_algorithm(info=True)

    print(cost.number_of_calls)
    plt.figure()
    # Plot on a log axis
    plt.plot(S)
    plt.yscale('log')
    plt.show()

