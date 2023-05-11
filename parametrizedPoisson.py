# We consider the following Poisson Equation with parametrized, spatially varying coefficients.
# On the computational unit domain \Omega = [0,1]^2, let u = u(s,x) be the solution to
#       - \nabla_s \cdot (a(s,x) \nabla_s u(s,x)) = 1 in \Omega                                     (1)
#                                          u(s,x) = 0 on \Gamma_D (Dirichlet boundary)                (2)
#                                \nabla u \cdot n = 0 on \Gamma_N (Neumann boundary)         (3)
# where s is the vector of the spatial coordinates and x are the parameters. \Gamma_N is the right
# boundary of the spatial domain and \Gamma_D = \partial \Omega \setminus \Gamma_N.

import numpy as np
import scipy.linalg as la

import ufl
import tqdm.autonotebook

from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem import FunctionSpace
from petsc4py.PETSc import ScalarType

import asfenicsx

# Create mesh
domain = mesh.create_unit_square(MPI.COMM_WORLD, 50, 50, mesh.CellType.quadrilateral)

num_nodes = domain.geometry.x.shape[0]
print("Number of nodes: ", num_nodes)

# Create function space with standard P1 linear Lagrangian elements
V = FunctionSpace(domain, ("CG", 1))

# Define homogeneous Dirichlet boundary conditions  on \Gamma_D 
tdim = domain.topology.dim
fdim = tdim - 1
def boundary_Dirichlet(x):
    return np.logical_and(np.logical_or(np.isclose(x[0], 0.0),np.isclose(x[1], 0.0),np.isclose(x[1], 1.0)), np.logical_not(np.isclose(x[0], 1.0)))

# Use only the Dirichlet boundary
dirichlet_dofs = fem.locate_dofs_geometrical(V, boundary_Dirichlet)
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

def calculate_eigenpairs(domain, beta):
    vertices = domain.geometry.x
    num_nodes = np.size(vertices,0)

    def correlation_operator(s, t, beta):
        # C(s,t) = exp(\beta^{-1} ||s-t||_1)
        return np.exp(np.sum(np.abs(s-t))/beta)
    
    # Calculate the correlation matrix for the mesh grid
    corr_mat = np.zeros((num_nodes, num_nodes))
    progress = tqdm.autonotebook.tqdm(desc="Constructing Correlation Matrix", total=(num_nodes**2-num_nodes)//2)
    for i in range(num_nodes):
        s = vertices[i,:tdim]
        for j in range(i,num_nodes):
            progress.update(1)
            t = vertices[j,:tdim]
            # Evaluate only upper triangular part of the correlation matrix, because the correlation matrix is symmetric
            corr_mat[i, j] = corr_mat[j, i] = correlation_operator(s, t, beta)

    # Calculate the eigenpairs of the correlation matrix
    eigenvalues, eigenvectors = la.eigh(corr_mat)
    sort_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_indices]
    eigenvectors = eigenvectors[:, sort_indices]
    return (eigenvalues, eigenvectors)

# Dimensions of parameter space
m = 50
M=1

(eigenvalues, eigenvectors)=calculate_eigenpairs(domain, 1.0)

def kl_expansion(x):
    log_a = np.zeros(num_nodes)
    for i in range(m):
        log_a += x[i]*eigenvalues[i]*eigenvectors[:,i]
    return log_a

f = fem.Constant(domain, ScalarType(1))
linear = f * v * ufl.dx
a = fem.Function(V)
bilinear = ufl.dot(a*ufl.grad(u), ufl.grad(v))*ufl.dx
  
def solve_problem(x):
    # KL expansion based on eigenpairs of the correlation operator
    a.interpolate(np.exp(kl_expansion(x)))
    
    problem = fem.petsc.LinearProblem(bilinear, linear, bcs=[bc_D],petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()
    return uh

def plot_solution(uh):
    import pyvista
    from dolfinx import plot
    u_topology, u_cell_types, u_geometry = plot.create_vtk_mesh(V)
    u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    u_grid.point_data["u"] = uh.x.array.real
    u_grid.set_active_scalars("u")
    u_plotter = pyvista.Plotter()
    u_plotter.add_mesh(u_grid, show_edges=True)
    u_plotter.view_xy()
    if not pyvista.OFF_SCREEN:
        u_plotter.show()

# The quantity of interest is the mean integral over the right boundary of the domain
#                f(x) = \frac{1}{|\Gamma_N|} \int_{\Gamma_N} u(s,x) ds                             (6)
# where |\Gamma_N| is the length of the right boundary of the domain \Omega.
# We approximate the linear functional using the symmetric mass matrix M
#      f(x) \approx c^T M u(x)                                                                     (7)
# and the components of c correspod to the mesh nodes on \Gamma_N that are equal to one with 
# the rest equal to zero.

def quantity_of_interest(uh):

    M = fem.assemble_matrix(ufl.inner(u, v)*ufl.dx)

    # TODO: Create vector c with ones on the right boundary and zeros elsewhere
    c = np.zeros()
    return ufl.dot(c, ufl.dot(M, uh.x))

if __name__ == "__main__":
    # Create random KL expansion
    sampling = asfenicsx.Sampling(M,m)

    # Solve the problem for each sample
    progress = tqdm.autonotebook.tqdm(desc="Solving PDE", total=M)
    for i in range(M):
        progress.update(1)
        uh = solve_problem(sampling.samples[i,:])
        print(uh.x.array.real)

    plot_solution(uh)
