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

from ASFEniCSx.sampling import Sampling
from ASFEniCSx.functional import Functional
from ASFEniCSx.asfenicsx import ASFEniCSx
from ASFEniCSx.FEniCSxSim import FEniCSxSim


gdim = 2
dir = os.path.dirname(__file__)

# Check if directory parametrizedPoisson exists if not create it
if not os.path.exists(os.path.join(dir,"parametrizedPoisson")):
    os.makedirs(os.path.join(dir,"parametrizedPoisson"))

class ParametrizedPoisson(FEniCSxSim):
    def __init__(self, beta): 
        super().__init__()  
        self.domain_markers = {"marker_Dirichlet": 1, "marker_Neumann": 2}
        self.physical_params = {"beta": beta}

    def create_mesh(self):
        # Create mesh
        gmsh_model_rank = 0
        gmsh.initialize()
        boundary_Dirichlet, boundary_Neumann = [], []
        marker_dirichlet, marker_neumann = self.domain_markers["marker_Dirichlet"], self.domain_markers["marker_Neumann"]
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

        self.mesh, self.cell_markers, self.facet_markers = gmshio.model_to_mesh(gmsh.model, self.comm, gmsh_model_rank, gdim=gdim)
        gmsh.finalize()

    def define_problem(self):
        """
        Calculate Eigendecomposition of the correlation matrix
        """
        tdim = self.mesh.topology.dim
        fdim = tdim - 1

        beta = self.physical_params["beta"]
        num_nodes = self.mesh.geometry.x.shape[0]
        vertices = self.mesh.geometry.x

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
        
        self.eigenvalues = eigenvalues  
        self.eigenvectors = eigenvectors

        # Plot eigenvalues on a log axis
        plt.figure()
        plt.plot(eigenvalues)
        plt.yscale('log')
        plt.xlabel("Eigenvalue Index")
        plt.ylabel("Eigenvalue")
        plt.title("Eigenvalues of the Correlation Matrix")
        plt.savefig(os.path.join(dir,"parametrizedPoisson/Correlation_eigenvalues.png"))
        plt.close("all")

        """
        Define the problem
        """
        V = FunctionSpace(self.mesh, ("CG", 1))
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        self.V = V

        # Use only the Dirichlet boundary
        dirichlet_dofs = fem.locate_dofs_topological(V, fdim, self.facet_markers.find(self.domain_markers["marker_Dirichlet"]))
        bc_D = fem.dirichletbc(ScalarType(0), dirichlet_dofs, V)


        # Set source term to f(x) = 1
        f = fem.Function(V)
        f.interpolate(lambda x: 0*x[0]+1)

        # Set diffusion field coefficients to a(x) = 1
        self._a = fem.Function(V)
        self._a.name="Diffusion Field"
        self._a.interpolate(lambda x: 0*x[0]+1)

        linear = f * v * ufl.dx
        bilinear = ufl.dot(self._a*ufl.grad(u), ufl.grad(v))*ufl.dx
        self.problem = fem.petsc.LinearProblem(bilinear, linear, bcs=[bc_D],petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

    def _update_problem(self, params):
        num_nodes = self.mesh.geometry.x.shape[0]
        log_a =np.zeros(num_nodes)
        for i in range(m):
            log_a += params[i]*self.eigenvalues[i]*self.eigenvectors[:,i]
        self._a.vector[:] = np.exp(log_a)

    def _solve(self):
        self._solution = self.problem.solve()

    def quantity_of_interest(self, params):
        self._update_problem(params)
        self._solve()
        # The quantity of interest is the mean integral over the right boundary of the domain
        #                f(x) = \frac{1}{|\Gamma_N|} \int_{\Gamma_N} u(s,x) ds                             (6)
        # where |\Gamma_N| is the length of the right boundary of the domain \Omega.
        # We approximate the linear functional using the symmetric mass matrix M
        #      f(x) \approx c^T M u(x)                                                                     (7)
        # and the components of c correspod to the mesh nodes on \Gamma_N that are equal to one with 
        # the rest equal to zero.
        c=fem.Function(self.V)
        c.interpolate(lambda x: np.isclose(x[0],1.0).astype(int))
        return fem.assemble_scalar(fem.form(ufl.inner(c,self._solution)*ufl.ds))

if __name__ == "__main__":
    # Dimensions of parameter space
    m = 100
    M = 300

    simulation = ParametrizedPoisson(0.1)
    simulation.create_mesh()
    simulation.define_problem()

    # Create test evaluation
    sample = np.random.uniform(-1,1, m)
    print(simulation.quantity_of_interest(sample))
    simulation.save_solution(os.path.join(dir,"parametrizedPoisson/solution.xdmf"), overwrite=True)

    # TODO: Check if the simulation is correct especially the correlation matrix

    # Set the parameter values
    samples = Sampling(M,m)
    samples.standard_gaussian()
    progress = tqdm.autonotebook.tqdm(desc="Solving Problem", total=M)
    for i in range(M):
        value = simulation.quantity_of_interest(samples.extract(i))
        samples.assign_value(i, value)
        progress.update(1)
    progress.close()

    cost = Functional(m, simulation.quantity_of_interest)

    active_subspace = ASFEniCSx(20, cost, samples)

    U, S = active_subspace.estimation()
    active_subspace.plot_eigenvalues(os.path.join(dir,"parametrizedPoisson/eigenvalues.pdf"))

