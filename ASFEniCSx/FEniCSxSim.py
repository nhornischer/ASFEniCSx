import os

from mpi4py import MPI
from abc import ABC, abstractmethod

class FEniCSxSim(ABC):
    """ Abstract Class for an arbitrary FEniCSx simulation

    Attributes:
    public:
        comm (MPI.COMM_WORLD): MPI communicator
        V (dolfinx.fem.FunctionSpace): Function space of the simulation
        mesh (dolfinx.mesh.Mesh): Mesh of the simulation
        _solution (dolfinx.fem.Function): Solution of the simulation
        _problem (dolfinx.fem.LinearProblem): Problem of the simulation
    """
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        pass

    @abstractmethod
    def quantity_of_interest(self, params):
        """ Calculates the quantity of interest of the simulation

        Args:
            params (numpy.ndarray): Parameters of the simulation
            
        Returns:
            float: Quantity of interest
        """
        pass

    @abstractmethod
    def create_mesh(self):
        pass

    @abstractmethod
    def define_problem(self):
        pass

    @abstractmethod
    def _update_problem(self, params):
        pass

    @abstractmethod
    def _solve(self):
        pass


    def save_mesh(self, filename = "mesh.xdmf", overwrite = False):
        """Saves the mesh to a file.

        Args:
            filename (str, optional): Name of the file. Defaults to "mesh.xdmf".
            overwrite (bool, optional): Overwrite the file if it already exists. Defaults to False.

        Raises:
            ValueError: If the mesh is not defined.
            FileExistsError: If the file already exists and overwrite is set to False.
        """
        from dolfinx.io import XDMFFile
        if not hasattr(self, "mesh"):
            raise ValueError("Mesh not defined. Create or load a mesh first.")
        dir = os.path.dirname(__file__)
        # Check if a file already exists
        if os.path.isfile(os.path.join(dir,filename)) and not overwrite:
            raise FileExistsError("File already exists. Set overwrite to True to overwrite the file.")
        with XDMFFile(MPI.COMM_WORLD, os.path.join(dir,filename),"w") as mesh_file_xdmf:
            mesh_file_xdmf.write_mesh(self.mesh)
            if hasattr(self, "cell_tags"):
                mesh_file_xdmf.write_meshtags(self.cell_tags)
            if hasattr(self, "facet_tags"):
                mesh_file_xdmf.write_meshtags(self.facet_tags)

    def save_solution(self, filename = "solution.xdmf", index = 0, overwrite = False):
        """Saves the solution to a file.

        Args:
            filename (str, optional): Name of the file. Defaults to "solution.xdmf".
            index (int, optional): Index of the solution to be saved. Defaults to 0.
            overwrite (bool, optional): Overwrite the file if it already exists. Defaults to False.

        Raises:
            ValueError: If the solution is not defined.
            ValueError: If the mesh is not defined.
            FileExistsError: If the file already exists and overwrite is set to False.
        """
        from dolfinx.io import XDMFFile
        if not hasattr(self, "solution"):
            raise ValueError("Solution not defined. Solve the problem first.")
        if not hasattr(self, "mesh"):
            raise ValueError("Mesh not defined. Create or load a mesh first.")
        if os.path.isfile(filename) and not overwrite:
            raise FileExistsError("File already exists. Set overwrite to True to overwrite the file.")
        _solutions = self.solution()
        xdmf = XDMFFile(self.mesh.comm, filename, "w")
        xdmf.write_mesh(self.mesh)
        for _solution in _solutions:
                xdmf.write_function(_solution, index)
        xdmf.close()

    def solution(self):
        """Returns a list of all the solution of the problem
        
        Returns:
            list: List of all the solutions as dolfinx.Function
        """
        sol = self._solution
        sol.name = "Solution"
        return [sol]