import numpy as np

from ASFEniCSx.utils import debug_info
from ASFEniCSx.sampling import sampling
from ASFEniCSx.functional import functional

class ASFEniCSx:
    """Class for constructing the active subspace in FeniCSx based on Constantine et al. 

    The class is based on the paper by Constantine et al. The class is constructed
    to be used with the FeniCSx library and requires a functional and a sampling object.
    It is important to know that the eigenvectors of the active subspace can be in the direction
    of positive impact on the cost function or negative impact. This is not known beforehand and
    is therefore not specified in the class. The user must therefore check the direction of the
    eigenvectors and change the sign if needed.

    Attributes:
    public:
        k (int): Number of eigenvalues of interest
        function (functional): functional describing the quantity of interest
        samples (sampling): sampling object containing the samples
        eigenvalues (numpy.ndarray): Eigenvalues of the covariance matrix (if created)
        eigenvectors (numpy.ndarray): Eigenvectors of the covariance matrix (if created)

    Methods:
    public:
        evaluate_gradients(info : bool, optional) : Evaluates the gradients of the function at the samples
        covariance(info : bool, optional): Approximates the covariance matrix of the gradient of the function
        random_sampling_algorithm() : Performs the random sampling algorithm to construct the active subspace
        partition(n : int) : Partitions the active subspace into the active and inactive subspace
        bootstrap(n : int, info : bool, optional) : Performs the bootstrap algorithm to estimate the error
        calculate_eigenpairs(matrix : np.ndarray) : Calculates the eigenpairs of the given matrix
        plot_eigenvalues() : Plots the eigenvalues of the covariance matrix
        plot_subspace() : Plots distance of the active subspace using bootstrap
    Example:
        >>> from ASFEniCSx import ASFEniCSx, sampling, functional
        >>> def f(x): return x[0]**2 + x[1]**2
        >>> def dfdx(x): return [2*x[0], 2*x[1]]
        >>> samples = sampling(100, 2)
        >>> function = functional(2, f)
        >>> function.get_derivative(dfdx)                           # Optional but sets the derivative of the function to the analytical solution
        >>> asfenicsx = ASFEniCSx(1, function, samples)
        >>> U, S = asfenicsx.random_sampling_algorithm()

    Version:
        0.1
    Contributors:
        Niklas Hornischer (nh605@cam.ac.uk)
    """

    def __init__(self, k : int, function : functional, samples : sampling, debug = True):
        """Constructor for the ASFEniCSx class

        Args:
            k (int): Number of eigenvalues of interest
            function (functional): functional describing the quantity of interest
            samples (sampling): sampling object containing the samples
            debug (bool, optional): If True, debug information is printed. Defaults to False.

        Raises:
            ValueError: If n is larger than the number of dimensions of the parameter space
        """
        assert k <= samples.m, "n must be smaller than the number of dimensions of the parameter space"
        self.k = k
        self.function = function
        self.samples = samples
        self._debug = debug

    def eigenvalues(self):
        """Returns the eigenvalues of the covariance matrix

        Returns:
            np.ndarray: Eigenvalues of the covariance matrix
        """
        if not hasattr(self, 'eigenvalues'):
            raise ValueError("Eigenvalues not calculated yet. Run the random sampling algorithm first.")
        return np.copy(self._eigenvalues)

    def evaluate_gradients(self, **kwargs):
        """Evaluates the gradients of the function at the samples

        Args:
            
        Returns:
            np.ndarray: Matrix containing the gradients of the function at the samples in the rows
        """

        # Check if additional arguments are given
        debug_info(self._debug, "Evaluating gradients for active subspace construction")
        gradients = np.zeros([self.samples.M, self.samples.m])
        for i in range(self.samples.M):
            gradients[i] = self.function.gradient(self.samples.extract(i), self.samples, **kwargs)
        self.gradients = gradients

        # Normalize the gradients accroding to the chain rule with the bounds from the sampling space to the range [-1, 1]
        if hasattr(self.samples, '_bounds'):
            for i in range(self.samples.M):
                for j in range(self.samples.m):
                    gradients[i,j] = gradients[i,j] * (self.samples._bounds[j,1] - self.samples._bounds[j,0]) / 2
        return gradients

    def covariance(self, gradients : np.ndarray):
        """Approximates the covariance matrix of the gradient of the function

        The calculation of the gradient is defined directly in the functional.
        The covariance matrix is approximated by the outer product of the gradient.
        
        Args:
            gradients (numpy.ndarray): Matrix containing the gradients of the function at the samples in the rows

        Returns:
            np.ndarray: Approximated covariance matrix with dimensions m x m    
        """
        covariance = np.zeros([self.samples.m, self.samples.m])
        for i in range(self.samples.M):
            covariance += np.outer(gradients[i,:], gradients[i,:])
        covariance = covariance / self.samples.M

        return covariance
    
    def random_sampling_algorithm(self):
        """Calculates the active subspace using the random sampling algorithm of Constantine et al.
        Corresponds to Algorithm 3.1 in the book of Constantine et al.

        Args:
        
        Returns:
            np.ndarray: Matrix of eigenvectors stored in the columns
            np.ndarray: Vector of eigenvalues
        """

        # Evaluate the gradients of the function at the samples
        debug_info(self._debug, "Constructing the active subspace using the random sampling algorithm")
        if not hasattr(self, 'gradients'):
            debug_info(self._debug, "Evaluating gradients for active subspace construction")
            self.evaluate_gradients()
        else:
            print("WARNING: Gradients already evaluated, skipping evaluation. Make sure the gradients are up to date.")

        # Construct the covariance matrix
        convariance_matrix = self.covariance(self.gradients)

        # Calculate the eigenvalues and eigenvectors of the covariance matrix
        S, U = self.calculate_eigenpairs(convariance_matrix)

        self._eigenvalues = S
        self._eigenvectors = U

        debug_info(self._debug, f"Active subspace constructed")

        return (self._eigenvectors, self._eigenvalues)
    
    def partition(self, n : int):
        """Partitions the active subspace into two subspaces of dimension n and m-n

        Args:
            n (int): Dimension of the active subspace

        Returns:
            np.ndarray: Matrix containing the active subspace of dimension n
            np.ndarray: Matrix containing the inactive subspace of dimension m-n
        """
        W1 = self._eigenvectors[:,:n]
        W2 = self._eigenvectors[:,n:]
        return (W1, W2)
    
    def bootstrap(self, M_boot : int):
        """ Compute the bootstrap values for the eigenvalues
        
        Args:
            M_boot (int): Number of bootstrap samples
            
        Returns:
            np.ndarray: Bootstrap lower and upper bounds for the eigenvalues
            np.ndarray: Bootstrap lower and upper bounds for the subspace distances
        """
        if not hasattr(self, 'gradients'):
            self.evaluate_gradients()

        if not hasattr(self, 'eigenvalues'):
            self.random_sampling_algorithm()

        # Loop over the number of bootstrap samples
        eigenvalues = np.zeros([self.samples.m, M_boot])
        subspace_distances = np.zeros([self.samples.m, M_boot])
        for i in range(M_boot):
            # Construct bootstrap replicate
            bootstrap_indices = np.random.randint(0, self.samples.M, size = self.samples.M)
            bootstrap_replicate = self.gradients[bootstrap_indices,:].copy()

            # Compute the bootstraped singular value decomposition
            S, U = self.calculate_eigenpairs(self.covariance(bootstrap_replicate))

            for j in range(self.samples.m-1):
                subspace_distances[j,i] = np.linalg.norm(np.dot(self._eigenvectors[:,:j+1].T, U[:,j+1:]), ord=2)
            eigenvalues[:,i] = S
        sub_max = np.max(subspace_distances, axis=1)
        sub_min = np.min(subspace_distances, axis=1)
        sub_mean = np.mean(subspace_distances, axis=1)

        # Compute the max and min of the eigenvalues over all bootstrap samples
        e_max = np.max(eigenvalues, axis=1)
        e_min = np.min(eigenvalues, axis=1)

        self.e_boot = [e_max, e_min]
        self.sub_boot = [sub_max, sub_min, sub_mean]

        debug_info(self._debug, f"Bootstrap values calculated")

        return [e_max, e_min], [sub_max, sub_min, sub_mean]
    
    def calculate_eigenpairs(self, matrix : np.ndarray):
        """Calculates the eigenvalues and eigenvectors of a matrix

        Args:
            matrix (np.ndarray): Matrix to calculate the eigenvalues and eigenvectors of

        Returns:
            np.ndarray: Vector of eigenvalues
            np.ndarray: Matrix of eigenvectors stored in the columns
        """
        e, W = np.linalg.eigh(matrix)
        e = abs(e)
        idx = e.argsort()[::-1]
        e = e[idx]
        W = W[:,idx]
        normalization = np.sign(W[0,:])
        normalization[normalization == 0] = 1
        W = W * normalization
        return e, W
    
    def plot_eigenvalues(self, filename = "eigenvalues.png", true_eigenvalues = None, ylim=None):
        """Plots the eigenvalues of the covariance matrix on a logarithmic scale

        Args:
            filename (str, optional): Filename of the plot. Defaults to "eigenvalues.png".
            true_eigenvalues (np.ndarray, optional): True eigenvalues of the covariance matrix. Defaults to None.
        Raises:
            ValueError: If the covariance matrix is not defined
        """
        if not hasattr(self, "_eigenvectors"):
            raise ValueError("Eigendecomposition of the covariance matrix is not defined. Calculate it first.")
        import matplotlib.pyplot as plt
        fig = plt.figure(filename)
        ax = fig.gca()
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        if true_eigenvalues is not None:
            ax.plot(range(1, self.k+1), true_eigenvalues[:self.k], marker="o", fillstyle="none", label="True")
        ax.plot(range(1, self.k+1), self._eigenvalues[:self.k], marker="x", fillstyle="none", label="Est")
        if hasattr(self, "e_boot"):
            debug_info(self._debug, "Plotting bootstrap bounds for eigenvalues")
            ax.fill_between(range(1, self.k+1), self.e_boot[0][:self.k], self.e_boot[1][:self.k], alpha=0.5, label = "BI")
        plt.yscale("log")
        plt.xlabel("Index")
        plt.ylabel("Eigenvalue")
        plt.legend()
        plt.grid()
        if ylim is not None:
            plt.ylim(ylim)
        plt.savefig(filename)
        plt.close()

    def plot_subspace(self, filename = "subspace", true_subspace = None, ylim=None):
        """Plots the subspace distances

        Args:
            filename (str, optional): Filename of the plot. Defaults to "subspace.png".
        Raises:
            ValueError: If the covariance matrix is not defined
        """
        if not hasattr(self, "_eigenvectors"):
            raise ValueError("Eigendecomposition of the covariance matrix is not defined. Calculate it first.")
        import matplotlib.pyplot as plt
        fig = plt.figure(filename)
        ax = fig.gca()
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        if true_subspace is not None:
            ax.plot(range(1, self.k), true_subspace[:self.k-1], marker="o", fillstyle="none", label="True")
        ax.plot(range(1, self.k), self.sub_boot[2][:self.k-1], marker="x", fillstyle="none", label="Est")
        if hasattr(self, "sub_boot"):
            debug_info(self._debug, "Plotting bootstrap bounds for subspace distances")
            ax.fill_between(range(1, self.k), self.sub_boot[0][:self.k-1], self.sub_boot[1][:self.k-1], alpha=0.5, label = "BI")
        plt.xlabel("Subspace Dimension")
        plt.yscale("log")
        plt.ylabel("Subspace Error")
        plt.legend()
        plt.grid()
        if ylim is not None:
            plt.ylim(ylim)
        plt.savefig(filename)
        plt.close()



# TODO: Check if private/protected variales are returned as objects or as copys.