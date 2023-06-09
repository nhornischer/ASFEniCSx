import logging, logging.config
import numpy as np
import json

from ASFEniCSx.utils import NumpyEncoder, normalizer

logging.config.fileConfig("logging.conf")
logger = logging.getLogger('Sampling')

class Sampling:
    """Class for sampling the domain of a parameter space

    This class produces an object containing the samples of the domain as well
    as the number of samples and the dimension of the parameter space.

    Important remarks: No mather what probability density function is used to 
    generate the samples, the samples must always be normalized to the interval
    [-1,1] in order to be used in the active subspace method. This needs to be
    done by the user, since the samples are stored unnormalized.

    The samples are stored in a numpy array of shape (M,m) where M is the number
    of samples and m is the dimension of the parameter space.

    The class also provides a method to extract a single sample from the array and
    a method to get the whole sampling array.

    Attributes:
    public:
        M (int): Number of samples
        m (int): Dimension of the parameter space
    private:
        _array (numpy.ndarray): Array containing the samples with shape (M,m)
        _bounds (numpy.ndarray): Array containing the bounds of the original domain with shape (m,2)
        _object_type (str): Type of the object (sampling or clustering) used for saving and loading.
    
    Methods:
    public:
        random_uniform() -> None: Generates the samples using a uniform distribution
        standard_gaussian(mean : float, variance : float) -> None: Generates the samples using a standard gaussian distribution
        samples() -> numpy.ndarray: Returns the sampling array
        extract(index : int) -> numpy.ndarray: Extracts the sample at the given index
        set_domainBounds(bounds : numpy.ndarray) -> None: Sets the boundaries of the sampling domain
        assign_values(f : callable) -> None: Assigns the values of the function f to the samples
        assign_value(index : int, value : float) -> None: Assigns the value to the sample at the given index
        extract_values(index : int) -> float: Extracts the values of the sample at the given index
        values() -> numpy.ndarray: Returns the values of the samples
        save(filename : str) -> float: Saves the sampling object to a json file
    private:
        _load(data : dict, overwrite : bool) -> float: Loads the sampling object from a json file

    Example:
        >>> samples = sampling(100, 10)
        >>> samples.random_uniform()

    Version:
        0.2
    Contributors:
        Niklas Hornischer (nh605@cam.ac.uk)
    """
    def __init__(self, M : int, m : int) -> None:
        """Constructor for the sampling object

        Sets the sampling attributes M and m to the values passed to the
        constructor.

        Args:
            M (int): Number of samples
            m (int): Dimension of the parameter space

        Raises: 
            AssertionError: If M or m are not greater than 0
        """
        assert M > 0, "Number of samples must be greater than 0"
        assert m > 0, "Dimension of parameter space must be greater than 0"

        self._object_type = "Sampling"
        self.M = M
        self.m = m
    
    def random_uniform(self) -> None:
        """Generates the samples using a uniform distribution
        
        Generates the samples using a uniform distribution with values from the defined domain boundaries
        or the default values [-1,1] if no bounds are specified.
        """
        if not hasattr(self, "_bounds"):
            self._bounds = np.array([[-1.0]*self.m, [1.0]*self.m]).T
            logger.info("No bounds were defined default bounds will be used [-1,1] for all variables")
        if not hasattr(self, "_array"):
            self._array = np.zeros((self.M, self.m))
        else:
            logger.warning("Samples already exist and are overwritten.")
        
        for i in range(self.m):
            self._array[:,i] = np.random.uniform(self._bounds[i,0], self._bounds[i,1], self.M)

    def standard_gaussian(self, mean : float or np.ndarray = 0.0, variance : float or np.ndarray = 1.0) -> None:
        """Generates the samples using a standard gaussian distribution
        
        Generates the samples using a standard gaussian distribution and normalizes .

        
        Args:
            mean (float or numpy.ndarray, optional): Mean of the gaussian distribution. Default is 0.0.
            variance (float or numpy.ndarray, optional): Variance of the gaussian distribution. Default is 1.0.
            
        """
        try:
            mean.shape
        except AttributeError:
            logger.info("Only one mean value defined. Using same mean for all parameters")
            mean = np.array([mean]*self.m)
        else:
            assert mean.shape == (self.m,), "Mean has wrong shape. Expected ({},) or (1,), got {}".format(self.m, mean.shape)

        try:
            variance.shape
        except AttributeError:
            logger.info("Only one variance value defined. Using same variance for all parameters")
            variance = np.array([variance]*self.m)
        else:
            assert variance.shape == (self.m,), "Variance has wrong shape. Expected ({},) or (1,), got {}".format(self.m, variance.shape)

        if not hasattr(self, "_array"):
            self._array = np.zeros((self.M, self.m))
        else:
            logger.warning("Samples already exist and are overwritten")
        for i in range(self.m):
            self._array[:,i] = np.random.normal(mean[i], variance[i], self.M)

        if not hasattr(self, "_bounds"):
            self._bounds = np.array([[np.min(self._array[:,i]), np.max(self._array[:,i])] for i in range(self.m)])
            logger.info("No bounds were defined default bounds [min, max] will be used for all variables")
        else:
            self._array = normalizer(self._array, None , self._bounds)

    def samples(self) -> np.ndarray:
        """Returns the sampling array
        
        Returns:
            numpy.ndarray: The sampling array
        """
        return np.copy(self._array)

    def extract(self, index : int) -> np.ndarray:
        """Extracts the sample at the given index.
        
        Args:   
            index (int): Index of the sample to be extracted
        
        Returns:
            numpy.ndarray: Sample at the given index

        Raises:
            AssertionError: If the index is out of bounds
        """
        assert 0<= index < self.M, "Index out of bounds"

        return self._array[index,:]
        
    def set_domainBounds(self, bounds : np.ndarray) -> None:
        """Sets the boundaries of the sampling domain
        
        Args:
            bounds (numpy.ndarray): Array containing the boundaries of the original unnormalized domain
        
        Raises:
            AssertionError: If the bounds have the wrong shape
        """
        assert bounds.shape == (self.m, 2), "Bounds have wrong shape. Expected ({},2), got {}".format(self.m, bounds.shape)
        if hasattr(self, "_array"):
            logger.error("Samples already exist and thus have bounds already. Bounds are overwritten.")
        self._bounds = bounds

    def assign_values(self, f : callable) -> None:
        """Assigns values to the sampling object

        Assigns values to the sampling object by evaluating the given function at the samples.

        Args:
            f (callable): The function to be evaluated

        Raises:
            TypeError: If the function is not callable
        """
        assert callable(f), "Function must be callable"
        for i in range(self.M):
            self.assign_value(i, f(self.extract(i)))

    def assign_value(self, index : int, value : float) -> None:
        """Assigns a value to the sample at given index
        
        Args:
            index (int): Index of the sample
            value (float): The value to be assigned to the sample
        
        Raises:
            AssertionError: If the index is out of bounds
        """
        assert 0<= index < self.M, "Index out of bounds"
        if not hasattr(self, "_values"):
            self._values = np.zeros(self.M)
        self._values[index] = value

    def extract_value(self, index : int) -> float:
        """Returns the value assigned to the sample at given index
        
        Args:
            index (int): Index of the sample
        
        Returns:
            float: The value assigned to the sample at the given index

        Raises:
            AssertionError: If the index is out of bounds
            AttributeError: If the values have not been assigned yet
        """
        assert 0<= index < self.M, "Index out of bounds"
        assert hasattr(self, "_values"), "Values have not been assigned yet"
        return self._values[index]

    def values(self) -> np.ndarray: 
        """Returns the values assigned to the samples
        
        Returns:
            numpy.ndarray: The values assigned to the samples
            
        Raises:
            AttributeError: If the values have not been assigned yet
        """
        assert hasattr(self, "_values"), "Values have not been assigned yet"
        return self._values
    
    def save(self, filename : str) -> None:
        """Saves the sampling object to a json file

        Saves the sampling object to a json file. The file is saved in the current working directory.

        Args:
            filename (str): Name of the file to be saved

        Raises:
            TypeError: If the filename is not a string
        """
        assert isinstance(filename, str), "Filename must be a string"
        with open(filename, "w") as f:
            json.dump(self.__dict__, f, cls=NumpyEncoder, indent = 3)

    def _load(self, data : dict, overwrite : bool = False) -> None:
        """Loads array data into the sampling object

        Loads array data from dictionary into the sampling object. The array must have the shape (M,m) where M is the number of samples
        and m is the dimension of the parameter space.

        Args:
            data (dict): Dictionary containing numpy.ndarray data
            overwrite (bool, optional): If True, overwrites the existing samples. Default is False.

        Raises:
            AssertionError: If the array has the wrong shape

        """
        array = np.asarray(data["_array"])
        assert array.shape == (self.M, self.m), "Array has wrong shape"
        if not hasattr(self, "_array") or overwrite:
            # Check if the array is normalized
            if np.any(array > 1) or np.any(array < -1):
                raise ValueError("Array is not normalized")
            self._array = array
            if "_values" in data:
                self._values = np.asarray(data["_values"])
            if "_bounds" in data:
                self._bounds = np.asarray(data["_bounds"])
        else:
            raise AttributeError("Samples already exist. Use overwrite=True to overwrite them")

class Clustering(Sampling):
    """Class for creating clustered samples of a parameter space as a subclass of sampling

    This class produces as sampling object that contains clustered samples of a parameter space in addition
    to the unclustered data. The clustering is done using the k-means algorithm and done on the normalized data.

    Attributes:
    public:
        M (int): Number of samples
        m (int): Dimension of the parameter space
        k (int): Number of clusters
    private:
        _array (numpy.ndarray): Array containing the samples
        _max_iter (int): Maximum number of iterations for the k-means algorithm
        _centroids (numpy.ndarray): Array containing the centroids of the clusters
        _clusters (list): List of index lists of each clusters

    Methods:
    public:
        detect(): Detects the clusters
        plot(filename : str): Plots the clusters
        clusters() -> list: Returns the clusters
        cluster_index(x : numpy.ndarray) -> int: Returns the index of the cluster the sample belongs to
    private:
        _assign_clusters(data : numpy.ndarray) -> list: Assigns the samples to the clusters
        _update_centroids(clusters : list) -> None: Updates the centroids of the clusters
        _load(data : dict, overwrite = False) -> None: Loads array data into the sampling object

    Example:
        >>> kmeans = Clustering(100, 2, 5)
        >>> kmeans.random_uniform()
        >>> kmeans.detect()
        >>> kmeans.plot("2D.pdf")

    Version:
        0.2
    Contributors:
        Niklas Hornischer (nh605@cam.ac.uk)
    """
    def __init__(self, M : int, m : int,  k : int, max_iter = 1000) -> None:
        """Constructor of the clustering object

        Args:
            M (int): Number of samples
            m (int): Dimension of the parameter space
            k (int): Number of clusters
            max_iter (int, optional): Maximum number of iterations for the k-means algorithm. Default is 1000.
        
        Raises:
            AssertionError: If k is not greater than 0 and less than M
        """
        assert 0 < k < M, "Number of clusters must be greater than 0 and less than the number of samples"
        super().__init__(M, m)
        self._object_type = "Clustering"
        self.k = k
        self._max_iter = max_iter
    
    def detect(self, centroids : None or np.ndarray = None) -> None:
        """
        Detects the clusters using the k-means algorithm

        Args:
            centroids (None or numpy.ndarray, optional): If None, the centroids are initialized randomly. 
            If numpy.ndarray, the centroids are initialized with the given array. Default is None.

        """
        # Initialize centroids as random for each parameter
        if centroids is None:
            centroids = np.zeros((self.k, self.m))
            for i in range(self.m):
                centroids[:,i] = np.random.uniform(self._bounds[i,0], self._bounds[i,1], self.k)
            self._centroids = centroids
            logger.info("Centroids initialized randomly")
        else:
            assert centroids.shape == (self.k, self.m), "Centroids have wrong shape"
            self._centroids = centroids

        _prev_centroids = np.zeros((self.k, self.m))
        _iter=0
        while not np.isclose(self._centroids, _prev_centroids).all() and _iter < self._max_iter:
            _prev_centroids = self._centroids.copy()
            _clusters = self._assign_clusters(self._array)
            self._update_centroids(_clusters)
            logger.debug( f"Iteration: {_iter + 1}, Maximum centroid update: {np.max(abs(_prev_centroids - self._centroids))}, Are previous and current centroid not equal? {np.not_equal(self._centroids, _prev_centroids).any()}, Are previous and current centroid equal? {np.equal(self._centroids, _prev_centroids).all()}")
            _iter += 1
        if _iter == self._max_iter:
            logger.warning("Maximum number of iterations reached")
        else:
            logger.info(f"Convergence reached after {_iter} iterations")
        self._clusters = _clusters
    
    def clusters(self) -> list:
        """Returns the clusters

        Returns:
            list: List of clusters containing a list of the indices of the samples belonging to the clusters
        """
        return self._clusters

    def _assign_clusters(self, data : np.ndarray) -> list:
        """Assigns the samples to the clusters

        This method can be used to assign samples to the clusters and is called by the detect method.
        
        Args:
            data (numpy.ndarray): Array containing the samples but has to be normalized
        
        Returns:
            List: List of the clusters containing a list of the indices of the samples belonging to the clusters

        Raises:
            AssertionError: If the centroids have not been initialized or the dimension of the data does not match the dimension of the parameter space

        Note:
            It is possible to use this method outside the class to assign a arbitrary data set to the defined clusters, but in this case the sample space
            is not updated. To update the sample space, use the detect method. Using this method outside the class is not recommended.
        """
        assert hasattr(self, "_centroids"), "Centroids have not been initialized"
        assert np.shape(data)[1] == self.m, "Dimension of data does not match dimension of parameter space"
        _clusters=[[] for _ in range(self.k)]
        for i,x in enumerate(data):
            idx = self.cluster_index(x)
            _clusters[idx].append(i)
        return _clusters

    def cluster_index(self, x : np.ndarray) -> int:
        """Returns the index of the cluster to which the sample belongs

        Args:
            x (numpy.ndarray): Normalized sample to be assigned to a cluster

        Returns:
            int: Index of the cluster to which the sample belongs

        Raises:
            AssertionError: If the centroids have not been initialized
            AssertionError: If the dimension of the data does not match the dimension of the parameter space
        """
        assert hasattr(self, "_centroids"), "Centroids have not been initialized"
        assert np.shape(x) == (self.m,), "Dimension of data does not match dimension of parameter space"
        normalized_x = normalizer(x, self._bounds)
        normalized_centroids = normalizer(self._centroids, self._bounds)
        distances = np.linalg.norm(normalized_centroids-normalized_x, axis=1)
        cluster_idx = np.argmin(distances)
        return cluster_idx

    def _update_centroids(self, _clusters : list) -> None:
        """Updates the centroids of the clusters

        This method can be used to update the centroids of the clusters and is called by the detect method.

        Args:
            _clusters (list): List of clusters of lists containing the indices of the samples belonging to the clusters

        Note:
            It is possible to use this method outside the class to update the centroids of the clusters, but in this case the sample space
            is not updated, but the centroids are. To update the sample space, use the detect method. Using this method outside the class
            is not recommended.
        """
        for i, centroid in enumerate(self._centroids):
            cluster_data = np.asarray([self.extract(idx) for idx in _clusters[i]])
            _new_centroid = np.mean(cluster_data, axis=0)
            if not np.isnan(centroid).any():
                self._centroids[i] = _new_centroid
    
    def plot(self, filename : str = "kmeans.pdf") -> None:
        """Plots the clusters in the parameter space
        
        To visualize the figures, use plt.show() after calling this method. 
        This is especially useful when plotting 3D parameter spaces.

        Args:
            filename (str, optional): Name of the file to save the plot to. Default is kmeans.pdf
        
        Raises:
            AssertionError: If the dimension of the parameter space is greater than 3
        """
        import os
        import matplotlib.pyplot as plt
        from matplotlib import colors
        from matplotlib import cm
        dir = os.path.dirname(__file__)
        cmap = plt.get_cmap('hsv')
        scalarMap = cm.ScalarMappable(colors.Normalize(vmin=0, vmax=self.k),cmap=cmap)
        cluster_data = [np.asarray([self.extract(idx) for idx in self._clusters[i]]) for i in range(self.k)]
        if self.m == 1:
            plt.figure(figsize=(8,6))
            for i in range(self.k):
                plt.plot(self._centroids[i,0], 0, 'x', color=scalarMap.to_rgba(i))
                plt.scatter(cluster_data[i][:,0], np.zeros(cluster_data[i].shape[0]),color=scalarMap.to_rgba(i))
            plt.xlabel(r'$x_1$')
            plt.xlim(self._bounds[0,0], self._bounds[0,1])
        elif self.m == 2:
            plt.figure(figsize=(8,6))
            for i in range(self.k):
                plt.plot(self._centroids[i,0], self._centroids[i,1], 'x', color=scalarMap.to_rgba(i))
                plt.scatter(cluster_data[i][:,0], cluster_data[i][:,1],color=scalarMap.to_rgba(i))
            plt.xlabel(r'$x_1$')
            plt.ylabel(r'$x_2$')
            plt.xlim(self._bounds[0,0], self._bounds[0,1])
            plt.ylim(self._bounds[1,0], self._bounds[1,1])
        elif self.m ==3:
            plt.figure(figsize=(8,6))
            ax = plt.axes(projection='3d')
            for i in range(self.k):
                ax.scatter3D(cluster_data[i][:,0], cluster_data[i][:,1], cluster_data[i][:,2],color=scalarMap.to_rgba(i))
                ax.scatter3D(self._centroids[i,0], self._centroids[i,1], self._centroids[i,2], marker='x',color=scalarMap.to_rgba(i))
            ax.set_xlabel(r'$x_1$')
            ax.set_ylabel(r'$x_2$')
            ax.set_zlabel(r'$x_3$')
            ax.set_xlim(self._bounds[0,0], self._bounds[0,1])
            ax.set_ylim(self._bounds[1,0], self._bounds[1,1])
            ax.set_zlim(self._bounds[2,0], self._bounds[2,1])
        else:
            raise ValueError("Cannot plot more than 3 dimensions")
        plt.tight_layout()
        plt.savefig(filename, format="pdf")

    def _load(self, data : dict, overwrite : bool = False) -> None:
        """
        Loads the data into the clustering object
        
        Args:
            data (numpy.ndarray): Data to be loaded into the clustering object
            centroids (numpy.ndarray): Centroids of the clusters
            clusters (list): List of clusters of lists containing the indices of the samples belonging to the clusters
            overwrite (bool, optional): If True, the data will be overwritten. Default is False
        
        Raises:
            ValueError: If the centroids have already been initialized and overwrite is False
            ValueError: If the clusters have already been initialized and overwrite is False
            
        """
        super()._load(data, overwrite=True)
        if hasattr(self, "_centroids") and not overwrite:
            raise ValueError("Centroids have already been initialized. Set overwrite=True to overwrite the data.")
        else:
            self._centroids = np.asarray(data["_centroids"])
        if hasattr(self, "_clusters") and not overwrite:
            raise ValueError("Clusters have already been initialized. Set overwrite=True to overwrite the data.")
        else:
            self._clusters = []
            clusters = data["_clusters"]
            for i in range(len(clusters)):
                self._clusters.append(np.asarray(clusters[i]))