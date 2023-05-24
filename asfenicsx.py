import numpy as np
import json
import scipy

class NumpyEncoder(json.JSONEncoder):
    """Class for encoding numpy arrays to json
    
    This class is used to encode numpy arrays to json. It is used in the save method of the sampling and clustering classes.
    
    Methods:
    public:
        default(obj) -> json: Encodes the given object to json
        
    Example:
        >>> json.dumps(np.array([1,2,3]), cls=NumpyEncoder)
        '[1, 2, 3]'
    
    Version:
        0.1
    Contributors:
        Niklas Hornischer (nh605@cam.ac.uk)
    """
    def default(self, obj : object):
        """Returns a converted object that can be converted to json
        
        Args:
            obj (object): The object to be encoded
            
        Returns:
            (object) object that can be converted to json
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()

class utils:
    """Class for utility functions

    This class provides utility functions for the active subspace method.

    Methods:
    public:
        load(filename : str) -> object: Loads a sampling object from a json file

    Example:
        >>> utils.load("sampling.json")
        <asfenicsx.sampling object at 0x7f8b1c0b6a90>
    
    Version:
        0.1
    Contributors:
        Niklas Hornischer (nh605@cam.ac.uk)
    """
    def load(filename : str):
        """Loads a sampling object from a json file
        
        Args:
            filename (str): The name of the file to be loaded
            
        Returns:
            object: The sampling object
        
        Raises:
            FileNotFoundError: If the file does not exist
        """
        try:
            with open(filename, "r") as f:
                data=json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("File not found")
        data_type = data["object_type"]
        if data_type == "sampling":
            object = sampling(data["M"], data["m"])
            object.load(data["_array"])
        elif data_type=="clustering":
            object = clustering(data["M"], data["m"], data["k"], data["_max_iter"])
            object.load(data["_array"], data["_centroids"], data["_clusters"])
        return object

class sampling:
    """Class for sampling the domain of a parameter space

    This class produces an object containing the samples of the domain as well
    as the number of samples and the dimension of the parameter space.

    Per default the domain is sampled using a uniform distribution with values
    between -1 and 1.

    Important remarks: No mather what probability density function is used to 
    generate the samples, the samples must always be normalized to the interval
    [-1,1] in order to be used in the active subspace method.

    The samples are stored in a numpy array of shape (M,m) where M is the number
    of samples and m is the dimension of the parameter space.

    The class also provides a method to extract a single sample from the array and
    a method to get the whole sampling array.

    Attributes:
    public:
        M (int): Number of samples
        m (int): Dimension of the parameter space
    private:
        _array (numpy.ndarray): Array containing the samples
    
    Methods:
    public:
        random_uniform(overwrite : bool) -> None: Generates the samples using a uniform distribution  
        extract(index : int) -> numpy.ndarray: Extracts a single sample from the array
        replace(index : int, sample : numpy.ndarray) -> None: Replaces a single sample in the array
        samples() -> numpy.ndarray: Returns the sampling array
        assign_values(f : callable) -> None: Assigns values to the samples using a function
        assign_value(index : int, value : float) -> None: Assigns a value to a single sample
        extract_value(index : int) -> numpy.ndarray: Extracts the value of the sample at the given index
        values() -> numpy.ndarray: Returns the array containing the values of the samples
        index(sample : numpy.ndarray) -> int: Returns the index of the given sample in the sampling array
        save(filename : str) -> None: Saves the sampling object to a json file
        load(data : numpy.ndarray, overwrite : boolean) -> None: Loads the sampling object from a numpy array

    Example:
        >>> samples = sampling(100, 10)

    Version:
        0.1
    Contributors:
        Niklas Hornischer (nh605@cam.ac.uk)
    """
    def __init__(self, M : int, m : int):
        """Constructor for the sampling object

        Sets the sampling attributes M and m to the values passed to the
        constructor and calls the random_uniform method to generate the samples.

        Args:
            M (int): Number of samples
            m (int): Dimension of the parameter space

        Raises: 
            AssertionError: If M or m are not greater than 0
        """
        assert M > 0, "Number of samples must be greater than 0"
        assert m > 0, "Dimension of parameter space must be greater than 0"

        self.object_type = "sampling"
        self.M = M
        self.m = m
        self.random_uniform()

    def random_uniform(self, overwrite = False):
        """Generates the samples using a uniform distribution
        
        Generates the samples using a uniform distribution with values between -1 and 1.
        
        Args:
            overwrite (bool, optional): If True, overwrites the existing samples. Default is False.
            
        Raises:
            AttributeError: If the samples already exist and overwrite is False
            
        """
        if not hasattr(self, "_array") or overwrite:
            self._array = np.random.uniform(-1, 1, (self.M,self.m))
        else:
            raise AttributeError("Samples already exist. Use overwrite=True to overwrite them")
    
    def extract(self, index : int):
        """Extracts a single sample from the array
        
        Args:   
            index (int): Index of the sample to be extracted
        
        Returns:
            numpy.ndarray: The sample at the given index

        Raises:
            AssertionError: If the index is out of bounds
        """
        assert 0<= index < self.M, "Index out of bounds"
        return self._array[index,:]
    
    def replace(self, index : int, sample : np.ndarray):
        """Replaces a single sample in the array
        
        Args:
            index (int): Index of the sample to be replaced
            sample (numpy.ndarray): The new sample
        
        Raises:
            AssertionError: If the index is out of bounds
        """
        assert 0<= index < self.M, "Index out of bounds"
        assert sample.shape == (self.m,), "Sample has wrong shape"
        self._array[index,:] = sample
    
    def samples(self):
        """Returns the sampling array
        
        Returns:
            numpy.ndarray: The sampling array
        """
        return self._array

    def assign_values(self, f : callable):
        """Assigns values to the sampling object

        Assigns values to the sampling object by evaluating the given function at the samples.

        Args:
            f (callable): The function to be evaluated

        Raises:
            TypeError: If the function is not callable
        """
        assert callable(f), "Function must be callable"
        self._values = np.apply_along_axis(f, 1, self._array)

    def assign_value(self, index : int, value : float):
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

    def extract_value(self, index : int):
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

    def values(self): 
        """Returns the values assigned to the samples
        
        Returns:
            numpy.ndarray: The values assigned to the samples
            
        Raises:
            AttributeError: If the values have not been assigned yet
        """
        assert hasattr(self, "_values"), "Values have not been assigned yet"
        return self._values
    
    def index(self, sample : np.ndarray):
        """ Returns the index of the given sample in the sampling array
        
        Args:
            sample (numpy.ndarray): The sample
            
        Returns:
            int: The index of the sample in the sampling array
            
        Raises:
            AssertionError: If the sample has the wrong shape
            AssertionError: If the sample is not in the sampling array
        """
        assert sample.shape == (self.m,), "Sample has wrong shape"
        assert sample in self._array, "Sample is not in the sampling array"
        return np.where(self._array == sample)[0][0]
    
    def save(self, filename : str):
        """Saves the sampling object to a json file

        Saves the sampling object to a json file. The file is saved in the current working directory.

        Args:
            filename (str): Name of the file to be saved

        Raises:
            TypeError: If the filename is not a string
        """
        assert isinstance(filename, str), "Filename must be a string"
        with open(filename, "w") as f:
            json.dump(self.__dict__, f, cls=NumpyEncoder)

    def load(self, data : np.ndarray, overwrite = False):
        """Loads array data into the sampling object

        Loads array data into the sampling object. The array must have the shape (M,m) where M is the number of samples
        and m is the dimension of the parameter space.

        Args:
            data (numpy.ndarray): Array containing the samples
            overwrite (bool, optional): If True, overwrites the existing samples. Default is False.

        Raises:
            AssertionError: If the array has the wrong shape


        """
        assert data.shape == (self.M, self.m), "Array has wrong shape"
        if not hasattr(self, "_array") or overwrite:
            self._array = np.asarray(data)
        else:
            raise AttributeError("Samples already exist. Use overwrite=True to overwrite them")

class clustering(sampling):
    """Class for creating clustered samples of a parameter space as a subclass of sampling

    This class produces as sampling object that contains clustered samples of a parameter space in addition
    to the unclustered data. The clustering is done using the k-means algorithm.

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
        assign_clusters(data : numpy.ndarray) -> list: Assigns the samples to the clusters
        update_centroids(clusters : list): Updates the centroids of the clusters
        plot(filename : str): Plots the clusters
        clusters() -> list: Returns the clusters
        centroids() -> numpy.ndarray: Returns the centroids of the clusters
        cluster_index(x : numpy.ndarray) -> int: Returns the index of the cluster the sample belongs to

    Example:
        >>> kmeans = clustering(100, 2, 5)
        >>> kmeans.detect()
        >>> kmeans.plot("2D.pdf")

    Version:
        0.1
    Contributors:
        Niklas Hornischer (nh605@cam.ac.uk)
    """
    def __init__(self, M : int, m : int,  k : int, max_iter = 1000):
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
        self.object_type = "clustering"
        self.k = k
        self._max_iter = max_iter
    
    def detect(self):
        """
        Detects the clusters using the k-means algorithm
        """
        _min, _max = np.min(self._array), np.max(self._array)
        self._centroids = np.random.uniform(_min, _max, (self.k, self.m))
        _prev_centroids=None
        _iter=0
        while np.not_equal(self._centroids, _prev_centroids).any() and _iter < self._max_iter:
            _prev_centroids = self._centroids.copy()
            _clusters = self.assign_clusters(self._array)
            self.update_centroids(_clusters)
            _iter += 1
        self._clusters = _clusters
    
    def clusters(self):
        """Returns the clusters

        Returns:
            list: List of cluster containing a list of the indices of the samples belonging to the clusters
        """
        return self._clusters
    
    def centroids(self):
        """Returns the centroids of the clusters

        Returns:
            numpy.ndarray: Array containing the centroids of the clusters
        """
        return self._centroids

    def assign_clusters(self, data : np.ndarray):
        """Assigns the samples to the clusters

        This method can be used to assign samples to the clusters and is called by the detect method.
        It is possible to assign a arbitrary data set to the defined clusters, but in this case the sample space
        is not updated.
        
        Args:
            data (numpy.ndarray): Array containing the samples
        
        Returns:
            List: List of the clusters containing a list of the indices of the samples belonging to the clusters

        Raises:
            AssertionError: If the centroids have not been initialized or the dimension of the data does not match the dimension of the parameter space
        """
        assert hasattr(self, "_centroids"), "Centroids have not been initialized"
        assert np.shape(data)[1] == self.m, "Dimension of data does not match dimension of parameter space"
        _clusters=[[] for _ in range(self.k)]
        for i,x in enumerate(data):
            idx = self.cluster_index(x)
            _clusters[idx].append(i)
        return _clusters

    def cluster_index(self, x : np.ndarray):
        """Returns the index of the cluster to which the sample belongs

        Args:
            x (numpy.ndarray): Sample to be assigned to a cluster

        Returns:
            int: Index of the cluster to which the sample belongs

        Raises:
            AssertionError: If the centroids have not been initialized
            AssertionError: If the dimension of the data does not match the dimension of the parameter space
        """
        assert hasattr(self, "_centroids"), "Centroids have not been initialized"
        assert np.shape(x)[0] == self.m, "Dimension of data does not match dimension of parameter space"
        distances = np.linalg.norm(self._centroids-x, axis=1)
        cluster_idx = np.argmin(distances)
        return cluster_idx

    def update_centroids(self, _clusters : list):
        """Updates the centroids of the clusters

        This method can be used to update the centroids of the clusters and is called by the detect method.
        It is not recommended to use this method on its own, as it does not assign the samples to the clusters,
        but changes the centroids of the cluster.

        Args:
            _clusters (list): List of clusters of lists containing the indices of the samples belonging to the clusters
        """
        for i, centroid in enumerate(self._centroids):
            cluster_data = np.asarray([self.extract(idx) for idx in _clusters[i]])
            _new_centroid = np.mean(cluster_data, axis=0)
            if not np.isnan(centroid).any():
                self._centroids[i] = _new_centroid
    
    def plot(self, filename = "kmeans.pdf"):
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
            plt.figure("K-means clustering (1D)")
            for i in range(self.k):
                plt.plot(self._centroids[i,0], 0, 'x', color=scalarMap.to_rgba(i))
                plt.scatter(cluster_data[i][:,0], np.zeros(cluster_data[i].shape[0]),color=scalarMap.to_rgba(i))
            plt.xlabel(r'$x_1$')
        elif self.m == 2:
            plt.figure("K-means clustering (2D)")
            for i in range(self.k):
                plt.plot(self._centroids[i,0], self._centroids[i,1], 'x', color=scalarMap.to_rgba(i))
                plt.scatter(cluster_data[i][:,0], cluster_data[i][:,1],color=scalarMap.to_rgba(i))
            plt.xlabel(r'$x_1$')
            plt.ylabel(r'$x_2$')
        elif self.m ==3:
            plt.figure("K-means clustering (3D)")
            ax = plt.axes(projection='3d')
            for i in range(self.k):
                ax.scatter3D(cluster_data[i][:,0], cluster_data[i][:,1], cluster_data[i][:,2],color=scalarMap.to_rgba(i))
                ax.scatter3D(self._centroids[i,0], self._centroids[i,1], self._centroids[i,2], marker='x',color=scalarMap.to_rgba(i))
            ax.set_xlabel(r'$x_1$')
            ax.set_ylabel(r'$x_2$')
            ax.set_zlabel(r'$x_3$')
        else:
            raise ValueError("Cannot plot more than 3 dimensions")
        if not os.path.exists(os.path.join(dir,"figures")):
            os.makedirs(os.path.join(dir,"figures"))
        plt.savefig(os.path.join(dir, "figures", filename), dpi=300, format="pdf")

    def load(self, data : np.ndarray, centroids : np.ndarray, clusters : list, overwrite = False):
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
        super().load(data)
        if hasattr(self, "_centroids") and not overwrite:
            raise ValueError("Centroids have already been initialized. Set overwrite=True to overwrite the data.")
        else:
            self._centroids = np.asarray(centroids)
        if hasattr(self, "_clusters") and not overwrite:
            raise ValueError("Clusters have already been initialized. Set overwrite=True to overwrite the data.")
            self._clusters = []
            for i in range(len(clusters)):
                self._clusters.append(np.asarray(clusters[i]))

class functional:
    """ Class for constructing a functional, in order to evaluate a function, its derivative and interpolated values.

    Attributes:
    public:
        m (int): Dimension of the parameter space
        f (function): Function to be evaluated
        use_clusters (boolean): If True, the interpolant will be evaluated using the clusters of the clustering object (if created)
    private:
        _number_of_calls (int): Number of calls to the function
        _derivative (callable): Analytical derivative of the function (if created)
        _interpolant (callable): Interpolant of the function (if created)
        _interpolants (list): List of callable interpolants of the function (if created)
        _derivatives (list): List of callable derivatives of the function (if created)

    Methods:
    public:
        number_of_calls() -> int: Returns the number of calls to the function
        reset_number_of_calls(): Resets the number of calls to the function
        evaluate(x : numpy.ndarrray) -> float: Evaluates the function at the point x
        get_derivative(dfdx : callable): Set the analytical derivative of the function
        get_gradient_method(method : str): Sets the method for calculating the gradient
        interpolation(samling : sampling): Calculates the interpolant and its derivative of the given function
        multivariate_interpolation(samples : numpy.ndarray, values : numpy.ndarray) -> numpy.ndarray, numpy.ndarray: Calculates the coefficients and exponents of the interpolant
        multivariate_polynomial(coefficients : numpy.ndarray, exponents : numpy.ndarray) -> callable: Returns a callable function of the interpolant
        multivariate_polynomial_derivative(coefficients : numpy.ndarray, exponents : numpy.ndarray) -> callable: Returns a callable function of the derivative of the interpolant
        evaluate_interpolant(x : numpy.ndarray) -> float: Evaluates the interpolant at the point x
        gradient(x : numpy.ndarray) -> numpy.ndarray: Calculates the gradient of the function at the point x

    private:
        _finite_differences(x : numpy.ndarray, h : float) -> numpy.ndarray: Calculates the finite difference of the function at the point x

    Example:
        >>> def f(x): return x[0]**2 + x[1]**2
        >>> func = functional(2, f)
        >>> x = np.array([1,2])
        >>> func.evaluate(x)
        5
        >>> func.get_derivative(lambda x: [2*x[0], 2*x[1]])
        >>> func.get_gradient_method("A")
        >>> func.gradient(x)
        array([2, 4])
        >>> func.interpolation(sampling(10, 2), interpolation_method="LS")
        >>> func.get_gradient_method("I")
        >>> func.evaluate_interpolant(x)
        4.9999
        >>> func.gradient(x)
        array([1.9999, 3.9999])

    Version:
        0.1
    Contributors:
        Niklas Hornischer (nh605@cam.ac.uk)
    """
    def __init__(self, m : int, f : callable):
        """Constructor of the functional class
        
        Args:
            m (int): Dimension of the parameter space
            f (function): Function to be evaluated  
        
        Raises:
            AssertionError: If the dimension of the parameter space is not positive
            AssertionError: If the function is not callable
        """
        assert m>0, "m must be positive"
        assert callable(f), "f must be a callable function" 
        self.m = m
        self.f = f
        self._number_of_calls = 0

    def number_of_calls(self):
        """Returns the number of calls to the function
        
        Returns:
            int: Number of calls to the function
        """
        return self._number_of_calls
    
    def reset_number_of_calls(self):
        """Resets the number of calls to the function"""
        self._number_of_calls = 0

    def evaluate(self, x : np.ndarray):
        """ Evaluates the function at a given point

        Args:
            x (numpy.ndarray): Point at which the function is evaluated

        Returns:
            float: Value of the function at x

        Raises:
            AssertionError: If the dimension of x does not match the dimension of the parameter space
        """
        assert np.shape(x) == (self.m,), "x must be a vector of dimension m" 
        self._number_of_calls += 1
        return self.f(x)

    def get_derivative(self, dfdx : callable):
        """Sets the explicitly formulated derivative of the function

        This method can be used to set the derivative of the functional object to 
        a explicitly formulated function. This can either be a analytical form or
        an interpolated function.

        Raises:
            AssertionError: If the derivative is not callable
        """
        assert callable(dfdx), "dfdx must be a callable function"
        self._derivative=dfdx

    def get_gradient_method(self, method : str):
        """Sets the method used to calculate the gradient of the function

        Args:
            method (str): Method used to calculate the gradient. Possible values are
                            'FD' (finite differences), 'I' (interpolation) and 'A' (analytical)

        Raises:
            ValueError: If the method is not valid
            ValueError: If the analytical method is used but no derivative has been set
        """
        if not method in [None, 'FD', 'I', 'A']:
            raise ValueError("Invalid method")
        if method == 'A' and not hasattr(self, '_derivative'):
            raise ValueError("No derivative has been set. Please define the derivative before using the analytical method")
        self.gradient_method=method
    
    def interpolation(self, sampling : sampling, order = 2, interpolation_method = 'default', overwrite = False, clustering = False):
        """Calculates a polynomial interpolant (globally or locally) based on given samples.

        This function calculates a polynomial based on the multivariate interpolation function
        and sets the interpolant and its derivative as attributes of the functional object.

        Args:
            sampling (sampling): sampling object containing the samples
            order (int, optional): Order of the polynomial interpolant. Defaults to 2.
            interpolation_method (str, optional): Method used to calculate the interpolant. Defaults to 'default'.
                                                    Possible values are 'default', 'LS' (least squares).
            overwrite (bool, optional): If True, the interpolant will be overwritten if it has already been calculated. Defaults to False.
            clustering (bool, optional): If True, the samples will be clustered and multiple local interpolants will be calculated. Defaults to False.

        Raises:
            AssertionError: If the dimension of the samples does not match the dimension of the parameter space
            ValueError: If the samples object does not contain any clusters
            ValueError: If the interpolant has already been calculated but overwrite is set to False
        """
        assert sampling.m == self.m, "The dimension of the samples must match the dimension of the parameter space"
        self.use_clusters = clustering

        # Calculates the global interpolant
        if not clustering:
            _data = sampling.samples()
            if hasattr(sampling, "_values"):
                _values = sampling.values()
                print("Has values")
            else:
                _values = None
            coefficients, exponents = self.multivariate_interpolation(_data, _values, order = order, method = interpolation_method)
            if hasattr(self, "_interpolant") and not overwrite:
                raise ValueError("The interpolant has already been calculated. Please set overwrite to True to overwrite the interpolant")
            self._interpolant = self.multivariate_polynomial(coefficients, exponents)
            self._derivative = self.multivariate_polynomial_derivative(coefficients, exponents)
        # Calculates multiple local interpolates based on the defined clusters
        else: 
            if not hasattr(sampling, "_clusters"):
                raise ValueError("The samples object does not contain any clusters. Please set clustering to False")
            if hasattr(self, "_interpolants") and not overwrite:
                raise ValueError("The interpolants have already been calculated. Please set overwrite to True to overwrite the interpolants")
            self._interpolants = []
            self._derivatives = []
            for index_list in sampling.clusters():
                _data = np.asarray([sampling.extract(i) for i in index_list])
                # Check if values have already been calculated
                if hasattr(sampling, "_values"):
                    _values = np.asarray([sampling.extract_value(i) for i in index_list])
                else:
                    _values = None
                coefficients, exponents = self.multivariate_interpolation(_data, _values, order = order, method = interpolation_method) 
                self._interpolants.append(self.multivariate_polynomial(coefficients, exponents))
                self._derivatives.append(self.multivariate_polynomial_derivative(coefficients, exponents))

    def multivariate_interpolation(self, samples : np.ndarray, values : np.ndarray, order = 2, method = 'default'):
        """Calculates the coefficients of a multivariate polynomial interpolation.

        This method can be used to construct a multivariate polynomial for arbitrary samples and data.
        The usage is not restricted to the functional class but can be used for any multivariate polynomial interpolation.
        If no values are given, the values are calculated by evaluating the function at the given samples.

        This method is not computationally efficient since it explicitly constructs a possibly dense matrix
        with high computational costs to solve the resulting linear system of equations.

        The polynomial is of the form
            f(x) = sum_{i=1}^{n} c_i * prod_{j=1}^{m} x_j^{e_{ij}}
        where n is the number of coefficients, m is the dimension of the parameter space,
        c_i are the coefficients and e_{ij} are the exponents of the summands.
        The exponents have a maximal total order of the given order, meaning that
        the sum of the exponents of each summand is smaller or equal to the given order.

        The coefficients are calculated by solving the linear system of equations
            A * c = v
        where A is a matrix with the samples as rows and the summands as columns, c is a vector of the coefficients
        and v is a vector of the values of the function at the samples.

        Args:
            samples (numpy.ndarray): Samples at which the function is evaluated
            values (numpy.ndarray): Values of the function at the samples
            order (int): Order of the polynomial
            method (str): Method used to calculate the coefficients. Possible values are
                            'default' and 'LS' (uses a least squares approximation)

        Returns:
            np.ndarray: Coefficients of the multivariate polynomial interpolation
            np.ndarray: Exponents of the summands
            
        Raises:
            AssertionError: If the number of samples and values is not equal
            AssertionError: If the number of coefficients is greater than the number of samples
        """
        from itertools import product
        # Creates a list of all the possible exponents of the summands for the mutlivariate interpolation
        # The exponents have a maximal total order of the given order, meaning that 
        # the sum of the exponents of each summand is smaller or equal to the given order.
        exponents=list(product(*([list(range(order+1))]*self.m)))
        _remove=[]
        for exponent in exponents:
            if sum(exponent)>order:
                _remove.append(exponent)
        for exponent in _remove:
            exponents.remove(exponent)

        # Creates the matrix A to solve the linear system of equations to determine the coefficients by evaluating
        # the polynomial at the given samples.
        # If the least squares method is used as many samples as possible are used to determine the coefficients
        # otherwise the number of samples is equal to the number of coefficients.
        if method=='LS':
            number_of_samples=np.shape(samples)[0]
        else:
            number_of_samples = len(exponents)

        # If no values are given evaluate the function at the samples
        if values is None:
            assert(np.shape(samples)[0] >= number_of_samples), "The number of samples must be greater or equal to the number of coefficients."
            values = np.zeros(number_of_samples)
            for i in range(number_of_samples):
                values[i] = self.evaluate(samples[i,:])
        else:
            assert len(exponents) <= len(values), "The number of samples must be greater or equal to the number of coefficients."
            assert np.shape(samples)[0] == len(values), "The number of samples and values must be equal"
            # Fake set the numer of calls to the function
            if method == 'LS':
                self._number_of_calls = len(values)
            else:
                self._number_of_calls = len(exponents)
        
        A=np.ones([number_of_samples, len(exponents)])
        for i in range(number_of_samples):
            for j, exponent in enumerate(exponents):
                A[i,j]=np.prod(samples[i,:]**exponent)
                
        if method == 'LS':
            c,_,_,_ = np.linalg.lstsq(A, values[:number_of_samples], rcond=None)
        else:
            c = np.linalg.solve(A, values[:number_of_samples])
        return c, np.asarray(exponents)

    def multivariate_polynomial(self, coefficients : np.ndarray, exponents : np.ndarray):
        """Constructs a multivariate polynomial from the coefficients and exponents of the summands.

        Args:
            coefficients (numpy.ndarray): Coefficients of the summands
            exponents (numpy.ndarray): Exponents of the summands

        Returns:
            function: Multivariate polynomial
        """
        return lambda x: np.dot([np.prod(np.power(x,exponents[k,:])) for k in range(len(exponents))], coefficients)   
    
    def multivariate_polynomial_derivative(self, coefficients : np.ndarray, exponents : np.ndarray):
        """Constructs the derivative of a multivariate polynomial from the coefficients and exponents of the summands.

        Args:
            coefficients (numpy.ndarray): Coefficients of the summands
            exponents (numpy.ndarray): Exponents of the summands

        Returns:
            function: Derivative of the multivariate polynomial
        """
        _dim = len(exponents[0])
        return lambda x: [np.dot([np.prod(np.power(x[0:k], exponents[j,0:k])) * exponents[j,k]*x[k]**max(exponents[j,k]-1,0) * np.product(np.power(x[k+1:_dim], exponents[j,k+1:_dim])) for j in range(len(coefficients))], coefficients) for k in range(_dim)]  

    def evaluate_interpolant(self, x : np.ndarray, sampling = None):
        """Evaluates the interpolant at the given point, either locally or globally.

        Args:
            x (numpy.ndarray): Point at which the interpolant is evaluated
            sampling (sampling, optional): sampling object that includes the clusters. Defaults to None.
                                            If this argument is not none, the evaluation is done locally.
        
        Returns:
            float: Value of the interpolant at the given point

        Raises:
            AssertionError: If the dimension of the point is not equal to the dimension of the interpolant
            ValueError: If no interpolant is found
        """
        assert len(x)==self.m, "x must have dimension m"
        if hasattr(self, '_interpolant') and not self.use_clusters:
            return self._interpolant(x)
        elif hasattr(self, '_interpolants') and self.use_clusters:
            if sampling is None or not hasattr(sampling, '_clusters'):
                raise ValueError("No clusters given. Construct global interpolant or specify cluster.")
            cluster_idx = sampling.cluster_index(x)
            return self._interpolants[cluster_idx](x)
        else:
            raise ValueError("No interpolant found")
    
    def gradient(self, x : np.ndarray, sampling = None, order = 2):
        """Calculates the gradient at the given point with the specified method.

        Args:
            x (numpy.ndarray): Point at which the gradient is calculated
            sampling (sampling, optional): sampling object that includes the clusters. Defaults to None.
                                            If this argument is not none, the gradient is calculated locally.
            order (int, optional): Order of the finite difference method. Defaults to 2.

        Returns:
            numpy.ndarray: Gradient at the given point

        Raises:
            AssertionError: If the dimension of the point is not equal to the dimension of the interpolant
            ValueError: If no gradient method is specified
        """
        assert(len(x)==self.m)
        if not hasattr(self, 'gradient_method'):
            raise ValueError("No gradient method specified")
        if self.gradient_method == "FD":
            return self._finite_differences(x, order = order)
        elif self.gradient_method == "A":
            return self._derivative(x)
        elif self.gradient_method == "I":
            if hasattr(self, '_interpolant') and not self.use_clusters:
                return self._derivative(x)
            elif hasattr(self, '_interpolants') and self.use_clusters:
                if sampling == None or not hasattr(sampling, '_clusters'):
                    raise ValueError("No clusters given. Construct global interpolant or specify cluster.")
                cluster_idx = sampling.cluster_index(x)
                return self._derivatives[cluster_idx](x)
            else:
                raise ValueError("No interpolant found")
        else:
            raise ValueError("No allowed gradient method specified")

    def _finite_differences(self, x : np.ndarray, h = 1e-6, order = 2):
        """Calculates the gradient of the interpolant at the given point using finite differences.

        Args:
            x (numpy.ndarray): Point at which the gradient is calculated
            h (float, optional): Step size for the finite differences. Defaults to 1e-6.
            order (int, optional): Order of the finite differences. Defaults to 2.
        Returns:
            np.ndarray: Gradient of the interpolant at the given point

        Raises:
            AssertionError: If the dimension of the point is not equal to the dimension of the functional
            AssertionError: If the step size is not positive
        """

        assert len(x)==self.m, "x must have dimension m"
        assert h>0, "h must be positive" 

        dfdx = np.zeros(self.m)
        if order == 1:
            f_0 = self.evaluate(x)
        for i in range(self.m):
            if order ==2:
                #f(x+e_ih)
                x[i] += h
                f_1 = self.evaluate(x)
                
                #f(x-e_ih)
                x[i] -= 2*h
                f_2 = self.evaluate(x)

                # Second-order central finite differences (ZFD)
                dfdx[i] = (f_1-f_2) / (2*h)

                # Reset x
                x[i] += h
            elif order == 1:
                x[i] += h
                f_1 = self.evaluate(x)

                # First-order forward finite differences (FFD)
                dfdx[i] = (f_1-f_0) / h

                # Reset x
                x[i] -= h
            else:
                raise ValueError("No implemented order of finite differences")
        return dfdx

class ASFEniCSx:
    """Class for constructing the active subspace in FeniCSx based on Constantine et al. 

    The class is based on the paper by Constantine et al. The class is constructed
    to be used with the FeniCSx library and requires a functional and a sampling object.

    Attributes:
    public:
        n (int): Desired Dimension of the active subspace
        function (functional): functional describing the quantity of interest
        samples (sampling): sampling object containing the samples
        eigenvalues (numpy.ndarray): Eigenvalues of the covariance matrix (if created)
        eigenvectors (numpy.ndarray): Eigenvectors of the covariance matrix (if created)

    Methods:
    public:
        covariance(info : bool, optional): Approximates the covariance matrix of the gradient of the function
        eigendecomposition(matris : numpy.ndarray): Calculates the eigendecomposition of a matrix
        random_sampling_algorithm() : Performs the random sampling algorithm to construct the active subspace

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

    def __init__(self, n : int, function : functional, samples : sampling):
        """Constructor for the ASFEniCSx class

        Args:
            n (int): Desired Dimension of the active subspace
            function (functional): functional describing the quantity of interest
            samples (sampling): sampling object containing the samples

        Raises:
            ValueError: If n is larger than the number of dimensions of the parameter space
        """
        assert n <= samples.m, "n must be smaller than the number of dimensions of the parameter space"
        self.n = n
        self.function = function
        self.samples = samples
        
    def covariance(self, info = False):
        """Approximates the covariance matrix of the gradient of the function

        The calculation of the gradient is defined directly in the functional.
        The covariance matrix is approximated by the outer product of the gradient.
        
        Args:
            info (bool, optional): If True, a progress bar is shown. Defaults to False.

        Returns:
            np.ndarray: Approximated covariance matrix with dimensions m x m    
        """
        covariance = np.zeros([self.samples.m, self.samples.m])
        if info:
            import tqdm.autonotebook
            progress = tqdm.autonotebook.tqdm(desc="Approximating Covariance Matrix", total=self.samples.M)
        for i in range(self.samples.M):
            grad = self.function.gradient(self.samples.extract(i), self.samples)
            covariance += np.outer(grad, grad)
            if info:
                progress.update(1)
        if info:
            progress.close()
        covariance = covariance / self.samples.M
        return covariance
    
    def eigendecomposition(self, matrix : np.ndarray):
        """Calculates the eigendecomposition of a matrix

        Args:
            matrix (numpy.ndarray): Matrix to be decomposed

        Returns:
            np.ndarray: Matrix of eigenvectors stored in the columns
            np.ndarray: Vector of eigenvalues
        """
        U, S, _ = scipy.linalg.svd(matrix, full_matrices=True)
        return U, S**2
    
    def random_sampling_algorithm(self, info = False):
        """Calculates the active subspace using the random sampling algorithm of Constantine et al.

        Args:
            info (bool, optional): If True, a progress bar is shown. Defaults to False.
        
        Returns:
            np.ndarray: Matrix of eigenvectors stored in the columns
            np.ndarray: Vector of eigenvalues
        """
        covariance = self.covariance(info = info)
        U, S = self.eigendecomposition(covariance)
        self.eigenvectors = U[:,0:self.n]
        self.eigenvalues = S[0:self.n]
        return (U[:,0:self.n], S[0:self.n]**2)
