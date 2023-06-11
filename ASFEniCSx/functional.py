import numpy as np

from ASFEniCSx.utils import debug_info
from ASFEniCSx.sampling import sampling

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
    def __init__(self, m : int, f : callable, debug = True):
        """Constructor of the functional class
        
        Args:
            m (int): Dimension of the parameter space
            f (function): Function to be evaluated  
            debug (bool, optional): If True, debug information will be printed. Default is False
        
        Raises:
            AssertionError: If the dimension of the parameter space is not positive
            AssertionError: If the function is not callable
        """
        assert m>0, "m must be positive"
        assert callable(f), "f must be a callable function" 
        self.m = m
        self.f = f
        self._number_of_calls = 0
        self._debug = debug
        debug_info(self._debug, f"New functional object created with pointer {self}")
            
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
        if self._debug:
            print(f"Derivative set to {dfdx}")
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
        if self._debug:
            try:
                print(f"Gradient method set to {method}, was {self.gradient_method}")
            except:
                print(f"Gradient method set to {method}, was not set before")
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

        if self._debug:
            print(f"Interpolation method set to {interpolation_method}")
            print(f"Order of the interpolant set to {order}")

        # Calculates the global interpolant
        if not clustering:
            _data = sampling.samples()
            if hasattr(sampling, "_values"):
                _values = sampling.values()
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
            assert(np.shape(samples)[0] >= number_of_samples), f"The number of samples must be greater or equal to the number of coefficients. Is {np.shape(samples)[0]} but should be {number_of_samples}"
            if self._debug:

                print(f"fCalculating {number_of_samples} values of the function, because none where given.")
            values = np.zeros(number_of_samples)
            for i in range(number_of_samples):
                values[i] = self.evaluate(samples[i,:])
        else:
            assert len(exponents) <= len(values), f"The number of samples must be greater or equal to the number of coefficients. Is {len(values)} but should be {len(exponents)}"
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
    
    def gradient(self, x : np.ndarray, sampling = None, order = 2, **kwargs):
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
            return self._finite_differences(x, order = order, **kwargs)
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

# TODO tabulate_dof_coordinate instead mesh.geometry.x

    def _finite_differences(self, x : np.ndarray, h = 1e-6, order = 2, **kwargs):
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
                raise ValueError(f"No implemented order of finite differences. Given order: {order}")
        return dfdx