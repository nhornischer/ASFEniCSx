import numpy as np
# TODO tabulate_dof_coordinate instead mesh.geometry.x

from ASFEniCSx.utils import debug_info
from ASFEniCSx.sampling import Sampling

class Functional:
    """ Class for constructing a functional, in order to evaluate a function.

    Attributes:
    public:
        m (int): Dimension of the parameter space
        f (function): Function to be evaluated
    private:
        _number_of_calls (int): Number of calls to the function
        _debug (bool): If True, debug information will be printed

    Methods:
    public:
        number_of_calls() -> int: Returns the number of calls to the function
        reset_number_of_calls(): Resets the number of calls to the function
        evaluate(x : numpy.ndarrray) -> float: Evaluates the function at the point x
        gradient(x : numpy.ndarray) -> numpy.ndarray: Calculates the gradient of the function at the point x
    private:
        _finite_differences(x : numpy.ndarray, h : float) -> numpy.ndarray: Calculates the finite difference of the function at the point x

    Example:
        >>> def f(x): return x[0]**2 + x[1]**2
        >>> func = functional(2, f)
        >>> x = np.array([1,2])
        >>> func.evaluate(x)
        5

    Notes:
        The function must be defined in the original parameter domain not in a normalized one.

    Version:
        0.2
    Contributors:
        Niklas Hornischer (nh605@cam.ac.uk)
    """
    def __init__(self, m : int, f : callable, debug = True) -> None:
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
            
    def number_of_calls(self) -> int:
        """Returns the number of calls to the function
        
        Returns:
            int: Number of calls to the function
        """
        return self._number_of_calls
    
    def reset_number_of_calls(self) -> None:
        """Resets the number of calls to the function"""
        self._number_of_calls = 0

    def evaluate(self, x : np.ndarray) -> float:
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

    def gradient(self, x : np.ndarray, **kwargs) -> np.ndarray:
        """Calculates the gradient of the function at the given point/points using finite differences.

        Args:
            x (numpy.ndarray): Point or points at which the gradient is calculated.
                                Must be shape (n,m) or (m, ) where n is the number 
                                of points and m is the dimension of the function.
            **kwargs: Keyword arguments for the finite difference method. Can be:
                        h (float): Step size for the finite difference method. Default is 1e-6
                        order (int): Order of the finite difference method. Default is 2
        Returns:
            numpy.ndarray: Gradient or gradients of the function at the given point or points.
                            The shape is (n,m) or (m, ) depending on the size of the input.

        Raises:
            AssertionError: If the dimension of the point is not equal to the dimension of the function
        """
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        assert x.shape[1] == self.m, "x must have dimension m"

        gradient = np.zeros((x.shape[0], self.m))
        for i in range(x.shape[0]):
            gradient[i,:] = self._finite_differences(x[i,:], **kwargs)

        if gradient.shape[0] == 1:
            gradient = gradient.reshape(-1)

        return gradient

    def _finite_differences(self, x : np.ndarray, h = 1e-6, order = 2 **kwargs) -> np.ndarray:
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
    
class Analytical(Functional):
    """
    Class for functions with analytical and explicit derivatives.
    
    Attributes:
    public:
        m (int): Dimension of the function
        f (callable): Function
        df (callable): Derivative of the function
    private:
        _number_of_calls (int): Number of calls to the function
        _debug (bool): Debug flag
    
    Methods:
    public:
        number_of_calls() -> int: Returns the number of calls to the function
        reset_number_of_calls() -> None: Resets the number of calls to the function
        evaluate(x : np.ndarray) -> float: Evaluates the function at a given point
        gradient(x : np.ndarray) -> np.ndarray: Calculates the gradient of the function at the given point/points
    private:
        _finite_differences(x : np.ndarray, h = 1e-6, order = 2) -> np.ndarray: Calculates the gradient of the interpolant at the given point using finite differences.

    Version:
        0.2

    Contributors:
        Niklas Hornischer (nh605@cam.ac.uk)
    """

    def __init__(self, m : int, f: callable, df: callable, debug : bool = True) -> None:
        """
        Constructor for the Analytical class.
        
        Args:
            m (int): Dimension of the function
            f (callable): Function
            df (callable): Derivative of the function
            debug (bool, optional): Debug flag. Defaults to True.
            
        Raises:
            AssertionError: If the derivative is not callable
        """
        super().__init__(m, f, debug = debug)

        assert callable(df), "df must be a callable function"
        self.df = df

    def gradient(self, x : np.ndarray) -> np.ndarray:
        """Calculates the gradient of the function at the given point/points.

        Args:
            x (numpy.ndarray): Point or points at which the gradient is calculated.
                                Must be shape (n,m) or (m, ) where n is the number 
                                of points and m is the dimension of the function.
        Returns:
            numpy.ndarray: Gradient or gradients of the function at the given point or points.
                            The shape is (n,m) or (m, ) depending on the size of the input.

        Raises:
            AssertionError: If the dimension of the point is not equal to the dimension of the function
        """
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        assert x.shape[1] == self.m, "x must have dimension m"

        gradient = np.zeros((x.shape[0], self.m))
        for i in range(x.shape[0]):
            gradient[i,:] = self.df(x[i,:])

        if gradient.shape[0] == 1:
            gradient = gradient.reshape(-1)

        return gradient

class Interpolation(Functional):
    """
    Class for the interpolation of a function using a given set of points and values.
    
    Attributes:
    public:
        m (int): Dimension of the parameter space
        f (function): Function to be evaluated
        samples (Sampling): Sampling object that may includes the clusters
    private:
        _number_of_calls (int): Number of calls to the function
        _debug (bool): If True, debug information will be printed
        _polynomial (callable): Polynomial object that is created after the interpolation
        _derivative (callable): Derivative object that is created after the interpolation

    Methods:
    public:
        gradient(x : np.ndarray) -> np.ndarray: Calculates the gradient at the given point or points.



    Version:
        0.2
        
    Contributors:
        Niklas Hornischer (nh605@cam.ac.uk)
    """

    def __init__(self, m : int, f : callable, samples : Sampling, debug = True) -> None:

        super().__init__(m, f, debug)
        self.samples = samples

    def gradient(self, x : np.ndarray) -> np.ndarray:
        """Calculates the gradient at the given point or points by evaluating the polynomial.

        Args:
            x (numpy.ndarray): Point at which the gradient is calculated
            sampling (sampling, optional): sampling object that includes the clusters. Defaults to None.
                                            If this argument is not none, the gradient is calculated locally.
            order (int, optional): Order of the finite difference method. Defaults to 2.

        Returns:
            numpy.ndarray: Gradient at the given point

        Raises:
            AssertionError: If the dimension of the point is not equal to the dimension of the interpolant
            ValueError: If no derivative is found
        """
        if not hasattr(self, '_derivative'):
            raise ValueError("No derivative found")

        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        assert x.shape[1] == self.m, "x must have dimension m"

        gradient = np.zeros((x.shape[0], self.m))
        for i in range(x.shape[0]):
            if  not self.use_clustering:
                gradient[i,:] = self._derivative(x[i,:])
            else:
                cluster_idx = self.samples.obtain_index(x[i,:])
                gradient[i,:] = self._derivative[cluster_idx](x[i,:])

        if gradient.shape[0] == 1:
            gradient = gradient.reshape(-1)

        return gradient

    def interpolate(self, order : int = 2, overwrite : bool = False, use_clustering : bool = True,  **kwargs):
        assert self.samples.m == self.m, "The dimension of the samples must match the dimension of the parameter space"

        # Calculates the global interpolant
        if not hasattr(self.samples, "_clusters") or not use_clustering:
            exponents = self._create_exponents(order, **kwargs)
            debug_info (self._debug, f"Calculating global interpolant with order {order} and {len(exponents)} exponents")

            # Obtain the data for the interpolation
            _data = self.samples.samples()[:len(exponents),:]
            if hasattr(self.samples, "_values"):
                _values = self.samples.values()[:len(exponents)]
            else:
                _values = np.asarray([self.evaluate(_data[i,:]) for i in range(len(exponents))])

            coefficients = self.multivariate_interpolation(_data, _values, exponents)
            if hasattr(self, "_polynomial") and not overwrite:
                raise ValueError("The interpolant has already been calculated. Please set overwrite to True to overwrite the interpolant")
            self._polynomial = self.multivariate_polynomial(coefficients, np.asarray(exponents))
            self._derivative = self.multivariate_polynomial_derivative(coefficients, np.asarray(exponents))
            self.use_clustering = False
        
        # Calculates multiple local interpolates based on the defined clusters
        else: 
            if hasattr(self, "_polynomial") and not overwrite:
                raise ValueError("The interpolant have already been calculated. Please set overwrite to True to overwrite the interpolant")            
            self._polynomial = []
            self._derivative = []
            for index_list in self.samples.clusters():
                # Set number of coefficients to the minimum of the elements in the cluster 
                # and the number of exponents given in **kwargs
                exponents = self._create_exponents(order,maximal_samples = len(index_list), **kwargs)
                debug_info (self._debug, f"Calculating local interpolant for cluster {index_list} with order {order} and {len(exponents)} exponents")

                _data = np.asarray([self.samples.extract(i) for i in index_list])
                # Check if values have already been calculated
                if hasattr(self.samples, "_values"):
                    _values = np.asarray([self.samples.extract_value(i) for i in index_list])
                else:
                    _values = np.asarray([self.evaluate(_data[i,:]) for i in range(len(exponents))])

                _data = _data[:len(exponents),:]
                _values = _values[:len(exponents)]
                
                coefficients = self.multivariate_interpolation(_data, _values, exponents) 
                self._polynomial.append(self.multivariate_polynomial(coefficients, np.asarray(exponents)))
                self._derivative.append(self.multivariate_polynomial_derivative(coefficients, np.asarray(exponents)))
            self.use_clustering = True

    def _create_exponents(self, order : int,  number_of_exponents = None, maximal_samples = None) -> list:
        """Creates a list of all the possible exponents of the summands for the mutlivariate interpolation
        The exponents have a maximal total order of the given order, meaning that 
        the sum of the exponents of each summand is smaller or equal to the given order.

        Args:
            order (int): Order of the interpolant
        Returns:
            list: List of exponents
        """
        from itertools import product
        exponents=list(product(*([list(range(order+1))]*self.m)))
        _remove=[]
        for exponent in exponents:
            if sum(exponent)>order:
                _remove.append(exponent)
        for exponent in _remove:
            exponents.remove(exponent)

        if number_of_exponents == None:
            number_of_exponents = self.samples.M

        if maximal_samples != None:
            number_of_exponents = min(number_of_exponents, maximal_samples)


        if len(exponents)>number_of_exponents:
            exponents = exponents[:number_of_exponents]
        return exponents

    
    def multivariate_interpolation(self, samples : np.ndarray, values : np.ndarray, exponents : list):
        # Creates the matrix A to solve the linear system of equations to determine the coefficients by evaluating
        # the polynomial at the given samples.

        assert np.shape(samples)[0] == len(values), f"The number of samples and values must be equal, but they are {np.shape(samples)} and {len(values)} respectively"

        A=np.ones([len(exponents), len(exponents)])
        for i in range(len(exponents)):
            for j, exponent in enumerate(exponents):
                A[i,j]=np.prod(samples[i,:]**exponent)
                
        c = np.linalg.solve(A, values)
        return c

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

    def approximate(self, x : np.ndarray) -> np.ndarray:
        """Calculates the polynomial at the given point with the specified method.

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
        if not hasattr(self, '_polynomial'):
            raise ValueError("No polynomial found")

        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        assert x.shape[1] == self.m, "x must have dimension m"

        values = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            if  not self.use_clustering:
                values[i] = self._polynomial(x[i,:])
            else:
                cluster_idx = self.samples.obtain_index(x[i,:])
                values[i] = self._polynomial[cluster_idx](x[i,:])

        if values.shape[0] == 1:
            values = values.reshape(-1)

        return values

class Regression(Functional):
    def __init__(self, m: int, f: callable, samples : np.ndarray,  debug=True) -> None:
        super().__init__(m, f, debug=debug)
        self.samples = samples

    
    def gradient(self, x : np.ndarray) -> np.ndarray:
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
        if not hasattr(self, '_derivative'):
            raise ValueError("No derivative found")

        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        assert x.shape[1] == self.m, "x must have dimension m"

        gradient = np.zeros((x.shape[0], self.m))
        for i in range(x.shape[0]):
            if  not self.use_clustering:
                gradient[i,:] = self._derivative(x[i,:])
            else:
                cluster_idx = self.samples.obtain_index(x[i,:])
                gradient[i,:] = self._derivative[cluster_idx](x[i,:])

        if gradient.shape[0] == 1:
            gradient = gradient.reshape(-1)

        return gradient

    def regression(self, order : int = 2, overwrite : bool = False, use_clustering : bool = True,  number_of_samples = None):
        assert self.samples.m == self.m, "The dimension of the samples must match the dimension of the parameter space"

        if number_of_samples == None:
            number_of_samples = self.samples.M

        # Calculates the global regressant
        if not hasattr(self.samples, "_clusters") or not use_clustering:
            exponents = self._create_exponents(order)
            debug_info (self._debug, f"Calculating global regressant with order {order} and {len(exponents)} exponents and {number_of_samples} samples")

            # Obtain the data for the interpolation
            _data = self.samples.samples()[:number_of_samples,:]
            if hasattr(self.samples, "_values"):
                _values = self.samples.values()[:number_of_samples]
            else:
                _values = np.asarray([self.evaluate(_data[i,:]) for i in range(number_of_samples)])

            coefficients = self.multivariate_interpolation(_data, _values, exponents)
            if hasattr(self, "_polynomial") and not overwrite:
                raise ValueError("The interpolant has already been calculated. Please set overwrite to True to overwrite the interpolant")
            self._polynomial = self.multivariate_polynomial(coefficients, np.asarray(exponents))
            self._derivative = self.multivariate_polynomial_derivative(coefficients, np.asarray(exponents))
            self.use_clustering = False
        
        # Calculates multiple local regressant based on the defined clusters
        else: 
            if hasattr(self, "_polynomial") and not overwrite:
                raise ValueError("The interpolant have already been calculated. Please set overwrite to True to overwrite the interpolant")            
            self._polynomial = []
            self._derivative = []
            for index_list in self.samples.clusters():
                # Set number of coefficients to the minimum of the elements in the cluster 
                # and the number of exponents given in **kwargs
                exponents = self._create_exponents(order)
                debug_info (self._debug, f"Calculating local regressant for cluster {index_list} with order {order} and {len(exponents)} exponents")

                _data = np.asarray([self.samples.extract(i) for i in index_list])
                _data = _data[:min(len(index_list), number_of_samples),:]
                # Check if values have already been calculated
                if hasattr(self.samples, "_values"):
                    _values = np.asarray([self.samples.extract_value(i) for i in index_list])
                    _values = _values[:min(len(index_list), number_of_samples)]
                else:
                    _values = np.asarray([self.evaluate(_data[i,:]) for i in range(len(exponents))])

                _data = _data[:len(exponents),:]
                _values = _values[:len(exponents)]
                
                coefficients = self.multivariate_interpolation(_data, _values, exponents) 
                self._polynomial.append(self.multivariate_polynomial(coefficients, np.asarray(exponents)))
                self._derivative.append(self.multivariate_polynomial_derivative(coefficients, np.asarray(exponents)))
            self.use_clustering = True

    def _create_exponents(self, order : int) -> list:
        """Creates a list of all the possible exponents of the summands for the mutlivariate interpolation
        The exponents have a maximal total order of the given order, meaning that 
        the sum of the exponents of each summand is smaller or equal to the given order.

        Args:
            order (int): Order of the interpolant
        Returns:
            list: List of exponents
        """
        from itertools import product
        exponents=list(product(*([list(range(order+1))]*self.m)))
        _remove=[]
        for exponent in exponents:
            if sum(exponent)>order:
                _remove.append(exponent)
        for exponent in _remove:
            exponents.remove(exponent)
        return exponents

    
    def multivariate_interpolation(self, samples : np.ndarray, values : np.ndarray, exponents : list):
        # Creates the matrix A to solve the linear system of equations to determine the coefficients by evaluating
        # the polynomial at the given samples.

        assert np.shape(samples)[0] == len(values), f"The number of samples and values must be equal, but they are {np.shape(samples)} and {len(values)} respectively"

        A=np.ones([samples.shape[0], len(exponents)])
        for i in range(samples.shape[0]):
            for j, exponent in enumerate(exponents):
                A[i,j]=np.prod(samples[i,:]**exponent)
                
        c,_,_,_ = np.linalg.lstsq(A, values, rcond=None)
        return c

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

    def approximate(self, x : np.ndarray) -> np.ndarray:
        """Calculates the polynoiaml at the given point.

        Args:
            x (numpy.ndarray): Point at which the polynomial is calculated
            sampling (sampling, optional): sampling object that includes the clusters. Defaults to None.
                                            If this argument is not none, the polynomial is calculated locally.
            order (int, optional): Order of the finite difference method. Defaults to 2.

        Returns:
            numpy.ndarray: Polynomial at the given point

        Raises:
            AssertionError: If the dimension of the point is not equal to the dimension of the interpolant
            ValueError: If no polynomial is specified
        """
        if not hasattr(self, '_polynomial'):
            raise ValueError("No polynomial found")

        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        assert x.shape[1] == self.m, "x must have dimension m"

        values = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            if  not self.use_clustering:
                values[i] = self._polynomial(x[i,:])
            else:
                cluster_idx = self.samples.obtain_index(x[i,:])
                values[i] = self._polynomial[cluster_idx](x[i,:])

        if values.shape[0] == 1:
            values = values.reshape(-1)

        return values
    

    
    

