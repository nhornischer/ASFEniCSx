import logging, logging.config
import numpy as np
import ASFEniCSx.utils as utils
# TODO tabulate_dof_coordinate instead mesh.geometry.x

from ASFEniCSx.sampling import Sampling

logging.config.fileConfig("logging.conf")
logger = logging.getLogger('Functional')

class Functional:
    """ Class for constructing a functional, in order to evaluate a function.

    Attributes:
    public:
        m (int): Dimension of the parameter space
        f (callable): Function to be evaluated
    private:
        _number_of_calls (int): Number of calls to the function

    Methods:
    public:
        number_of_calls() -> int: Returns the number of calls to the function
        reset_number_of_calls(): Resets the number of calls to the function
        evaluate(x : numpy.ndarrray) -> float or numpy.ndarray: Evaluates the function at the point x
        gradient(x : numpy.ndarray) -> numpy.ndarray: Calculates the gradient of the function at the point x
    private:
        _finite_differences(x : numpy.ndarray, h : float, order : int) -> numpy.ndarray: Calculates the finite difference of the function at the point x

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
    
    def __init__(self, m : int, f : callable) -> None:
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

    def number_of_calls(self) -> int:
        """Returns the number of calls to the function
        
        Returns:
            int: Number of calls to the function
        """
        return self._number_of_calls
    
    def reset_number_of_calls(self) -> None:
        """Resets the number of calls to the function"""
        self._number_of_calls = 0

    def evaluate(self, x : np.ndarray) -> float or np.ndarray:
        """ Evaluates the function at a given point or points

        Args:
            x (numpy.ndarray): Point or points at which the function is evaluated.
                                Must be shape (n,m) or (m, ) where n is the number 
                                of points and m is the dimension of the function.
        Returns:
            float or numpy.ndarray: Value of the function at x

        Raises:
            AssertionError: If the dimension of x does not match the dimension of the parameter space
        """
        if type(x) != np.ndarray:
            x = np.array([x])
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        assert x.shape[1] == self.m, "x must be of dimension m"

        evaluations= np.zeros(x.shape[0], dtype = x.dtype)
        for i in range(x.shape[0]):
            _value = self.f(x[i,:])
            if _value.dtype != x.dtype:
                logger.warning(f"Function returns value of different datatype. Converting from {_value.dtype} to {x.dtype}")
            evaluations[i] = _value
        self._number_of_calls += x.shape[0]
        logger.debug(f"Datatype of the input : {x.dtype}, datatype of the output : {evaluations.dtype}")
        if evaluations.shape[0] == 1:
            logger.debug(f"Converting array of shape {evaluations.shape} to float")
            if x.dtype == np.float64:
                evaluations = float(evaluations.reshape(-1)[0])
            elif x.dtype == np.complex128:
                evaluations = complex(evaluations.reshape(-1)[0])
        return evaluations

    def gradient(self, x : float or np.ndarray,  method = 'FD', **kwargs) -> np.ndarray:
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
        if type(x) != np.ndarray:
            x = np.array([x])
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        assert x.shape[1] == self.m, "x must have dimension m"

        gradient = np.zeros((x.shape[0], self.m))
        for i in range(x.shape[0]):
            if method == 'FD':
                gradient[i,:] = self._finite_differences(x[i,:], **kwargs)
            if method == 'CSM':
                gradient[i,:] = self._complex_step_method(x[i,:], **kwargs)

        if gradient.shape[0] == 1:
            gradient = gradient.reshape(-1)

        if gradient.shape[0] == 1:
            logger.debug(f"Converting array of shape {gradient.shape} to float")
            if x.dtype == np.float64:
                gradient = float(gradient.reshape(-1)[0])
            elif x.dtype == np.complex128:
                gradient = complex(gradient.reshape(-1)[0])

        return gradient

    def _finite_differences(self, x : np.ndarray, h = 1e-6, order = 2, **kwargs) -> np.ndarray:
        """Calculates the gradient of the interpolant at the given point using finite differences.

        Args:
            x (numpy.ndarray): Point at which the gradient is calculated
            h (float, optional): Step size for the finite differences. Defaults to 1e-6.
            order (int, optional): Order of the finite differences. Defaults to 2.
        Returns:
            np.ndarray: Gradient of the interpolant at the given point

        Raises:
            AssertionError: If the step size is not positive
        """
        assert h>0, "h must be positive" 

        #TODO: Check central finite differences. The error does not decrease with h^2.

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
    
    def _complex_step_method(self, x : np.ndarray, h = 1e-6) -> np.ndarray:
        """Calculates the gradient of the interpolant at the given point using the complex step method.

        Args:
            x (numpy.ndarray): Point at which the gradient is calculated
            h (float, optional): Step size for the complex. Defaults to 1e-6.
        Returns:
            np.ndarray: Gradient of the interpolant at the given point

        Raises:
            AssertionError: If the step size is not positive
        """
        assert h>0, "h must be positive" 
        
        # Convert x to complex
        x = np.asarray(x, dtype=np.complex128)
        dfdx = np.zeros(self.m)
        for i in range(self.m):
            x[i] += h*1j
            f_1 = self.evaluate(x)
            dfdx[i] = f_1.imag / h
            x[i] -= h*1j
        return dfdx
    
class Analytical(Functional):
    """
    Class for functions with analytical and explicit derivatives.
    
    ***Inherited from Functional***

    Attributes:
    public:
        df (callable): Derivative of the function

    Methods:
    public:
        gradient(x : np.ndarray) -> np.ndarray: Calculates the gradient of the function at the given point/points. (OVERWRITTEN)
    private:
        _finite_differences(x : np.ndarray, h = 1e-6, order = 2) -> np.ndarray: Calculates the gradient of the interpolant at the given point using finite differences.

    Version:
        0.2

    Contributors:
        Niklas Hornischer (nh605@cam.ac.uk)
    """

    def __init__(self, m : int, f: callable, df: callable) -> None:
        """
        Constructor for the Analytical class.
        
        ***Inherited from Functional***

        Args:
            df (callable): Derivative of the function

        Raises:
            AssertionError: If the derivative is not callable
        """
        assert callable(df), "df must be a callable function"
        
        super().__init__(m, f)
        self.df = df

    def gradient(self, x : np.ndarray) -> np.ndarray:
        """Calculates the gradient of the function at the given point/points.

        (OVERWRITTEN)

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

    ***Inherited from Functional***
    
    Attributes:
    public:
        samples (Sampling): Sampling object that may includes the clusters
    private:
        _polynomial (callable): Polynomial object that is created after the interpolation
        _derivative (callable): Derivative object that is created after the interpolation
        _use_clustering (bool): If True, the clustering is used to construct and evaluate the interpolant

    Methods:
    public:
        interpolate(order : int, use_clustering : bool) -> None: Interpolates the function using the sampling object.
        approximate(x : np.ndarray) -> np.ndarray: Approximates the function at the given point or points by evaluating the interpolant.
        gradient(x : np.ndarray) -> np.ndarray: Calculates the gradient at the given point or points. (OVERWRITTEN)
    private:
        _create_exponents(order : int, number_of_eponents : int, maximal_samples : int) -> list: Creates the exponents for the polynomial.
        _multivariate_interpolation(samples : np.ndarray, values : np.ndarray, exponents : list) -> np.ndarray: Calculates the coefficients of the interpolant.

    Version:
        0.2
        
    Contributors:
        Niklas Hornischer (nh605@cam.ac.uk)
    """

    def __init__(self, m : int, f : callable, samples : Sampling) -> None:
        """
        Constructor for the Interpolation class.

        ***Inherited from Functional***
        
        Args:
            samples (Sampling): Sampling object that may includes the clusters

        Raises:
            AssertionError: If the dimension of the samples does not match the dimension of the parameter space
        """
        super().__init__(m, f)
        assert samples.m == m, "The dimension of the samples must match the dimension of the parameter space"
        self.samples = samples

    def interpolate(self, order : int = 2, use_clustering : bool = True,  **kwargs) -> None:
        """Calculates the interpolant of the function using the sampling attribute and creates the polynomial and its derivative.

        Args:
            order (int, optional): Order of the interpolant. Defaults to 2.
            use_clustering (bool, optional): If True, the clustering is used to construct and evaluate the interpolant. Defaults to True.
            kwargs: Additional keyword arguments for the _create_exponents method
        """

        # Calculates the global interpolant
        if not hasattr(self.samples, "_clusters") and use_clustering:
            logger.warning("No clusters found. Interpolating without clustering")
            use_clustering = False
        if not use_clustering:
            _exponents = self._create_exponents(order, **kwargs)
            if len(_exponents) > self.samples.M:
                raise ValueError(f"The number of exponents ({len(_exponents)}) is larger than the number of samples ({self.samples.M}). Not possible to solve an underdetermined system of equations")
            logger.info(f"Calculating global interpolant of order {order} with {len(_exponents)} exponents")

            # Obtain the data for the interpolation
            _data = self.samples.samples()[:len(_exponents),:]
            if hasattr(self.samples, "_values"):
                _values = self.samples.values()[:len(_exponents)]
            else:
                _values = self.evaluate(_data)
            
            # Calculate the coefficients and create the interpolant
            coefficients = self._multivariate_interpolation(_data, _values, _exponents)
            if hasattr(self, "_polynomial"):
                logger.warning("The interpolant has already been calculated. Overwriting the interpolant")
            self._polynomial = utils.create_polynomial(coefficients, np.asarray(_exponents))
            self._derivative = utils.create_polynomial_derivative(coefficients, np.asarray(_exponents))
        
        # Calculates multiple local interpolates based on the defined clusters
        else: 
            if hasattr(self, "_polynomial"):
                logger.warning("The interpolants have already been calculated. Overwriting the interpolant")           
            self._polynomial = []
            self._derivative = []
            for cluster_index,index_list in enumerate(self.samples.clusters()):
                # Set number of coefficients to the minimum of the elements in the cluster 
                # and the number of exponents given in **kwargs
                _exponents = self._create_exponents(order, maximal_samples = len(index_list), **kwargs)
                logger.info(f"Calculating local interpolant for cluster {cluster_index} with order {order} and {len(_exponents)} exponents")

                _data = np.asarray([self.samples.extract(i) for i in index_list])
                # Check if values have already been calculated
                if hasattr(self.samples, "_values"):
                    _values = np.asarray([self.samples.extract_value(i) for i in index_list])
                else:
                    _values = self.evaluate(_data)

                _data = _data[:len(_exponents),:]
                _values = _values[:len(_exponents)]
                
                coefficients = self._multivariate_interpolation(_data, _values, _exponents) 
                self._polynomial.append(utils.create_polynomial(coefficients, np.asarray(_exponents)))
                self._derivative.append(utils.create_polynomial_derivative(coefficients, np.asarray(_exponents)))
        self._use_clustering = use_clustering

    def _create_exponents(self, order : int,  number_of_exponents : int or None = None, maximal_samples : int or None = None) -> list:
        """Creates a list of all the possible exponents of the summands for the mutlivariate interpolation
        The exponents have a maximal total order of the given order, meaning that 
        the sum of the exponents of each summand is smaller or equal to the given order.

        Args:
            order (int): Order of the interpolant
            number_of_exponents (int or None, optional): Number of exponents to be calculated. Defaults to None.
            maximal_samples (int or None, optional): Maximal number of exponents to be calculated. Defaults to None.

        Returns:
            list: List of exponents
        """
        from itertools import product
        _exponents=list(product(*([list(range(order+1))]*self.m)))
        _remove=[]
        for exponent in _exponents:
            if sum(exponent)>order:
                _remove.append(exponent)
        for exponent in _remove:
            _exponents.remove(exponent)

        if number_of_exponents == None:
            number_of_exponents = self.samples.M

        if maximal_samples != None:
            number_of_exponents = min(number_of_exponents, maximal_samples)


        if len(_exponents)>number_of_exponents:
            logger.warning(f"Number of exponents ({len(_exponents)}) is larger than the number of samples ({number_of_exponents}). Using only the first {number_of_exponents} exponents")
            _exponents = _exponents[:number_of_exponents]
        return _exponents

    def _multivariate_interpolation(self, samples : np.ndarray, values : np.ndarray, exponents : list) -> np.ndarray:
        """Calculates the coefficients of the multivariate polynomial by solving a linear system of equations.

        Args:
            samples (numpy.ndarray): Samples of the function
            values (numpy.ndarray): Values of the function at the samples
            exponents (list): List of exponents of the summands

        Returns:
            numpy.ndarray: Coefficients of the multivariate polynomial
        """

        assert np.shape(samples)[0] == len(values), f"The number of samples and values must be equal, but they are {np.shape(samples)} and {len(values)} respectively"

        A=np.ones([len(exponents), len(exponents)])
        for i in range(len(exponents)):
            for j, exponent in enumerate(exponents):
                A[i,j]=np.prod(samples[i,:]**exponent)
                
        c = np.linalg.solve(A, values)
        return c

    def approximate(self, x : np.ndarray) -> float or np.ndarray:
        """Calculates the polynomial at the given point with the specified method.

        Args:
            x (numpy.ndarray): Point or points at which the polynomial is evaluated.
                                Must be shape (n,m) or (m, ) where n is the number 
                                of points and m is the dimension of the function.
        Returns:
            float or numpy.ndarray: Value or values of the function at the given point or points.
                                    The shape is (n,) or float depending on the size of the input.

        Raises:
            AssertionError: If the dimension of the point is not equal to the dimension of the interpolant
            ValueError: If no polynomial was found
        """
        if not hasattr(self, '_polynomial'):
            raise ValueError("No polynomial found")

        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        assert x.shape[1] == self.m, "x must have dimension m"

        values = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            if  not self._use_clustering:
                values[i] = self._polynomial(x[i,:])
            else:
                cluster_idx = self.samples.cluster_index(x[i,:])
                values[i] = self._polynomial[cluster_idx](x[i,:])

        if values.shape[0] == 1:
            values = float(values.reshape(-1)[0])

        return values
    
    def gradient(self, x : np.ndarray) -> np.ndarray:
        """Calculates the gradient at the given point or points by evaluating the polynomial.

        (OVERWRITTEN)

        Args:
            x (numpy.ndarray): Point or points at which the gradient is calculated.
                                Must be shape (n,m) or (m, ) where n is the number 
                                of points and m is the dimension of the function.
        Returns:
            numpy.ndarray: Gradient or gradients of the function at the given point or points.
                            The shape is (n,m) or (m, ) depending on the size of the input.

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
            if  not self._use_clustering:
                gradient[i,:] = self._derivative(x[i,:])
            else:
                cluster_idx = self.samples.cluster_index(x[i,:])
                gradient[i,:] = self._derivative[cluster_idx](x[i,:])

        if gradient.shape[0] == 1:
            gradient = gradient.reshape(-1)

        return gradient

class Regression(Functional):
    """
    Class for multivariate regression.

    ***Inherited from Functional***

    Attributes:
    public:
        samples (Sampling): Sampling object that may includes the clusters
    private:
        _polynomial (callable): Polynomial that approximates the function
        _derivative (callable): Gradient of the polynomial
        _use_clustering (bool): Whether to use clustering or not
    
    Methods:
    public:
        regress(order : inter, use_clustering : bool, number_of_samples : int or None) -> None: Regresses the function using the sampling object
        approximate(x : numpy.ndarray) -> float or numpy.ndarray: Calculates the polynomial at the given point.
        gradient(x : numpy.ndarray) -> numpy.ndarray: Calculates the gradient at the given point. (OVERWRITTEN)
    private:
        _create_exponents(number_of_exponents : int or None, maximal_samples : int or None) -> list: Creates the exponents of the summands of the polynomial
        _multivariate_regression(samples : numpy.ndarray, values, numpy.ndarray, exponents : list) -> numpy.ndarray: Calculates the coefficients of the multivariate polynomial by solving a linear system of equations.

    Version:
        0.2

    Contributors:
        Niklas Hornischer (nh605@cam.ac.uk)
    """

    def __init__(self, m: int, f: callable, samples : np.ndarray) -> None:
        """
        Constructor for the Regression class.

        ***Inherited from Functional***
        
        Args:
            samples (Sampling): Sampling object that may includes the clusters

        Raises:
            AssertionError: If the dimension of the samples does not match the dimension of the parameter space
        """
        super().__init__(m, f)
        assert samples.m == m, "The dimension of the samples must match the dimension of the parameter space"
        self.samples = samples

    def regress(self, order : int = 2, use_clustering : bool = True,  number_of_samples : int or None = None) -> None:
        """Calculates the coefficients of the multivariate polynomial by solving a linear system of equations.

        Args:
            order (int, optional): Order of the polynomial. Defaults to 2.
            use_clustering (bool, optional): Whether to use clustering or not. Defaults to True.
            number_of_samples (int, optional): Number of samples to use. Defaults to None.
        """

        if number_of_samples == None:
            number_of_samples = self.samples.M

        if not hasattr(self.samples, "_clusters") and use_clustering:
            logger.warning("No clusters found. Regression without clustering")
            use_clustering = False
        if not use_clustering:
            _exponents = self._create_exponents(order)
            logger.info(f"Calculating global regressant with order {order} and {len(_exponents)} exponents and {number_of_samples} samples")

            if len(_exponents) > number_of_samples:
                logger.warning(f"The number of exponents ({len(_exponents)}) is greater than the number of samples ({number_of_samples}). The system is underdetermined")
            # Obtain the data for the interpolation
            _data = self.samples.samples()[:number_of_samples,:]
            if hasattr(self.samples, "_values"):
                _values = self.samples.values()[:number_of_samples]
            else:
                _values = np.asarray(self.evaluate(_data))

            coefficients = self._multivariate_regression(_data, _values, _exponents)
            if hasattr(self, "_polynomial"):
                logger.warning("The polynomial has already been calculated. Overwriting the interpolant")
            self._polynomial = utils.create_polynomial(coefficients, np.asarray(_exponents))
            self._derivative = utils.create_polynomial_derivative(coefficients, np.asarray(_exponents))
            self._use_clustering = False
        
        # Calculates multiple local regressant based on the defined clusters
        else: 
            if hasattr(self, "_polynomial"):
                logger.warning("The polynomials have already been calculated. Overwriting the interpolant")           
            self._polynomial = []
            self._derivative = []
            for cluster_index,index_list in enumerate(self.samples.clusters()):
                # Set number of coefficients to the minimum of the elements in the cluster 
                # and the number of exponents given in **kwargs
                _exponents = self._create_exponents(order)
                if len(_exponents)>len(index_list):
                    logger.warning("The number of exponents is larger than the number of samples in the cluster. The regression is underdetermined.")
                _data = np.asarray([self.samples.extract(i) for i in index_list])
                _data = _data[:min(len(index_list), number_of_samples),:]
            
                logger.info(f"Calculating local polynomial for cluster {cluster_index} with order {order} and {len(_exponents)} exponents and number of samples {min(len(index_list), number_of_samples)}")

                # Check if values have already been calculated
                if hasattr(self.samples, "_values"):
                    _values = np.asarray([self.samples.extract_value(i) for i in index_list])
                    _values = _values[:min(len(index_list), number_of_samples)]
                else:
                    _values = self.evaluate(_data)
                
                coefficients = self._multivariate_regression(_data, _values, _exponents) 
                self._polynomial.append(utils.create_polynomial(coefficients, np.asarray(_exponents)))
                self._derivative.append(utils.create_polynomial_derivative(coefficients, np.asarray(_exponents)))
            self._use_clustering = True

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

    def _multivariate_regression(self, samples : np.ndarray, values : np.ndarray, exponents : list) -> np.ndarray:
        """Calculates the coefficients of the multivariate polynomial by solving a linear system of equations.

        Args:
            samples (numpy.ndarray): Samples of the function
            values (numpy.ndarray): Values of the function at the samples
            exponents (list): List of exponents of the summands

        Returns:
            numpy.ndarray: Coefficients of the multivariate polynomial
        """
        assert np.shape(samples)[0] == len(values), f"The number of samples and values must be equal, but they are {np.shape(samples)} and {len(values)} respectively"

        A=np.ones([samples.shape[0], len(exponents)])
        for i in range(samples.shape[0]):
            for j, exponent in enumerate(exponents):
                A[i,j]=np.prod(samples[i,:]**exponent)
                
        c,_,_,_ = np.linalg.lstsq(A, values, rcond=None)
        return c
    
    def approximate(self, x : np.ndarray) -> float or np.ndarray:
        """Calculates the polynomial at the given point with the specified method.

        Args:
            x (numpy.ndarray): Point or points at which the polynomial is evaluated.
                                Must be shape (n,m) or (m, ) where n is the number 
                                of points and m is the dimension of the function.
        Returns:
            float or numpy.ndarray: Value or values of the function at the given point or points.
                                    The shape is (n,) or float depending on the size of the input.

        Raises:
            AssertionError: If the dimension of the point is not equal to the dimension of the interpolant
            ValueError: If no polynomial was found
        """
        if not hasattr(self, '_polynomial'):
            raise ValueError("No polynomial found")

        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        assert x.shape[1] == self.m, "x must have dimension m"

        values = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            if  not self._use_clustering:
                values[i] = self._polynomial(x[i,:])
            else:
                cluster_idx = self.samples.cluster_index(x[i,:])
                values[i] = self._polynomial[cluster_idx](x[i,:])

        if values.shape[0] == 1:
            values = float(values.reshape(-1)[0])

        return values

    def gradient(self, x : np.ndarray) -> np.ndarray:
        """Calculates the gradient at the given point or points by evaluating the polynomial.

        (OVERWRITTEN)

        Args:
            x (numpy.ndarray): Point or points at which the gradient is calculated.
                                Must be shape (n,m) or (m, ) where n is the number 
                                of points and m is the dimension of the function.
        Returns:
            numpy.ndarray: Gradient or gradients of the function at the given point or points.
                            The shape is (n,m) or (m, ) depending on the size of the input.

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
            if  not self._use_clustering:
                gradient[i,:] = self._derivative(x[i,:])
            else:
                cluster_idx = self.samples.cluster_index(x[i,:])
                gradient[i,:] = self._derivative[cluster_idx](x[i,:])

        if gradient.shape[0] == 1:
            gradient = gradient.reshape(-1)

        return gradient