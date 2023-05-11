import numpy as np
import scipy

class ASFEniCSx:

    def __init__(self, n, cost, samples):  
        self.n = n
        self.cost = cost
        self.samples = samples

    def covariance(self):
        covariance = np.zeros([self.cost.m, self.cost.m])
        for i in range(self.samples.M):
            grad= self.cost.grad_f(self.samples.samples[i])
            covariance += np.outer(grad, grad)
        covariance = covariance / self.samples.M
        return covariance
    
    def eigendecomposition(self, matrix):
        U, S, _ = scipy.linalg.svd(matrix, full_matrices=True)
        return U, S**2
    
    def random_sampling_algorithm(self):
        covariance = self.covariance()
        U, S = self.eigendecomposition(covariance)
        return (U[:,0:self.n], S[0:self.n]**2)
    


class Sampling:
    def __init__(self, M, m, func = None):
        self.M = M
        self.m = m
        if hasattr(func, '__call__'):
            self.samples = self.random_custom(M, m, func)
        else:   
            self.samples = self.random_uniform(m)

    def random_uniform(self, m):
        samples = np.zeros([self.M, m])
        for i in range(self.M):
            samples[i] = np.random.uniform(-1, 1, m)
        return samples
    
    def print_samples(self):
        print(f"Samples of the active subspace method: [M x m]=[{self.M},{self.m}] \n{self.samples}")

    def random_custom(self, M, m , func):
# TODO: Implement custom sampling
        return None
    

class CostFunctional:    

    __methods = list(["FFD", "BFD", "ZFD"])
    __approximative_approach=False
    def __init__(self, m, f, grad_f=None, method = "FFD", order=1):
        assert(m>0, "m must be positive")
        self.m = m         # dimension of parameter space
        self.f = f
        if hasattr(grad_f, '__call__'):
            self.grad_f = grad_f
        else: 
            self.__approximative_approach=True
            if grad_f in self.__methods:
                self.approximation_method = method
                self.approximation_order = order
            else:
                Warning("Invalid approximation method. Using FD")

    
    def get_quantity_of_interest(self, func):
        self.f = func

    def get_gradient(self, func):
        self.grad_f = func

    def gradient(self, x):
        assert(len(x)==self.m, "x must have dimension m")
        if self.__approximative_approach:
            return self.approximate_gradient(x)
        else:
            return self.evaluate_gradient(x)

    def finite_differences(self, x, h=1e-6, axis=0, method = "ZFD"):
        assert(len(x)==self.m, "x must have dimension m")
        assert(h>0, "h must be positive")
        assert(axis<len(x), "Axis out of range")
        
        # f(x)
        f_0= self.f(x)

        #f(x+e_ih)
        x[axis] += h
        f_1 = self.f(x)
        
        #f(x-e_ih)
        x[axis] -= 2*h
        f_2 = self.f(x)

        if method == "FFD":
            # First-order forward finite differences (FFD) 
            # \frac{\partial f(x)}{\partial x_i} = (f(x+e_ih)-f(x))/h + O(h)
            return (f_1-f_0) / h
        elif method == "BFD":
            # First-order backward finite differences (BFD)
            # \frac{\partial f(x)}{\partial x_i} = (f(x)-f(x-e_ih))/h + O(h)
            return (f_0-f_2) / h
        else:
            # Second-order central finite differences (ZFD)
            # \frac{\partial f(x)}{\partial x_i} = (f(x+e_ih)-f(x-e_ih))/(2h) + O(h^2)
            return (f_1-f_2) / (2*h)

        
    def evaluate_gradient(self, x):
        assert(len(x)==self.m, "x must have dimension m")
        return self.grad_f(x)
    
    def approximate_gradient(self, x, method = "ZFD"):
        local_gradient=np.zeros(self.m)
        for i in range(self.m):
            local_gradient[i] = self.finite_differences(x, axis=i, method=method)
        return local_gradient
    



    