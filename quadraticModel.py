import numpy as np
import math
import os
import matplotlib.pyplot as plt
from ASFEniCSx.sampling import Sampling
from ASFEniCSx.functional import Functional, Analytical, Regression, Interpolation
from ASFEniCSx.asfenicsx import ASFEniCSx
import ASFEniCSx.utils as utils

dir = os.path.dirname(__file__)

# Check if directory parametrizedPoisson exists if not create it
if not os.path.exists(os.path.join(dir,"quadraticModel")):
    os.makedirs(os.path.join(dir,"quadraticModel"))

os.chdir("quadraticModel")
dir = os.path.join(os.path.dirname(__file__), "quadraticModel")

m = 10          # Number of parameters
alpha = 2       # Oversampling factor (originally 2)
k = 6           # Number of eigenvalues of interest

M = int(alpha * k * np.log(m)) 

samples = Sampling(M, m)
samples.random_uniform()

# Construct eigenvectors by generating an orthogonal basis from a random matrix and normalize them
eigenvectors, _ = np.linalg.qr(np.random.rand(m,m))
normalization = np.sign(eigenvectors[0,:])
normalization[normalization==0] = 1
eigenvectors = eigenvectors * normalization

"""
Eigenvalues with constant decay
"""
# Define the eigenvalues of the matrix A using a exponential decay with constant rate k
rate = -1.125
eigenvalues_constant = np.exp(rate*np.arange(1,m+1)+5.56)

# Define the matrix A
A = eigenvectors @ np.diag(eigenvalues_constant) @ eigenvectors.T

# Define the eigenvalues of the matrix A using a exponential decay with constant rate k but with a larger gap between the first and second eigenvalue
eigenvalues_gap = np.copy(eigenvalues_constant)
eigenvalues_gap[1:] = np.exp(rate*np.arange(2,m+1))

# Define the matrix A
A_gap = eigenvectors @ np.diag(eigenvalues_gap) @ eigenvectors.T

# Define the eigenvalues of the matrix A using a exponential decay with constant rate k but with a larger gap between the third and fourth eigenvalue
eigenvalues_gap2 = np.copy(eigenvalues_constant)
eigenvalues_gap2[3:] = np.exp(rate*np.arange(4,m+1))

# Define the matrix A
A_gap2 = eigenvectors @ np.diag(eigenvalues_gap2) @ eigenvectors.T

######################################################################
# Test the ASFEniCSx class with the quadratic model
######################################################################


"""
Constant Decay
"""
function = lambda x: 0.5 * x.T @ A @ x
grad = lambda x: A @ x

analytical = Analytical(m, function, grad)
asfenicsx = ASFEniCSx(k, analytical, samples)
U_A, S_A = asfenicsx.estimation()
asfenicsx.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
asfenicsx.plot_eigenvalues(true_eigenvalues=1/3*eigenvalues_constant**2, filename=os.path.join(dir,"constantDecay_eigenvalues.pdf"), ylim=[1e-8,1e4])
asfenicsx.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"constantDecay_subspace.pdf"), ylim=[1e-6,1])

"""
Eigenvalues with gap between first and second
"""
function = lambda x: 0.5 * x.T @ A_gap @ x
grad = lambda x: A_gap @ x
analytical = Analytical(m, function, grad)

asfenicsx = ASFEniCSx(k, analytical, samples)
U_A, S_A = asfenicsx.estimation()
asfenicsx.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
asfenicsx.plot_eigenvalues(true_eigenvalues=1/3*eigenvalues_gap**2, filename=os.path.join(dir,"gap_eigenvalues.pdf"), ylim=[1e-8,1e4])
asfenicsx.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"gap_subspace.pdf"), ylim=[1e-6,1])

"""
Eigenvalues with gap between third and fourth
"""

function = lambda x: 0.5 * x.T @ A_gap2 @ x
grad = lambda x: A_gap2 @ x
analytical = Analytical(m, function, grad)

asfenicsx = ASFEniCSx(k, analytical, samples)
U_A, S_A = asfenicsx.estimation()
asfenicsx.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
asfenicsx.plot_eigenvalues(true_eigenvalues=1/3*eigenvalues_gap2**2, filename=os.path.join(dir,"gap2_eigenvalues.pdf"), ylim=[1e-8,1e4])
asfenicsx.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"gap2_subspace.pdf"), ylim=[1e-6,1])

plt.close('all')


######################################################################
# Test the ASFEniCSx class with the quadratic model and Finite Differences
######################################################################

"""
Constant Decay
"""
function = lambda x: 0.5 * x.T @ A @ x
grad = lambda x: A @ x


# Finite Differences
func = Functional(m, function)

asfenicsx = ASFEniCSx(k, func, samples)
asfenicsx.evaluate_gradients(h=1e-1, order = 1)
U_A, S_A = asfenicsx.estimation()
asfenicsx.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
asfenicsx.plot_eigenvalues(true_eigenvalues=1/3*eigenvalues_constant**2, filename=os.path.join(dir,"constantDecay_eigenvalues_FD_1e-1.pdf"), ylim=[1e-8,1e4])
asfenicsx.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"constantDecay_subspace_FD_1e-1.pdf"), ylim=[1e-6,1])

asfenicsx.evaluate_gradients(h=1e-3, order = 1)
U_A, S_A = asfenicsx.estimation()
asfenicsx.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
asfenicsx.plot_eigenvalues(true_eigenvalues=1/3*eigenvalues_constant**2, filename=os.path.join(dir,"constantDecay_eigenvalues_FD_1e-3.pdf"), ylim=[1e-8,1e4])
asfenicsx.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"constantDecay_subspace_FD_1e-3.pdf"), ylim=[1e-6,1])

asfenicsx.evaluate_gradients(h=1e-5, order = 1)
U_A, S_A = asfenicsx.estimation()
asfenicsx.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
asfenicsx.plot_eigenvalues(true_eigenvalues=1/3*eigenvalues_constant**2, filename=os.path.join(dir,"constantDecay_eigenvalues_FD_1e-5.pdf"), ylim=[1e-8,1e4])
asfenicsx.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"constantDecay_subspace_FD_1e-5.pdf"), ylim=[1e-6,1])

"""
Eigenvalues with gap between first and second
"""
function = lambda x: 0.5 * x.T @ A_gap @ x
grad = lambda x: A_gap @ x
analytical = Analytical(m, function, grad)

# Finite Differences
func = Functional(m, function)
asfenicsx = ASFEniCSx(k, func, samples)
asfenicsx.evaluate_gradients(h=1e-1, order = 1)
U_A, S_A = asfenicsx.estimation()
asfenicsx.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
asfenicsx.plot_eigenvalues(true_eigenvalues=1/3*eigenvalues_gap**2, filename=os.path.join(dir,"gap_eigenvalues_FD_1e-1.pdf"), ylim=[1e-8,1e4])
asfenicsx.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"gap_subspace_FD_1e-1.pdf"), ylim=[1e-6,1])

asfenicsx.evaluate_gradients(h=1e-3, order = 1)
U_A, S_A = asfenicsx.estimation()
asfenicsx.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
asfenicsx.plot_eigenvalues(true_eigenvalues=1/3*eigenvalues_gap**2, filename=os.path.join(dir,"gap_eigenvalues_FD_1e-3.pdf"), ylim=[1e-8,1e4])
asfenicsx.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"gap_subspace_FD_1e-3.pdf"), ylim=[1e-6,1])

asfenicsx.evaluate_gradients(h=1e-5, order = 1)
U_A, S_A = asfenicsx.estimation()
asfenicsx.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
asfenicsx.plot_eigenvalues(true_eigenvalues=1/3*eigenvalues_gap**2, filename=os.path.join(dir,"gap_eigenvalues_FD_1e-5.pdf"), ylim=[1e-8,1e4])
asfenicsx.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"gap_subspace_FD_1e-5.pdf"), ylim=[1e-6,1])

"""
Eigenvalues with gap between third and fourth
"""

function = lambda x: 0.5 * x.T @ A_gap2 @ x
grad = lambda x: A_gap2 @ x
analytical = Analytical(m, function, grad)

# Finite Differences
func = Functional(m, function)
asfenicsx = ASFEniCSx(k, func, samples)
asfenicsx.evaluate_gradients(h=1e-1, order = 1)
U_A, S_A = asfenicsx.estimation()
asfenicsx.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
asfenicsx.plot_eigenvalues(true_eigenvalues=1/3*eigenvalues_gap2**2, filename=os.path.join(dir,"gap2_eigenvalues_FD_1e-1.pdf"), ylim=[1e-8,1e4])
asfenicsx.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"gap2_subspace_FD_1e-1.pdf"), ylim=[1e-6,1])

asfenicsx.evaluate_gradients(h=1e-3, order = 1)
U_A, S_A = asfenicsx.estimation()
asfenicsx.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
asfenicsx.plot_eigenvalues(true_eigenvalues=1/3*eigenvalues_gap2**2, filename=os.path.join(dir,"gap2_eigenvalues_FD_1e-3.pdf"), ylim=[1e-8,1e4])
asfenicsx.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"gap2_subspace_FD_1e-3.pdf"), ylim=[1e-6,1])

asfenicsx.evaluate_gradients(h=1e-5, order = 1)
U_A, S_A = asfenicsx.estimation()
asfenicsx.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
asfenicsx.plot_eigenvalues(true_eigenvalues=1/3*eigenvalues_gap2**2, filename=os.path.join(dir,"gap2_eigenvalues_FD_1e-5.pdf"), ylim=[1e-8,1e4])
asfenicsx.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"gap2_subspace_FD_1e-5.pdf"), ylim=[1e-6,1])

plt.close('all')

######################################################################
# Test the gradient methods for the quadratic model with regression
######################################################################

# Define the function and its gradient
function = lambda x: 0.5 * x.T @ A @ x
grad = lambda x: A @ x

analytical = Analytical(m, function, grad)

# Analyse regression
samples_regression = Sampling(100, m)
samples_regression.random_uniform()

A_data = analytical.gradient(samples_regression.samples())

regressant = Regression(m, function, samples_regression)
utils.evaluate_derivative_regression(regressant,2 ,False, os.path.join(dir,"constantDecay_regression.pdf"), A_data)

# Regression
func = Regression(m, function, samples)
func.regression(2, overwrite = True)
asfenicsx = ASFEniCSx(k, func, samples)
U_A, S_A = asfenicsx.estimation()
asfenicsx.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
asfenicsx.plot_eigenvalues(true_eigenvalues=1/3*eigenvalues_constant**2, filename=os.path.join(dir,"constantDecay_eigenvalues_Regression.pdf"), ylim=[1e-8,1e4])
asfenicsx.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"constantDecay_subspace_Regression.pdf"), ylim=[1e-6,1])

func = Regression(m, function, samples_regression)
func.regression(2, overwrite = True)
asfenicsx = ASFEniCSx(k, func, samples)
U_A, S_A = asfenicsx.estimation()
asfenicsx.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
asfenicsx.plot_eigenvalues(true_eigenvalues=1/3*eigenvalues_constant**2, filename=os.path.join(dir,"constantDecay_eigenvalues_Regression_high.pdf"), ylim=[1e-8,1e4])
asfenicsx.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"constantDecay_subspace_Regression_high.pdf"), ylim=[1e-6,1])

# Define the function and its gradient
function = lambda x: 0.5 * x.T @ A_gap @ x
grad = lambda x: A_gap @ x

analytical = Analytical(m, function, grad)

# Analyse regression
A_data = analytical.gradient(samples_regression.samples())

regressant = Regression(m, function, samples_regression)
utils.evaluate_derivative_regression(regressant,2 ,False, os.path.join(dir,"gap_regression.pdf"), A_data)


# Regression
func = Regression(m, function, samples)
func.regression(2, overwrite = True)
asfenicsx = ASFEniCSx(k, func, samples)
U_A, S_A = asfenicsx.estimation()
asfenicsx.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
asfenicsx.plot_eigenvalues(true_eigenvalues=1/3*eigenvalues_gap**2, filename=os.path.join(dir,"gap_eigenvalues_Regression.pdf"), ylim=[1e-8,1e4])
asfenicsx.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"gap_subspace_Regression.pdf"), ylim=[1e-6,1])

func = Regression(m, function, samples_regression)
func.regression(2, overwrite = True)
asfenicsx = ASFEniCSx(k, func, samples)
U_A, S_A = asfenicsx.estimation()
asfenicsx.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
asfenicsx.plot_eigenvalues(true_eigenvalues=1/3*eigenvalues_gap**2, filename=os.path.join(dir,"gap_eigenvalues_Regression_high.pdf"), ylim=[1e-8,1e4])
asfenicsx.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"gap_subspace_Regression_high.pdf"), ylim=[1e-6,1])

# Define the function and its gradient
function = lambda x: 0.5 * x.T @ A_gap2 @ x
grad = lambda x: A_gap2 @ x

analytical = Analytical(m, function, grad)

# Analyse regression
A_data = analytical.gradient(samples_regression.samples())

regressant = Regression(m, function, samples_regression)
utils.evaluate_derivative_regression(regressant,2 ,False, os.path.join(dir,"gap2_regression.pdf"), A_data)


# Regression
func = Regression(m, function, samples)
func.regression(2, overwrite = True)
asfenicsx = ASFEniCSx(k, func, samples)
U_A, S_A = asfenicsx.estimation()
asfenicsx.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
asfenicsx.plot_eigenvalues(true_eigenvalues=1/3*eigenvalues_gap2**2, filename=os.path.join(dir,"gap2_eigenvalues_Regression.pdf"), ylim=[1e-8,1e4])
asfenicsx.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"gap2_subspace_Regression.pdf"), ylim=[1e-6,1])

func = Regression(m, function, samples_regression)
func.regression(2, overwrite = True)
asfenicsx = ASFEniCSx(k, func, samples)
U_A, S_A = asfenicsx.estimation()
asfenicsx.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
asfenicsx.plot_eigenvalues(true_eigenvalues=1/3*eigenvalues_gap2**2, filename=os.path.join(dir,"gap2_eigenvalues_Regression_high.pdf"), ylim=[1e-8,1e4])
asfenicsx.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"gap2_subspace_Regression_high.pdf"), ylim=[1e-6,1])

######################################################################
# Test the gradient methods for the quadratic model with interpolation
######################################################################

# Define the function and its gradient
function = lambda x: 0.5 * x.T @ A @ x
grad = lambda x: A @ x

analytical = Analytical(m, function, grad)

# Analyse interpolation
A_data = analytical.gradient(samples_regression.samples())
samples_regression.assign_values(function)

interpolant = Interpolation(m, function, samples_regression)
utils.evaluate_derivative_interpolation(interpolant, 2 ,False, os.path.join(dir,"constantDecay_interpolation.pdf"), A_data)


# Interpolation
func = Interpolation(m, function, samples)
func.interpolate(1, overwrite = True)
asfenicsx = ASFEniCSx(k, func, samples)
U_A, S_A = asfenicsx.estimation()
asfenicsx.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
asfenicsx.plot_eigenvalues(true_eigenvalues=1/3*eigenvalues_constant**2, filename=os.path.join(dir,"eigenvalues_Interpolation_1.pdf"), ylim=[1e-8,1e4])
asfenicsx.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"subspace_Interpolation_1.pdf"), ylim=[1e-6,1])

func.interpolate(2, overwrite = True)
asfenicsx = ASFEniCSx(k, func, samples)
U_A, S_A = asfenicsx.estimation()
asfenicsx.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
asfenicsx.plot_eigenvalues(true_eigenvalues=1/3*eigenvalues_constant**2, filename=os.path.join(dir,"eigenvalues_Interpolation_2.pdf"), ylim=[1e-8,1e4])
asfenicsx.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"subspace_Interpolation_2.pdf"), ylim=[1e-6,1])

# Define the function and its gradient
function = lambda x: 0.5 * x.T @ A_gap @ x
grad = lambda x: A_gap @ x

analytical = Analytical(m, function, grad)
samples_regression.assign_values(function, overwrite = True)

# Analyse interpolation
A_data = analytical.gradient(samples_regression.samples())

interpolant = Interpolation(m, function, samples_regression)
utils.evaluate_derivative_interpolation(interpolant, 2 ,False, os.path.join(dir,"gap_interpolation.pdf"), A_data)


# Interpolation
func = Interpolation(m, function, samples)
func.interpolate(1, overwrite = True)
asfenicsx = ASFEniCSx(k, func, samples)
U_A, S_A = asfenicsx.estimation()
asfenicsx.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
asfenicsx.plot_eigenvalues(true_eigenvalues=1/3*eigenvalues_gap**2, filename=os.path.join(dir,"gap_eigenvalues_Interpolation_1.pdf"), ylim=[1e-8,1e4])
asfenicsx.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"gap_subspace_Interpolation_1.pdf"), ylim=[1e-6,1])

func.interpolate(2, overwrite = True)
asfenicsx = ASFEniCSx(k, func, samples)
U_A, S_A = asfenicsx.estimation()
asfenicsx.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
asfenicsx.plot_eigenvalues(true_eigenvalues=1/3*eigenvalues_gap**2, filename=os.path.join(dir,"gap_eigenvalues_Interpolation_2.pdf"), ylim=[1e-8,1e4])
asfenicsx.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"gap_subspace_Interpolation_2.pdf"), ylim=[1e-6,1])

# Define the function and its gradient
function = lambda x: 0.5 * x.T @ A_gap2 @ x
grad = lambda x: A_gap2 @ x

analytical = Analytical(m, function, grad)
samples_regression.assign_values(function, overwrite = True)

# Analyse interpolation
A_data = analytical.gradient(samples_regression.samples())

interpolant = Interpolation(m, function, samples_regression)
utils.evaluate_derivative_interpolation(interpolant, 2 ,False, os.path.join(dir,"gap2_interpolation.pdf"), A_data)


# Interpolation
func = Interpolation(m, function, samples)
func.interpolate(1, overwrite = True)
asfenicsx = ASFEniCSx(k, func, samples)
U_A, S_A = asfenicsx.estimation()
asfenicsx.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
asfenicsx.plot_eigenvalues(true_eigenvalues=1/3*eigenvalues_gap2**2, filename=os.path.join(dir,"gap2_eigenvalues_Interpolation_1.pdf"), ylim=[1e-8,1e4])
asfenicsx.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"gap2_subspace_Interpolation_1.pdf"), ylim=[1e-6,1])

