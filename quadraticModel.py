import numpy as np
import os
import scipy
import matplotlib.pyplot as plt
from asfenicsx import sampling, functional, ASFEniCSx

dir = os.path.dirname(__file__)

# Check if directory parametrizedPoisson exists if not create it
if not os.path.exists(os.path.join(dir,"quadraticModel")):
    os.makedirs(os.path.join(dir,"quadraticModel/quadraticModel"))


m = 10          # Number of parameters
alpha = 2       # Oversampling factor (originally 2)
k = 6           # Number of eigenvalues of interest

M = int(alpha * k * np.log(m)) 

samples = sampling(M, m)

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

################################################################################
# Sanity Check
################################################################################
C = 1/3 * A @ A
eigs_C = np.linalg.eigvalsh(C)
idx = eigs_C.argsort()[::-1]
eigs_C = eigs_C[idx]

eigs_A = np.linalg.eigvalsh(A)
idx = eigs_A.argsort()[::-1]
eigs_A = eigs_A[idx]

ax = plt.figure().gca()
ax.plot(range(1,m+1),eigs_C, marker="o", fillstyle="none", label="eig(C)")
ax.plot(range(1,m+1),eigenvalues_constant**2, marker="*", fillstyle="none", linestyle="--", label="eig(A)^2")
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.yscale('log')
plt.grid()
plt.xlabel("Eigenvalue Index")
plt.ylabel("Eigenvalue")
plt.title("Eigenvalues of the Correlation Matrix")
plt.legend()

# TODO: Why are the eigenvalues of C not equivalent to the squared eigenvalues of A when they should be

######################################################################
# Test the gradient methods for the quadratic model
######################################################################

# Define the function and its gradient
function = lambda x: 0.5 * x.T @ A @ x
grad = lambda x: A @ x

# Compute the true gradients
func = functional(m, function)
func.get_derivative(grad)
func.get_gradient_method('A')

true_gradients = np.zeros((m,M))
for i in range(M):
    true_gradients[:,i] = func.gradient(samples.extract(i))

func.get_gradient_method('FD')
FD_gradients = np.zeros((m,M))
for i in range(M):
    FD_gradients[:,i] = func.gradient(samples.extract(i), h=1e-3, order=1)

FD_gradients_2 = np.zeros((m,M))
for i in range(M):
    FD_gradients_2[:,i] = func.gradient(samples.extract(i), h=1e-3, order=2)

func.get_gradient_method('I')
func.interpolation(samples, order=1)
I_gradients = np.zeros((m,M))
for i in range(M):
    I_gradients[:,i] = func.gradient(samples.extract(i))

func.interpolation(samples, interpolation_method='LS', order = 1, overwrite=True)
LS_gradients = np.zeros((m,M))
for i in range(M):
    LS_gradients[:,i] = func.gradient(samples.extract(i))

func.interpolation(samples, interpolation_method='LS', order = 2, overwrite=True)
LS_gradients2 = np.zeros((m,M))
for i in range(M):
    LS_gradients2[:,i] = func.gradient(samples.extract(i))

# Compute the errors
FD_error = np.zeros(M)
FD_error2 = np.zeros(M)
I_error = np.zeros(M)
LS_error = np.zeros(M)
LS_error2 = np.zeros(M)
for i in range(M):
    FD_error[i] = np.linalg.norm(FD_gradients[:,i] - true_gradients[:,i], ord=2)/ np.linalg.norm(true_gradients[:,i], ord=2)
    FD_error2[i] = np.linalg.norm(FD_gradients_2[:,i] - true_gradients[:,i], ord=2)/ np.linalg.norm(true_gradients[:,i], ord=2)
    I_error[i] = np.linalg.norm(I_gradients[:,i] - true_gradients[:,i], ord=2)/ np.linalg.norm(true_gradients[:,i], ord=2)
    LS_error[i] = np.linalg.norm(LS_gradients[:,i] - true_gradients[:,i], ord=2) / np.linalg.norm(true_gradients[:,i], ord=2)
    LS_error2[i] = np.linalg.norm(LS_gradients2[:,i] - true_gradients[:,i], ord=2) / np.linalg.norm(true_gradients[:,i], ord=2)

# Plot the errors
ax = plt.figure().gca()
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax.plot(range(1,M+1), FD_error, marker="o", fillstyle="none", label="FD-order1")
ax.plot(range(1,M+1), FD_error2, marker="s", fillstyle="none", label="FD-order2")
ax.plot(range(1,M+1), I_error, marker="*", fillstyle="none", label="I")
ax.plot(range(1,M+1), LS_error, marker="x", fillstyle="none", label="LS-order1")
ax.plot(range(1,M+1), LS_error2, marker="^", fillstyle="none", label="LS-order2")
plt.yscale('log')
plt.xlabel("Sample Index")
plt.ylabel("Error")
plt.legend()


# Define the function and its gradient
function = lambda x: 0.5 * x.T @ A_gap @ x
grad = lambda x: A_gap @ x

# Compute the true gradients
func = functional(m, function)
func.get_derivative(grad)
func.get_gradient_method('A')

true_gradients = np.zeros((m,M))
for i in range(M):
    true_gradients[:,i] = func.gradient(samples.extract(i))

func.get_gradient_method('FD')
FD_gradients = np.zeros((m,M))
for i in range(M):
    FD_gradients[:,i] = func.gradient(samples.extract(i), h=1e-3, order=1)

FD_gradients_2 = np.zeros((m,M))
for i in range(M):
    FD_gradients_2[:,i] = func.gradient(samples.extract(i), h=1e-3, order=2)

func.get_gradient_method('I')
func.interpolation(samples, order=1)
I_gradients = np.zeros((m,M))
for i in range(M):
    I_gradients[:,i] = func.gradient(samples.extract(i))

func.interpolation(samples, interpolation_method='LS', order = 1, overwrite=True)
LS_gradients = np.zeros((m,M))
for i in range(M):
    LS_gradients[:,i] = func.gradient(samples.extract(i))

func.interpolation(samples, interpolation_method='LS', order = 2, overwrite=True)
LS_gradients2 = np.zeros((m,M))
for i in range(M):
    LS_gradients2[:,i] = func.gradient(samples.extract(i))

# Compute the errors
FD_error = np.zeros(M)
FD_error2 = np.zeros(M)
I_error = np.zeros(M)
LS_error = np.zeros(M)
LS_error2 = np.zeros(M)
for i in range(M):
    FD_error[i] = np.linalg.norm(FD_gradients[:,i] - true_gradients[:,i], ord=2)/ np.linalg.norm(true_gradients[:,i], ord=2)
    FD_error2[i] = np.linalg.norm(FD_gradients_2[:,i] - true_gradients[:,i], ord=2)/ np.linalg.norm(true_gradients[:,i], ord=2)
    I_error[i] = np.linalg.norm(I_gradients[:,i] - true_gradients[:,i], ord=2)/ np.linalg.norm(true_gradients[:,i], ord=2)
    LS_error[i] = np.linalg.norm(LS_gradients[:,i] - true_gradients[:,i], ord=2) / np.linalg.norm(true_gradients[:,i], ord=2)
    LS_error2[i] = np.linalg.norm(LS_gradients2[:,i] - true_gradients[:,i], ord=2) / np.linalg.norm(true_gradients[:,i], ord=2)

# Plot the errors
ax = plt.figure().gca()
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax.plot(range(1,M+1), FD_error, marker="o", fillstyle="none", label="FD-order1")
ax.plot(range(1,M+1), FD_error2, marker="s", fillstyle="none", label="FD-order2")
ax.plot(range(1,M+1), I_error, marker="*", fillstyle="none", label="I")
ax.plot(range(1,M+1), LS_error, marker="x", fillstyle="none", label="LS-order1")
ax.plot(range(1,M+1), LS_error2, marker="^", fillstyle="none", label="LS-order2")
plt.yscale('log')
plt.xlabel("Sample Index")
plt.ylabel("Error")
plt.legend()

# Define the function and its gradient
function = lambda x: 0.5 * x.T @ A_gap2 @ x
grad = lambda x: A_gap2 @ x

# Compute the true gradients
func = functional(m, function)
func.get_derivative(grad)
func.get_gradient_method('A')

true_gradients = np.zeros((m,M))
for i in range(M):
    true_gradients[:,i] = func.gradient(samples.extract(i))

func.get_gradient_method('FD')
FD_gradients = np.zeros((m,M))
for i in range(M):
    FD_gradients[:,i] = func.gradient(samples.extract(i), h=1e-3, order=1)

FD_gradients_2 = np.zeros((m,M))
for i in range(M):
    FD_gradients_2[:,i] = func.gradient(samples.extract(i), h=1e-3, order=2)

func.get_gradient_method('I')
func.interpolation(samples, order=1)
I_gradients = np.zeros((m,M))
for i in range(M):
    I_gradients[:,i] = func.gradient(samples.extract(i))

func.interpolation(samples, interpolation_method='LS', order = 1, overwrite=True)
LS_gradients = np.zeros((m,M))
for i in range(M):
    LS_gradients[:,i] = func.gradient(samples.extract(i))

func.interpolation(samples, interpolation_method='LS', order = 2, overwrite=True)
LS_gradients2 = np.zeros((m,M))
for i in range(M):
    LS_gradients2[:,i] = func.gradient(samples.extract(i))

# Compute the errors
FD_error = np.zeros(M)
FD_error2 = np.zeros(M)
I_error = np.zeros(M)
LS_error = np.zeros(M)
LS_error2 = np.zeros(M)
for i in range(M):
    FD_error[i] = np.linalg.norm(FD_gradients[:,i] - true_gradients[:,i], ord=2)/ np.linalg.norm(true_gradients[:,i], ord=2)
    FD_error2[i] = np.linalg.norm(FD_gradients_2[:,i] - true_gradients[:,i], ord=2)/ np.linalg.norm(true_gradients[:,i], ord=2)
    I_error[i] = np.linalg.norm(I_gradients[:,i] - true_gradients[:,i], ord=2)/ np.linalg.norm(true_gradients[:,i], ord=2)
    LS_error[i] = np.linalg.norm(LS_gradients[:,i] - true_gradients[:,i], ord=2) / np.linalg.norm(true_gradients[:,i], ord=2)
    LS_error2[i] = np.linalg.norm(LS_gradients2[:,i] - true_gradients[:,i], ord=2) / np.linalg.norm(true_gradients[:,i], ord=2)

# Plot the errors
ax = plt.figure().gca()
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax.plot(range(1,M+1), FD_error, marker="o", fillstyle="none", label="FD-order1")
ax.plot(range(1,M+1), FD_error2, marker="s", fillstyle="none", label="FD-order2")
ax.plot(range(1,M+1), I_error, marker="*", fillstyle="none", label="I")
ax.plot(range(1,M+1), LS_error, marker="x", fillstyle="none", label="LS-order-1")
ax.plot(range(1,M+1), LS_error2, marker="^", fillstyle="none", label="LS-order2")
plt.yscale('log')
plt.xlabel("Sample Index")
plt.ylabel("Error")
plt.legend()

######################################################################
# Test the ASFEniCSx class with the quadratic model
######################################################################


function = lambda x: 0.5 * x.T @ A @ x
grad = lambda x: A @ x

func = functional(m, function)
func.get_derivative(grad)

func.get_gradient_method('A')
as_A = ASFEniCSx(k, func, samples)
U_A, S_A = as_A.random_sampling_algorithm()
[e_max_A, e_min_A],[sub_max_A, sub_min_A, sub_mean_A] = as_A.bootstrap(100)

# Define true subspace error
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)

as_A.plot_eigenvalues(true_eigenvalues=eigenvalues_constant**2, filename=os.path.join(dir,"quadraticModel/constant_decay"), ylim=[1e-8,1e4])
as_A.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"quadraticModel/subspace_constant_decay"), ylim=[1e-6,1])

func.get_gradient_method('FD')
as_A = ASFEniCSx(k, func, samples)
as_A.evaluate_gradients(h=1e-1, order = 1)
U_A, S_A = as_A.random_sampling_algorithm()
as_A.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
as_A.plot_eigenvalues(true_eigenvalues=eigenvalues_constant**2, filename=os.path.join(dir,"quadraticModel/constant_decay_FD_1e-1_1"), ylim=[1e-8,1e4])
as_A.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"quadraticModel/subspace_constant_decay_FD_1e-1_1"), ylim=[1e-6,1])

as_A.evaluate_gradients(h=1e-3, order = 1)
U_A, S_A = as_A.random_sampling_algorithm()
as_A.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
as_A.plot_eigenvalues(true_eigenvalues=eigenvalues_constant**2, filename=os.path.join(dir,"quadraticModel/constant_decay_FD_1e-3_1"), ylim=[1e-8,1e4])
as_A.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"quadraticModel/subspace_constant_decay_FD_1e-3_1"), ylim=[1e-6,1])

as_A.evaluate_gradients(h=1e-5, order = 1)
U_A, S_A = as_A.random_sampling_algorithm()
as_A.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
as_A.plot_eigenvalues(true_eigenvalues=eigenvalues_constant**2, filename=os.path.join(dir,"quadraticModel/constant_decay_FD_1e-5_1"), ylim=[1e-8,1e4])
as_A.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"quadraticModel/subspace_constant_decay_FD_1e-5_1"), ylim=[1e-6,1])

as_A.evaluate_gradients(h=1e-1, order = 2)
U_A, S_A = as_A.random_sampling_algorithm()
as_A.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
as_A.plot_eigenvalues(true_eigenvalues=eigenvalues_constant**2, filename=os.path.join(dir,"quadraticModel/constant_decay_FD_1e-1_2"), ylim=[1e-8,1e4])
as_A.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"quadraticModel/subspace_constant_decay_FD_1e-1_2"), ylim=[1e-6,1])

func.get_gradient_method('I')
func.interpolation(samples, interpolation_method = 'LS', order = 1)
as_A = ASFEniCSx(k, func, samples)
as_A.evaluate_gradients()
U_A, S_A = as_A.random_sampling_algorithm()
as_A.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
as_A.plot_eigenvalues(true_eigenvalues=eigenvalues_constant**2, filename=os.path.join(dir,"quadraticModel/constant_decay_I_1"), ylim=[1e-8,1e4])
as_A.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"quadraticModel/subspace_constant_decay_I_1"), ylim=[1e-6,1])

func.get_gradient_method('I')
func.interpolation(samples, interpolation_method = 'LS', order = 2, overwrite=True)
as_A = ASFEniCSx(k, func, samples)
as_A.evaluate_gradients()
U_A, S_A = as_A.random_sampling_algorithm()
as_A.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
as_A.plot_eigenvalues(true_eigenvalues=eigenvalues_constant**2, filename=os.path.join(dir,"quadraticModel/constant_decay_I_2"), ylim=[1e-8,1e4])
as_A.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"quadraticModel/subspace_constant_decay_I_2"), ylim=[1e-6,1])


"""
Eigenvalues with gap between first and second
"""
function = lambda x: 0.5 * x.T @ A_gap @ x
grad = lambda x: A_gap @ x

func = functional(m, function)
func.get_derivative(grad)

func.get_gradient_method('A')
as_A = ASFEniCSx(k, func, samples)
U_A, S_A = as_A.random_sampling_algorithm()
as_A.bootstrap(100)

# Define true subspace error
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)

as_A.plot_eigenvalues(true_eigenvalues=eigenvalues_gap**2, filename=os.path.join(dir,"quadraticModel/gap_decay"), ylim=[1e-8,1e4])
as_A.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"quadraticModel/subspace_gap_decay"), ylim=[1e-6,1])

function = lambda x: 0.5 * x.T @ A_gap @ x

func = functional(m, function)

func.get_gradient_method('FD')
as_A = ASFEniCSx(k, func, samples)
as_A.evaluate_gradients(h=1e-1, order = 1)
U_A, S_A = as_A.random_sampling_algorithm()
as_A.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
as_A.plot_eigenvalues(true_eigenvalues=eigenvalues_gap**2, filename=os.path.join(dir,"quadraticModel/gap_FD_1e-1_1"), ylim=[1e-8,1e4])
as_A.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"quadraticModel/subspace_gap_FD_1e-1_1"), ylim=[1e-6,1])

as_A.evaluate_gradients(h=1e-3, order = 1)
U_A, S_A = as_A.random_sampling_algorithm()
as_A.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
as_A.plot_eigenvalues(true_eigenvalues=eigenvalues_gap**2, filename=os.path.join(dir,"quadraticModel/gap_FD_1e-3_1"), ylim=[1e-8,1e4])
as_A.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"quadraticModel/subspace_gap_FD_1e-3_1"), ylim=[1e-6,1])

as_A.evaluate_gradients(h=1e-5, order = 1)
U_A, S_A = as_A.random_sampling_algorithm()
as_A.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
as_A.plot_eigenvalues(true_eigenvalues=eigenvalues_gap**2, filename=os.path.join(dir,"quadraticModel/gap_FD_1e-5_1"), ylim=[1e-8,1e4])
as_A.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"quadraticModel/subspace_gap_FD_1e-5_1"), ylim=[1e-6,1])

as_A.evaluate_gradients(h=1e-1, order = 2)
U_A, S_A = as_A.random_sampling_algorithm()
as_A.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
as_A.plot_eigenvalues(true_eigenvalues=eigenvalues_gap**2, filename=os.path.join(dir,"quadraticModel/gap_FD_1e-1_2"), ylim=[1e-8,1e4])
as_A.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"quadraticModel/subspace_gap_FD_1e-1_2"), ylim=[1e-6,1])

func.get_gradient_method('I')
func.interpolation(samples, interpolation_method='LS', order = 1)
as_I = ASFEniCSx(k, func, samples)
U_I, S_I = as_I.random_sampling_algorithm()
as_I.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_I[:,i+1:]), ord=2)
as_I.plot_eigenvalues(true_eigenvalues=eigenvalues_gap**2, filename=os.path.join(dir,"quadraticModel/gap_I_1"), ylim=[1e-8,1e4])
as_I.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"quadraticModel/subspace_gap_I_1"), ylim=[1e-6,1])

func.get_gradient_method('I')
func.interpolation(samples, interpolation_method='LS', order = 2, overwrite=True)
as_I = ASFEniCSx(k, func, samples)
U_I, S_I = as_I.random_sampling_algorithm()
as_I.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_I[:,i+1:]), ord=2)
as_I.plot_eigenvalues(true_eigenvalues=eigenvalues_gap**2, filename=os.path.join(dir,"quadraticModel/gap_I_2"), ylim=[1e-8,1e4])
as_I.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"quadraticModel/subspace_gap_I_2"), ylim=[1e-6,1])

"""Eigenvalues with gap between third and fourth"""

function = lambda x: 0.5 * x.T @ A_gap2 @ x
grad = lambda x: A_gap2 @ x

func = functional(m, function)
func.get_derivative(grad)

func.get_gradient_method('A')
as_A = ASFEniCSx(k, func, samples)
U_A, S_A = as_A.random_sampling_algorithm()
as_A.bootstrap(100)

# Define true subspace error
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)

as_A.plot_eigenvalues(true_eigenvalues=eigenvalues_gap2**2, filename=os.path.join(dir,"quadraticModel/gap2_decay"), ylim=[1e-8,1e4])
as_A.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"quadraticModel/subspace_gap2_decay"), ylim=[1e-6,1])

func.get_gradient_method('FD')
as_A = ASFEniCSx(k, func, samples)
as_A.evaluate_gradients(h=1e-1, order = 1)
U_A, S_A = as_A.random_sampling_algorithm()
as_A.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
as_A.plot_eigenvalues(true_eigenvalues=eigenvalues_gap2**2, filename=os.path.join(dir,"quadraticModel/gap2_FD_1e-1_1"), ylim=[1e-8,1e4])
as_A.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"quadraticModel/subspace_gap2_FD_1e-1_1"), ylim=[1e-6,1])

as_A.evaluate_gradients(h=1e-3, order = 1)
U_A, S_A = as_A.random_sampling_algorithm()
as_A.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
as_A.plot_eigenvalues(true_eigenvalues=eigenvalues_gap2**2, filename=os.path.join(dir,"quadraticModel/gap2_FD_1e-3_1"), ylim=[1e-8,1e4])
as_A.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"quadraticModel/subspace_gap2_FD_1e-3_1"), ylim=[1e-6,1])

as_A.evaluate_gradients(h=1e-5, order = 1)
U_A, S_A = as_A.random_sampling_algorithm()
as_A.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
as_A.plot_eigenvalues(true_eigenvalues=eigenvalues_gap2**2, filename=os.path.join(dir,"quadraticModel/gap2_FD_1e-5_1"), ylim=[1e-8,1e4])
as_A.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"quadraticModel/subspace_gap2_FD_1e-5_1"), ylim=[1e-6,1])

as_A.evaluate_gradients(h=1e-1, order = 2)
U_A, S_A = as_A.random_sampling_algorithm()
as_A.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
as_A.plot_eigenvalues(true_eigenvalues=eigenvalues_gap2**2, filename=os.path.join(dir,"quadraticModel/gap2_FD_1e-1_2"), ylim=[1e-8,1e4])
as_A.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"quadraticModel/subspace_gap2_FD_1e-1_2"), ylim=[1e-6,1])

func.get_gradient_method('I')
func.interpolation(samples, order=1, interpolation_method="LS", overwrite=True)

as_I = ASFEniCSx(k, func, samples)
U_I, S_I = as_I.random_sampling_algorithm()
as_I.bootstrap(100)

# Define true subspace error
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_I[:,i+1:]), ord=2)

as_I.plot_eigenvalues(true_eigenvalues=eigenvalues_gap2**2, filename=os.path.join(dir,"quadraticModel/gap2_LS_1"), ylim=[1e-8,1e4])
as_I.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"quadraticModel/subspace_gap2_LS_1"), ylim=[1e-6,1])

func.get_gradient_method('I')
func.interpolation(samples, order=2, interpolation_method="LS", overwrite=True)

as_I = ASFEniCSx(k, func, samples)
U_I, S_I = as_I.random_sampling_algorithm()
as_I.bootstrap(100)

# Define true subspace error
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_I[:,i+1:]), ord=2)

as_I.plot_eigenvalues(true_eigenvalues=eigenvalues_gap2**2, filename=os.path.join(dir,"quadraticModel/gap2_LS_2"), ylim=[1e-8,1e4])
as_I.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"quadraticModel/subspace_gap2_LS_2"), ylim=[1e-6,1])


plt.show()
