import numpy as np
import os
import matplotlib.pyplot as plt
from ASFEniCSx.sampling import sampling
from ASFEniCSx.functional import functional
from ASFEniCSx.asfenicsx import ASFEniCSx

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

grads = np.asarray([A @ samples.extract(i) for i in range(M)])
# Define the function and its gradient
function = lambda x: 0.5 * x.T @ A @ x
grad = lambda x: A @ x

# Compute the true gradients
func = functional(m, function)
func.get_derivative(grad)
func.get_gradient_method('A')

test_asfenicsx = ASFEniCSx(k, func, samples)
cov = test_asfenicsx.covariance(grads)
eigs_cov = np.linalg.eigvalsh(cov)
idx = eigs_cov.argsort()[::-1]
eigs_cov = eigs_cov[idx]

eigs_A = np.linalg.eigvalsh(A)
idx = eigs_A.argsort()[::-1]
eigs_A = eigs_A[idx]

_, eigs_asfenicsx = test_asfenicsx.random_sampling_algorithm()

ax = plt.figure().gca()
ax.plot(range(1,m+1),eigs_C, marker="o", fillstyle="none", label="eig(C)")
ax.plot(range(1,m+1),eigs_cov, marker="+", fillstyle="none", label="eig(cov)")
ax.plot(range(1,m+1),eigs_A**2, marker="*", fillstyle="none", label="eig(A)^2")
ax.plot(range(1,m+1),eigenvalues_constant**2, marker="*", fillstyle="none", linestyle="--", label="eigs^2")
ax.plot(range(1,m+1),eigs_asfenicsx, marker="x", fillstyle="none", linestyle="--", label="eigs_asfenicsx")
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.yscale('log')
plt.grid()
plt.xlabel("Eigenvalue Index")
plt.ylabel("Eigenvalue")
plt.title("Eigenvalues of the Correlation Matrix")
plt.legend()


plt.show()
exit()

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
plt.savefig("quadraticModel/constantDecay_gradient_errors")


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
plt.savefig("quadraticModel/gap_gradient_errors")

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
plt.savefig("quadraticModel/gap2_gradient_errors")

######################################################################
# Test the ASFEniCSx class with the quadratic model
######################################################################


"""
Constant Decay
"""
function = lambda x: 0.5 * x.T @ A @ x
grad = lambda x: A @ x
func = functional(m, function)
func.get_derivative(grad)

func.get_gradient_method('A')
asfenicsx = ASFEniCSx(k, func, samples)
U_A, S_A = asfenicsx.random_sampling_algorithm()
asfenicsx.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
asfenicsx.plot_eigenvalues(true_eigenvalues=eigenvalues_constant**2, filename=os.path.join(dir,"quadraticModel/constantDecay_eigenvalues"), ylim=[1e-8,1e4])
asfenicsx.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"quadraticModel/constantDecay_subspace"), ylim=[1e-6,1])

# Finite Differences
func.get_gradient_method('FD')
asfenicsx.evaluate_gradients(h=1e-1, order = 1)
U_A, S_A = asfenicsx.random_sampling_algorithm()
asfenicsx.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
asfenicsx.plot_eigenvalues(true_eigenvalues=eigenvalues_constant**2, filename=os.path.join(dir,"quadraticModel/constantDecay_eigenvalues_FD_1e-1_1"), ylim=[1e-8,1e4])
asfenicsx.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"quadraticModel/constantDecay_subspace_FD_1e-1_1"), ylim=[1e-6,1])

asfenicsx.evaluate_gradients(h=1e-3, order = 1)
U_A, S_A = asfenicsx.random_sampling_algorithm()
asfenicsx.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
asfenicsx.plot_eigenvalues(true_eigenvalues=eigenvalues_constant**2, filename=os.path.join(dir,"quadraticModel/constantDecay_eigenvalues_FD_1e-3_1"), ylim=[1e-8,1e4])
asfenicsx.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"quadraticModel/constantDecay_subspace_FD_1e-3_1"), ylim=[1e-6,1])

asfenicsx.evaluate_gradients(h=1e-5, order = 1)
U_A, S_A = asfenicsx.random_sampling_algorithm()
asfenicsx.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
asfenicsx.plot_eigenvalues(true_eigenvalues=eigenvalues_constant**2, filename=os.path.join(dir,"quadraticModel/constantDecay_eigenvalues_FD_1e-5_1"), ylim=[1e-8,1e4])
asfenicsx.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"quadraticModel/constantDecay_subspace_FD_1e-5_1"), ylim=[1e-6,1])

asfenicsx.evaluate_gradients(h=1e-1, order = 2)
U_A, S_A = asfenicsx.random_sampling_algorithm()
asfenicsx.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
asfenicsx.plot_eigenvalues(true_eigenvalues=eigenvalues_constant**2, filename=os.path.join(dir,"quadraticModel/constantDecay_eigenvalues_FD_1e-1_2"), ylim=[1e-8,1e4])
asfenicsx.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"quadraticModel/constantDecay_subspace_FD_1e-1_2"), ylim=[1e-6,1])

# Interpolation
func.get_gradient_method('I')
func.interpolation(samples, interpolation_method = 'LS', order = 1)
asfenicsx.evaluate_gradients()
U_A, S_A = asfenicsx.random_sampling_algorithm()
asfenicsx.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
asfenicsx.plot_eigenvalues(true_eigenvalues=eigenvalues_constant**2, filename=os.path.join(dir,"quadraticModel/constantDecay_eigenvalues_I_1"), ylim=[1e-8,1e4])
asfenicsx.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"quadraticModel/constantDecay_subspace_I_1"), ylim=[1e-6,1])

func.interpolation(samples, interpolation_method = 'LS', order = 2, overwrite=True)
asfenicsx.evaluate_gradients()
U_A, S_A = asfenicsx.random_sampling_algorithm()
asfenicsx.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
asfenicsx.plot_eigenvalues(true_eigenvalues=eigenvalues_constant**2, filename=os.path.join(dir,"quadraticModel/constantDecay_eigenvalues_I_2"), ylim=[1e-8,1e4])
asfenicsx.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"quadraticModel/constantDecay_subspace_I_2"), ylim=[1e-6,1])


"""
Eigenvalues with gap between first and second
"""
function = lambda x: 0.5 * x.T @ A_gap @ x
grad = lambda x: A_gap @ x
func = functional(m, function)
func.get_derivative(grad)

func.get_gradient_method('A')
asfenicsx = ASFEniCSx(k, func, samples)
U_A, S_A = asfenicsx.random_sampling_algorithm()
asfenicsx.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
asfenicsx.plot_eigenvalues(true_eigenvalues=eigenvalues_gap**2, filename=os.path.join(dir,"quadraticModel/gap_eigenvalues"), ylim=[1e-8,1e4])
asfenicsx.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"quadraticModel/gap_subspace"), ylim=[1e-6,1])

# Finite Differences
func.get_gradient_method('FD')
asfenicsx.evaluate_gradients(h=1e-1, order = 1)
U_A, S_A = asfenicsx.random_sampling_algorithm()
asfenicsx.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
asfenicsx.plot_eigenvalues(true_eigenvalues=eigenvalues_gap**2, filename=os.path.join(dir,"quadraticModel/gap_eigenvalues_FD_1e-1_1"), ylim=[1e-8,1e4])
asfenicsx.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"quadraticModel/gap_subspace_FD_1e-1_1"), ylim=[1e-6,1])

asfenicsx.evaluate_gradients(h=1e-3, order = 1)
U_A, S_A = asfenicsx.random_sampling_algorithm()
asfenicsx.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
asfenicsx.plot_eigenvalues(true_eigenvalues=eigenvalues_gap**2, filename=os.path.join(dir,"quadraticModel/gap_eigenvalues_FD_1e-3_1"), ylim=[1e-8,1e4])
asfenicsx.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"quadraticModel/gap_subspace_FD_1e-3_1"), ylim=[1e-6,1])

asfenicsx.evaluate_gradients(h=1e-5, order = 1)
U_A, S_A = asfenicsx.random_sampling_algorithm()
asfenicsx.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
asfenicsx.plot_eigenvalues(true_eigenvalues=eigenvalues_gap**2, filename=os.path.join(dir,"quadraticModel/gap_eigenvalues_FD_1e-5_1"), ylim=[1e-8,1e4])
asfenicsx.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"quadraticModel/gap_subspace_FD_1e-5_1"), ylim=[1e-6,1])

asfenicsx.evaluate_gradients(h=1e-1, order = 2)
U_A, S_A = asfenicsx.random_sampling_algorithm()
asfenicsx.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
asfenicsx.plot_eigenvalues(true_eigenvalues=eigenvalues_gap**2, filename=os.path.join(dir,"quadraticModel/gap_eigenvalues_FD_1e-1_2"), ylim=[1e-8,1e4])
asfenicsx.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"quadraticModel/gap_subspace_FD_1e-1_2"), ylim=[1e-6,1])

# Interpolation
func.get_gradient_method('I')
func.interpolation(samples, interpolation_method='LS', order = 1, overwrite=True)
asfenicsx.evaluate_gradients()
U_A, S_A = asfenicsx.random_sampling_algorithm()
asfenicsx.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
asfenicsx.plot_eigenvalues(true_eigenvalues=eigenvalues_gap**2, filename=os.path.join(dir,"quadraticModel/gap_eigenvalues_I_1"), ylim=[1e-8,1e4])
asfenicsx.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"quadraticModel/gap_subspace_I_1"), ylim=[1e-6,1])

func.interpolation(samples, interpolation_method='LS', order = 2, overwrite=True)
asfenicsx.evaluate_gradients()
U_A, S_A = asfenicsx.random_sampling_algorithm()
asfenicsx.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
asfenicsx.plot_eigenvalues(true_eigenvalues=eigenvalues_gap**2, filename=os.path.join(dir,"quadraticModel/gap_eigenvalues_I_2"), ylim=[1e-8,1e4])
asfenicsx.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"quadraticModel/gap_subspace_I_2"), ylim=[1e-6,1])

"""
Eigenvalues with gap between third and fourth
"""

function = lambda x: 0.5 * x.T @ A_gap2 @ x
grad = lambda x: A_gap2 @ x
func = functional(m, function)
func.get_derivative(grad)

func.get_gradient_method('A')
asfenicsx = ASFEniCSx(k, func, samples)
U_A, S_A = asfenicsx.random_sampling_algorithm()
asfenicsx.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
asfenicsx.plot_eigenvalues(true_eigenvalues=eigenvalues_gap2**2, filename=os.path.join(dir,"quadraticModel/gap2_eigenvalues"), ylim=[1e-8,1e4])
asfenicsx.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"quadraticModel/gap2_subspace"), ylim=[1e-6,1])

# Finite Differences
func.get_gradient_method('FD')
asfenicsx = ASFEniCSx(k, func, samples)
asfenicsx.evaluate_gradients(h=1e-1, order = 1)
U_A, S_A = asfenicsx.random_sampling_algorithm()
asfenicsx.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
asfenicsx.plot_eigenvalues(true_eigenvalues=eigenvalues_gap2**2, filename=os.path.join(dir,"quadraticModel/gap2_eigenvalues_FD_1e-1_1"), ylim=[1e-8,1e4])
asfenicsx.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"quadraticModel/gap2_subspace_FD_1e-1_1"), ylim=[1e-6,1])

asfenicsx.evaluate_gradients(h=1e-3, order = 1)
U_A, S_A = asfenicsx.random_sampling_algorithm()
asfenicsx.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
asfenicsx.plot_eigenvalues(true_eigenvalues=eigenvalues_gap2**2, filename=os.path.join(dir,"quadraticModel/gap2_eigenvalues_FD_1e-3_1"), ylim=[1e-8,1e4])
asfenicsx.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"quadraticModel/gap2_subspace_FD_1e-3_1"), ylim=[1e-6,1])

asfenicsx.evaluate_gradients(h=1e-5, order = 1)
U_A, S_A = asfenicsx.random_sampling_algorithm()
asfenicsx.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
asfenicsx.plot_eigenvalues(true_eigenvalues=eigenvalues_gap2**2, filename=os.path.join(dir,"quadraticModel/gap2_eigenvalues_FD_1e-5_1"), ylim=[1e-8,1e4])
asfenicsx.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"quadraticModel/gap2_subspace_FD_1e-5_1"), ylim=[1e-6,1])

asfenicsx.evaluate_gradients(h=1e-1, order = 2)
U_A, S_A = asfenicsx.random_sampling_algorithm()
asfenicsx.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
asfenicsx.plot_eigenvalues(true_eigenvalues=eigenvalues_gap2**2, filename=os.path.join(dir,"quadraticModel/gap2_eigenvalues_FD_1e-1_2"), ylim=[1e-8,1e4])
asfenicsx.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"quadraticModel/gap2_subspace_FD_1e-1_2"), ylim=[1e-6,1])

# Interpolation
func.get_gradient_method('I')
func.interpolation(samples, order=1, interpolation_method="LS", overwrite=True)
asfenicsx.evaluate_gradients()
asfenicsx = ASFEniCSx(k, func, samples)
U_A, S_A = asfenicsx.random_sampling_algorithm()
asfenicsx.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
asfenicsx.plot_eigenvalues(true_eigenvalues=eigenvalues_gap2**2, filename=os.path.join(dir,"quadraticModel/gap2_eigenvalues_LS_1"), ylim=[1e-8,1e4])
asfenicsx.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"quadraticModel/gap2_subspace_LS_1"), ylim=[1e-6,1])

func.interpolation(samples, order=2, interpolation_method="LS", overwrite=True)
asfenicsx.evaluate_gradients()
asfenicsx = ASFEniCSx(k, func, samples)
U_A, S_A = asfenicsx.random_sampling_algorithm()
asfenicsx.bootstrap(100)
sub_error = np.zeros(m-1)
for i in range(m-1):
    sub_error[i] = np.linalg.norm(np.dot(eigenvectors[:,:i+1].T, U_A[:,i+1:]), ord=2)
asfenicsx.plot_eigenvalues(true_eigenvalues=eigenvalues_gap2**2, filename=os.path.join(dir,"quadraticModel/gap2_eigenvalues_LS_2"), ylim=[1e-8,1e4])
asfenicsx.plot_subspace(true_subspace=sub_error, filename=os.path.join(dir,"quadraticModel/gap2_subspace_LS_2"), ylim=[1e-6,1])

plt.close('all')
