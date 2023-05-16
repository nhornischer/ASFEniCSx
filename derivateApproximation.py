import math
import time
import numpy as np
import matplotlib.pyplot as plt

from asfenicsx import Functional, Sampling

def test_function(x : list or np.array):
    f = math.exp(0.7 * x[0] + 0.3 * x[1])
    return f

def test_derivative(x : list or np.array):
    df_dx = np.zeros(len(x))
    df_dx[0] = 0.7 * math.exp(0.7 * x[0] + 0.3 * x[1])
    df_dx[1] = 0.3 * math.exp(0.7 * x[0] + 0.3 * x[1])

    return df_dx

cost = Functional(2, test_function)

M=100
samples = Sampling(M, 2)

t=time.time()
cost.interpolation(samples.samples)
print(time.time()-t)

grid = np.linspace(-1,1,1000)

plt.figure()
# Plot test_function over the two-dimensional grid
X, Y = np.meshgrid(grid, grid)
Z = np.zeros(X.shape)
t=time.time()
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i,j] = test_function([X[i,j], Y[i,j]])
print(time.time()-t)
plt.contourf(X,Y,Z,levels=100)
plt.colorbar()
for i in range(M):
    plt.plot(samples.samples[i,0], samples.samples[i,1], 'x', color='white')
plt.title("Test function")

# Plot the evaluation of the cost.Interpolator over the two-dimensional grid
plt.figure()
X, Y = np.meshgrid(grid, grid)
Z = np.zeros(X.shape)
t=time.time()
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i,j] = cost.interpolator([X[i,j], Y[i,j]])
print(time.time()-t)
plt.contourf(X,Y,Z,levels=100)  
plt.colorbar()
plt.title("Interpolated function")
cost.interpolation(samples.samples, method='multivariate')
# Plot the evaluation of the cost.Interpolator over the two-dimensional grid
plt.figure()
X, Y = np.meshgrid(grid, grid)
Z = np.zeros(X.shape)
t=time.time()
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i,j] = cost.interpolator([X[i,j], Y[i,j]])
print(time.time()-t)
plt.contourf(X,Y,Z,levels=100)  
plt.colorbar()
plt.title("Bivariate inteprolated function")
plt.show()

exit()
errors= np.zeros([M,2])
for i in range(M):
    x = samples.extract_sample(i)
    real = test_derivative(x)
    FD = cost.finite_differences(x)
    errors[i,:] = real-FD

print(f"Maximal error: {np.max(errors)}")




