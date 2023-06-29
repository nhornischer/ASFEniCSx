import numpy as np

from ASFEniCSx.functional import Functional

# UNIVARIATE FUNCTION

f = lambda x: np.exp(x) / np.sqrt(np.sin(x)**3 + np.cos(x)**3)
dfdx = lambda x: np.exp(x) *(2*np.sin(x)**3 +2*np.cos(x)**3 + 3 * np.sin(x)*np.cos(x)**2 - 3*np.sin(x)**2*np.cos(x))/(2*np.sqrt(np.sin(x)**3 + np.cos(x)**3)**3)

function = Functional(1,f)

x = 1.5

print("f(x) = ", function.evaluate(x))
print("f'(x) = ", function.gradient(x))


derivative = np.zeros((20,3))
h_steps = 10.0**(-np.arange(0,20))
np.sort(h_steps)
real_derivative = dfdx(x)
for i,h in enumerate(h_steps):
    derivative[i,0] = np.abs(function.gradient(x, h = h, order = 1) - real_derivative)/np.abs(real_derivative)
    derivative[i,1] = np.abs(function.gradient(x, h = h, order = 2) - real_derivative)/np.abs(real_derivative)
    derivative[i,2] = np.abs(function.gradient(x, 'CSM', h = h) - real_derivative)/np.abs(real_derivative)
import matplotlib.pyplot as plt

plt.figure()
plt.title("Univariate function")
plt.plot(derivative[:,0], label = 'Forward-difference')
plt.plot(derivative[:,1], label = 'Central-difference')
plt.plot(derivative[:,2], label = 'Complex-step')
plt.ylabel(r"Normalized error, $\frac{|f'(x) - f'_{approx}(x)|}{|f'(x)|}$")
plt.xlabel("Step size, h")
plt.xticks(range(0,20,2),h_steps[::2])
plt.yscale('log')
plt.ylim([1e-20,1e1])
plt.legend()
plt.tight_layout()

# BIVARIATE FUNCTION
f = lambda x: np.exp(0.7 * x[0] + 0.3 * x[1])
dfdx = lambda x: [0.7 * np.exp(0.7 * x[0] + 0.3 * x[1]),
                   0.3 * np.exp(0.7 * x[0] + 0.3 * x[1])]
function = Functional(2,f)

x = np.asarray([1.5, 1.5])
real_derivative = dfdx(x)
derivative = np.zeros((20,2,3))
h_steps = 10.0**(-np.arange(0,20))
np.sort(h_steps)
for i,h in enumerate(h_steps):
    derivative[i,:,0] = np.abs(function.gradient(x, h = h, order = 1) - real_derivative)/np.abs(real_derivative)
    derivative[i,:,1] = np.abs(function.gradient(x, h = h, order = 2) - real_derivative)/np.abs(real_derivative)
    derivative[i,:,2] = np.abs(function.gradient(x,'CSM', h = h) - real_derivative)/np.abs(real_derivative)
plt.figure(figsize=(12.8,4.8))
plt.subplot(1,2,1)
plt.title("Bivariate function (x1)")
plt.plot(derivative[:,0,0], label = 'Forward-difference')
plt.plot(derivative[:,0,1], label = 'Central-difference')
plt.plot(derivative[:,0,2], label = 'Complex-step')
plt.ylabel(r"Normalized error, $\frac{|f'(x) - f'_{approx}(x)|}{|f'(x)|}$")
plt.xlabel("Step size, h")
plt.xticks(range(0,20,2),h_steps[::2])
plt.yscale('log')
plt.ylim([1e-20,1e1])
plt.legend()
plt.tight_layout()
plt.subplot(1,2,2)
plt.title("Bivariate function (x2)")
plt.plot(derivative[:,1,0], label = 'Forward-difference')
plt.plot(derivative[:,1,1], label = 'Central-difference')
plt.plot(derivative[:,1,2], label = 'Complex-step')
plt.ylabel(r"Normalized error, $\frac{|f'(x) - f'_{approx}(x)|}{|f'(x)|}$")
plt.xlabel("Step size, h")
plt.xticks(range(0,20,2),h_steps[::2])
plt.yscale('log')
plt.ylim([1e-20,1e1])
plt.legend()
plt.tight_layout()

plt.show()