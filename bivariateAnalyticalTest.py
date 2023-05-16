import numpy as np
import math
import scipy
import time
from asfenicsx import ASFEniCSx, Sampling, Functional

import matplotlib.pyplot as plt
from matplotlib import cm

def evaluate_f(x : list or np.array):
    f = math.exp(0.7 * x[0] + 0.3 * x[1])
    return f


def calculate_df_dx(x : list or np.array):
    df_dx = np.zeros(len(x))
    df_dx[0] = 0.7 * math.exp(0.7 * x[0] + 0.3 * x[1])
    df_dx[1] = 0.3 * math.exp(0.7 * x[0] + 0.3 * x[1])
    return df_dx

if __name__ == '__main__':
    data = np.zeros([100, 100])
    x_range = np.linspace(-1, 1, 100)
    y_range = np.linspace(-1, 1, 100)

    samples = Sampling(100, 2)

    cost = Functional(2, evaluate_f)
    cost.get_derivative(calculate_df_dx)  
    cost.interpolation(samples.samples)

    for i,x in enumerate(x_range):
        for j,y in enumerate(y_range):
             data[i,j]= evaluate_f([x, y])

    fig= plt.figure()

    X,Y = np.meshgrid(x_range, y_range)
    plt.contourf(X,Y, data, 100)

    asfenicsx = ASFEniCSx(2, cost, samples)
    t=time.time()
    U, S = asfenicsx.random_sampling_algorithm()
    print(f"Time A {time.time()-t}")
    

    cost.get_gradient_method('A')

    plt.arrow(0,0, U[0,0]/2, U[1,0]/2, color='r', width=0.02)
    plt.arrow(0,0, U[0,1]/2, U[1,1]/2, color='r', width=0.02)

    cost.get_gradient_method('I')

    t=time.time()
    
    U_1,S_1=asfenicsx.random_sampling_algorithm()
    print(f"Time I {time.time()-t}")
    plt.arrow(0,0, U_1[0,0]/2, U_1[1,0]/2, color='b', width=0.01)
    plt.arrow(0,0, U_1[0,1]/2, U_1[1,1]/2, color='b', width=0.01)

    t=time.time()
    cost.interpolation(samples.samples, method='multivariate')
    U_3,S_3=asfenicsx.random_sampling_algorithm()
    print(f"Time I {time.time()-t}")
    plt.arrow(0,0, U_3[0,0]/2, U_3[1,0]/2, color='c', width=0.01)
    plt.arrow(0,0, U_3[0,1]/2, U_3[1,1]/2, color='c', width=0.01)

    cost.get_gradient_method('FD')
    t=time.time()
    U_2,S_2=asfenicsx.random_sampling_algorithm()
    print(f"Time FD {time.time()-t}")
    plt.arrow(0,0, U_2[0,0]/2, U_2[1,0]/2, color='g', width=0.01)
    plt.arrow(0,0, U_2[0,1]/2, U_2[1,1]/2, color='g', width=0.01)

    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    print("Interpolator", U_1-U, S_1-S)
    print("Bivariate", U_3-U, S_3-S)
    print("FiniteDifferences",U_2-U, S_2-S)

    plt.show()