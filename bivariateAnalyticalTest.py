import numpy as np
import math
import os

from ASFEniCSx.sampling import Clustering, Sampling
# from ASFEniCSx.functional import Functional
from ASFEniCSx.functional import Analytical, Functional, Interpolation, Regression
from ASFEniCSx.asfenicsx import ASFEniCSx
import ASFEniCSx.utils as utils

import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16})

if not os.path.exists("bivariateAnalyticalTest"):
    os.mkdir("bivariateAnalyticalTest")
if not os.path.exists("bivariateAnalyticalTestNoise"):
    os.mkdir("bivariateAnalyticalTestNoise")

os.chdir("bivariateAnalyticalTest")
dir = os.path.join(os.path.dirname(__file__), "bivariateAnalyticalTest")

def evaluate_f(x : list or np.array):
    f = math.exp(0.7 * x[0] + 0.3 * x[1])
    return f

def noisy_evaluate_f(x : list or np.array):
    f = math.exp(0.7 * x[0] + 0.3 * x[1])
    return f + np.random.uniform(-0.05 * np.max(f), 0.05 * np.max(f))

def calculate_df_dx(x : list or np.array):
    df_dx = np.zeros(len(x))
    df_dx[0] = 0.7 * math.exp(0.7 * x[0] + 0.3 * x[1])
    df_dx[1] = 0.3 * math.exp(0.7 * x[0] + 0.3 * x[1])
    return df_dx

if __name__ == '__main__':
    # Plot the function
    data = np.zeros([100, 100])
    x_range = np.linspace(-1, 1, 100)
    y_range = np.linspace(-1, 1, 100)
    for i,x in enumerate(x_range):
        for j,y in enumerate(y_range):
             data[i,j]= evaluate_f([x, y])
    fig= plt.figure("Bivariate test function", figsize=(8,6))
    X,Y = np.meshgrid(x_range, y_range)
    plt.contourf(X,Y, data.T, 100)
    plt.colorbar()
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')

    plt.arrow(0, 0, 0.7, 0.3, width = 0.02, color = 'r')
    plt.arrow(0.0, 0.0, -0.3, 0.7, width = 0.02, color = 'b')
    plt.tight_layout()
    plt.savefig("2D_function.pdf")
    plt.clf()

    # Define clustered samples
    samples = Clustering(100, 2, 5)
    samples.random_uniform()
    samples.detect()
    samples.plot(os.path.join(dir,"2D_samples.pdf"))
    samples.save(os.path.join(dir,"bivariate_samples.json"))
    

    print("##############################################################Standard Test##############################################################")
    samples.assign_values(evaluate_f)

    """ 
    Perform a validation test of the differentiation methods
    by comparing the analytical derivative with the numerical
    """
    # For the errors we investigate the mean value of the 
    # Frobenious norm and maximum norm over all samples 

    # Analytical
    analytical = Analytical(2, evaluate_f, calculate_df_dx)

    A_data = analytical.gradient(samples.samples())

    # Function
    function = Functional(2, evaluate_f)

    utils.evaluate_derivative_FD(function, samples, os.path.join(dir,"derivatives_FD.pdf"), A_data)

    # Interpolation
    interpolant = Interpolation(2, evaluate_f, samples)
    utils.evaluate_derivative_interpolation(interpolant, 3, False, os.path.join(dir,"derivatives_I.pdf"), A_data)

    # Local Interpolation
    utils.evaluate_derivative_interpolation(interpolant, 3, True, os.path.join(dir,"derivatives_I_local.pdf"), A_data)

    # Regression
    regressant = Regression(2, evaluate_f, samples)
    utils.evaluate_derivative_regression(regressant, 3, False, os.path.join(dir,"derivatives_R.pdf"), A_data)

    # Local Regression
    utils.evaluate_derivative_regression(regressant, 3, True, os.path.join(dir,"derivatives_R_local.pdf"), A_data)

    """
    Active Subspace Construction
    """
    print("Active Subspace")
    n= 2
    asfenicsx = ASFEniCSx(n, analytical, samples)
    U, S = asfenicsx.estimation()

    asfenicsx = ASFEniCSx(n, function, samples)
    U_FD, S_FD = asfenicsx.estimation()
    print(f"\tError FD (EV, EW): {np.linalg.norm(U-U_FD)} , {np.linalg.norm(S-S_FD)}")

    interpolant.interpolate(order = 2, use_clustering = False)
    asfenicsx = ASFEniCSx(n, interpolant, samples)
    U_I, S_I = asfenicsx.estimation()
    print(f"\tError I (EV, EW): {np.linalg.norm(U-U_I)} , {np.linalg.norm(S-S_I)}")

    regressant.regress(order = 2, use_clustering = False)
    asfenicsx = ASFEniCSx(n, regressant, samples)
    U_LS, S_LS = asfenicsx.estimation()
    print(f"\tError LS (EV, EW): {np.linalg.norm(U-U_LS)} , {np.linalg.norm(S-S_LS)}")

    interpolant.interpolate(order = 2, use_clustering = True)
    asfenicsx = ASFEniCSx(n, interpolant, samples)
    U_I_local, S_I_local = asfenicsx.estimation()
    print(f"\tError I local (EV, EW): {np.linalg.norm(U-U_I_local)} , {np.linalg.norm(S-S_I_local)}")

    regressant.regress(order = 2, use_clustering = True)
    asfenicsx = ASFEniCSx(n, regressant, samples)
    U_LS_local, S_LS_local = asfenicsx.estimation()
    print(f"\tError LS local (EV, EW): {np.linalg.norm(U-U_LS_local)} , {np.linalg.norm(S-S_LS_local)}")

    fig= plt.figure("Active Subspace", figsize=(12,9))
    X,Y = np.meshgrid(x_range, y_range)
    plt.contourf(X,Y, data.T, 100)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')

    for i in range(n):
        if i ==0:
            plt.arrow(0, 0, U[0,i]/2, U[1,i]/2, width = 0.02, color = 'r', label="Analytical")
        else:
            plt.arrow(0, 0, U[0,i]/2, U[1,i]/2, width = 0.02, color = 'r')
    
    for i in range(n):
        if i ==0:
            plt.arrow(0, 0, U_FD[0,i]/2, U_FD[1,i]/2, width = 0.01, alpha = 0.5, linewidth = 5, color = 'g', label="FD")
        else:
            plt.arrow(0, 0, U_FD[0,i]/2, U_FD[1,i]/2, width = 0.01, alpha = 0.5, linewidth = 5, color = 'g')
    
    for i in range(n):
        if i ==0:
            plt.arrow(0, 0, U_I[0,i]/2, U_I[1,i]/2, width = 0.01, alpha = 0.5, linewidth = 5, color = 'b', label="Global Interpolation")
        else:
            plt.arrow(0, 0, U_I[0,i]/2, U_I[1,i]/2, width = 0.01, alpha = 0.5, linewidth = 5, color = 'b')

    for i in range(n):
        if i ==0:
            plt.arrow(0, 0, U_LS[0,i]/2, U_LS[1,i]/2, width = 0.01, alpha = 0.5, linewidth = 5, color = 'y', label="Global Regression")
        else:
            plt.arrow(0, 0, U_LS[0,i]/2, U_LS[1,i]/2, width = 0.01, alpha = 0.5, linewidth = 5, color = 'y')

    for i in range(n):
        if i ==0:
            plt.arrow(0, 0, U_I_local[0,i]/2, U_I_local[1,i]/2, width = 0.01, alpha = 0.5, linewidth = 5, color = 'c', label="Local Interpolation")
        else:
            plt.arrow(0, 0, U_I_local[0,i]/2, U_I_local[1,i]/2, width = 0.01, alpha = 0.5, linewidth = 5, color = 'c')

    for i in range(n):
        if i ==0:
            plt.arrow(0, 0, U_LS_local[0,i]/2, U_LS_local[1,i]/2, width = 0.01, alpha = 0.5, linewidth = 5, color = 'm', label="Local Regression")
        else:
            plt.arrow(0, 0, U_LS_local[0,i]/2, U_LS_local[1,i]/2, width = 0.01, alpha = 0.5, linewidth = 5, color = 'm')
    
    plt.legend(loc='upper left')
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.savefig("2D_active_subspace.pdf")
    
    plt.figure("Active Subspace Eigenvalues")
    plt.plot(S, 'r', linewidth = 4)
    plt.plot(S_FD, 'g')
    plt.plot(S_I, 'b')
    plt.plot(S_LS, 'y')
    plt.plot(S_I_local, 'c')
    plt.plot(S_LS_local, 'm')
    plt.legend(["Analytical", "FD", "Interpolation", "Least Squares", "Local Interpolation", "Local Least Squares"])
    plt.savefig("2D_active_subspace_eigenvalues.pdf")
    plt.close("all")

    '''
    Noise Test
    '''
    
    print("##############################################################Noise Test##############################################################")



    os.chdir("..")
    os.chdir("bivariateAnalyticalTestNoise")
    dir = os.path.join(os.path.dirname(__file__), "bivariateAnalyticalTestNoise")

    if not os.path.exists("figures"):
        os.mkdir("figures")

    samples.assign_values(noisy_evaluate_f)

    """ 
    Perform a validation test of the differentiation methods
    by comparing the analytical derivative with the numerical
    """
    # For the errors we investigate the mean value of the 
    # Frobenious norm and maximum norm over all samples 

    # Analytical
    analytical = Analytical(2, noisy_evaluate_f, calculate_df_dx)

    A_data = analytical.gradient(samples.samples())

    # Function
    function = Functional(2, noisy_evaluate_f)

    # First Order Finite Differences
    step_width = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    FD1_errors = np.zeros([len(step_width), 2])
    for i, h in enumerate(step_width):
        FD_data = function.gradient(samples.samples(), order = 1, h = h)
        FD1_errors[i, 0] = np.linalg.norm(A_data-FD_data)/np.linalg.norm(A_data)
        FD1_errors[i, 1] = np.max(np.abs(A_data-FD_data))/np.max(A_data)
    
    # Second Order Finite Differences
    FD2_errors = np.zeros([len(step_width), 2])
    for i, h in enumerate(step_width):
        FD_data = function.gradient(samples.samples(), order = 2, h = h)
        FD2_errors[i, 0] = np.linalg.norm(A_data-FD_data)/np.linalg.norm(A_data)
        FD2_errors[i, 1] = np.max(np.abs(A_data-FD_data))/np.max(A_data)

    # Plot the errors
    plt.figure(figsize=(8,6))
    plt.plot(FD1_errors[:,0],color = "r", label = "1st-order")
    plt.plot(FD1_errors[:,1],color = "r", linestyle = "dotted")
    plt.plot(FD2_errors[:,0],color = "b", label = "2nd-order")
    plt.plot(FD2_errors[:,1],color = "b", linestyle = "dotted")
    plt.xlabel(r'$h$')
    plt.xticks(np.arange(len(step_width)), step_width)
    plt.yscale('log')
    plt.ylabel(r'$\frac{|| \nabla_{FD}f-\nabla_{A} f ||}{||\nabla_{A} f ||}$')
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.savefig(os.path.join(dir,"derivatives_FD.pdf"))

    # Interpolation
    interpolant = Interpolation(2, noisy_evaluate_f, samples)
    plt.figure(figsize=(8,6))
    colors = ["r", "b", "g", "k"]
    for order in range(3, 0, -1):
        number_of_coefficients = range(1, math.comb(2+order, 2))
        I_errors = np.zeros([len(number_of_coefficients), 2])
        for i, n_coef in enumerate(number_of_coefficients):
            interpolant.interpolate(order = order, number_of_exponents = n_coef, use_clustering=False)
            _data = interpolant.gradient(samples.samples())
            I_errors[i, 0] = np.linalg.norm(A_data-_data)/np.linalg.norm(A_data)
            I_errors[i, 1] = np.max(np.abs(A_data-_data))/np.max(A_data)

        plt.plot(number_of_coefficients, I_errors[:,0],color= colors[order-1], label = str(order)+"-order")
        plt.plot(number_of_coefficients, I_errors[:,1],color= colors[order-1], linestyle = "dotted")
    plt.xlabel(r'$n_{coef}$')
    number_of_coefficients = range(1, math.comb(2+3, 2))
    plt.xticks(range(1, len(number_of_coefficients)+1), number_of_coefficients)
    plt.yscale('log')
    plt.ylim([1e-3, 1e3])
    plt.ylabel(r'$\frac{|| \nabla_{I}f-\nabla_{A} f ||}{||\nabla_{A} f ||}$')
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.savefig(os.path.join(dir,"derivatives_I.pdf"))

    # Local Interpolation
    interpolant = Interpolation(2, noisy_evaluate_f, samples)
    plt.figure(figsize=(8,6))
    colors = ["r", "b", "g", "k"]
    for order in range(3, 0, -1):
        number_of_coefficients = range(1, math.comb(2+order, 2))
        I_errors = np.zeros([len(number_of_coefficients), 2])
        for i, n_coef in enumerate(number_of_coefficients):
            interpolant.interpolate(order = order, number_of_exponents = n_coef, use_clustering=True)
            _data = interpolant.gradient(samples.samples())
            I_errors[i, 0] = np.linalg.norm(A_data-_data)/np.linalg.norm(A_data)
            I_errors[i, 1] = np.max(np.abs(A_data-_data))/np.max(A_data)

        plt.plot(number_of_coefficients, I_errors[:,0],color= colors[order-1], label = str(order)+"-order")
        plt.plot(number_of_coefficients, I_errors[:,1],color= colors[order-1], linestyle = "dotted")
    plt.xlabel(r'$n_{coef}$')
    number_of_coefficients = range(1, math.comb(2+3, 2))
    plt.xticks(range(1, len(number_of_coefficients)+1), number_of_coefficients)
    plt.yscale('log')
    plt.ylim([1e-3, 1e3])
    plt.ylabel(r'$\frac{|| \nabla_{I}f-\nabla_{A} f ||}{||\nabla_{A} f ||}$')
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.savefig(os.path.join(dir,"derivatives_I_local.pdf"))

    # Regression
    regressant = Regression(2, noisy_evaluate_f, samples)
    plt.figure(figsize=(8,6))
    colors = ["r", "b", "g", "k"]
    for order in range(3, 0, -1):
        R_errors = np.zeros([100, 2])
        for i in range(100):
            regressant.regress(order = order, number_of_samples = i+1, use_clustering=False)
            _data = regressant.gradient(samples.samples())
            R_errors[i, 0] = np.linalg.norm(A_data-_data)/np.linalg.norm(A_data)
            R_errors[i, 1] = np.max(np.abs(A_data-_data))/np.max(A_data)

        plt.plot(range(1, 101), R_errors[:,0],color= colors[order-1], label = str(order)+"-order")
        plt.plot(range(1, 101), R_errors[:,1],color= colors[order-1], linestyle = "dotted")
    plt.xlabel(r'$n_{samples}$')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim([1e-3, 1e3])
    plt.ylabel(r'$\frac{|| \nabla_{R}f-\nabla_{A} f ||}{||\nabla_{A} f ||}$')
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.savefig(os.path.join(dir,"derivatives_R.pdf"))

    # Local Regression
    regressant = Regression(2, noisy_evaluate_f, samples)
    plt.figure(figsize=(8,6))
    colors = ["r", "b", "g", "k"]
    for order in range(3, 0, -1):
        R_errors = np.zeros([100, 2])
        for i in range(100):
            regressant.regress(order = order, number_of_samples = i+1, use_clustering=True)
            _data = regressant.gradient(samples.samples())
            R_errors[i, 0] = np.linalg.norm(A_data-_data)/np.linalg.norm(A_data)
            R_errors[i, 1] = np.max(np.abs(A_data-_data))/np.max(A_data)

        plt.plot(range(1, 101), R_errors[:,0],color= colors[order-1], label = str(order)+"-order")
        plt.plot(range(1, 101), R_errors[:,1],color= colors[order-1], linestyle = "dotted")
    plt.xlabel(r'$n_{samples}$')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$\frac{|| \nabla_{R}f-\nabla_{A} f ||}{||\nabla_{A} f ||}$')
    plt.ylim([1e-3, 1e3])
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.savefig(os.path.join(dir,"derivatives_R_local.pdf"))
    
    
    """
    Active Subspace Construction
    """
    print("Active Subspace")
    n= 2

    asfenicsx = ASFEniCSx(n, function, samples)
    U_FD, S_FD = asfenicsx.estimation()
    print(f"\tError FD (EV, EW): {np.linalg.norm(U-U_FD)} , {np.linalg.norm(S-S_FD)}")

    interpolant.interpolate(order = 2, use_clustering = False)
    asfenicsx = ASFEniCSx(n, interpolant, samples)
    U_I, S_I = asfenicsx.estimation()
    print(f"\tError I (EV, EW): {np.linalg.norm(U-U_I)} , {np.linalg.norm(S-S_I)}")

    regressant.regress(order = 2, use_clustering = False)
    asfenicsx = ASFEniCSx(n, regressant, samples)
    U_LS, S_LS = asfenicsx.estimation()
    print(f"\tError LS (EV, EW): {np.linalg.norm(U-U_LS)} , {np.linalg.norm(S-S_LS)}")

    interpolant.interpolate(order = 2, use_clustering = True)
    asfenicsx = ASFEniCSx(n, interpolant, samples)
    U_I_local, S_I_local = asfenicsx.estimation()
    print(f"\tError I local (EV, EW): {np.linalg.norm(U-U_I_local)} , {np.linalg.norm(S-S_I_local)}")

    regressant.regress(order = 2, use_clustering = True)
    asfenicsx = ASFEniCSx(n, regressant, samples)
    U_LS_local, S_LS_local = asfenicsx.estimation()
    print(f"\tError LS local (EV, EW): {np.linalg.norm(U-U_LS_local)} , {np.linalg.norm(S-S_LS_local)}")

    fig= plt.figure("Active Subspace", figsize=(12,9))
    X,Y = np.meshgrid(x_range, y_range)
    plt.contourf(X,Y, data.T, 100)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')

    for i in range(n):
        if i ==0:
            plt.arrow(0, 0, U[0,i]/2, U[1,i]/2, width = 0.02, color = 'r', label="Analytical")
        else:
            plt.arrow(0, 0, U[0,i]/2, U[1,i]/2, width = 0.02, color = 'r')
    
    for i in range(n):
        if i ==0:
            plt.arrow(0, 0, U_FD[0,i]/2, U_FD[1,i]/2, width = 0.01, alpha = 0.5, linewidth = 5, color = 'g', label="FD")
        else:
            plt.arrow(0, 0, U_FD[0,i]/2, U_FD[1,i]/2, width = 0.01, alpha = 0.5, linewidth = 5, color = 'g')
    
    for i in range(n):
        if i ==0:
            plt.arrow(0, 0, U_I[0,i]/2, U_I[1,i]/2, width = 0.01, alpha = 0.5, linewidth = 5, color = 'b', label="Global Interpolation")
        else:
            plt.arrow(0, 0, U_I[0,i]/2, U_I[1,i]/2, width = 0.01, alpha = 0.5, linewidth = 5, color = 'b')

    for i in range(n):
        if i ==0:
            plt.arrow(0, 0, U_LS[0,i]/2, U_LS[1,i]/2, width = 0.01, alpha = 0.5, linewidth = 5, color = 'y', label="Global Regression")
        else:
            plt.arrow(0, 0, U_LS[0,i]/2, U_LS[1,i]/2, width = 0.01, alpha = 0.5, linewidth = 5, color = 'y')

    for i in range(n):
        if i ==0:
            plt.arrow(0, 0, U_I_local[0,i]/2, U_I_local[1,i]/2, width = 0.01, alpha = 0.5, linewidth = 5, color = 'c', label="Local Interpolation")
        else:
            plt.arrow(0, 0, U_I_local[0,i]/2, U_I_local[1,i]/2, width = 0.01, alpha = 0.5, linewidth = 5, color = 'c')

    for i in range(n):
        if i ==0:
            plt.arrow(0, 0, U_LS_local[0,i]/2, U_LS_local[1,i]/2, width = 0.01, alpha = 0.5, linewidth = 5, color = 'm', label="Local Regression")
        else:
            plt.arrow(0, 0, U_LS_local[0,i]/2, U_LS_local[1,i]/2, width = 0.01, alpha = 0.5, linewidth = 5, color = 'm')
    
    plt.legend(loc='upper left')
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.savefig("2D_active_subspace.pdf")
    
    plt.figure("Active Subspace Eigenvalues")
    plt.plot(S, 'r', linewidth = 4)
    plt.plot(S_FD, 'g')
    plt.plot(S_I, 'b')
    plt.plot(S_LS, 'y')
    plt.plot(S_I_local, 'c')
    plt.plot(S_LS_local, 'm')
    plt.legend(["Analytical", "FD", "Global Interpolation", "Global Regression", "Local Interpolation", "Local Regression"])
    plt.savefig("2D_active_subspace_eigenvalues.pdf")
    plt.close("all")