import numpy as np
import math
import os

from ASFEniCSx.sampling import Clustering, Sampling
from ASFEniCSx.functional import Functional
from ASFEniCSx.asfenicsx import ASFEniCSx

import matplotlib.pyplot as plt

if not os.path.exists("bivariateAnalyticalTest"):
    os.mkdir("bivariateAnalyticalTest")
if not os.path.exists("bivariateAnalyticalTestNoise"):
    os.mkdir("bivariateAnalyticalTestNoise")

os.chdir("bivariateAnalyticalTest")
dir = os.path.join(os.path.dirname(__file__), "bivariateAnalyticalTest")

if not os.path.exists("figures"):
    os.mkdir("figures")

def evaluate_f(x : list or np.array):
    f = math.exp(0.7 * x[0] + 0.3 * x[1])
    return f

def noisy_evaluate_f(x : list or np.array):
    f = math.exp(0.7 * x[0] + 0.3 * x[1])
    return f + np.random.uniform(-0.1 * np.max(f), 0.1 * np.max(f))

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
    fig= plt.figure("Bivariate test function")
    X,Y = np.meshgrid(x_range, y_range)
    plt.contourf(X,Y, data.T, 100)
    plt.colorbar()
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')

    plt.arrow(0, 0, 0.7, 0.3, width = 0.02, color = 'r')
    plt.arrow(0.0, 0.0, -0.3, 0.7, width = 0.02, color = 'b')

    plt.savefig("figures/2D_function.pdf")
    plt.clf()

    # Define clustered samples
    samples = Clustering(100, 2, 5)
    samples.random_uniform()
    samples.detect()
    samples.plot(os.path.join(dir,"figures/2D_samples.pdf"))
    samples.save(os.path.join(dir,"bivariate_samples.json"))

    # Plot clustering of the parameter space
    plt.figure("Clustering of the parameter space")
    from matplotlib import colors
    from matplotlib import cm
    cmap = plt.get_cmap('hsv')
    scalarMap = cm.ScalarMappable(colors.Normalize(vmin=0, vmax=samples.k),cmap=cmap)
    for x in x_range:
        for y in y_range:
            plt.plot(x,y,'o', color = scalarMap.to_rgba(samples.obtain_index(np.asarray([x,y]))))
    plt.savefig(os.path.join(dir,"figures/2D_clustering.pdf"))
    

    print("##############################################################Standard Test##############################################################")
    samples.assign_values(evaluate_f)

    # Define the cost function
    cost = Functional(2, evaluate_f)
    cost.get_derivative(calculate_df_dx)

    """ 
    Perform a validation test of the differentiation methods
    by comparing the analytical derivative with the numerical
    """

    # Analytical
    cost.get_gradient_method('A')
    A_data = np.zeros([100, 100, 2])
    for i,x in enumerate(x_range):
        for j,y in enumerate(y_range):
            A_data[i,j,:] = cost.gradient([x, y])

    # First Order Finite Differences
    cost.get_gradient_method('FD')
    FD_data = np.zeros([100, 100, 2])
    for i,x in enumerate(x_range):
        for j,y in enumerate(y_range):
            FD_data[i,j,:] = cost.gradient([x, y], order = 1)
    print("First Order Finite Differences")
    print(f"\tError (Frobenius-Norm): {np.linalg.norm(A_data-FD_data)/np.linalg.norm(A_data)}")
    print(f"\tError (inf-Norm): {np.max(A_data-FD_data)/np.max(A_data)}")
    print(f"\tNumber of function evaluations: {cost.number_of_calls()}" )
    cost.reset_number_of_calls()

    # Second Order Finite Differences
    cost.get_gradient_method('FD')
    FD_data = np.zeros([100, 100, 2])
    for i,x in enumerate(x_range):
        for j,y in enumerate(y_range):
            FD_data[i,j,:] = cost.gradient([x, y], order = 2)
    print("Second Order Finite Differences")
    print(f"\tError (Frobenius-Norm): {np.linalg.norm(A_data-FD_data)/np.linalg.norm(A_data)}")
    print(f"\tError (inf-Norm): {np.max(A_data-FD_data)/np.max(A_data)}")
    print(f"\tNumber of function evaluations: {cost.number_of_calls()}" )
    cost.reset_number_of_calls()

    # Interpolation (global)
    cost.get_gradient_method('I')
    maximal_order = 3

    # Default interpolation method (NOT LEAST SQUARES)
    interpolation_errors = np.zeros([maximal_order, 2])
    interpolation_dev_errors = np.zeros([maximal_order, 2])
    for k in range(1,maximal_order+1):
        cost.interpolation(samples,order = k, overwrite=True)
        data_interpolated = np.zeros([100, 100])
        I_data = np.zeros([100, 100, 2])
        for i,x in enumerate(x_range):
            for j,y in enumerate(y_range):
                data_interpolated[i,j] = cost.evaluate_interpolant([x, y])
                I_data[i,j,:] = cost.gradient([x, y])

        interpolation_errors[k-1,0] = np.linalg.norm(data-data_interpolated)/np.linalg.norm(data)
        interpolation_errors[k-1,1] = np.max(data-data_interpolated)/np.max(data)
        interpolation_dev_errors[k-1,0] = np.linalg.norm(A_data-I_data)/np.linalg.norm(A_data)
        interpolation_dev_errors[k-1,1] = np.max(A_data-I_data)/np.max(A_data)
        print(f"Interpolation (order = {k})")
        print(f"\tInterpolation Error (Frobenius-Norm): {interpolation_errors[k-1,0]}")
        print(f"\tInterpolation Error (inf-Norm): {interpolation_errors[k-1,1]}")
        print(f"\tError (Frobenius-Norm): {interpolation_dev_errors[k-1,0]}")
        print(f"\tError (inf-Norm): {interpolation_dev_errors[k-1,1]}")
        print(f"\tNumber of function evaluations: {cost.number_of_calls()}" )

        plt.figure("Default, global interpolation")
        plt.contourf(X,Y, data_interpolated, 100)
        plt.colorbar()
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.savefig(f"figures/2D_interpolation_order_{k}.pdf")
        plt.clf()

        cost.reset_number_of_calls()

    # Least squares interpolation
    least_squares_errors = np.zeros([maximal_order, 2])
    least_squares_dev_errors = np.zeros([maximal_order, 2])
    for k in range(1, maximal_order+1):
        cost.interpolation(samples,order = k, interpolation_method="LS", overwrite=True)
        data_LS = np.zeros([100, 100])
        LS_data = np.zeros([100, 100, 2])
        for i,x in enumerate(x_range):
            for j,y in enumerate(y_range):
                data_LS[i,j] = cost.evaluate_interpolant([x, y])
                LS_data[i,j,:] = cost.gradient([x, y])
        least_squares_errors[k-1,0] = np.linalg.norm(data-data_LS)/np.linalg.norm(data)
        least_squares_errors[k-1,1] = np.max(data-data_LS)/np.max(data)
        least_squares_dev_errors[k-1,0] = np.linalg.norm(A_data-LS_data)/np.linalg.norm(A_data)
        least_squares_dev_errors[k-1,1] = np.max(A_data-LS_data)/np.max(A_data)
        print(f"Least Squares Interpolation (order = {k})")
        print(f"\tInterpolation Error (Frobenius-Norm): {least_squares_errors[k-1,0]}")
        print(f"\tInterpolation Error (inf-Norm): {least_squares_errors[k-1,1]}")
        print(f"\tError (Frobenius-Norm): {least_squares_dev_errors[k-1,0]}")
        print(f"\tError (inf-Norm): {least_squares_dev_errors[k-1,1]}")
        print(f"\tNumber of function evaluations: {cost.number_of_calls()}" )

        plt.figure("Least sqaures, global interpolation")
        plt.contourf(X,Y, data_interpolated, 100)
        plt.colorbar()
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.savefig(f"figures/2D_least_squares_interpolation_order_{k}.pdf")
        plt.clf()

        cost.reset_number_of_calls()

    # Local interpolation
    local_errors = np.zeros([maximal_order, 2])
    local_dev_errors = np.zeros([maximal_order, 2])
    for k in range(1, maximal_order+1):
        cost.interpolation(samples,order=k, interpolation_method="local", overwrite=True, clustering = True)
        data_interpolated_local = np.zeros([100, 100])
        I_data_local = np.zeros([100, 100, 2])
        for i,x in enumerate(x_range):
            for j,y in enumerate(y_range):
                data_interpolated_local[i,j] = cost.evaluate_interpolant([x, y], samples)
                I_data_local[i,j,:] = cost.gradient([x, y], samples)
        local_errors[k-1,0] = np.linalg.norm(data-data_interpolated_local)/np.linalg.norm(data)
        local_errors[k-1,1] = np.max(data-data_interpolated_local)/np.max(data)
        local_dev_errors[k-1,0] = np.linalg.norm(A_data-I_data_local)/np.linalg.norm(A_data)
        local_dev_errors[k-1,1] = np.max(A_data-I_data_local)/np.max(A_data)
        print(f"Local Interpolation (order = {k})")
        print(f"\tInterpolation Error (Frobenius-Norm): {local_errors[k-1,0]}")
        print(f"\tInterpolation Error (inf-Norm): {local_errors[k-1,1]}")
        print(f"\tError (Frobenius-Norm): {local_dev_errors[k-1,0]}")
        print(f"\tError (inf-Norm): {local_dev_errors[k-1,1]}")
        print(f"\tNumber of function evaluations: {cost.number_of_calls()}" )

        plt.figure("Local interpolation")
        plt.contourf(X,Y, data_interpolated_local, 100)
        plt.colorbar()
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.savefig(f"figures/2D_local_interpolation_order_{k}.pdf")
        plt.clf()

        cost.reset_number_of_calls()


    # Local least squares interpolation
    local_least_squares_errors = np.zeros([maximal_order, 2])
    local_least_squares_dev_errors = np.zeros([maximal_order, 2])
    for k in range(1,   maximal_order+1):
        cost.interpolation(samples, order=k, interpolation_method="LS", overwrite=True, clustering = True)
        data_LS_local = np.zeros([100, 100])
        LS_data_local = np.zeros([100, 100, 2])
        for i,x in enumerate(x_range):
            for j,y in enumerate(y_range):
                data_LS_local[i,j] = cost.evaluate_interpolant([x, y], samples)
                LS_data_local[i,j,:] = cost.gradient([x, y], samples)
        local_least_squares_errors[k-1,0] = np.linalg.norm(data-data_LS_local)/np.linalg.norm(data)
        local_least_squares_errors[k-1,1] = np.max(data-data_LS_local)/np.max(data)
        local_least_squares_dev_errors[k-1,0] = np.linalg.norm(A_data-LS_data_local)/np.linalg.norm(A_data)
        local_least_squares_dev_errors[k-1,1] = np.max(A_data-LS_data_local)/np.max(A_data)
        print(f"Local Least Squares Interpolation (order = {k})")
        print(f"\tInterpolation Error (Frobenius-Norm): {local_least_squares_errors[k-1,0]}")
        print(f"\tInterpolation Error (inf-Norm): {local_least_squares_errors[k-1,1]}")
        print(f"\tError (Frobenius-Norm): {local_least_squares_dev_errors[k-1,0]}")
        print(f"\tError (inf-Norm): {local_least_squares_dev_errors[k-1,1]}")
        print(f"\tNumber of function evaluations: {cost.number_of_calls()}" )

        plt.figure("Local Least squares interpolation")
        plt.contourf(X,Y, data_LS_local, 100)
        plt.colorbar()
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.savefig(f"figures/2D_local_LS_order_{k}.pdf")
        plt.clf()

        cost.reset_number_of_calls()

    fig,ax = plt.subplots()
    ax.plot(range(1, maximal_order+1), interpolation_errors[:,0], label = "Global Interpolation")
    ax.plot(range(1, maximal_order+1), least_squares_errors[:,0], label = "Global Least Squares Interpolation")
    ax.plot(range(1, maximal_order+1), local_errors[:,0], label = "Local Interpolation")
    ax.plot(range(1, maximal_order+1), local_least_squares_errors[:,0], label = "Local Least Squares Interpolation")
    ax.set_xlabel("Order")
    ax.set_xticks(range(1, maximal_order+1))
    ax.set_ylabel("Frobenius Norm")
    ax.legend()
    ax2= ax.twinx()
    ax2.plot(range(1, maximal_order+1), interpolation_errors[:,1], label = "Global Interpolation", linestyle = ":")
    ax2.plot(range(1, maximal_order+1), least_squares_errors[:,1], label = "Global Least Squares Interpolation", linestyle = ":")
    ax2.plot(range(1, maximal_order+1), local_errors[:,1], label = "Local Interpolation", linestyle = ":")
    ax2.plot(range(1, maximal_order+1), local_least_squares_errors[:,1], label = "Local Least Squares Interpolation", linestyle = ":")
    ax2.set_ylabel("inf Norm (dotted)")
    
    fig.savefig("figures/2D_interpolation_errors.pdf")

    fig,ax = plt.subplots()
    ax.plot(range(1, maximal_order+1), interpolation_dev_errors[:,0], label = "Global Interpolation")
    ax.plot(range(1, maximal_order+1), least_squares_dev_errors[:,0], label = "Global Least Squares Interpolation")
    ax.plot(range(1, maximal_order+1), local_dev_errors[:,0], label = "Local Interpolation")
    ax.plot(range(1, maximal_order+1), local_least_squares_dev_errors[:,0], label = "Local Least Squares Interpolation")
    ax.set_xlabel("Order")
    ax.set_xticks(range(1, maximal_order+1))
    ax.set_ylabel("Frobenius Norm")
    ax.legend()
    ax2= ax.twinx()
    ax2.plot(range(1, maximal_order+1), interpolation_dev_errors[:,1], label = "Global Interpolation", linestyle = ":")
    ax2.plot(range(1, maximal_order+1), least_squares_dev_errors[:,1], label = "Global Least Squares Interpolation", linestyle = ":")
    ax2.plot(range(1, maximal_order+1), local_dev_errors[:,1], label = "Local Interpolation", linestyle = ":")
    ax2.plot(range(1, maximal_order+1), local_least_squares_dev_errors[:,1], label = "Local Least Squares Interpolation", linestyle = ":")
    ax2.set_ylabel("inf Norm (dotted)")


    fig.savefig("figures/2D_interpolation_dev_errors.pdf")
    """
    Active Subspace Construction
    """
    print("Active Subspace")
    n= 2
    asfenicsx = ASFEniCSx(n, cost, samples)

    # Reset to analytical gradient
    cost.get_derivative(calculate_df_dx)
    cost.get_gradient_method('A')

    U, S = asfenicsx.estimation()

    cost.get_gradient_method('FD')
    asfenicsx.evaluate_gradients()
    U_FD, S_FD = asfenicsx.estimation()
    print(f"\tNumer of function evaluations for FD: {cost.number_of_calls()}")
    print(f"\tError FD (EV, EW): {np.linalg.norm(U-U_FD)} , {np.linalg.norm(S-S_FD)}")
    cost.reset_number_of_calls()

    cost.interpolation(samples, overwrite = True)
    cost.get_gradient_method('I')
    asfenicsx.evaluate_gradients()
    U_I, S_I = asfenicsx.estimation()
    print(f"\tNumer of function evaluations for interpolation: {cost.number_of_calls()}")
    print(f"\tError I (EV, EW): {np.linalg.norm(U-U_I)} , {np.linalg.norm(S-S_I)}")
    cost.reset_number_of_calls()

    cost.interpolation(samples, interpolation_method = "LS", overwrite = True)
    cost.get_gradient_method('I')
    asfenicsx.evaluate_gradients()
    U_LS, S_LS = asfenicsx.estimation()
    print(f"\tNumer of function evaluations for least squares interpolation: {cost.number_of_calls()}")
    print(f"\tError LS (EV, EW): {np.linalg.norm(U-U_LS)} , {np.linalg.norm(S-S_LS)}")
    cost.reset_number_of_calls()

    cost.interpolation(samples, overwrite = True, clustering = True)
    cost.get_gradient_method('I')
    asfenicsx.evaluate_gradients()
    U_I_local, S_I_local = asfenicsx.estimation()
    print(f"\tNumer of function evaluations for interpolation: {cost.number_of_calls()}")
    print(f"\tError I local (EV, EW): {np.linalg.norm(U-U_I_local)} , {np.linalg.norm(S-S_I_local)}")
    cost.reset_number_of_calls()

    cost.interpolation(samples, interpolation_method = "LS", overwrite = True, clustering = True)
    cost.get_gradient_method('I')
    asfenicsx.evaluate_gradients()
    U_LS_local, S_LS_local = asfenicsx.estimation()
    print(f"\tNumer of function evaluations for local least squares interpolation: {cost.number_of_calls()}")
    print(f"\tError LS local (EV, EW): {np.linalg.norm(U-U_LS_local)} , {np.linalg.norm(S-S_LS_local)}")
    cost.reset_number_of_calls()

    fig= plt.figure("Active Subspace")
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
            plt.arrow(0, 0, U_I[0,i]/2, U_I[1,i]/2, width = 0.01, alpha = 0.5, linewidth = 5, color = 'b', label="Interpolation")
        else:
            plt.arrow(0, 0, U_I[0,i]/2, U_I[1,i]/2, width = 0.01, alpha = 0.5, linewidth = 5, color = 'b')

    for i in range(n):
        if i ==0:
            plt.arrow(0, 0, U_LS[0,i]/2, U_LS[1,i]/2, width = 0.01, alpha = 0.5, linewidth = 5, color = 'y', label="Least Squares")
        else:
            plt.arrow(0, 0, U_LS[0,i]/2, U_LS[1,i]/2, width = 0.01, alpha = 0.5, linewidth = 5, color = 'y')

    for i in range(n):
        if i ==0:
            plt.arrow(0, 0, U_I_local[0,i]/2, U_I_local[1,i]/2, width = 0.01, alpha = 0.5, linewidth = 5, color = 'c', label="Local Interpolation")
        else:
            plt.arrow(0, 0, U_I_local[0,i]/2, U_I_local[1,i]/2, width = 0.01, alpha = 0.5, linewidth = 5, color = 'c')

    for i in range(n):
        if i ==0:
            plt.arrow(0, 0, U_LS_local[0,i]/2, U_LS_local[1,i]/2, width = 0.01, alpha = 0.5, linewidth = 5, color = 'm', label="Local Least Squares")
        else:
            plt.arrow(0, 0, U_LS_local[0,i]/2, U_LS_local[1,i]/2, width = 0.01, alpha = 0.5, linewidth = 5, color = 'm')
    
    plt.legend(loc='lower right')
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.savefig("figures/2D_active_subspace.pdf")
    
    plt.figure("Active Subspace Eigenvalues")
    plt.plot(S, 'r', linewidth = 4)
    plt.plot(S_FD, 'g')
    plt.plot(S_I, 'b')
    plt.plot(S_LS, 'y')
    plt.plot(S_I_local, 'c')
    plt.plot(S_LS_local, 'm')
    plt.legend(["Analytical", "FD", "Interpolation", "Least Squares", "Local Interpolation", "Local Least Squares"])
    plt.savefig("figures/2D_active_subspace_eigenvalues.pdf")
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

    noise_data= np.zeros((len(x_range), len(y_range)))
    for i,x in enumerate(x_range):
        for j,y in enumerate(y_range):
             noise_data[i,j]= noisy_evaluate_f([x, y])
    fig= plt.figure("Bivariate test function")
    X,Y = np.meshgrid(x_range, y_range)
    plt.contourf(X,Y, noise_data, 100)
    plt.colorbar()
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.savefig("figures/2D_function.pdf")
    plt.clf()

    samples.assign_values(noisy_evaluate_f, overwrite = True)

    # Define the cost function
    cost = Functional(2, noisy_evaluate_f)

    """ 
    Perform a validation test of the differentiation methods
    by comparing the analytical derivative with the numerical
    """

    # First Order Finite Differences
    cost.get_gradient_method('FD')
    FD_data = np.zeros([100, 100, 2])
    for i,x in enumerate(x_range):
        for j,y in enumerate(y_range):
            FD_data[i,j,:] = cost.gradient([x, y], order = 1)
    print("First Order Finite Differences")
    print(f"\tError (Frobenius-Norm): {np.linalg.norm(A_data-FD_data)/np.linalg.norm(A_data)}")
    print(f"\tError (inf-Norm): {np.max(A_data-FD_data)/np.max(A_data)}")
    print(f"\tNumber of function evaluations: {cost.number_of_calls()}" )
    cost.reset_number_of_calls()

    # Second Order Finite Differences
    cost.get_gradient_method('FD')
    FD_data = np.zeros([100, 100, 2])
    for i,x in enumerate(x_range):
        for j,y in enumerate(y_range):
            FD_data[i,j,:] = cost.gradient([x, y], order = 2)
    print("Second Order Finite Differences")
    print(f"\tError (Frobenius-Norm): {np.linalg.norm(A_data-FD_data)/np.linalg.norm(A_data)}")
    print(f"\tError (inf-Norm): {np.max(A_data-FD_data)/np.max(A_data)}")
    print(f"\tNumber of function evaluations: {cost.number_of_calls()}" )
    cost.reset_number_of_calls()

    # Interpolation (global)
    cost.get_gradient_method('I')
    maximal_order = 3

    # Default interpolation method (NOT LEAST SQUARES)
    interpolation_errors = np.zeros([maximal_order, 2])
    interpolation_dev_errors = np.zeros([maximal_order, 2])
    for k in range(1,maximal_order+1):
        cost.interpolation(samples,order = k, overwrite=True)
        data_interpolated = np.zeros([100, 100])
        I_data = np.zeros([100, 100, 2])
        for i,x in enumerate(x_range):
            for j,y in enumerate(y_range):
                data_interpolated[i,j] = cost.evaluate_interpolant([x, y])
                I_data[i,j,:] = cost.gradient([x, y])

        interpolation_errors[k-1,0] = np.linalg.norm(data-data_interpolated)/np.linalg.norm(data)
        interpolation_errors[k-1,1] = np.max(data-data_interpolated)/np.max(data)
        interpolation_dev_errors[k-1,0] = np.linalg.norm(A_data-I_data)/np.linalg.norm(A_data)
        interpolation_dev_errors[k-1,1] = np.max(A_data-I_data)/np.max(A_data)
        print(f"Interpolation (order = {k})")
        print(f"\tInterpolation Error (Frobenius-Norm): {interpolation_errors[k-1,0]}")
        print(f"\tInterpolation Error (inf-Norm): {interpolation_errors[k-1,1]}")
        print(f"\tError (Frobenius-Norm): {interpolation_dev_errors[k-1,0]}")
        print(f"\tError (inf-Norm): {interpolation_dev_errors[k-1,1]}")
        print(f"\tNumber of function evaluations: {cost.number_of_calls()}" )

        plt.figure("Default, global interpolation")
        plt.contourf(X,Y, data_interpolated, 100)
        plt.colorbar()
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.savefig(f"figures/2D_interpolation_order_{k}.pdf")
        plt.clf()

        cost.reset_number_of_calls()

    # Least squares interpolation
    least_squares_errors = np.zeros([maximal_order, 2])
    least_squares_dev_errors = np.zeros([maximal_order, 2])
    for k in range(1, maximal_order+1):
        cost.interpolation(samples,order = k, interpolation_method="LS", overwrite=True)
        data_LS = np.zeros([100, 100])
        LS_data = np.zeros([100, 100, 2])
        for i,x in enumerate(x_range):
            for j,y in enumerate(y_range):
                data_LS[i,j] = cost.evaluate_interpolant([x, y])
                LS_data[i,j,:] = cost.gradient([x, y])
        least_squares_errors[k-1,0] = np.linalg.norm(data-data_LS)/np.linalg.norm(data)
        least_squares_errors[k-1,1] = np.max(data-data_LS)/np.max(data)
        least_squares_dev_errors[k-1,0] = np.linalg.norm(A_data-LS_data)/np.linalg.norm(A_data)
        least_squares_dev_errors[k-1,1] = np.max(A_data-LS_data)/np.max(A_data)
        print(f"Least Squares Interpolation (order = {k})")
        print(f"\tInterpolation Error (Frobenius-Norm): {least_squares_errors[k-1,0]}")
        print(f"\tInterpolation Error (inf-Norm): {least_squares_errors[k-1,1]}")
        print(f"\tError (Frobenius-Norm): {least_squares_dev_errors[k-1,0]}")
        print(f"\tError (inf-Norm): {least_squares_dev_errors[k-1,1]}")
        print(f"\tNumber of function evaluations: {cost.number_of_calls()}" )

        plt.figure("Least sqaures, global interpolation")
        plt.contourf(X,Y, data_interpolated, 100)
        plt.colorbar()
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.savefig(f"figures/2D_least_squares_interpolation_order_{k}.pdf")
        plt.clf()

        cost.reset_number_of_calls()

    # Local interpolation
    local_errors = np.zeros([maximal_order, 2])
    local_dev_errors = np.zeros([maximal_order, 2])
    for k in range(1, maximal_order+1):
        cost.interpolation(samples,order=k, interpolation_method="local", overwrite=True, clustering = True)
        data_interpolated_local = np.zeros([100, 100])
        I_data_local = np.zeros([100, 100, 2])
        for i,x in enumerate(x_range):
            for j,y in enumerate(y_range):
                data_interpolated_local[i,j] = cost.evaluate_interpolant([x, y], samples)
                I_data_local[i,j,:] = cost.gradient([x, y], samples)
        local_errors[k-1,0] = np.linalg.norm(data-data_interpolated_local)/np.linalg.norm(data)
        local_errors[k-1,1] = np.max(data-data_interpolated_local)/np.max(data)
        local_dev_errors[k-1,0] = np.linalg.norm(A_data-I_data_local)/np.linalg.norm(A_data)
        local_dev_errors[k-1,1] = np.max(A_data-I_data_local)/np.max(A_data)
        print(f"Local Interpolation (order = {k})")
        print(f"\tInterpolation Error (Frobenius-Norm): {local_errors[k-1,0]}")
        print(f"\tInterpolation Error (inf-Norm): {local_errors[k-1,1]}")
        print(f"\tError (Frobenius-Norm): {local_dev_errors[k-1,0]}")
        print(f"\tError (inf-Norm): {local_dev_errors[k-1,1]}")
        print(f"\tNumber of function evaluations: {cost.number_of_calls()}" )

        plt.figure("Local interpolation")
        plt.contourf(X,Y, data_interpolated_local, 100)
        plt.colorbar()
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.savefig(f"figures/2D_local_interpolation_order_{k}.pdf")
        plt.clf()

        cost.reset_number_of_calls()


    # Local least squares interpolation
    local_least_squares_errors = np.zeros([maximal_order, 2])
    local_least_squares_dev_errors = np.zeros([maximal_order, 2])
    for k in range(1,   maximal_order+1):
        cost.interpolation(samples, order=k, interpolation_method="LS", overwrite=True, clustering = True)
        data_LS_local = np.zeros([100, 100])
        LS_data_local = np.zeros([100, 100, 2])
        for i,x in enumerate(x_range):
            for j,y in enumerate(y_range):
                data_LS_local[i,j] = cost.evaluate_interpolant([x, y], samples)
                LS_data_local[i,j,:] = cost.gradient([x, y], samples)
        local_least_squares_errors[k-1,0] = np.linalg.norm(data-data_LS_local)/np.linalg.norm(data)
        local_least_squares_errors[k-1,1] = np.max(data-data_LS_local)/np.max(data)
        local_least_squares_dev_errors[k-1,0] = np.linalg.norm(A_data-LS_data_local)/np.linalg.norm(A_data)
        local_least_squares_dev_errors[k-1,1] = np.max(A_data-LS_data_local)/np.max(A_data)
        print(f"Local Least Squares Interpolation (order = {k})")
        print(f"\tInterpolation Error (Frobenius-Norm): {local_least_squares_errors[k-1,0]}")
        print(f"\tInterpolation Error (inf-Norm): {local_least_squares_errors[k-1,1]}")
        print(f"\tError (Frobenius-Norm): {local_least_squares_dev_errors[k-1,0]}")
        print(f"\tError (inf-Norm): {local_least_squares_dev_errors[k-1,1]}")
        print(f"\tNumber of function evaluations: {cost.number_of_calls()}" )

        plt.figure("Local Least squares interpolation")
        plt.contourf(X,Y, data_LS_local, 100)
        plt.colorbar()
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.savefig(f"figures/2D_local_LS_order_{k}.pdf")
        plt.clf()

        cost.reset_number_of_calls()

    fig,ax = plt.subplots()
    ax.plot(range(1, maximal_order+1), interpolation_errors[:,0], label = "Global Interpolation")
    ax.plot(range(1, maximal_order+1), least_squares_errors[:,0], label = "Global Least Squares Interpolation")
    ax.plot(range(1, maximal_order+1), local_errors[:,0], label = "Local Interpolation")
    ax.plot(range(1, maximal_order+1), local_least_squares_errors[:,0], label = "Local Least Squares Interpolation")
    ax.set_xlabel("Order")
    ax.set_xticks(range(1, maximal_order+1))
    ax.set_ylabel("Frobenius Norm")
    ax.legend()
    ax2= ax.twinx()
    ax2.plot(range(1, maximal_order+1), interpolation_errors[:,1], label = "Global Interpolation", linestyle = ":")
    ax2.plot(range(1, maximal_order+1), least_squares_errors[:,1], label = "Global Least Squares Interpolation", linestyle = ":")
    ax2.plot(range(1, maximal_order+1), local_errors[:,1], label = "Local Interpolation", linestyle = ":")
    ax2.plot(range(1, maximal_order+1), local_least_squares_errors[:,1], label = "Local Least Squares Interpolation", linestyle = ":")
    ax2.set_ylabel("inf Norm (dotted)")
    
    fig.savefig("figures/2D_interpolation_errors.pdf")

    fig,ax = plt.subplots()
    ax.plot(range(1, maximal_order+1), interpolation_dev_errors[:,0], label = "Global Interpolation")
    ax.plot(range(1, maximal_order+1), least_squares_dev_errors[:,0], label = "Global Least Squares Interpolation")
    ax.plot(range(1, maximal_order+1), local_dev_errors[:,0], label = "Local Interpolation")
    ax.plot(range(1, maximal_order+1), local_least_squares_dev_errors[:,0], label = "Local Least Squares Interpolation")
    ax.set_xlabel("Order")
    ax.set_xticks(range(1, maximal_order+1))
    ax.set_ylabel("Frobenius Norm")
    ax.legend()
    ax2= ax.twinx()
    ax2.plot(range(1, maximal_order+1), interpolation_dev_errors[:,1], label = "Global Interpolation", linestyle = ":")
    ax2.plot(range(1, maximal_order+1), least_squares_dev_errors[:,1], label = "Global Least Squares Interpolation", linestyle = ":")
    ax2.plot(range(1, maximal_order+1), local_dev_errors[:,1], label = "Local Interpolation", linestyle = ":")
    ax2.plot(range(1, maximal_order+1), local_least_squares_dev_errors[:,1], label = "Local Least Squares Interpolation", linestyle = ":")
    ax2.set_ylabel("inf Norm (dotted)")


    fig.savefig("figures/2D_interpolation_dev_errors.pdf")
    """
    Active Subspace Construction
    """
    print("Active Subspace")
    n= 2
    asfenicsx = ASFEniCSx(n, cost, samples)

    # Reset to analytical gradient
    cost.get_derivative(calculate_df_dx)
    cost.get_gradient_method('A')

    U, S = asfenicsx.estimation()

    cost.get_gradient_method('FD')
    asfenicsx.evaluate_gradients()
    U_FD, S_FD = asfenicsx.estimation()
    print(f"\tNumer of function evaluations for FD-2: {cost.number_of_calls()}")
    print(f"\tError FD (EV, EW): {np.linalg.norm(U-U_FD)} , {np.linalg.norm(S-S_FD)}")
    cost.reset_number_of_calls()

    cost.interpolation(samples, overwrite = True)
    cost.get_gradient_method('I')
    asfenicsx.evaluate_gradients()
    U_I, S_I = asfenicsx.estimation()
    print(f"\tNumer of function evaluations for interpolation-2: {cost.number_of_calls()}")
    print(f"\tError I (EV, EW): {np.linalg.norm(U-U_I)} , {np.linalg.norm(S-S_I)}")
    cost.reset_number_of_calls()

    cost.interpolation(samples, interpolation_method = "LS", overwrite = True)
    cost.get_gradient_method('I')
    asfenicsx.evaluate_gradients()
    U_LS, S_LS = asfenicsx.estimation()
    print(f"\tNumer of function evaluations for least squares interpolation-2: {cost.number_of_calls()}")
    print(f"\tError LS (EV, EW): {np.linalg.norm(U-U_LS)} , {np.linalg.norm(S-S_LS)}")
    cost.reset_number_of_calls()

    cost.interpolation(samples, overwrite = True, clustering = True)
    cost.get_gradient_method('I')
    asfenicsx.evaluate_gradients()
    U_I_local, S_I_local = asfenicsx.estimation()
    print(f"\tNumer of function evaluations for interpolation-2: {cost.number_of_calls()}")
    print(f"\tError I local (EV, EW): {np.linalg.norm(U-U_I_local)} , {np.linalg.norm(S-S_I_local)}")
    cost.reset_number_of_calls()

    cost.interpolation(samples, interpolation_method = "LS", overwrite = True, clustering = True)
    cost.get_gradient_method('I')
    asfenicsx.evaluate_gradients()
    U_LS_local, S_LS_local = asfenicsx.estimation()
    print(f"\tNumer of function evaluations for least squares interpolation-2: {cost.number_of_calls()}")
    print(f"\tError LS local (EV, EW): {np.linalg.norm(U-U_LS_local)} , {np.linalg.norm(S-S_LS_local)}")
    cost.reset_number_of_calls()

    fig= plt.figure("Active Subspace")
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
            plt.arrow(0, 0, U_I[0,i]/2, U_I[1,i]/2, width = 0.01, alpha = 0.5, linewidth = 5, color = 'b', label="Interpolation")
        else:
            plt.arrow(0, 0, U_I[0,i]/2, U_I[1,i]/2, width = 0.01, alpha = 0.5, linewidth = 5, color = 'b')

    for i in range(n):
        if i ==0:
            plt.arrow(0, 0, U_LS[0,i]/2, U_LS[1,i]/2, width = 0.01, alpha = 0.5, linewidth = 5, color = 'y', label="Least Squares")
        else:
            plt.arrow(0, 0, U_LS[0,i]/2, U_LS[1,i]/2, width = 0.01, alpha = 0.5, linewidth = 5, color = 'y')

    for i in range(n):
        if i ==0:
            plt.arrow(0, 0, U_I_local[0,i]/2, U_I_local[1,i]/2, width = 0.01, alpha = 0.5, linewidth = 5, color = 'c', label="Local Interpolation")
        else:
            plt.arrow(0, 0, U_I_local[0,i]/2, U_I_local[1,i]/2, width = 0.01, alpha = 0.5, linewidth = 5, color = 'c')

    for i in range(n):
        if i ==0:
            plt.arrow(0, 0, U_LS_local[0,i]/2, U_LS_local[1,i]/2, width = 0.01, alpha = 0.5, linewidth = 5, color = 'm', label="Local Least Squares")
        else:
            plt.arrow(0, 0, U_LS_local[0,i]/2, U_LS_local[1,i]/2, width = 0.01, alpha = 0.5, linewidth = 5, color = 'm')
    
    plt.legend(loc='lower right')
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.savefig("figures/2D_active_subspace.pdf")
    
    plt.figure("Active Subspace Eigenvalues")
    plt.plot(S, 'r', linewidth = 4)
    plt.plot(S_FD, 'g')
    plt.plot(S_I, 'b')
    plt.plot(S_LS, 'y')
    plt.plot(S_I_local, 'c')
    plt.plot(S_LS_local, 'm')
    plt.legend(["Analytical", "FD", "Interpolation", "Least Squares", "Local Interpolation", "Local Least Squares"])
    plt.savefig("figures/2D_active_subspace_eigenvalues.pdf")

    plt.close('all')