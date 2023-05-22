import numpy as np
import math
from asfenicsx import ASFEniCSx, Clustering, Functional

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
    # Plot the funciton
    data = np.zeros([100, 100])
    x_range = np.linspace(-1, 1, 100)
    y_range = np.linspace(-1, 1, 100)
    for i,x in enumerate(x_range):
        for j,y in enumerate(y_range):
             data[i,j]= evaluate_f([x, y])
    fig= plt.figure("Bivariate test function")
    X,Y = np.meshgrid(x_range, y_range)
    plt.contourf(X,Y, data, 100)
    plt.colorbar()
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.savefig("figures/2D_function.pdf")
    plt.clf()

    # Define clustered samples
    samples = Clustering(100, 2, 5)
    samples.detect()
    samples.plot("2D_samples.pdf")

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
    print(f"\tError (Frobenius-Norm): {np.linalg.norm(A_data-FD_data)}")
    print(f"\tError (inf-Norm): {np.max(A_data-FD_data)}")
    print(f"\tNumber of function evaluations: {cost.number_of_calls()}" )
    cost.reset_number_of_calls()

    # Second Order Finite Differences
    cost.get_gradient_method('FD')
    FD_data = np.zeros([100, 100, 2])
    for i,x in enumerate(x_range):
        for j,y in enumerate(y_range):
            FD_data[i,j,:] = cost.gradient([x, y], order = 2)
    print("Second Order Finite Differences")
    print(f"\tError (Frobenius-Norm): {np.linalg.norm(A_data-FD_data)}")
    print(f"\tError (inf-Norm): {np.max(A_data-FD_data)}")
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

        interpolation_errors[k-1,0] = np.linalg.norm(data-data_interpolated)
        interpolation_errors[k-1,1] = np.max(data-data_interpolated)
        interpolation_dev_errors[k-1,0] = np.linalg.norm(A_data-I_data)
        interpolation_dev_errors[k-1,1] = np.max(A_data-I_data)
        print(f"Interpolation (order = {k})")
        print(f"\tInterpolation Error (Frobenius-Norm): {np.linalg.norm(data-data_interpolated)}")
        print(f"\tInterpolation Error (inf-Norm): {np.max(data-data_interpolated)}")
        print(f"\tError (Frobenius-Norm): {np.linalg.norm(A_data-I_data)}")
        print(f"\tError (inf-Norm): {np.max(A_data-I_data)}")
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
        least_squares_errors[k-1,0] = np.linalg.norm(data-data_LS)
        least_squares_errors[k-1,1] = np.max(data-data_LS)
        least_squares_dev_errors[k-1,0] = np.linalg.norm(A_data-LS_data)
        least_squares_dev_errors[k-1,1] = np.max(A_data-LS_data)
        print(f"Least Squares Interpolation (order = {k})")
        print(f"\tInterpolation Error (Frobenius-Norm): {np.linalg.norm(data-data_LS)}")
        print(f"\tInterpolation Error (inf-Norm): {np.max(data-data_LS)}")
        print(f"\tError (Frobenius-Norm): {np.linalg.norm(A_data-LS_data)}")
        print(f"\tError (inf-Norm): {np.max(A_data-LS_data)}")
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
        local_errors[k-1,0] = np.linalg.norm(data-data_interpolated_local)
        local_errors[k-1,1] = np.max(data-data_interpolated_local)
        local_dev_errors[k-1,0] = np.linalg.norm(A_data-I_data_local)
        local_dev_errors[k-1,1] = np.max(A_data-I_data_local)
        print(f"Local Interpolation (order = {k})")
        print(f"\tInterpolation Error (Frobenius-Norm): {np.linalg.norm(data-data_interpolated_local)}")
        print(f"\tInterpolation Error (inf-Norm): {np.max(data-data_interpolated_local)}")
        print(f"\tError (Frobenius-Norm): {np.linalg.norm(A_data-I_data_local)}")
        print(f"\tError (inf-Norm): {np.max(A_data-I_data_local)}")
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
        local_least_squares_errors[k-1,0] = np.linalg.norm(data-data_LS_local)
        local_least_squares_errors[k-1,1] = np.max(data-data_LS_local)
        local_least_squares_dev_errors[k-1,0] = np.linalg.norm(A_data-LS_data_local)
        local_least_squares_dev_errors[k-1,1] = np.max(A_data-LS_data_local)
        print(f"Local Least Squares Interpolation (order = {k})")
        print(f"\tInterpolation Error (Frobenius-Norm): {np.linalg.norm(data-data_LS_local)}")
        print(f"\tInterpolation Error (inf-Norm): {np.max(data-data_LS_local)}")
        print(f"\tError (Frobenius-Norm): {np.linalg.norm(A_data-LS_data_local)}")
        print(f"\tError (inf-Norm): {np.max(A_data-LS_data_local)}")
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

    U, S = asfenicsx.random_sampling_algorithm()

    cost.get_gradient_method('FD')
    U_FD, S_FD = asfenicsx.random_sampling_algorithm()
    print(f"\tNumer of function evaluations for FD: {cost.number_of_calls()}")
    print(f"\tError FD (EV, EW): {np.linalg.norm(U-U_FD)} , {np.linalg.norm(S-S_FD)}")
    cost.reset_number_of_calls()

    cost.interpolation(samples, overwrite = True)
    cost.get_gradient_method('I')
    U_I, S_I = asfenicsx.random_sampling_algorithm()
    print(f"\tNumer of function evaluations for integration: {cost.number_of_calls()}")
    print(f"\tError I (EV, EW): {np.linalg.norm(U-U_I)} , {np.linalg.norm(S-S_I)}")
    cost.reset_number_of_calls()

    cost.interpolation(samples, interpolation_method = "LS", overwrite = True)
    cost.get_gradient_method('I')
    U_LS, S_LS = asfenicsx.random_sampling_algorithm()
    print(f"\tNumer of function evaluations for least squares integration: {cost.number_of_calls()}")
    print(f"\tError LS (EV, EW): {np.linalg.norm(U-U_LS)} , {np.linalg.norm(S-S_LS)}")
    cost.reset_number_of_calls()

    cost.interpolation(samples, overwrite = True, clustering = True)
    cost.get_gradient_method('I')
    U_I_local, S_I_local = asfenicsx.random_sampling_algorithm()
    print(f"\tNumer of function evaluations for integration: {cost.number_of_calls()}")
    print(f"\tError I local (EV, EW): {np.linalg.norm(U-U_I_local)} , {np.linalg.norm(S-S_I_local)}")
    cost.reset_number_of_calls()

    cost.interpolation(samples, interpolation_method = "LS", overwrite = True, clustering = True)
    cost.get_gradient_method('I')
    U_LS_local, S_LS_local = asfenicsx.random_sampling_algorithm()
    print(f"\tNumer of function evaluations for least squares integration: {cost.number_of_calls()}")
    print(f"\tError LS local (EV, EW): {np.linalg.norm(U-U_LS_local)} , {np.linalg.norm(S-S_LS_local)}")
    cost.reset_number_of_calls()

    fig= plt.figure("Active Subspace")
    X,Y = np.meshgrid(x_range, y_range)
    plt.contourf(X,Y, data, 100)
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
            plt.arrow(0, 0, U_I[0,i]/2, U_I[1,i]/2, width = 0.01, alpha = 0.5, linewidth = 5, color = 'b', label="Integration")
        else:
            plt.arrow(0, 0, U_I[0,i]/2, U_I[1,i]/2, width = 0.01, alpha = 0.5, linewidth = 5, color = 'b')

    for i in range(n):
        if i ==0:
            plt.arrow(0, 0, U_LS[0,i]/2, U_LS[1,i]/2, width = 0.01, alpha = 0.5, linewidth = 5, color = 'y', label="Least Squares")
        else:
            plt.arrow(0, 0, U_LS[0,i]/2, U_LS[1,i]/2, width = 0.01, alpha = 0.5, linewidth = 5, color = 'y')

    for i in range(n):
        if i ==0:
            plt.arrow(0, 0, U_I_local[0,i]/2, U_I_local[1,i]/2, width = 0.01, alpha = 0.5, linewidth = 5, color = 'c', label="Local Integration")
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
    plt.legend(["Analytical", "FD", "Integration", "Least Squares", "Local Integration", "Local Least Squares"])
    plt.savefig("figures/2D_active_subspace_eigenvalues.pdf")

    
