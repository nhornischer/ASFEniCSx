import os
import numpy as np
import matplotlib.pyplot as plt

from ASFEniCSx.sampling import Sampling
from ASFEniCSx.functional import Functional, Analytical
from ASFEniCSx.asfenicsx import ASFEniCSx
from ASFEniCSx.utils import debug_info, normalizer, denormalizer

dir = os.path.dirname(__file__)
_debug = True

# Check if directory wingWeight exists if not create it
if not os.path.exists(os.path.join(dir,"wingWeight")):
    os.makedirs(os.path.join(dir,"wingWeight"))

##############################################################################################################
# Define the function of interest and its gradient
##############################################################################################################

# In this example we are interested in the wing weight as a function of the following inputs:
# Sw: wing area (ft^2)                          [150, 200]
# Wfw: wing fuel weight (lb)                    [220, 300]
# A: aspect ratio                               [6,10]
# Lambda: quarter-chord sweep angle (deg)       [-10, 10]
# q: dynamic pressure (lb/ft^2)                 [16, 45]
# lambda: taper ratio                           [.5, 1]
# tc: thickness-to-chord ratio                  [.08, .18]
# Nz: ultimate load factor                      [2.5, 6]  
# Wdg: design gross weight (lb)                 [1700, 2500]
# Wp: wing primary structure weight (lb)        [0.025, 0.08]

# This simulation experiment is obtained from http://www.sfu.ca/~ssurjano/wingweight.html

def wing(x):    
    
    Sw = x[0]; Wfw = x[1]; A = x[2]; L = x[3]*np.pi/180.; q = x[4]
    l = x[5]; tc = x[6]; Nz = x[7]; Wdg = x[8]; Wp = x[9]
    
    return (.036*Sw**.758*Wfw**.0035*A**.6*np.cos(L)**-.9*q**.006*l**.04*100**-.3*tc**-.3*Nz**.49*Wdg**.49 + Sw*Wp)

def wing_grad(x):
    #each row of xx should be [Sw. Wfw, A, Lambda, q, lambda, tc, Nz, Wdg, Wp] in the normalized input space
    #returns matrix whose ith row is gradient of wing function at ith row of inputs
    
    Sw = x[0]; Wfw = x[1]; A = x[2]; L = x[3]*np.pi/180.; q = x[4]
    l = x[5]; tc = x[6]; Nz = x[7]; Wdg = x[8]; Wp = x[9]
    
    Q = .036*Sw**.758*Wfw**.0035*A**.6*np.cos(L)**-.9*q**.006*l**.04*100**-.3*tc**-.3*Nz**.49*Wdg**.49 #Convenience variable
    
    dfdSw = (.758*Q/Sw + Wp)
    dfdWfw = (.0035*Q/Wfw)
    dfdA = (.6*Q/A)
    dfdL = (.9*Q*np.sin(L)/np.cos(L))
    dfdq = (.006*Q/q)
    dfdl = (.04*Q/l)
    dfdtc = (-.3*Q/tc)
    dfdNz = (.49*Q/Nz)
    dfdWdg = (.49*Q/Wdg)
    dfdWp = (Sw)
        
    return np.asarray([dfdSw, dfdWfw, dfdA, dfdL, dfdq, dfdl, dfdtc, dfdNz, dfdWdg, dfdWp])

##############################################################################################################
# Construction of the active subspace
##############################################################################################################

"""
Defining of the helper functions and parameters
"""

# Define parameters for the active subspace method
M = 1000    # Number of points to sample
m = 10      # Dimensions of the parameter space

# Sample the input space
samples = Sampling(M, m, 5)
samples.set_domainBounds(np.array([[150, 200], [220, 300], [6,10], [-10, 10], [16, 45], [.5, 1], [.08, .18], [2.5, 6], [1700, 2500], [.025, .08]]))
samples.random_uniform()

func = Analytical(m, wing, wing_grad)

asfenicsx = ASFEniCSx(m, func, samples)
asfenicsx.estimation()
asfenicsx.bootstrap(100)

asfenicsx.plot_eigenvalues(os.path.join(dir,"wingWeight","analytical_eigenvalues.pdf"))
asfenicsx.plot_subspace(os.path.join(dir,"wingWeight","analytical_subspace.pdf"))
asfenicsx.plot_eigenvectors(os.path.join(dir,"wingWeight","analytical_eigenvectors.pdf"), n = 2)
asfenicsx.partition(2)
asfenicsx.plot_sufficient_summary(os.path.join(dir,"wingWeight","analytical_sufficient_summary"))