import math

from asfenicsx import clustering, functional, ASFEniCSx
import airfoilNavierStokes as ANS

# Define parameter space
m = 8               # Number of parameters
alpha = 2           # Oversampling factor
n = m + 1           # Largest dimension we can handle
k = 5               # Number of clusters 

M = int(alpha * n * math.log(m))        # Number of samples

samples = clustering(M, m, k)
samples.detect()

# Assing values to the samples in the parameter space
for i in range(1):
    try:
        value = ANS.quantity_of_interest(samples.extract(i))
    except:
        Warning("Error in the simulation. Continuing with next sample.")
    samples.assign_value(i, value)

exit()

cost = functional(m, ANS.quantity_of_interest)

cost.interpolation(samples)
cost.get_gradient_method("I")

asfenicsx = ASFEniCSx(m, cost, samples)
U, S = asfenicsx.random_sampling_algorithm()