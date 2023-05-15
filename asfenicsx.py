import numpy as np
import scipy

class ASFEniCSx:

    def __init__(self, n, cost, samples):  
        self.n = n
        self.cost = cost
        self.samples = samples

    def covariance(self):
        covariance = np.zeros([self.cost.m, self.cost.m])
        import tqdm.autonotebook
        progress = tqdm.autonotebook.tqdm(desc="Approximating Covariance Matrix", total=self.samples.M)
        for i in range(self.samples.M):
            grad= self.cost.gradient(self.samples.samples[i])
            covariance += np.outer(grad, grad)
            progress.update(1)
        covariance = covariance / self.samples.M
        return covariance
    
    def eigendecomposition(self, matrix):
        U, S, _ = scipy.linalg.svd(matrix, full_matrices=True)
        return U, S**2
    
    def random_sampling_algorithm(self):
        covariance = self.covariance()
        U, S = self.eigendecomposition(covariance)
        return (U[:,0:self.n], S[0:self.n]**2)

class Sampling:
    def __init__(self, M, m, func = None):
        self.M = M
        self.m = m
        self.samples = self.random_uniform()

    def random_uniform(self):
        return np.random.uniform(-1, 1, (self.M,self.m))
    
    def extract_sample(self, index):
        return self.samples[index,:]
    
class Clustering:
    def __init__(self, k=5, max_iter=1000):
        self.k = k
        self.max_iter = max_iter
    
    def detect(self, data):
        min_, max_ = np.min(data), np.max(data)
        self.centroids = np.random.uniform(min_, max_, (self.k, data.shape[1]))
        prev_centroids_=None
        iter_=0
        while np.not_equal(self.centroids, prev_centroids_).any() and iter_ < self.max_iter:
            prev_centroids_ = self.centroids.copy()
            self.clusters = self.assign_clusters(data)
            self.centroids = self.update_centroids(prev_centroids_)
            iter_ += 1
        return self.clusters, self.centroids
    
    def assign_clusters(self, data):
        clusters=[[] for _ in range(self.k)]
        for x in data:
            distances = np.linalg.norm(self.centroids-x, axis=1)
            cluster = np.argmin(distances)
            clusters[cluster].append(x)
        for i, cluster in enumerate(clusters):
            clusters[i]= np.asarray(cluster)
        return clusters

    def update_centroids(self, prev_centroids):
        centroids = [np.mean(cluster, axis=0) for cluster in self.clusters]
        for i, centroid in enumerate(self.centroids):
            if np.isnan(centroid).any():
                centroids[i] = prev_centroids[i]
        return np.asarray(centroids)
    
    def evaluate(self, data):
        centroids = []
        centroid_idxs = []
        for x in data:
            dists = np.linalg.norm(x, self.centroids)
            centroid_idx = np.argmin(dists)
            centroids.append(self.centroids[centroid_idx])
            centroid_idxs.append(centroid_idx)
        return centroids, centroid_idx
    

class Functional:
    def __init__(self, m, f):
        assert(m>0, "m must be positive")
        self.m = m         # dimension of parameter space
        self.f = f
    
    def interpolation(self, order=1):
        
        pass

    def finite_differences(self, x, h=1e-6):
        assert(len(x)==self.m, "x must have dimension m")
        assert(h>0, "h must be positive")

        # f(x)
        f_0= self.f(x)
        dfdx = np.zeros(self.m)
        for i in range(self.m):
            #f(x+e_ih)
            x[i] += h
            f_1 = self.f(x)
            
            #f(x-e_ih)
            x[i] -= 2*h
            f_2 = self.f(x)

            # Second-order central finite differences (ZFD)
            dfdx[i] = (f_1-f_2) / (2*h)

        return dfdx



