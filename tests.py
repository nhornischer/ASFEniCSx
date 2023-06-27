import unittest
import os
import numpy as np
from ASFEniCSx.utils import normalizer, denormalizer, load
from ASFEniCSx.sampling import Sampling, Clustering
from ASFEniCSx.functional import Interpolation, Regression

dir = os.path.dirname(__file__)

if not os.path.exists(os.path.join(dir, 'test_data')):
    os.makedirs(os.path.join(dir, 'test_data'))

os.chdir(os.path.join(dir, 'test_data'))
dir = os.path.join(dir, 'test_data')

class UtilsTest(unittest.TestCase):
    """
    Test the utils module
    """
    
    def test_normalizer(self):
        # Test the normalizer function by checking if the normalized values are within the interval
        a_original = np.random.uniform(-1.3, 2., (10, 3))
        bounds = np.vstack([np.array([-1.3, 2.0]), np.array([-1.3, 2.]), np.array([-1.3, 2.])])
        interval = np.array([-0.5, 0.5])
        normalized_a = normalizer(a_original, bounds, interval)

        self.assertTrue(np.all(normalized_a >= interval[0]))
        self.assertTrue(np.all(normalized_a <= interval[1]))

        # Test the normalizer with a singel sample
        a_original = np.array([-1.0, 1.0, -1.0])
        bounds = np.vstack([np.array([-1.0, 2.0]), np.array([-2.0, 2.]), np.array([-4.0, 0.0])])
        interval = np.array([-1.0, 1.0])
        normalized_a = normalizer(a_original, bounds, interval)

        self.assertTrue(np.all(normalized_a == np.array([-1.0, 0.5, 0.5])))

        # Test the normalizer function by checking if a value is mapped to the correct normalized value
        a_original = np.array([[-1.0, 1.0, -1.0], [0.5, 2., -0.5]])
        bounds = np.vstack([np.array([-1.0, 2.0]), np.array([-2.0, 2.]), np.array([-4.0, 0.0])])
        interval = np.array([-1.0, 1.0])
        normalized_a = normalizer(a_original, bounds, interval)

        self.assertTrue(np.all(normalized_a == np.array([[-1.0, 0.5, 0.5], [0.0, 1.0, 0.75]])))

    def test_denormalizer(self):
        # Test the denormalizer function by checking if the denormalized values are within the bounds
        normalized_a = np.random.uniform(-0.5,0.5, (10, 3))
        bounds = np.vstack([np.array([-1.3, 2.0]), np.array([-1.3, 2.]), np.array([-1.3, 2.])])
        interval = np.array([-0.5, 0.5])
        denormalized_a = denormalizer(normalized_a, bounds, interval)

        self.assertTrue(np.all(denormalized_a >= bounds[:, 0]))
        self.assertTrue(np.all(denormalized_a <= bounds[:, 1]))

        # Test the denormalizer with a singel sample
        normalized_a = np.array([-1.0, 0.5, 0.5])
        bounds = np.vstack([np.array([-1.0, 2.0]), np.array([-2.0, 2.]), np.array([-4.0, 0.0])])
        interval = np.array([-1.0, 1.0])
        denormalized_a = denormalizer(normalized_a, bounds, interval)

        self.assertTrue(np.all(denormalized_a == np.array([-1.0, 1.0, -1.0])))

        # Test the denormalizer function by checking if a value is mapped to the correct denormalized value
        normalized_a = np.array([[-1.0, 0.5, 0.5], [0.0, 1.0, 0.75]])
        bounds = np.vstack([np.array([-1.0, 2.0]), np.array([-2.0, 2.]), np.array([-4.0, 0.0])])
        interval = np.array([-1.0, 1.0])
        denormalized_a = denormalizer(normalized_a, bounds, interval)

        self.assertTrue(np.all(denormalized_a == np.array([[-1.0, 1.0, -1.0], [0.5, 2., -0.5]])))

    def test_load(self):
        # Test storing and loading of a sampling object
        samples = Sampling(100, 10)
        samples.random_uniform()
        samples.save('test_sampling')
        loaded_samples = load('test_sampling')

        self.assertTrue(np.all(samples._array == loaded_samples._array))
        self.assertTrue(loaded_samples._object_type == 'Sampling')

        # Test storing and loading of a clustering object
        samples = Clustering(100, 10, 5)
        samples.random_uniform()
        samples.detect()
        samples.save('test_clustering')
        loaded_samples = load('test_clustering')

        self.assertTrue(np.all(samples._array == loaded_samples._array))
        self.assertTrue(loaded_samples._object_type == 'Clustering')

class SamplingTest(unittest.TestCase):
    """
    Test the sampling module
    """
    def test_shape_random_uniform(self):
        # Testing if the shape of a random uniform sampling is correct
        samples = Sampling(100, 10)
        samples.random_uniform()

        self.assertEqual(np.shape(samples._array), (100, 10))

    def test_normalized_random_uniform(self):
        # Testing if the random uniform sampling is normalized
        samples = Sampling(100, 10)
        samples.random_uniform()

        self.assertTrue(np.all(samples._array >= -1.0))
        self.assertTrue(np.all(samples._array <= 1.0))

    def test_random_uniform_with_equal_bounds(self):
        # Testing if the random uniform sampling is within the bounds
        samples = Sampling(100, 10)
        bounds = np.vstack([np.array([-0.5, 0.5])] * 10)
        samples.set_domainBounds(bounds)
        samples.random_uniform()

        self.assertTrue(np.all(samples._array >= bounds[:, 0]))
        self.assertTrue(np.all(samples._array <= bounds[:, 1]))

    def test_random_uniform_with_different_bounds(self):
        # Testing if the random uniform sampling is within the bounds
        samples = Sampling(100, 10)
        bounds = np.zeros((10, 2))
        bounds[:,0] = np.random.uniform(-1,-0.01, 10)
        bounds[:,1] = np.random.uniform(0.01,1, 10)
        samples.set_domainBounds(bounds)
        samples.random_uniform()
        
        self.assertTrue(np.all(samples._array >= bounds[:, 0]))
        self.assertTrue(np.all(samples._array <= bounds[:, 1]))

    def test_shape_standard_gaussian(self):
        # Testing if the shape of a standard gaussian sampling is correct
        samples = Sampling(100, 10)
        samples.standard_gaussian()

        self.assertEqual(np.shape(samples._array), (100, 10))
    
    def test_normalized_standard_gaussian(self):
        # Testing if the standard gaussian sampling is normalized to the domain boundaries
        samples = Sampling(100, 10)
        samples.set_domainBounds(np.vstack([np.array([-1.0, 1.0])] * 10))
        samples.standard_gaussian()

        self.assertTrue(np.all(samples._array >= -1.0))
        self.assertTrue(np.all(samples._array <= 1.0))

    def test_sampling_extract(self):
        # Testing if the extract and index functions work correctly
        samples = Sampling(100, 10)
        samples.random_uniform()
        test_values = np.random.uniform(-10.0, 10.0, 10)
        samples._array[3,:] = np.copy(test_values)

        self.assertTrue(np.all(samples.extract(3) == test_values))

    def test_value_assignment(self):
        # Testing if the value assignment works correctly
        samples = Sampling(100, 10)
        samples.random_uniform()
        f = lambda x: np.sum(x)
        samples.assign_values(f)

        f_true = np.zeros(100)
        for i in range(100):
            f_true[i] = f(samples.extract(i))
        self.assertTrue(np.all(samples.values() == f_true))

class ClusteringTest(unittest.TestCase):
    def setUp(self):
        self.m = 3
        self.M_per_cluster = 100
        centroids = np.asarray([[-5, -5, -5],
                                [5, 5, 5],
                                [5, -5, 5],
                                [-5, 5, -5],
                                [-5, -5, 5],
                                [5, 5, -5],
                                [-5, 5, 5],
                                [5, -5, -5]], dtype=np.float64)
        data = []
        for i in range(np.shape(centroids)[0]):
            cluster_data = np.random.uniform(-3, 3, (self.M_per_cluster, self.m)) + centroids[i,:]
            centroids[i,:] = np.mean(cluster_data, axis=0)
            data.append(cluster_data)
        
        self.data = np.concatenate(data)
        self.centroids = centroids

        self.clusters = np.reshape(np.arange(self.M_per_cluster*np.shape(centroids)[0]), (np.shape(centroids)[0],self.M_per_cluster))
        self.real_indices = np.asarray([0, 1, 2, 3, 4, 5, 6, 7]*self.M_per_cluster).reshape((self.M_per_cluster,8)).T.reshape((self.M_per_cluster*8,))

    def test_cluster_index(self):
        # Testing if the index of the cluster is returned correctly
        samples = Clustering(np.shape(self.data)[0], self.m, np.shape(self.centroids)[0])
        samples.set_domainBounds(np.vstack([np.array([-10.0, 10.0])] * self.m))
        samples._array = np.copy(self.data)
        samples._centroids = np.copy(self.centroids)
        samples._clusters = np.copy(self.clusters)

        test_indices = np.zeros(np.shape(self.data)[0])
        for i in range(np.shape(self.data)[0]):
            test_indices[i] = samples.cluster_index(self.data[i,:])

        self.assertTrue(np.all(test_indices == self.real_indices))

    def test_assign_clusters(self):
        # Testing if the clustering assigns the correct clusters
        samples = Clustering(np.shape(self.data)[0], self.m, np.shape(self.centroids)[0])
        samples.set_domainBounds(np.vstack([np.array([-10.0, 10.0])] * self.m))
        samples._array = np.copy(self.data)
        samples._centroids = np.copy(self.centroids)
        samples._clusters = samples._assign_clusters(samples.samples())

        self.assertTrue(np.all(np.asarray(samples._clusters) == self.clusters))

    def test_update_centroids(self):
        # Testing if the centroids are updated correctly
        # Nothing should be updated in this case
        samples = Clustering(np.shape(self.data)[0], self.m, np.shape(self.centroids)[0])
        samples.set_domainBounds(np.vstack([np.array([-10.0, 10.0])] * self.m))
        samples._array = np.copy(self.data)
        samples._centroids = np.copy(self.centroids)
        samples._clusters = np.copy(self.clusters)
        samples._update_centroids(samples._clusters)

        self.assertTrue(np.all(np.asarray(samples._centroids) == self.centroids))

    def test_detect_clusters(self):
        # Testing if the clustering detects the correct clusters
        samples = Clustering(np.shape(self.data)[0], self.m, np.shape(self.centroids)[0])
        samples.set_domainBounds(np.vstack([np.array([-10.0, 10.0])] * self.m))
        samples._array = np.copy(self.data)
        # Giving guesses for the centroids to see if updating them works corretctly
        samples.detect(self.centroids+np.random.uniform(-1.0,1.0,np.shape(self.centroids)))

        self.assertTrue(np.all(np.asarray(samples._clusters) == self.clusters))
        self.assertTrue(np.all(np.asarray(samples._centroids) == self.centroids))

class NonuniformClusteringTest(unittest.TestCase):
    def setUp(self):
        self.m = 2
        self.M_per_cluster = 100
        centroids = np.asarray([[-5, -100],
                                [5, 100],
                                [5, -100],
                                [-5, 100]], dtype=np.float64)
        data = []
        for i in range(np.shape(centroids)[0]):
            cluster_data = np.asarray([np.random.uniform(-4, 4, (self.M_per_cluster)) + centroids[i,0], np.random.uniform(-90, 90, (self.M_per_cluster)) + centroids[i,1]]).T
            centroids[i,:] = np.mean(cluster_data, axis=0)
            data.append(cluster_data)
        
        self.data = np.concatenate(data)
        self.centroids = centroids
        self.clusters = np.reshape(np.arange(self.M_per_cluster*np.shape(centroids)[0]), (np.shape(centroids)[0],self.M_per_cluster))
        self.real_indices = np.asarray([0, 1, 2, 3]*self.M_per_cluster).reshape((self.M_per_cluster,4)).T.reshape((self.M_per_cluster*4,))

    def test_nonuniform_cluster_index(self):
        # Testing if the index of the cluster is returned correctly
        samples = Clustering(np.shape(self.data)[0], self.m, np.shape(self.centroids)[0])
        samples.set_domainBounds(np.array([[-10.0, 10.0], [-200.0, 200.0]]))
        samples._array = np.copy(self.data)
        samples._centroids = np.copy(self.centroids)
        samples._clusters = np.copy(self.clusters)

        test_indices = np.zeros(np.shape(self.data)[0])
        for i in range(np.shape(self.data)[0]):
            test_indices[i] = samples.cluster_index(self.data[i,:])
        self.assertTrue(np.all(test_indices == self.real_indices))
        
    def test_nonuniform_assign_clusters(self):
        # Testing if the clustering assigns the correct clusters
        samples = Clustering(np.shape(self.data)[0], self.m, np.shape(self.centroids)[0])
        samples.set_domainBounds(np.array([[-10.0, 10.0], [-200.0, 200.0]]))
        samples._array = np.copy(self.data)
        samples._centroids = np.copy(self.centroids)
        samples._clusters = samples._assign_clusters(samples.samples())

        self.assertTrue(np.all(np.asarray(samples._clusters) == self.clusters))

    def test_nonuniform_update_centroids(self):
        # Testing if the centroids are updated correctly
        # Nothing should be updated in this case
        samples = Clustering(np.shape(self.data)[0], self.m, np.shape(self.centroids)[0])
        samples.set_domainBounds(np.array([[-10.0, 10.0], [-200.0, 200.0]]))
        samples._array = np.copy(self.data)
        samples._centroids = np.copy(self.centroids)
        samples._clusters = np.copy(self.clusters)
        samples._update_centroids(samples._clusters)

        self.assertTrue(np.allclose(np.asarray(samples._centroids),self.centroids))

    def test_nonuniform_detect_clusters(self):
        # Testing if the clustering detects the correct clusters
        samples = Clustering(np.shape(self.data)[0], self.m, np.shape(self.centroids)[0])
        samples.set_domainBounds(np.array([[-10.0, 10.0], [-200.0, 200.0]]))
        samples._array = np.copy(self.data)
        samples._clusters = self.clusters
        samples._centroids = self.centroids

        samples.detect(self.centroids+np.random.uniform(-1.0,1.0,np.shape(self.centroids)))

        self.assertTrue(np.all(np.asarray(samples._clusters) == self.clusters))
        self.assertTrue(np.allclose(np.asarray(samples._centroids), self.centroids))

class InterpolationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.linear_function = lambda x: np.sum(x)
        self.quadratic_function = lambda x: np.sum(x**2)

    def test_linear_interpolation(self):
        samples = Sampling(100, 10)
        samples.random_uniform()
        samples.assign_values(self.linear_function)

        interpolant = Interpolation(10, self.linear_function, samples)
        interpolant.interpolate(order = 1, use_clustering = False)
        self.assertTrue(np.isclose(samples.values(), interpolant.approximate(samples.samples())).all())

    def test_quadratic_interpolation(self):
        samples = Sampling(100, 10)
        samples.random_uniform()
        samples.assign_values(self.quadratic_function)

        interpolant = Interpolation(10, self.quadratic_function, samples)
        interpolant.interpolate(order = 2, use_clustering = False)
        self.assertTrue(np.isclose(samples.values(), interpolant.approximate(samples.samples())).all())

    def test_linear_interpolation_with_clustering(self):
        samples = Clustering(100, 10, 5)
        samples.random_uniform()
        samples.assign_values(self.linear_function)

        interpolant = Interpolation(10, self.linear_function, samples)
        interpolant.interpolate(order = 1, use_clustering = True)
        self.assertTrue(np.isclose(samples.values(), interpolant.approximate(samples.samples())).all())

    def test_quadratic_interpolation_with_clustering(self):
        samples = Clustering(100, 10, 5)
        samples.random_uniform()
        samples.assign_values(self.quadratic_function)

        interpolant = Interpolation(10, self.quadratic_function, samples)
        interpolant.interpolate(order = 2, use_clustering = True)
        self.assertTrue(np.isclose(samples.values(), interpolant.approximate(samples.samples())).all())

class RegressionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.linear_function = lambda x: np.sum(x)
        self.quadratic_function = lambda x: np.sum(x**2)

    def test_linear_regression(self):
        samples = Sampling(100, 10)
        samples.random_uniform()
        samples.assign_values(self.linear_function)

        regression = Regression(10, self.linear_function, samples)
        regression.regression(order = 1, use_clustering = False)
        self.assertTrue(np.isclose(samples.values(), regression.approximate(samples.samples())).all())

    def test_quadratic_regression(self):
        samples = Sampling(100, 10)
        samples.random_uniform()
        samples.assign_values(self.quadratic_function)

        regression = Regression(10, self.quadratic_function, samples)
        regression.regression(order = 2, use_clustering = False)
        self.assertTrue(np.isclose(samples.values(), regression.approximate(samples.samples())).all())

    def test_linear_regression_with_clustering(self):
        samples = Clustering(100, 10, 5)
        samples.random_uniform()
        samples.assign_values(self.linear_function)

        regression = Regression(10, self.linear_function, samples)
        regression.regression(order = 1, use_clustering = True)
        self.assertTrue(np.isclose(samples.values(), regression.approximate(samples.samples())).all())

    def test_quadratic_regression_with_clustering(self):
        samples = Clustering(100, 10, 5)
        samples.random_uniform()
        samples.assign_values(self.quadratic_function)

        regression = Regression(10, self.quadratic_function, samples)
        regression.regression(order = 2, use_clustering = True)
        self.assertTrue(np.isclose(samples.values(), regression.approximate(samples.samples())).all())

if __name__ == '__main__':
    unittest.main()