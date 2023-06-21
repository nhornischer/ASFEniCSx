import unittest
import numpy as np
from ASFEniCSx.utils import normalizer, denormalizer, load
from ASFEniCSx.sampling import Sampling, Clustering


class UtilsTest(unittest.TestCase):
    
    def test_normalizer(self):
        a_original = np.random.uniform(-1.3, 2., (10, 3))
        bounds = np.vstack([np.array([-1.3, 2.0]), np.array([-1.3, 2.]), np.array([-1.3, 2.])])
        interval = np.array([-0.5, 0.5])
        normalized_a = normalizer(a_original, bounds, interval)

        self.assertTrue(np.all(normalized_a >= interval[0]))
        self.assertTrue(np.all(normalized_a <= interval[1]))

    def test_denormalizer(self):
        normalized_a = np.random.uniform(-0.5,0.5, (10, 3))
        bounds = np.vstack([np.array([-1.3, 2.0]), np.array([-1.3, 2.]), np.array([-1.3, 2.])])
        interval = np.array([-0.5, 0.5])
        denormalized_a = denormalizer(normalized_a, bounds, interval)

        self.assertTrue(np.all(denormalized_a >= bounds[:, 0]))
        self.assertTrue(np.all(denormalized_a <= bounds[:, 1]))

    def test_normalizer_denormalizer(self):
        a_original = np.random.uniform(-1.3, 2., (10, 3))
        bounds = np.vstack([np.array([-1.3, 2.0]), np.array([-1.3, 2.]), np.array([-1.3, 2.])])
        interval = np.array([-0.5, 0.5])
        normalized_a = normalizer(a_original, bounds, interval)

        self.assertTrue(np.max(np.abs(a_original - denormalizer(normalized_a, bounds, interval))) < 1e-15)


class SamplingTest(unittest.TestCase):
    def test_normalized_random_uniform(self):
        samples = Sampling(100, 10)
        samples.random_uniform()

        self.assertEqual(np.shape(samples._array), (100, 10))
        self.assertTrue(np.all(samples._array >= -1.0))
        self.assertTrue(np.all(samples._array <= 1.0))

    def test_random_uniform_with_equal_bounds(self):
        samples = Sampling(100, 10)
        bounds = np.vstack([np.array([-1.3, 2.0])] * 10)
        samples.set_domainBounds(bounds)
        samples.random_uniform()

        self.assertEqual(np.shape(samples._array), (100, 10))
        self.assertTrue(np.all(samples._array >= bounds[:, 0]))
        self.assertTrue(np.all(samples._array <= bounds[:, 1]))
        # Should be statistically the case, but not guaranteed
        self.assertTrue(np.max(samples._array) >= 1.0)
        self.assertTrue(np.min(samples._array) <= -1.0)

    def test_random_uniform_with_different_bounds(self):
        samples = Sampling(100, 10)
        bounds = np.zeros((10, 2))
        bounds[:,0] = np.random.uniform(-10,-1, 10)
        bounds[:,1] = np.random.uniform(1,10, 10)
        samples.set_domainBounds(bounds)
        samples.random_uniform()
        
        self.assertTrue(np.all(samples._array >= bounds[:, 0]))
        self.assertTrue(np.all(samples._array <= bounds[:, 1]))
        # Should be statistically the case, but not guaranteed
        self.assertTrue(np.max(samples._array) >= 1.0)
        self.assertTrue(np.min(samples._array) <= -1.0)

    def test_sampling_extract_and_index(self):
        samples = Sampling(100, 10)
        samples.random_uniform()
        test_values = np.random.uniform(-10.0, 10.0, 10)
        samples._array[3,:] = np.copy(test_values)

        self.assertTrue(np.all(samples.extract(3) == test_values))
        self.assertTrue(samples.index(test_values) == 3)

    def test_value_assignment(self):
        samples = Sampling(100, 10)
        samples.random_uniform()
        f = lambda x: np.sum(x)
        samples.assign_values(f)

        f_true = np.zeros(100)
        for i in range(100):
            f_true[i] = f(samples.extract(i))
        self.assertTrue(np.all(samples.values() == f_true))

    def test_saving_and_loading(self):
        samples = Sampling(100, 10)
        samples.random_uniform()
        f = lambda x: np.sum(x)
        samples.assign_values(f)

        samples.save('test_samples')
        samples_loaded = load('test_samples')

        self.assertTrue(np.all(samples._array == samples_loaded._array))
        self.assertTrue(np.all(samples.values() == samples_loaded.values()))

class ClusteringTest(unittest.TestCase):
    def setUp(self) -> None:
        m = 2
        centroids = np.asarray([[-5, -5],
                                [-5, 5],
                                [5, -5],
                                [5, 5]])
        data = []
        for i in range(np.shape(centroids)[0]):
            data.append(np.random.uniform(-4, 4, (20, m)) + centroids[i,:])
        
        self.data = np.concatenate(data)
        self.centroids = np.copy(centroids)

    def test_kmeans(self):
        fails = 0
        successes = 0
        while (successes == 0 or fails > successes) and fails + successes < 100 :
            clustering = Clustering(np.shape(self.data)[0], np.shape(self.data)[1], np.shape(self.centroids)[0])
            clustering._array = np.copy(self.data)
            clustering.detect()

            # Compare the true centroids with the clustering._centroids by 
            # finding the closest centroid to each true centroid
            sorted_centroids = np.copy(clustering._centroids)
            for i in range(clustering.k):
                sorted_centroids[i,:] = clustering._centroids[np.argmin(np.linalg.norm(clustering._centroids - self.centroids[i,:], axis=1)),:]
            
            if np.all(np.abs(sorted_centroids - self.centroids) < 1):
                successes += 1
            else:
                fails += 1
            
        

        self.assertTrue(successes > 0 and fails < successes and fails + successes < 100)

if __name__ == '__main__':
    unittest.main()