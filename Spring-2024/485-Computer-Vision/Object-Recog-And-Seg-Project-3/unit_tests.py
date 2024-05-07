import unittest
import project3 as p3
import numpy as np

class TestVocabularyGeneration(unittest.TestCase):
    def test_k_means(self):
        points = np.array([[1, 2], [2, 1], [3, 4], [5, 6]])
        centers = p3.k_means(points, 2)
        self.assertEqual(centers, [[-1, -1], [-1, -1]])


# Define the main function to execute the tests
if __name__ == "__main__":
    unittest.main()
