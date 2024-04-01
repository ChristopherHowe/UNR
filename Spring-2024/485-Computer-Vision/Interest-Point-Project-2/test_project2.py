import unittest
import project2 as p2
import numpy as np


class TestStringMethods(unittest.TestCase):
    def test_upper(self):
        self.assertEqual("foo".upper(), "FOO")

    def test_safe_slice(self):
        arr = np.zeros((5, 5))
        self.assertEqual(p2.safe_slice(arr, 2, 4, 2, 4).shape, (2, 2))
        self.assertEqual(p2.safe_slice(arr, -1, 4, -1, 4).shape, (5, 5))
        self.assertEqual(p2.safe_slice(arr, 2, 4, -1, 4).shape, (2, 5))
        self.assertEqual(p2.safe_slice(arr, -1, 4, 2, 4).shape, (5, 2))
        self.assertEqual(p2.safe_slice(arr, 2, 5, 2, 5).shape, (3, 3))
        self.assertEqual(p2.safe_slice(arr, 2, 5, 2, 4).shape, (3, 2))
        self.assertEqual(p2.safe_slice(arr, 2, 4, 2, 5).shape, (2, 3))
        self.assertEqual(p2.safe_slice(arr, -1, 4, 2, 5).shape, (5, 3))
        self.assertEqual(p2.safe_slice(arr, 2, 5, -1, 4).shape, (3, 5))


if __name__ == "__main__":
    unittest.main()
