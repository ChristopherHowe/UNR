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

    def test_unpad_image(self):
        arr_grey = np.zeros((5, 5))
        arr_col = np.zeros((5, 5, 3))
        self.assertEqual(p2.unpad_img(arr_grey, ((1, 1), (1, 1))).shape, (3, 3))
        self.assertEqual(p2.unpad_img(arr_col, ((1, 1), (1, 1), (0, 0))).shape, (3, 3, 3))

    def test_get_binary_parrtern(self):
        test1 = np.array([[5, 5, 5], [5, 1, 5], [5, 5, 5]])
        self.assertEqual(p2.get_binary_pattern_as_base_10(test1, 1, 1), 255)
        test2 = np.array([[0, 5, 5], [5, 1, 5], [5, 5, 5]])
        self.assertEqual(p2.get_binary_pattern_as_base_10(test2, 1, 1), 127)
        test3 = np.array([[0, 0, 5], [5, 1, 5], [5, 0, 5]])
        self.assertEqual(p2.get_binary_pattern_as_base_10(test3, 1, 1), 61)


if __name__ == "__main__":
    unittest.main()
