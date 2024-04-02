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

    def test_extract_hog(self):
        array = np.zeros((20, 20), dtype=np.uint8)

        # Alternate between 0 and 255 every 4 spaces
        array[::2, ::4] = 255
        array[1::2, 1::4] = 255
        print(array)

        p2.extract_HOG()




if __name__ == "__main__":
    unittest.main()
