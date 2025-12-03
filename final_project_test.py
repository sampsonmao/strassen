"""
Final Project tests
Sampson Mao
"""

import sys
import time
import unittest

import numpy as np

from final_project import *


class MatrixTestCase(unittest.TestCase):
    def test_iter(self):
        data1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        a = Matrix(data1)

        result = [x for x in a]
        answer = []
        for row in data1:
            answer += row

        self.assertEqual(answer, result)

    def setUp(self):
        self.one_by_one_lst_a = [[11]]

        self.one_by_one_lst_b = [[-4]]

        self.two_by_two_lst_a = [[5, 12], [-9, 0]]

        self.two_by_two_lst_b = [[200, 7], [31, -100]]

        self.three_by_three_lst_a = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]

        self.three_by_three_lst_b = [[10, 10, 10], [20, 20, 20], [30, 30, 30]]

        self.two_by_three_lst_a = [[0, 1, 0], [1, 0, 1]]

        self.two_by_three_lst_b = [[1, 0, 1], [0, 1, 0]]

        self.one_by_five_lst = [[1, 2, 3, 4, 5]]

        self.five_by_three_lst = [
            [1, 2, 3],
            [6, 7, 8],
            [-1, -2, -3],
            [-99, 0, -99],
            [60, -23, 100],
        ]

        self.three_by_six_lst = [
            [1, 2, 3, 0, 0, 9],
            [-20, -1, -100, 0, 7, -38],
            [6, 7, 8, 2, 200, 1],
        ]

        self.one_by_one_mat_a = Matrix(self.one_by_one_lst_a)
        self.one_by_one_mat_b = Matrix(self.one_by_one_lst_b)
        self.two_by_two_mat_a = Matrix(self.two_by_two_lst_a)
        self.two_by_two_mat_b = Matrix(self.two_by_two_lst_b)
        self.three_by_three_mat_a = Matrix(self.three_by_three_lst_a)
        self.three_by_three_mat_b = Matrix(self.three_by_three_lst_b)
        self.two_by_three_mat_a = Matrix(self.two_by_three_lst_a)
        self.two_by_three_mat_b = Matrix(self.two_by_three_lst_b)
        self.one_by_five_mat = Matrix(self.one_by_five_lst)
        self.five_by_three_mat = Matrix(self.five_by_three_lst)
        self.three_by_six_mat = Matrix(self.three_by_six_lst)

        self.one_by_one_np_a = np.array(self.one_by_one_lst_a)
        self.one_by_one_np_b = np.array(self.one_by_one_lst_b)
        self.two_by_two_np_a = np.array(self.two_by_two_lst_a)
        self.two_by_two_np_b = np.array(self.two_by_two_lst_b)
        self.three_by_three_np_a = np.array(self.three_by_three_lst_a)
        self.three_by_three_np_b = np.array(self.three_by_three_lst_b)
        self.two_by_three_np_a = np.array(self.two_by_three_lst_a)
        self.two_by_three_np_b = np.array(self.two_by_three_lst_b)
        self.one_by_five_np = np.array(self.one_by_five_lst)
        self.five_by_three_np = np.array(self.five_by_three_lst)
        self.three_by_six_np = np.array(self.three_by_six_lst)

    def test_eq(self):

        self.assertEqual(self.two_by_two_mat_a, Matrix([[5, 12], [-9, 0]]))

        self.assertEqual(self.one_by_one_mat_a, self.one_by_one_mat_a)
        self.assertEqual(self.two_by_two_mat_a, self.two_by_two_mat_a)
        self.assertEqual(self.three_by_three_mat_a, self.three_by_three_mat_a)

        self.assertNotEqual(self.one_by_one_mat_a, self.one_by_one_mat_b)
        self.assertNotEqual(self.two_by_two_mat_a, self.two_by_two_mat_b)
        self.assertNotEqual(
            self.three_by_three_mat_a, self.three_by_three_mat_b
        )

    def test_add1(self):
        c = self.one_by_one_mat_a + self.one_by_one_mat_b

        correct_c = Matrix(self.one_by_one_np_a + self.one_by_one_np_b)

        self.assertEqual(c, correct_c)

    def test_add2(self):
        c = self.two_by_two_mat_a + self.two_by_two_mat_b

        correct_c = Matrix(self.two_by_two_np_a + self.two_by_two_np_b)

        self.assertEqual(c, correct_c)

    def test_add3(self):
        c = self.three_by_three_mat_a + self.three_by_three_mat_b

        correct_c = Matrix(self.three_by_three_np_a + self.three_by_three_np_b)

        self.assertEqual(c, correct_c)

    def test_add_rect(self):
        c = self.two_by_three_mat_a + self.two_by_three_mat_b

        correct_c = Matrix(self.two_by_three_np_a + self.two_by_three_np_b)
        self.assertEqual(c, correct_c)

    def test_add_invalid(self):
        with self.assertRaises(ValueError):
            c = self.three_by_three_mat_a + self.two_by_two_mat_b

    def test_sub1(self):
        c = self.one_by_one_mat_a - self.one_by_one_mat_b

        correct_c = Matrix(self.one_by_one_np_a - self.one_by_one_np_b)

        self.assertEqual(c, correct_c)

    def test_sub2(self):
        c = self.two_by_two_mat_a - self.two_by_two_mat_b

        correct_c = Matrix(self.two_by_two_np_a - self.two_by_two_np_b)

        self.assertEqual(c, correct_c)

    def test_sub3(self):
        c = self.three_by_three_mat_a - self.three_by_three_mat_b

        correct_c = Matrix(self.three_by_three_np_a - self.three_by_three_np_b)

        self.assertEqual(c, correct_c)

    def test_sub_rect(self):
        c = self.two_by_three_mat_a - self.two_by_three_mat_b

        correct_c = Matrix(self.two_by_three_np_a - self.two_by_three_np_b)
        self.assertEqual(c, correct_c)

    def test_sub_invalid(self):
        with self.assertRaises(ValueError):
            c = self.three_by_six_mat - self.two_by_two_mat_b

    def test_mat_mul1(self):
        c = self.one_by_one_mat_a @ self.one_by_one_mat_b

        correct_c = Matrix(self.one_by_one_np_a @ self.one_by_one_np_b)

        self.assertEqual(c, correct_c)

    def test_mat_mul2(self):
        c = self.two_by_two_mat_a @ self.two_by_two_mat_b

        correct_c = Matrix(self.two_by_two_np_a @ self.two_by_two_np_b)

        self.assertEqual(c, correct_c)

    def test_mat_mul3(self):
        c = self.three_by_three_mat_a @ self.three_by_three_mat_b

        correct_c = Matrix(self.three_by_three_np_a @ self.three_by_three_np_b)

        self.assertEqual(c, correct_c)

    def test_get_item(self):

        self.assertEqual(self.three_by_three_mat_a[1, 1], 2)
        self.assertEqual(self.three_by_three_mat_a[0], [1, 1, 1])

        with self.assertRaises(IndexError):
            x = self.three_by_three_mat_a[5]

        with self.assertRaises(IndexError):
            x = self.three_by_three_mat_a[1:]

    def test_pad(self):
        padded = self.one_by_one_mat_a.pad_to_next_pow_two(2)

        self.assertEqual(padded.nrows, 2)
        self.assertEqual(padded.ncols, 2)

        padded = self.one_by_one_mat_a.pad_to_next_pow_two(5)

        self.assertEqual(padded.nrows, 8)
        self.assertEqual(padded.ncols, 8)

    def test_partition(self):
        u, v, w, x = self.two_by_two_mat_a.partition_sq_matrix()

        self.assertEqual(u, Matrix([[self.two_by_two_mat_a[0, 0]]]))
        self.assertEqual(v, Matrix([[self.two_by_two_mat_a[0, 1]]]))
        self.assertEqual(w, Matrix([[self.two_by_two_mat_a[1, 0]]]))
        self.assertEqual(x, Matrix([[self.two_by_two_mat_a[1, 1]]]))

        four_by_four_mat = self.three_by_three_mat_a.pad_to_next_pow_two(4)
        u, v, w, x = four_by_four_mat.partition_sq_matrix()

        u_correct = Matrix([[1, 1], [2, 2]])
        v_correct = Matrix([[1, 0], [2, 0]])
        w_correct = Matrix([[3, 3], [0, 0]])
        x_correct = Matrix([[3, 0], [0, 0]])

        self.assertEqual(u, u_correct)
        self.assertEqual(v, v_correct)
        self.assertEqual(w, w_correct)
        self.assertEqual(x, x_correct)

    def test_rectangular1(self):
        # Tests non-power of 2 and rectangular
        c = self.one_by_five_mat @ self.five_by_three_mat

        correct_c = Matrix(self.one_by_five_np @ self.five_by_three_np)

        self.assertEqual(c, correct_c)

    def test_rectangular2(self):
        c = self.five_by_three_mat @ self.three_by_six_mat

        correct_c = Matrix(self.five_by_three_np @ self.three_by_six_np)

        self.assertEqual(c, correct_c)

    def test_rectangular_invalid(self):
        with self.assertRaises(ValueError):
            c = self.three_by_six_mat @ self.five_by_three_mat

    @unittest.skip("Takes a long time for matrices >=512")
    def test_div_conq_times(self):
        """
        Divide and conquer: size=2, duration=3.840000135824084e-05
        Divide and conquer: size=4, duration=0.00018869999985327013
        Divide and conquer: size=8, duration=0.0007310000000870787
        Divide and conquer: size=16, duration=0.0059089999995194376
        Divide and conquer: size=32, duration=0.04870429999937187
        Divide and conquer: size=64, duration=0.41397539999888977
        Divide and conquer: size=128, duration=3.2516961000001174
        Divide and conquer: size=256, duration=26.267951199999516
        Divide and conquer: size=512, duration=208.57386960000076
        Divide and conquer: size=1024, duration=1682.1576343000015
        """
        sys.setrecursionlimit(10**6)
        n = 2
        while n <= 1024:
            a = (np.random.rand(n, n) * 1000).tolist()
            b = (np.random.rand(n, n) * 1000).tolist()

            a = Matrix(a)
            b = Matrix(b)

            start = time.perf_counter()
            c = a @ b
            duration = time.perf_counter() - start
            print(f"Divide and conquer: size={n}, duration={duration}")
            n *= 2


class StrassenMatrixTestCase(MatrixTestCase):

    @unittest.skip("Takes a long time for matrices >=512")
    def test_strassen(self):
        """
        Strassen: size=2, duration=5.210000017541461e-05
        Strassen: size=4, duration=0.00019740000061574392
        Strassen: size=8, duration=0.0013484000010066666
        Strassen: size=16, duration=0.010044800001196563
        Strassen: size=32, duration=0.0698469999988447
        Strassen: size=64, duration=0.5107070000012754
        Strassen: size=128, duration=3.5746930000022985
        Strassen: size=256, duration=25.69689269999799
        Strassen: size=512, duration=180.04096420000133
        """
        sys.setrecursionlimit(10**6)
        n = 2
        while n <= 1024:
            a = (np.random.rand(n, n) * 1000).tolist()
            b = (np.random.rand(n, n) * 1000).tolist()

            a = StrassenMatrix(a)
            b = StrassenMatrix(b)

            start = time.perf_counter()
            c = a @ b
            duration = time.perf_counter() - start
            print(f"Strassen: size={n}, duration={duration}")
            n *= 2
