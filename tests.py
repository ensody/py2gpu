from py2gpu.api import blockwise, IntArray, FloatArray, compile_gpu_code
from numpy import array, hstack, vstack
from unittest import TestCase

@blockwise({('x', 'y'): (2, 2)}, {('x', 'y'): IntArray}, overlapping=False)
def f(x, y):
    x[0, 0] = 1 * y[0, 0]
    x[0, 1] = 2 * y[0, 1]
    x[1, 0] = 3 * y[1, 0]
    x[1, 1] = 4 * y[1, 1]

compile_gpu_code()

class GPUTest(TestCase):
    def test_increasing_multiplication(self):
        x = array([[1, 2, 5, 6],
                   [3, 4, 7, 8],
                   [1, 1, 2, 2],
                   [1, 1, 2, 2]])
        multiplier = array([[1, 2],
                            [3, 4]])
        multiplier = hstack([multiplier, multiplier])
        multiplier = vstack([multiplier, multiplier])
        y = x.copy()
        f(x, y)
        self.assertTrue((x == y * multiplier).all())
