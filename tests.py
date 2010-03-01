from py2gpu.api import blockwise, Int32Array, FloatArray, compile_gpu_code
from numpy import array, hstack, vstack, zeros, float32
from numpy.random import rand
from scipy.ndimage import sobel, convolve, label
from unittest import TestCase

@blockwise({('x', 'y'): (2, 2)}, {('x', 'y'): Int32Array}, overlapping=False)
def increasing_multiplication(x, y):
    x[0, 0] = 1 * y[0, 0]
    x[0, 1] = 2 * y[0, 1]
    x[1, 0] = 3 * y[1, 0]
    x[1, 1] = 4 * y[1, 1]

@blockwise({'x': (1, 1), 'y': (3, 3)}, {('x', 'y'): FloatArray}, overlapping=True)
def reducer(x, y):
    x[0, 0] = y[0, 1] + y[1, 2] + y[2, 1] + y[1, 0]


@blockwise({'x': (1, 1), 'y': (3, 3)}, {('x', 'y'): FloatArray},
    overlapping=True,  center_on_origin=True)
def center_reducer(x, y):
    x[0, 0] = 0
    if x.offset[0]:
        x[0, 0] += y[0, 1]
    if x.offset[1] < x.shape[1] - 1:
        x[0, 0] += y[1, 2]
    if x.offset[0] < x.shape[0] - 1:
        x[0, 0] += y[2, 1]
    if x.offset[1]:
        x[0, 0] += y[1, 0]

compile_gpu_code()

class GPUTest(TestCase):
    def assertEqualArrays(self, x, y):
        if not (x == y).all():
            self.fail('Arrays are not equal:\n%r\n!=\n%r' % (x, y))

    def test_increasing_multiplication(self):
        y = array([[1, 2, 5, 6],
                   [3, 4, 7, 8],
                   [1, 1, 2, 2],
                   [1, 1, 2, 2]])
        multiplier = array([[1, 2],
                            [3, 4]])
        multiplier = hstack([multiplier, multiplier])
        multiplier = vstack([multiplier, multiplier])
        x = y.copy()
        increasing_multiplication(x, y)
        self.assertEqualArrays(x, y * multiplier)

    def test_reducer(self):
        y = array([[1, 2, 5, 6, 9],
                   [3, 4, 7, 8, 4],
                   [1, 1, 2, 2, 6],
                   [1, 1, 2, 2, 5],
                   [6, 3, 8, 1, 4]], dtype=float32)
        x = zeros((3, 3), dtype=float32)
        reducer(x, y)
        result = array([[13, 19, 19],
                        [ 8, 12, 18],
                        [ 7, 13, 10]], dtype=float32)
        self.assertEqualArrays(x, result)

    def test_center_reducer(self):
        y = array([[1, 2, 5, 6, 9],
                   [3, 4, 7, 8, 4],
                   [1, 1, 2, 2, 6],
                   [1, 1, 2, 2, 5],
                   [6, 3, 8, 1, 4]], dtype=float32)
        x = zeros(y.shape, dtype=float32)
        center_reducer(x, y)
        result = array([[5, 10, 15, 22, 10],
                        [6, 13, 19, 19, 23],
                        [5,  8, 12, 18, 11],
                        [8,  7, 13, 10, 12],
                        [4, 15,  6, 14,  6]], dtype=float32)
        self.assertEqualArrays(x, result)
