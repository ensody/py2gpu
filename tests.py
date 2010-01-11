from py2gpu.api import blockwise, IntArray, FloatArray, compile_gpu_code
from numpy import array, hstack, vstack, zeros, float32
from numpy.random import rand
from scipy.ndimage import sobel, convolve, label
from unittest import TestCase

@blockwise({('x', 'y'): (2, 2)}, {('x', 'y'): IntArray}, overlapping=False)
def increasing_multiplication(x, y):
    x[0, 0] = 1 * y[0, 0]
    x[0, 1] = 2 * y[0, 1]
    x[1, 0] = 3 * y[1, 0]
    x[1, 1] = 4 * y[1, 1]

@blockwise({'x': (2, 1)}, {'x': IntArray}, overlapping=False)
def get_index(x):
    x[0, 0] = CPU_INDEX + 1
    x[1, 0] = THREAD_INDEX + 1001

@blockwise({'x': (2, 1)}, {('x', 'y'): IntArray}, threadmemory={'y': (5, 5)},
           overlapping=False)
def get_index_tm(x, y):
    x[0, 0] = CPU_INDEX + 1
    x[1, 0] = THREAD_INDEX + 1001
    y[0, 0] = CPU_INDEX + 1
    y[1, 0] = THREAD_INDEX + 1001

@blockwise({'x': (1, 1), 'y': (3, 3)}, {('x', 'y'): FloatArray}, overlapping=True)
def reducer(x, y):
    x[0, 0] = y[0, 1] + y[1, 2] + y[2, 1] + y[1, 0]

@blockwise({'x': (1, 1), 'y': (3, 3)}, {('x', 'y'): FloatArray}, overlapping=True,
           center_as_origin=True, out_of_bounds_value=0)
def center_reducer(x, y):
    x[0, 0] = y[-1, 0] + y[0, 1] + y[1, 0] + y[0, -1]

#@blockwise({'x': (1, 1), 'y': (3, 3)}, {('x', 'y'): FloatArray}, overlapping=True, center_as_origin=True)
def if_center_reducer(x, y):
    top = 0
    bottom = 0
    left = 0
    right = 0
    if y.offset[0] > 0:
        top = y[-1, 0]
    if y.offset[0] < y.dim[0]:
        bottom = y[1, 0]
    if y.offset[1] > 0:
        left = y[0, -1]
    if y.offset[1] < y.dim[1]:
        right = y[0, 1]
    x[0, 0] = top + right + bottom + left

def count_labels(img):
    labelmap = zeros((img.shape[0] * img.shape[1],), dtype=int)
    orig = img.copy()
    height, width = img.shape
    id = 0
    next = 0
    top = 0
    for row in range(height):
        left = 0
        for col in range(width):
            bottom = img[row, col]
            if row > 0:
                top = labelmap[img[row-1, col]]

            if not bottom:
                left = 0
                continue

            if top:
                if left:
                    if left > top:
                        # Merge from top
                        print col, 'mergetop |',
                        labelmap[(labelmap == left).nonzero()] = top
                        id = top
                    elif left < top:
                        # Merge from left
                        print col, 'mergelefttop |',
                        labelmap[(labelmap == top).nonzero()] = left
                else:
                    # Merge from top
                    id = top
            elif not left:
                print col, '++ |',
                next += 1
                labelmap[next] = id = next

            img[row, col] = id
            left = id

        c = 0
        for i in range(1, next + 1):
            if labelmap[i] >= i:
                c += 1
        print c

        l = label(orig[:row+1])
        if c != l[1]:
            a = array(range(10) + range(10) + range(10) + range(10))[:img.shape[1]]
            print '!!!', c, 'vs', l[1]
            print vstack([a, img[:row+1]])
            print vstack([a, l[0]])
            print 'lm'
            for i in range(len(labelmap)):
                if labelmap[i]:
                    print '%d: %d' % (i, labelmap[i])
            print '!!!\n'
            return img, next
    return img, c

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

    def test_get_index(self):
        height, width = 20, 21
        data = zeros((height, width), dtype=int)
        get_index(data)
        thread = 1001
        cpu = 0
        for y in range(0, height, 2):
            for x in range(width):
                if data[y, x] == cpu + 1:
                    thread = 1001
                    cpu += 1
                self.assertEquals(data[y, x], cpu)
                self.assertEquals(data[y+1, x], thread)
                thread += 1

    def test_get_index_tm(self):
        height, width = 200, 210
        data = zeros((height, width), dtype=int)
        tm = zeros((10, 20), dtype=int)
        get_index_tm(data, tm)

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

    def test_labels(self):
        for i in range(1):
            print 'round', i
            r = (rand(25, 24) * 2).round().astype(bool).astype(int)
            l = label(r)
            cl = count_labels(r)
            print cl[0]
            print l[0]
            self.assertEquals(l[1], cl[1])
