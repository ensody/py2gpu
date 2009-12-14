from py2gpu.api import blockwise, IntArray, FloatArray
from numpy import float32

@blockwise({'data': ('height', 'width')},
           {'data': IntArray, ('sum', 'return', 'height', 'width'): int})
def __array_sum2d(data, height, width):
    sum = 0
    for row in range(height):
        for col in range(width):
            sum += data[row, col]
    return sum

blockwise({'data': ('height', 'width')},
          {'data': FloatArray, ('height', 'width'): int, ('sum', 'return'): float32},
          name='__array_fsum2d')(__array_sum2d)

@blockwise({'data': ('height', 'width')},
           {'data': IntArray, ('height', 'width'): int, 'return': float32})
def __array_mean2d(data, height, width):
    return float(data.sum()) / (height * width)

blockwise({'data': ('height', 'width')},
          {'data': FloatArray, ('height', 'width'): int, 'return': float32},
          name='__array_fmean2d')(__array_mean2d)

@blockwise({'data': ('height', 'width')},
           {'data': FloatArray, 'result': IntArray, ('height', 'width'): int, 'value': float32})
def __array_fgreater2d(data, result, value, height, width):
    for row in range(height):
        for col in range(width):
            result[row, col] = data[row, col] > value

@blockwise({'data': ('height', 'width')},
           {'data': FloatArray, 'result': IntArray, ('height', 'width'): int, 'value': float32})
def __array_fless2d(data, result, value, height, width):
    for row in range(height):
        for col in range(width):
            result[row, col] = data[row, col] < value

@blockwise({'data': ('height', 'width')},
           {('data', 'result'): FloatArray, 'values': IntArray, ('height', 'width'): int})
def __array_fmapmul2d(data, result, values, height, width):
    for row in range(height):
        for col in range(width):
            result[row, col] = data[row, col] * values[row, col]

@blockwise({'data': ('height', 'width')},
           {'data': IntArray, ('height', 'width'): int, 'value': float32})
def __array_invert2d(data, height, width):
    for row in range(height):
        for col in range(width):
            data[row, col] = not data[row, col]

def array_do_nothing(x):
    pass

# TODO: Reductions aren't supported, yet
_reductions = {
    'sum': lambda a, b: a + b,
    'max': lambda a, b: max(a, b),
    'min': lambda a, b: min(a, b),
}

for name, func in _reductions.items():
    for dtype in (IntArray, FloatArray):
        name = '__%s_%s' % (dtype.__name__, name)
        globals()[name] = blockwise(
            {'x': (1,)}, {'x': dtype}, name=name, reduce=func)(array_do_nothing)
