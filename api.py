import ast
from functools import wraps
import inspect
import numpy
from . import driver
from .grammar import _gpu_funcs, convert, make_prototype
from .utils import get_arg_type, ArrayType
from textwrap import dedent

# TODO: implement reduction support

def _simplify(mapping):
    simple = {}
    for names, value in mapping.items():
        if not isinstance(names, (list, tuple, set)):
            names = (names,)
        for name in names:
            assert name not in simple, "Variable %s specified multiple times" % name
            simple[name] = value
    return simple

def _rename_func(tree, name):
    tree.body[0].name = name
    return tree

def blockwise(blockshapes, types, threadmemory={}, overlapping=True, center_on_origin=False,
              name=None, reduce=None):
    """
    Processes the image in parallel by splitting it into equally-sized blocks.

    threadmemory can be used to store thread-local temporary data.
    """
    blockshapes = _simplify(blockshapes)
    types = _simplify(types)
    threadmemory = _simplify(threadmemory)
    if overlapping is True:
        overlapping = blockshapes.keys()
    elif isinstance(overlapping, basestring):
        overlapping = (overlapping,)
    else:
        overlapping = ()
    assert overlapping or not center_on_origin, \
        "You can't have overlapping=False and center_on_origin=True"
    for varname, vartype in types.items():
        types[varname] = get_arg_type(vartype)
    def _blockwise(func):
        func = getattr(func, '_py2gpu_original', func)
        fname = name
        if not fname:
            fname = func.__name__
        assert fname not in _gpu_funcs, 'The function "%s" has already been registered!' % fname
        source = dedent(inspect.getsource(func))
        info = _gpu_funcs.setdefault(fname, {
            'func': func,
            'functype': 'blockwise',
            'blockshapes': blockshapes,
            'overlapping': overlapping,
            'center_on_origin': center_on_origin,
            'source': source,
            'threadmemory': threadmemory,
            'types': types,
            'tree': _rename_func(ast.parse(source), fname),
            'prototypes': {},
        })
        @wraps(func)
        def _call_blockwise(*args):
            assert 'gpufunc' in info, \
                'You have to call compile_gpu_code() before executing a GPU function.'
            gpufunc = info['gpufunc']
            return gpufunc(*args)
        _call_blockwise._py2gpu_original = func
        return _call_blockwise
    return _blockwise

_typedef_base = r'''
typedef struct /* __align__(16) */ {
    %(type)s *data;
    int shape[NDIMS];
    int offset[NDIMS];
} %(Type)sArrayStruct;
typedef %(Type)sArrayStruct* %(Type)sArray;
'''.lstrip()

_array_typedefs = (('int32', 'Int32'), ('uint32', 'UInt32'),
                   ('int8', 'Int8'), ('float', 'Float'))

_typedefs = r'''
typedef char int8;
typedef unsigned char uint8;
typedef short int16;
typedef unsigned short uint16;
typedef int int32;
typedef unsigned int uint32;

#define NDIMS 4
#define sync __syncthreads
#define CPU_INDEX blockIdx.x
#define CPU_COUNT gridDim.x
#define THREAD_INDEX threadIdx.x
#define THREAD_COUNT blockDim.x
#define INSTANCE CPU_INDEX * THREAD_COUNT + THREAD_INDEX
#define __py_int(x) ((int)(x))
#define __py_float(x) ((float)(x))
#define __py_sqrt(x) (sqrtf(x))
#define __py_log(x) (logf(x))
#define abs(x) fabs(x)

} // extern "C"

template <typename T>
__device__ T __py_max(T a, T b) {
    return max(a, b);
}

template <typename T>
__device__ T __py_min(T a, T b) {
    return min(a, b);
}

template <>
__device__ int32 __py_max(int32 a, int32 b) {
  return max(a, b);
}

template <>
__device__ float __py_max(float a, float b) {
  return fmax(a, b);
}

template <>
__device__ int32 __py_min(int32 a, int32 b) {
  return min(a, b);
}

template <>
__device__ float __py_min(float a, float b) {
  return fmin(a, b);
}

extern "C" {
'''.lstrip() + '\n'.join(_typedef_base % {'type': type_name, 'Type': class_name}
                for type_name, class_name in _array_typedefs) + '\n\n'

intpsize = numpy.intp(0).nbytes
int32size = numpy.int32(0).nbytes

class Int32Array(ArrayType):
    dtype = numpy.int32

class UInt32Array(ArrayType):
    dtype = numpy.uint32

class Int8Array(ArrayType):
    dtype = numpy.int8

class FloatArray(ArrayType):
    dtype = numpy.float32

def get_shape_dim(dim, argnames, args):
    if isinstance(dim, basestring):
        return args[argnames.index(dim)]
    return dim

def get_shape(shape, argnames, args):
    if not shape:
        return shape
    return tuple(get_shape_dim(dim, argnames, args) for dim in shape)

def make_gpu_func(func, name, info):
    if info['types'].get('return'):
        def _gpu_func(*args):
            raise ValueError("Functions with a return value can't be called from the CPU.")
        return _gpu_func
    blockshapes = info['blockshapes']
    threadmemory = info['threadmemory']
    overlapping = info['overlapping']
    center_on_origin = info['center_on_origin']
    types = info['types']
    argnames = inspect.getargspec(info['func']).args
    argtypes = ''.join(types[arg][0] for arg in argnames) + 'ii'
    func.prepare(argtypes)
    def _gpu_func(*args):
        kernel_args = []
        arrays = []
        grid = (1, 1, 1)
        count = threadcount = block = 0
        for argname, arg in zip(argnames, args):
            if isinstance(arg, numpy.ndarray):
                arg = GPUArray(arg, dtype=types[argname][2])
                arrays.append(arg)
            if isinstance(arg, GPUArray):
                assert arg.dtype == types[argname][2], \
                    "The data type (%s) of the argument %s doesn't match " \
                    'the function definition (%s)' % (arg.dtype, argname, types[argname][2])
                shape = get_shape(threadmemory.get(argname), argnames, args)
                if shape:
                    old_threadcount = threadcount
                    threadcount = arg.data.size / numpy.array(shape).prod()
                    if old_threadcount:
                        threadcount = min(old_threadcount, threadcount)
                shape = get_shape(blockshapes.get(argname), argnames, args)
                if shape:
                    if argname in overlapping and shape != (1,) * len(shape):
                        # TODO: read pixels into shared memory by running
                        # shape[1:].prod() threads that read contiguous lines
                        # of memory, sync(), and then let only the first
                        # thread in the block do the real calculation
                        # TODO: maybe we can reorder data if it's read-only?
                        if center_on_origin:
                            assert all(dim % 2 for dim in shape), \
                                'Block dimensions must be uneven when using ' \
                                'center_on_origin=True. Please check %s' % argname
                            blockcount = arg.data.size
                        else:
                            blockcount = (numpy.array(arg.data.shape) -
                                          (numpy.array(shape) - 1)).prod()
                    else:
                        assert not any(dim1 % dim2 for dim1, dim2
                                       in zip(arg.data.shape, shape)), \
                            'Size of argument "%s" must be an integer ' \
                            'multiple of its block size when using ' \
                            'non-overlapping blocks.' % argname
                        # TODO: reorder pixels for locality
                        blockcount = arg.data.size / numpy.array(shape).prod()
                    if count:
                        assert count == blockcount, \
                            'Number of blocks of argument "%s" (%d) ' \
                            "doesn't match the preceding blockwise " \
                            'arguments (%d).' % (argname, blockcount, count)
                    count = int(blockcount)
                kernel_args.extend((arg.pointer, arg.shape))
                continue
            kernel_args.append(arg)

        # Determine number of blocks
        if not threadcount:
            threadcount = count
        threadcount = int(threadcount)
        grid, block = driver.splay(threadcount)
        threadcount = min(grid[0] * block[0], threadcount)
        kernel_args.extend((count, threadcount))
        func(grid[0], block[0], *kernel_args)
        # Now copy temporary arrays back
        for gpuarray in arrays:
            # TODO: reverse pixel reordering if needed
            gpuarray.copy_from_device()
    return _gpu_func

def compile_gpu_code():
    source = ['\n\n']
    for name, info in _gpu_funcs.items():
        source.append(convert(info['tree'], info['source']))
        source.insert(0, ';\n'.join(info['prototypes'].values()) + ';\n')
    source.insert(0, _typedefs)
    source = ''.join(source)
    mod = driver.SourceModule(source)
    for name, info in _gpu_funcs.items():
        if 'return' in info['types']:
            continue
        func = mod.get_function('__kernel_' + name)
        info['gpufunc'] = make_gpu_func(func, name, info)
        info['gpumodule'] = mod

class GPUArray(object):
    def __init__(self, data, dtype=None, copy_to_device=True, stream=None):
        self.data = data
        if dtype:
            data = data.astype(dtype)
        else:
            dtype = data.dtype
        self.dtype = dtype

        self.shape = data.shape

        # data
        if copy_to_device:
            self.pointer = driver.to_device(data)
        else:
            self.pointer = driver.mem_alloc_like(data)

    def copy_from_device(self):
        driver.memcpy_dtoh(self.data, self.pointer)

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        shape = tuple(shape)
        shape += (0,) * (4 - len(shape))
        shape = numpy.array(shape, dtype=numpy.int32)

        if not hasattr(self, '_shape'):
            self._shape = driver.mem_alloc_like(shape)
        driver.memcpy_htod(self._shape, shape)

    def shrink_from_original_shape(self, shrink):
        shape = numpy.array(self.data.shape, dtype=numpy.int32)
        shrink = numpy.array(shrink, dtype=numpy.int32)
        assert (shrink >= 0).all()
        shape[:len(shrink)] -= shrink
        self.shape = shape

from . import arrayfuncs
def make_array_reduction(name):
    def _reduction(self):
        if self.dtype == dtype(numpy.int32):
            return getattr(arrayfuncs, '__IntArray_' + name)(self)
        elif self.dtype == dtype(numpy.float32):
            return getattr(arrayfuncs, '__FloatArray_' + name)(self)
        else:
            raise TypeError("Can't handle arrays of dtype %s" % self.dtype)
for name in arrayfuncs._reductions:
    setattr(GPUArray, name, make_array_reduction(name))
