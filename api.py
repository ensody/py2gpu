import ast
from functools import wraps
import inspect
import numpy
import pycuda.autoinit
from pycuda import driver
from pycuda.compiler import SourceModule
from pycuda.gpuarray import splay
from py2gpu.grammar import _gpu_funcs, convert, make_prototype, _no_bounds_check
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

def blockwise(blockshapes, types, overlapping=True, center_as_origin=False,
              out_of_bounds_value=_no_bounds_check, name=None, reduce=None):
    blockshapes = _simplify(blockshapes)
    types = _simplify(types)
    if overlapping is True:
        overlapping = blockshapes.keys()
    elif isinstance(overlapping, basestring):
        overlapping = (overlapping,)
    assert overlapping or not center_as_origin, \
        "You can't have overlapping=False and center_as_origin=True"
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
            'center_as_origin': center_as_origin,
            'out_of_bounds_value': out_of_bounds_value,
            'source': source,
            'types': types,
            'tree': _rename_func(ast.parse(source), fname),
            'prototypes': {},
        })
        def _call_blockwise(*args):
            assert 'gpufunc' in info, \
                'You have to call compile_gpu_code() before executing a GPU function.'
            gpufunc = info['gpufunc']
            return gpufunc(*args)
        _call_blockwise._py2gpu_original = func
        return wraps(func)(_call_blockwise)
    return _blockwise

_typedef_base = r'''
typedef struct __align__(64) {
    %(type)s *data;
    int dim[4];
    int offset[4];
    int ndim;
    int size;
} %(Type)sArrayStruct;
typedef %(Type)sArrayStruct* %(Type)sArray;
'''.lstrip()

_typedefs = r'''
#define sync __synchronize
#define CPU_INDEX blockIdx.x
#define CPU_COUNT gridDim.x
#define THREAD_INDEX threadIdx.x
#define THREAD_COUNT blockDim.x
#define __int(x) ((int)(x))
#define __float(x) ((float)(x))
#define imax(a, b) max(a, b)
#define imin(a, b) min(a, b)
#define abs(x) fabs(x)

'''.lstrip() + '\n'.join(_typedef_base % {'type': name, 'Type': name.capitalize()}
                for name in ['int', 'float']) + '\n\n'

intpsize = numpy.intp(0).nbytes
int32size = numpy.int32(0).nbytes

class ArrayType(object):
    pass

class IntArray(ArrayType):
    dtype = numpy.int32

class FloatArray(ArrayType):
    dtype = numpy.float32

def get_arg_type(arg):
    if issubclass(arg, ArrayType):
        return "P", arg.__name__, numpy.dtype(arg.dtype)
    dtype = numpy.dtype(arg)
    if issubclass(arg, numpy.int8):
        cname = 'char'
    elif issubclass(arg, numpy.uint8):
        cname = 'unsigned char'
    elif issubclass(arg, numpy.int16):
        cname = 'short'
    elif issubclass(arg, numpy.uint16):
        cname = 'unsigned short'
    elif issubclass(arg, (int, numpy.int32)):
        cname = 'int'
    elif issubclass(arg, numpy.uint32):
        cname = 'unsigned int'
    elif issubclass(arg, (long, numpy.int64)):
        cname = 'long'
    elif issubclass(arg, numpy.uint64):
        cname = 'unsigned long'
    elif issubclass(arg, numpy.float32):
        cname = 'float'
    elif issubclass(arg, (float, numpy.float64)):
        cname = 'double'
    else:
        raise ValueError("Unknown type '%r'" % tp)
    return dtype.char, cname, dtype

def make_gpu_func(mod, name, info):
    if info['types'].get('return'):
        def _gpu_func(*args):
            raise ValueError("Functions with a return value can't be called from the CPU.")
        return _gpu_func
    func = mod.get_function('_kernel_' + name)
    blockshapes = info['blockshapes']
    overlapping = info['overlapping']
    center_as_origin = info['center_as_origin']
    types = info['types']
    argnames = inspect.getargspec(info['func']).args
    argtypes = ''.join(types[arg][0] for arg in argnames) + 'i'
    func.prepare(argtypes, (1, 1, 1))
    def _gpu_func(*args):
        kernel_args = []
        arrays = []
        grid = (1, 1, 1)
        count, block = 0, 0
        for argname, arg in zip(argnames, args):
            if isinstance(arg, numpy.ndarray):
                arg = GPUArray(arg, dtype=types[argname][2])
                arrays.append(arg)
            if isinstance(arg, GPUArray):
                assert arg.dtype == types[argname][2], \
                    "The data type (%s) of the argument %s doesn't match " \
                    'the function definition (%s)' % (arg.dtype, argname, types[argname][2])
                shape = blockshapes.get(argname)
                if shape:
                    if argname in overlapping and shape != (1,) * len(shape):
                        # TODO: read pixels into shared memory by running
                        # shape[1:].prod() threads that read contiguous lines
                        # of memory, sync(), and then let only the first
                        # thread in the block do the real calculation
                        # TODO: maybe we can reorder data if it's read-only?
                        if center_as_origin:
                            assert all(dim % 2 for dim in shape), \
                                'Block dimensions must be uneven when using ' \
                                'center_as_origin=True. Please check %s' % argname
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
                arg = arg.pointer
            kernel_args.append(arg)
        # Determine number of blocks
        kernel_args.append(count)
        grid, block = splay(count)
        func.set_block_shape(*block)
        func.prepared_call(grid, *kernel_args)
        # Now copy temporary arrays back
        for gpuarray in arrays:
            # TODO: reverse pixel reordering if needed
            gpuarray.copy_from_device()
    return _gpu_func

def make_emulator_func(func):
    def emulator(*args):
        assert len(kwargs.keys()) == 1, 'Only "block" keyword argument is supported!'
        # TODO:
        raise NotImplementedError('Function emulation is not supported, yet')
    return emulator

def compile_gpu_code():
    source = ['\n\n']
    for name, info in _gpu_funcs.items():
        source.append(convert(info['tree'], info['source']))
        source.insert(0, ';\n'.join(info['prototypes'].values()) + ';\n')
    source.insert(0, _typedefs)
    source = ''.join(source)
    print '\n' + source
    mod = SourceModule(source)
    for name, info in _gpu_funcs.items():
        info['gpufunc'] = make_gpu_func(mod, name, info)

def emulate_gpu_code():
    for info in _gpu_funcs.values():
        info['gpufunc'] = make_emulator_func(info['func'])

class GPUArray(object):
    size = intpsize + (4 + 4 + 1 + 1) * int32size
    def __init__(self, data, dtype=None, copy_to_device=True):
        self.data = data
        if dtype:
            data = data.astype(dtype)
        else:
            dtype = data.dtype
        self.dtype = dtype

        # Copy array to device
        self.pointer = driver.mem_alloc(self.size)

        # data
        if copy_to_device:
            self.device_data = driver.to_device(data)
        else:
            self.device_data = driver.mem_alloc(data.nbytes)
        driver.memcpy_htod(int(self.pointer), numpy.intp(int(self.device_data)))

        # dim
        struct = data.shape
        struct += (4 - data.ndim) * (0,)
        # offset, ndim, size
        struct += 4 * (0,) + (data.ndim, data.size)
        struct = numpy.array(struct, dtype=numpy.int32)
        driver.memcpy_htod(int(self.pointer) + intpsize, buffer(struct))

    def copy_from_device(self):
        self.data[...] = driver.from_device_like(self.device_data, self.data)

from py2gpu import arrayfuncs
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
