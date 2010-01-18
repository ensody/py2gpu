import ast
from functools import wraps
import inspect
import numpy
try:
    import pycuda.autoinit
    from pycuda import driver
    from pycuda.compiler import SourceModule
    from pycuda.gpuarray import splay
except ImportError:
    import sys
    print >>sys.stderr, 'pycuda not found. Only emulation will work.'
    def splay(*args):
        return (512, 1), (32, 1, 1)
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

def blockwise(blockshapes, types, threadmemory={}, overlapping=True, center_as_origin=False,
              out_of_bounds_value=_no_bounds_check, name=None, reduce=None):
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
            'threadmemory': threadmemory,
            'types': types,
            'tree': _rename_func(ast.parse(source), fname),
            'prototypes': {},
        })
        def _call_blockwise(*args):
            global _emulating_call
            assert 'gpufunc' in info, \
                'You have to call compile_gpu_code() before executing a GPU function.'
            if _emulating_call:
                gpufunc = info['func']
            else:
                gpufunc = info['gpufunc']
            return gpufunc(*args)
        _call_blockwise._py2gpu_original = func
        return wraps(func)(_call_blockwise)
    return _blockwise

_typedef_base = r'''
typedef struct /* __align__(64) */ {
    %(type)s *data;
    int shape[NDIMS];
    int offset[NDIMS];
} %(Type)sArrayStruct;
typedef %(Type)sArrayStruct* %(Type)sArray;
'''.lstrip()

_typedefs = r'''
#define NDIMS 4
#define sync __syncthreads
#define CPU_INDEX blockIdx.x
#define CPU_COUNT gridDim.x
#define THREAD_INDEX threadIdx.x
#define THREAD_COUNT blockDim.x
#define INSTANCE CPU_INDEX * THREAD_COUNT + THREAD_INDEX
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
        return "PP", arg.__name__, numpy.dtype(arg.dtype), get_arg_type(arg.dtype)[1]
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
    return dtype.char, cname, dtype, cname

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
    center_as_origin = info['center_as_origin']
    types = info['types']
    argnames = inspect.getargspec(info['func']).args
    argtypes = ''.join(types[arg][0] for arg in argnames) + 'ii'
    func.prepare(argtypes, (1, 1, 1))
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
                kernel_args.extend((arg.pointer, arg.shape))
                continue
            kernel_args.append(arg)

        # Determine number of blocks
        if not threadcount:
            threadcount = count
        threadcount = int(threadcount)
        grid, block = splay(threadcount)
        print name, threadcount, grid[0] * block[0], grid[0], 'x', block[0]
        threadcount = min(grid[0] * block[0], threadcount)
        kernel_args.extend((count, threadcount))
        func.set_block_shape(*block)
        func.prepared_call(grid, *kernel_args)
        # Now copy temporary arrays back
        for gpuarray in arrays:
            # TODO: reverse pixel reordering if needed
            gpuarray.copy_from_device()
    return _gpu_func

_emulating_call = False
def make_emulator_func(func, name, info):
    blockshapes = info['blockshapes']
    threadmemory = info['threadmemory']
    overlapping = info['overlapping']
    center_as_origin = info['center_as_origin']
    out_of_bounds_value = info['out_of_bounds_value']
    types = info['types']
    argnames = inspect.getargspec(info['func']).args
    def _emulator_func(*args):
        global _emulating_call
        is_first_call = not _emulating_call
        _emulating_call = True
        blockcount, threadcount = args[-2:]
        args = args[:-2]
        blocks_per_thread = (blockcount + threadcount - 1) / threadcount
        # Start with last thread, so we're more likely to hit a bug
        for thread in reversed(range(threadcount)):
            start = thread * blocks_per_thread
            end = min(start + blocks_per_thread, blockcount)
            if start >= end:
                continue
            # print thread
            for block in range(start, end):
                print thread, block, threadcount, blockcount
                kernel_args = []
                for argname, arg in zip(argnames, args):
                    if isinstance(arg, GPUArray):
                        arg = arg.data
                        shape = get_shape(threadmemory.get(argname), argnames, args)
                        if shape:
                            # Emulate thread memory
                            size = numpy.array(shape).prod()
                            pool = arg.reshape(arg.size)
                            memory = pool[size*thread:size*(thread+1)]
                            arg = EmulatedArray(memory.reshape(shape), shape)
                        shape = get_shape(blockshapes.get(argname), argnames, args)
                        if shape:
                            blockshape = shape
                            shape = numpy.array(shape)
                            if center_as_origin:
                                shape[...] = 1
                            numblocks = (numpy.array(arg.shape) - (shape - 1))
                            if argname in overlapping:
                                shape[...] = 1
                            numblocks /= shape
                            offset = shape.copy() * 0
                            rest = block
                            for dim in range(len(shape[:-1])):
                                offset[dim], rest = divmod(rest, numblocks[dim+1:].prod())
                                offset[dim] *= shape[dim]
                            offset[-1] = rest * shape[-1]
                            print argname, offset, numblocks, blockshape, arg.shape
                            arg = EmulatedArray(arg, blockshape, offset=offset,
                                out_of_bounds_value=out_of_bounds_value)

                        if not isinstance(arg, EmulatedArray):
                            arg = EmulatedArray(arg, None)
                    kernel_args.append(arg)
                func(*kernel_args)
        if is_first_call:
            _emulating_call = False

    def do_nothing(*args):
        pass
    _emulator_func.prepare = do_nothing
    _emulator_func.set_block_shape = do_nothing
    def prepared_call(grid, *args):
        _emulator_func(*args)
    _emulator_func.prepared_call = prepared_call

    return _emulator_func

class EmulatedArray(object):
    def __init__(self, data, blockshape, offset=None, out_of_bounds_value=None):
        self.data = data
        self.blockshape = blockshape
        self.shape = data.shape
        if offset is None:
            offset = 4 * (0,)
        self.offset = offset
        self.out_of_bounds_value = out_of_bounds_value

    def _convert_index(self, index):
        if not isinstance(index, (tuple, list)):
            index = (index,)
        index = numpy.array(index)
        index += numpy.array(self.offset[:index.size])
        if self.out_of_bounds_value is not _no_bounds_check:
            if (index < 0).any() or (index >= numpy.array(self.shape[:index.size])).any():
                return None
        if index.size > 1:
            index = tuple(index)
        else:
            index = index[0]
        return index

    def __getitem__(self, index):
        index = self._convert_index(index)
        if index is None:
            return self.out_of_bounds_value
        return self.data[index]

    def __setitem__(self, index, value):
        index = self._convert_index(index)
        if index is None:
            return
        self.data[index] = value

    def __getattr__(self, name):
        kind = ''
        if self.data.dtype == numpy.dtype(numpy.float32):
            kind = 'f'
        dim = '%dd' % len(self.data.shape)
        name = '__array_' + kind + name + dim
        func = _gpu_funcs[name]['func']
        def arrayfunc(*args):
            args += tuple(self.blockshape)
            return func(self, *args)
        return arrayfunc

def compile_gpu_code(emulate=False):
    if emulate:
        global driver, SourceModule
        from py2gpu import c_driver as driver
        SourceModule = driver.SourceModule
    GPUArray.emulate = emulate
    source = ['\n\n']
    for name, info in _gpu_funcs.items():
        source.append(convert(info['tree'], info['source']))
        source.insert(0, ';\n'.join(info['prototypes'].values()) + ';\n')
    source.insert(0, _typedefs)
    source = ''.join(source)
    mod = SourceModule(source)
    for name, info in _gpu_funcs.items():
        if 'return' in info['types']:
            continue
        func = mod.get_function('__kernel_' + name)
        info['gpufunc'] = make_gpu_func(func, name, info)
        info['gpumodule'] = mod

def emulate_gpu_code():
    GPUArray.emulate = True
    for name, info in _gpu_funcs.items():
        func = make_emulator_func(info['func'], name, info)
        info['gpufunc'] = make_gpu_func(func, name, info)

class GPUArray(object):
    emulate = False

    def __init__(self, data, dtype=None, copy_to_device=True):
        self.data = data
        if dtype:
            data = data.astype(dtype)
        else:
            dtype = data.dtype
        self.dtype = dtype

        self.shape = data.shape

        if self.emulate:
            self.pointer = self
            return

        # data
        if copy_to_device:
            self.pointer = driver.to_device(data)
        else:
            self.pointer = driver.mem_alloc_like(data)

    def copy_from_device(self):
        if self.emulate:
            return
        self.data[...] = driver.from_device_like(self.pointer, self.data)

    def set_shape(self, shape):
        shape = tuple(shape)
        shape += (0,) * (4 - len(shape))
        shape = numpy.array(shape, dtype=numpy.int32)

        if not hasattr(self, '_shape'):
            if self.emulate:
                self._shape = numpy.array(shape, dtype=numpy.int32)
            else:
                self._shape = driver.to_device(shape)
            return

        if self.emulate:
            self._shape[:] = shape
        else:
            memcpy_htod(self._shape, buffer(shape))
    def get_shape(self):
        return self._shape
    shape = property(get_shape, set_shape)

    def shrink_from_original_shape(self, shrink):
        shape = numpy.array(self.data.shape, dtype=numpy.int32)
        shrink = numpy.array(shrink, dtype=numpy.int32)
        assert (shrink >= 0).all()
        shape[:len(shrink)] -= shrink
        self.shape = shape

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
