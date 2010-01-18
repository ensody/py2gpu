import ctypes
from py2gpu.api import GPUArray, get_arg_type
from py2gpu.grammar import _gpu_funcs, indent_source
import numpy
from numpy import ctypeslib
import subprocess

_base_source = r'''
extern "C" {
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int x, y, z;
} idx;
static idx threadIdx, blockIdx, blockDim, gridDim;

#define __global__
#define __device__

#ifdef _WIN32
#define fmax(a, b) (a > b ? a : b)
#define fmin(a, b) (a < b ? a : b)
#define max(a, b) (a > b ? a : b)
#define min(a, b) (a < b ? a : b)
#define EXPORT __declspec(dllexport)
void initgpucpu() {}
#else
#define EXPORT
#endif

%s

%s
}
'''.lstrip()

_caller = r'''
EXPORT void __caller__kernel_%(name)s(%(args)s, int __count, int __total_threads, int __cpu_count, int __thread_count) {
    threadIdx.y = threadIdx.z = blockIdx.y = blockIdx.z = 0;
    gridDim.y = gridDim.z = blockDim.y = blockDim.z = 1;
    gridDim.x = __cpu_count;
    blockDim.x = __thread_count;

    for (int __cpu_id=0; __cpu_id < __cpu_count; __cpu_id++) {
        blockIdx.x = __cpu_id;
        for (int __thread_id=0; __thread_id < __thread_count; __thread_id++) {
            threadIdx.x = __thread_id;
            __kernel_%(name)s(%(callargs)s, __count, __total_threads);
        }
    }
}
'''.lstrip()

_argtypes = {
    'i': ctypes.c_int,
    'l': ctypes.c_long,
    'f': ctypes.c_float,
    'd': ctypes.c_double,
}

class Function(object):
    def __init__(self, name, func):
        self.name = name
        self.func = func
        self.thread_count = 1
        self.cpu_count = 1

        func.restype = None

    def set_block_shape(self, *block):
        self.thread_count = block[0]

    def prepare(self, argtypes, *block):
        types = []
        for argtype in argtypes + 'ii':
            if argtype == 'P':
                types.append(ctypeslib.ndpointer())
            else:
                types.append(_argtypes[argtype])
        self.func.argtypes = types

    def prepared_call(self, grid, *args):
        self.cpu_count = grid[0]
        newargs = []

        for arg in args:
            if isinstance(arg, GPUArray):
                newargs.append(arg.data)
            else:
                newargs.append(arg)
        newargs.extend((self.cpu_count, self.thread_count))

        return self.func(*newargs)

class SourceModule(object):
    def __init__(self, source):
        callers = []
        for name, info in _gpu_funcs.items():
            if 'return' in info['types']:
                continue
            args = [arg.id for arg in info['funcnode'].args.args]
            types = info['types']
            funcargs = []
            callargs = []
            arrays = []
            for arg in args:
                kind = types[arg][1]
                if kind.endswith('Array'):
                    funcargs.append('%s *%s' % (get_arg_type(types[arg][2].type)[1], arg))
                    funcargs.append('int *__dim_%s' % arg)
                    callargs.append('%s, __dim_%s' % (arg, arg))
                else:
                    funcargs.append('%s %s' % (kind, arg))
                    callargs.append(arg)

            data = {
                'args': ', '.join(funcargs),
                'callargs': ', '.join(callargs),
                'name': name,
                'arrays': '\n'.join(arrays),
            }
            callers.append(_caller % data)
        self.source = _base_source % (source, '\n'.join(callers))
        with open('gpucpu.cpp', 'w') as fp:
            fp.write(self.source)
        if subprocess.call(['python', 'gpucpusetup.py',
                            'build_ext', '--inplace']) != 0:
            raise ValueError('Could not compile GPU/CPU code')
        self.lib = ctypeslib.load_library('gpucpu','.')

    def get_function(self, name):
        name = '__caller' + name
        return Function(name, self.lib[name])
