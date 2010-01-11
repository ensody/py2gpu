from py2gpu.grammar import _gpu_funcs
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

%s

%s
}
'''.lstrip()

_caller = r'''
void __caller__kernel_%(name)s(%(args)s, int __count, int __total_threads, int __cpu_count, int __thread_count) {
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


class Function(object):
    def __init__(self, func):
        self.func = func
        self.thread_count = 1
        self.cpu_count = 1

    def prepare(self, *args):
        pass

    def prepared_call(self, *args):
        pass

class SourceModule(object):
    def __init__(self, source):
        callers = []
        for name, info in _gpu_funcs.items():
            if 'return' in info['types']:
                continue
            args = [arg.id for arg in info['funcnode'].args.args]
            types = info['types']
            data = {
                'args': ', '.join(['%s %s' % (types[arg][1], arg) for arg in args]),
                'callargs': ', '.join(args),
                'name': name,
            }
            callers.append(_caller % data)
        self.source = _base_source % (source, '\n'.join(callers))
        with open('gpucpu.cpp', 'w') as fp:
            fp.write(self.source)
        subprocess.call(['python', 'gpucpusetup.py', 'build_ext', '--inplace'])
        self.lib = ctypeslib.load_library('gpucpu','.')

    def get_function(self, name):
        return Function(self.lib['__caller_' + name])
