from ctypes import c_int, c_long, c_float, c_void_p, c_double
import numpy
from numpy import ctypeslib
import os
import platform
import subprocess

from .grammar import _gpu_funcs, indent_source
from .utils import get_arg_type

LIBEXT = '.dll' if platform.system() == 'Windows' else '.so'

def import_driver():
    parent = os.path.dirname(__file__)
    source_path = os.path.join(parent, '_driver.cu')
    driver_path = os.path.join(parent, '_driver' + LIBEXT)
    options = ['--shared', '-deviceemu', '-DDEVICEEMU=1', '-o', driver_path]
    if subprocess.call(['nvcc'] + options + [source_path]):
        raise ValueError('Could not compile GPU/CPU code')
    return ctypeslib.load_library('_driver', parent)
_driver = import_driver()

emulating = c_int.in_dll(_driver, 'emulating')

_alloc = _driver.drv_alloc
_alloc.argtypes = [c_int]
_alloc.restype = c_void_p

_free = _driver.drv_free
_free.argtypes = [c_void_p]
_free.restype = c_int

_htod = _driver.drv_htod
_htod.argtypes = [c_void_p, ctypeslib.ndpointer(), c_int]
_htod.restype = c_int

_dtoh = _driver.drv_dtoh
_dtoh.argtypes = [ctypeslib.ndpointer(), c_void_p, c_int]
_dtoh.restype = c_int

class GPUError(Exception):
    pass

class GPUMemory(object):
    def __init__(self, data):
        self.data = data

    def __del__(self):
        if _free(self.data):
            raise GPUError('Freeing of GPU data failed!')

    @property
    def _as_parameter_(self):
        return self.data

def mem_alloc(size):
    data = _alloc(size)
    if data is None:
        raise GPUError('Memory allocation failed!')
    return GPUMemory(data)

def mem_alloc_like(data):
    return mem_alloc(data.nbytes)

def memcpy_htod(target, source):
    if _htod(target, source, source.nbytes):
        raise GPUError('Copying to GPU failed!')

def memcpy_dtoh(target, source):
    if _dtoh(target, source, target.nbytes):
        raise GPUError('Copying to host failed!')

def to_device(data):
    mem = mem_alloc_like(data)
    memcpy_htod(mem, data)
    return mem

def _splay_backend(n, dev):
    # heavily modified from cublas
    from pycuda.tools import DeviceData
    devdata = DeviceData(dev)

    min_threads = devdata.warp_size
    max_threads = 128
    max_blocks = 4 * devdata.thread_blocks_per_mp \
            * dev.get_attribute(drv.device_attribute.MULTIPROCESSOR_COUNT)

    if n < min_threads:
        block_count = 1
        threads_per_block = min_threads
    elif n < (max_blocks * min_threads):
        block_count = (n + min_threads - 1) // min_threads
        threads_per_block = min_threads
    elif n < (max_blocks * max_threads):
        block_count = max_blocks
        grp = (n + min_threads - 1) // min_threads
        threads_per_block = ((grp + max_blocks -1) // max_blocks) * min_threads
    else:
        block_count = max_blocks
        threads_per_block = max_threads

    #print "n:%d bc:%d tpb:%d" % (n, block_count, threads_per_block)
    return (block_count, 1), (threads_per_block, 1, 1)

def splay(*args):
    return (512, 1), (32, 1, 1)

_base_source = r'''
extern "C" {
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
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
    __kernel_%(name)s<<< __cpu_count, __thread_count >>>(%(callargs)s, __count, __total_threads);
}
'''.lstrip()

_argtypes = {
    'P': c_void_p,
    'i': c_int,
    'l': c_long,
    'f': c_float,
    'd': c_double,
}

class Function(object):
    def __init__(self, name, func):
        self.name = name
        self.func = func
        func.restype = None

    def prepare(self, argtypes):
        types = []
        for argtype in argtypes + 'ii':
            types.append(_argtypes[argtype])
        self.func.argtypes = types

    def __call__(self, cpu_count, thread_count, *args):
        args = list(args)
        args.extend((cpu_count, thread_count))
        print args
        return self.func(*args)

class SourceModule(object):
    def __init__(self, source, options=[]):
        options = options[:]
        if emulating:
            options.extend(['-deviceemu', '-DDEVICEEMU=1'])
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
        try:
            with open('gpucode.cu', 'r') as fp:
                changed = fp.read() == self.source
        except IOError:
            changed = True
        if changed:
            with open('gpucode.cu', 'w') as fp:
                fp.write(self.source)
            options.extend(['--shared', '-o', 'gpucode'+LIBEXT])
            if subprocess.call(['nvcc'] + options + ['gpucode.cu']):
                raise ValueError('Could not compile GPU/CPU code')
        self.lib = ctypeslib.load_library('gpucode', '.')

    def get_function(self, name):
        name = '__caller' + name
        return Function(name, self.lib[name])
