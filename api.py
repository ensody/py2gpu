import ast
from functools import wraps
import inspect
import pycuda.autoinit
from pycuda import driver
from pycuda.compiler import SourceModule
from py2gpu.grammar import _blockwise_funcs, convert

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

def blockwise(blocksize, types, name=None):
    blocksize = _simplify(blocksize)
    types = _simplify(types)
    def _blockwise(func):
        func = getattr(func, '_py2gpu_original', func)
        fname = name
        if not fname:
            fname = func.__name__
        assert fname not in _blockwise_funcs, 'The function "%s" has already been registered!' % fname
        info = _blockwise_funcs.setdefault(fname, {
            'func': func,
            'blocksize': blocksize,
            'types': types,
            'source': _rename_func(ast.parse(inspect.getsource(func)), fname),
        })
        def _call_blockwise(*args):
            assert 'gpufunc' in info, \
                'You have to call compile_gpu_code() before executing a GPU function.'
            gpufunc = info['gpufunc']
            kwargs = {'block': (256, 1)}
            return gpufunc(*args, **kwargs)
        _call_blockwise._py2gpu_original = func
        return wraps(func)(_call_blockwise)
    return _blockwise

def compile_gpu_code():
    source = []
    for name, info in _blockwise_funcs.items():
        source.append(convert(info['source']))
    print '\n'.join(source)
    mod = SourceModule('\n'.join(source))
    for name, info in _blockwise_funcs.items():
        info['gpufunc'] = mod.get_function(name)

def emulate_gpu_code():
    for info in _blockwise_funcs.values():
        def emulator(*args, **kwargs):
            assert 'block' in kwargs
            assert len(kwargs.keys()) == 1, 'Only "block" keyword argument is supported!'
            # TODO: Call with different blocks
            return info['func'](*args)
        info['gpufunc'] = emulator
