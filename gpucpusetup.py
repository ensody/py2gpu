#!/usr/bin/env python
from distutils.core import setup
from distutils.extension import Extension
import numpy

setup(name='optimizations',
      ext_modules=[Extension('gpucpu',
                             ['gpucpu.cpp'],
                             language='c++',
                             include_dirs = [numpy.get_include(), '.'],
                             extra_compile_args=['-O3'])],
)
