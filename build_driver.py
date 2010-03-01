import os
import platform
import subprocess
import sys

LIBEXT = '.dll' if platform.system() == 'Windows' else '.so'

def build_driver(emulate=False):
    print 'Building', 'in emulation mode' if emulate else 'for GPU (use --emulate for CPU emulation)'
    parent = os.path.dirname(__file__)
    source_path = os.path.join(parent, 'driver.cu')
    driver_path = os.path.join(parent, '_driver' + LIBEXT)
    options = ['--shared', '-O3', '-o', driver_path]
    if emulate:
        options.extend(['-deviceemu', '-DDEVICEEMU=1'])
    if subprocess.call(['nvcc'] + options + [source_path]):
        raise ValueError('Could not compile GPU/CPU code')
    print 'Done'

def usage():
    print 'Usage: build_driver.py [--emulate]'
    sys.exit(-1)

if __name__ == '__main__':
    emulate = False
    if len(sys.argv) > 2:
        usage()
    if len(sys.argv) == 2:
        if sys.argv[1] == '--emulate':
            emulate = True
        else:
            usage()
    build_driver(emulate=emulate)
