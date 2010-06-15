import os.path
import os
import platform
import subprocess
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LIBEXT = '.dll' if platform.system() == 'Windows' else '.so'

def build_module(names, emulate=False):
    print 'Building', 'in emulation mode' if emulate else 'for GPU (use --emulate for CPU emulation)'
    name, extra = names[0], names[1:]
    path = os.path.abspath(name)
    base = os.path.splitext(os.path.basename(path))[0]
    parent = os.path.dirname(path)
    source_path = name
    driver_path = os.path.join(parent, '_%s%s' % (base, LIBEXT))
    options = ['--shared', '-O3', '-o', driver_path]
    if emulate:
        options.extend(['-deviceemu', '-DDEVICEEMU=1'])
    if subprocess.call(['nvcc'] + options + [source_path] + extra):
        raise ValueError('Could not compile GPU/CPU code')
    print 'Done'

def usage():
    print 'Usage: %s <module.cu>... [--emulate]' % os.path.basename(sys.argv[0])
    sys.exit(-1)

if __name__ == '__main__':
    emulate = False
    if len(sys.argv) < 2:
        usage()
    names = sys.argv[1:]
    if sys.argv[-1] == '--emulate':
        emulate = True
        names = names[:-1]
    build_module(names, emulate=emulate)
