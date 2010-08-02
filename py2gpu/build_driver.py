import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py2gpu.build_module import build_module

def usage():
    print 'Usage: %s [--emulate]' % os.path.basename(sys.argv[0])
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
    build_module(['driver.cu'], emulate=emulate)
