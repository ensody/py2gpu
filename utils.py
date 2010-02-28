import numpy

class ArrayType(object):
    pass

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
