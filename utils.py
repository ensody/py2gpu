import numpy

class ArrayType(object):
    pass

def get_arg_type(arg):
    if issubclass(arg, ArrayType):
        dtype_chars = 'P' + 3*numpy.dtype(numpy.int).char
        return dtype_chars, arg.__name__, numpy.dtype(arg.dtype), \
               get_arg_type(arg.dtype)[1]
    dtype = numpy.dtype(arg)
    if issubclass(arg, numpy.int8):
        cname = 'int8'
    elif issubclass(arg, numpy.uint8):
        cname = 'uint8'
    elif issubclass(arg, numpy.int16):
        cname = 'int16'
    elif issubclass(arg, numpy.uint16):
        cname = 'uint16'
    elif issubclass(arg, (int, numpy.int32)):
        cname = 'int32'
    elif issubclass(arg, numpy.uint32):
        cname = 'uint32'
    elif issubclass(arg, (long, numpy.int64)):
        cname = 'int64'
    elif issubclass(arg, numpy.uint64):
        cname = 'uint64'
    elif issubclass(arg, numpy.float32):
        cname = 'float'
    elif issubclass(arg, (float, numpy.float64)):
        cname = 'double'
    else:
        raise ValueError("Unknown type '%r'" % tp)
    return dtype.char, cname, dtype, cname
