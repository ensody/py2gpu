"""
Python ast to C++ converter
"""
import ast
from pymeta.grammar import OMeta
from pymeta.runtime import ParseError, EOFError

# Globals for storing type information about registered functions
_gpu_funcs = {}

class Py2GPUParseError(ValueError):
    pass

py2gpu_grammar = r'''
node :name = :n ?(n.__class__.__name__ == name) -> n

add = node('Add') -> '+'
binop = node('BinOp'):n -> '(%s %s %s)' % (self.parse(n.left, 'op'), self.parse(n.op, 'op'), self.parse(n.right, 'op'))
div = node('Div') -> '/'
mult = node('Mult') -> '*'
name = node('Name'):n -> n.id
num = node('Num'):n -> str(n.n)
sub = node('Sub') -> '-'
call = node('Call'):n -> self.gen_call(n)

subscript = node('Subscript'):n !(self.parse(n.value, 'name')):name
    -> '%s->data[%s]' % (name, self.gen_subscript(name, self.parse(n.slice, 'index')))
index = node('Index'):n -> self.parse(n.value, 'subscriptindex')
subscriptindex = op:n -> [n]
               | tupleslice
tupleslice = node('Tuple'):n -> [self.parse(index, 'op') for index in n.elts]

op = add
   | binop
   | div
   | mult
   | name
   | num
   | sub
   | subscript
   | call

assign = node('Assign'):n ?(len(n.targets) == 1) -> '%s = %s' % (self.parse(n.targets[0], 'op'), self.parse(n.value, 'op'))
expr = node('Expr'):n -> self.parse(n.value, 'op')
functiondef 0 = node('FunctionDef'):n -> self.gen_func(n, 0)

bodyitem :i = (assign
               | expr):n -> indent(i) + n + ';'
            | functiondef(i)
body :i = bodyitem(i)+:xs -> '\n'.join(xs) + '\n'

grammar = node('Module'):n -> self.parse(n.body, 'body', 0)
'''

def convert(code):
    if isinstance(code, ast.AST):
        tree = code
    else:
        tree = ast.parse(code)
    try:
        converted = Py2GPUGrammar([]).parse(tree)
    except ParseError, e:
        lines = code.split('\n')
        start, stop = max(0, e.position-1), min(len(lines), e.position+2)
        snippet = '\n'.join(lines[start:stop])
        raise Py2GPUParseError('Parse error at line %d (%s):\n%s' % (e.position, str(e), snippet))
    return converted

def indent(level):
    return level * '    '

def indent_source(level, source=''):
    shift = level * '    '
    return '\n'.join([line and shift + line for line in source.split('\n')])

def p(*x):
    print x

vars = {
    'indent': indent,
    'p': p,
}

_func_prototype = r'''
%(func_type)s void %(name)s(%(args)s)
'''.strip()

_func_template = r'''
%(func)s {
%(body)s
}
'''.lstrip()

_shift_arg = r'''
%(type)sStruct __shifted_%(name)s = *%(name)s;
'''.lstrip();

_offset_template = r'''
%(name)s.offset[%(dim)d] = block / (%(dims)s / %(dimsize)d);
'''.lstrip();

_kernel_body = r'''
unsigned int thread_id = threadIdx.x;
unsigned int total_threads = gridDim.x*blockDim.x;
unsigned int memory_start = blockDim.x*blockIdx.x;
unsigned int block, blocks_per_thread, end;

%(declarations)s

if (count > total_threads)
    blocks_per_thread = (count + total_threads - 1) / total_threads;
else
    blocks_per_thread = 1;

block = memory_start + thread_id;
end = block + blocks_per_thread;

if (end > count)
    end = count;

for (; block < end; block++) {
    %(offset)s
    %(call)s
}
'''.lstrip()

def make_prototype(func, func_type, name, info):
        try:
            return info['prototypes'][func_type]
        except KeyError:
            pass
        args = [arg.id for arg in func.args.args]
        types = info['types']
        assert func.args.vararg is None
        assert func.args.kwarg is None
        assert func.args.defaults == []
        data = {
            'args': ', '.join(['%s %s' % (types[arg][1], arg) for arg in args]),
            'name': name,
            'func_type': func_type,
        }
        if func_type == '__global__':
             data['args'] += ', int count'
        prototype = _func_prototype % data
        info['prototypes'][func_type] = prototype
        return prototype

class Py2GPUGrammar(OMeta.makeGrammar(py2gpu_grammar, vars, name="Py2CGrammar")):
    def raise_parse_error(self, node, error, message=''):
        lineno = getattr(node, 'lineno', 1)
        col_offset = getattr(node, 'col_offset', 1)
        if message:
            message = ': ' + message
        raise ParseError(lineno, error,
            'Parse error at line %d, col %d (node %s)%s' % (
                lineno, col_offset, node.__class__.__name__, message))

    def parse(self, data, rule='grammar', *args):
        # print data, rule
        if not isinstance(data, (tuple, list)):
            data = (data,)
        try:
            grammar = self.__class__(data)
            grammar._func_name = getattr(self, '_func_name', None)
            result, error = grammar.apply(rule, *args)
        except ParseError:
            raise_parse_error(data, None, 'Unsupported node type')
        try:
            head = grammar.input.head()
        except EOFError:
            pass
        else:
            raise_parse_error(head[0], error)
        return result

    def gen_subscript(self, name, indices):
        dims = len(indices)
        access = []
        for dim, index in enumerate(indices):
            index = '(%s->offset[%d] + %s)' % (name, dim, index)
            access.append(' * '.join(['%s->dim[%d]' % (name, subdim)
                                      for subdim in range(dim+1, dims)] + [index]))
        return ' + '.join(access)

    def gen_func(self, func, level):
        name = func.name
        info = _gpu_funcs[name]

        # Store function name, so it can be reused in other parser rules
        self._func_name = name

        types = info['types']
        args = set(arg.id for arg in func.args.args)
        vars = set(types.keys()).symmetric_difference(args)
        vars = '\n'.join('%s %s;' % (types[var][1], var) for var in vars)
        if vars:
            vars += '\n\n'
        data = {
            'func': make_prototype(func, '__device__', name, info),
            'body': indent_source(level+1, vars) + self.parse(func.body, 'body', level+1),
        }
        source = _func_template % data

        # Generate kernel that shifts block by offset and calls device function
        args = []
        blockinit = []
        offsetinit = []
        blockshapes = info['blockshapes']
        for arg in func.args.args:
            arg = arg.id
            shape = blockshapes.get(arg)
            if shape:
                blockinit.append(_shift_arg % {'name': arg, 'type': types[arg][1]})
                arg = '__shifted_' + arg
                for dim in range(len(shape)):
                    dims = ' * '.join('%s.dim[%d]' % (arg, subdim)
                                      for subdim in range(dim+1, len(shape)))
                    offsetinit.append(_offset_template % {'name': arg, 'dim': dim,
                                      'dims': dims, 'dimsize': shape[dim]})
                arg = '&' + arg
            args.append(arg)
        bodydata = {
            'declarations': '%s' % ''.join(blockinit),
            'offset': indent_source(level + 1, '\n'.join(offsetinit)).lstrip(),
            'call': '%s(%s);' % (func.name, ', '.join(args)),
        }
        data['func'] = make_prototype(func, '__global__', '_kernel_' + name, info)
        data['body'] = indent_source(level+1, _kernel_body % bodydata)
        source += _func_template % data

        # Reset function context
        self._func_name = None

        return source

    def gen_call(self, call):
        name = self.parse(call.func, 'name')
        args = [self.parse(arg, 'op') for arg in call.args]
        assert call.starargs is None
        assert call.kwargs is None
        assert call.keywords == []
        return '%s(%s)' % (name, ', '.join(args))
