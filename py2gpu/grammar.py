"""
Python ast to C++ converter
"""
try:
    import ast
except ImportError:
    import _ast as ast
    def parse(expr, filename='<unknown>', mode='exec'):
        return compile(expr, filename, mode, ast.PyCF_ONLY_AST)
    ast.parse = parse
from pymeta.grammar import OMeta
from pymeta.runtime import ParseError, EOFError
from math import sqrt
import numpy

# Globals for storing type information about registered functions
_gpu_funcs = {}

class Py2GPUParseError(ValueError):
    pass

py2gpu_grammar = r'''
node :name = :n ?(n.__class__.__name__ == name) -> n

attribute = node('Attribute'):n -> '%s->%s' % (self.parse(n.value, 'name'), n.attr)
name = node('Name'):n -> name_mapper.get(n.id, n.id)
varaccess = attribute | name
num = node('Num'):n -> str(n.n)+'f' if isinstance(n.n, float) else str(n.n)
str = node('Str'):n -> to_string(n.s)
anyvar = varaccess | num | str

binop = node('BinOp'):n -> '(%s %s %s)' % (self.parse(n.left, 'op'), self.parse(n.op, 'anybinaryop'), self.parse(n.right, 'op'))
anybinaryop = add | div | mult | sub
add = node('Add') -> '+'
div = node('Div') -> '/'
mult = node('Mult') -> '*'
sub = node('Sub') -> '-'

unaryop = node('UnaryOp'):n -> '%s(%s)' % (self.parse(n.op, 'anyunaryop'), self.parse(n.operand, 'op'))
anyunaryop = usub | not
not = node('Not') -> '!'
usub = node('USub') -> '-'

boolop = node('BoolOp'):n -> (' %s ' % self.parse(n.op, 'anyboolop')).join(['(%s)' % self.parse(value, 'op') for value in n.values])
anyboolop = and | or
and = node('And') -> '&&'
or = node('Or') -> '||'

compare = node('Compare'):n -> '(%s %s %s)' % (self.parse(n.left, 'op'), self.parse(n.ops[0], 'compareop'), self.parse(n.comparators[0], 'op'))
compareop = eq | noteq | gt | gte | lt | lte
eq = (node('Eq') | node('Is')) -> '=='
noteq = (node('NotEq') | node('IsNot')) -> '!='
gt = node('Gt') -> '>'
gte = node('GtE') -> '>='
lt = node('Lt') -> '<'
lte = node('LtE') -> '<='

call = node('Call'):n -> self.gen_call(n)

subscript :assign = node('Subscript'):n !(self.parse(n.value, 'varaccess')):name
    -> self.gen_subscript(name, self.parse(n.slice, 'index'), assign)
index = node('Index'):n -> self.parse(n.value, 'subscriptindex')
subscriptindex = op:n -> [n]
               | tupleslice
tupleslice = node('Tuple'):n -> [self.parse(index, 'op') for index in n.elts]

op = binop
   | unaryop
   | boolop
   | subscript(False)
   | anyvar
   | call
   | compare

assignop = name | subscript(True)
assign = node('Assign'):n -> '%s = %s' % (' = '.join([self.parse(target, 'assignop') for target in n.targets]), self.parse(n.value, 'oporassign'))
augassign = node('AugAssign'):n -> '%s %s= %s' % (self.parse(n.target, 'assignop'), self.parse(n.op, 'anybinaryop'), self.parse(n.value, 'op'))
oporassign = op | assign
expr = node('Expr'):n -> self.parse(n.value, 'op')
return = node('Return'):n -> 'return %s' % self.parse(n.value, 'op') if n.value else 'return'
pass = node('Pass') -> ''
functiondef 0 = node('FunctionDef'):n -> self.gen_func(n, 0)

if :i = node('If'):n -> 'if (%s) {\n%s%s}%s' % (self.parse(n.test, 'op'), self.parse(n.body, 'body', i+1), indent(i), self.parse(n, 'else', i))
else :i = node('If'):n -> ' else {\n%s%s}' % (self.parse(n.orelse, 'body', i+1), indent(i)) if n.orelse else ''

for :i = node('For'):n -> self.gen_for(n, i)
range = node('Call'):n ?(self.parse(n.func, 'name') == 'range') ?(not n.keywords and not n.starargs and not n.kwargs) -> [self.parse(arg, 'op') for arg in n.args]

while :i = node('While'):n -> self.gen_while(n, i)

continue = node('Continue') -> 'continue'
break = node('Break') -> 'break'

print = node('Print') -> ''

bodyitem :i = (assign | augassign | expr | return | pass | continue | break):n -> indent(i) + n + ';'
            | (if(i) | for(i) | while(i)):n -> indent(i) + n
            | functiondef(i)
            | print
body :i = bodyitem(i)+:xs -> '\n'.join(xs) + '\n'

grammar = node('Module'):n -> self.parse(n.body, 'body', 0)
'''

def convert(code, codestring=''):
    if isinstance(code, ast.AST):
        tree = code
    else:
        tree = ast.parse(code)
    try:
        converted = Py2GPUGrammar([]).parse(tree)
    except ParseError, e:
        lines = codestring.split('\n')
        start, stop = max(0, e.position-1), min(len(lines), e.position+2)
        snippet = '\n'.join(lines[start:stop])
        raise Py2GPUParseError('Parse error at line %d (%s):\n%s' % (e.position, str(e), snippet))
    return converted

def indent(level):
    return level * '    '

def indent_source(level, source=''):
    shift = level * '    '
    return '\n'.join([line and shift + line for line in source.split('\n')])

def to_string(string):
    string = repr(string)
    if string[0] == "'":
        string = '"' + string[1:-1].replace('"', r'\"') + '"'
    return string

def p(*x):
    print x

name_mapper = {
    'None': 'NULL',
    'True': 'true',
    'False': 'false',
}

vars = {
    'indent': indent,
    'p': p,
    'name_mapper': name_mapper,
    'to_string': to_string,
}

_for_loop = r'''
for (int %(name)s=%(start)s; %(name)s < %(stop)s; %(name)s += %(step)s) {
%(body)s%(indent)s}
'''.strip()

_while_loop = r'''
while (%(test)s) {
%(body)s%(indent)s}
'''.strip()

_func_prototype = r'''
%(func_type)s %(return_type)s %(name)s(%(args)s)
'''.strip()

_func_template = r'''
%(func)s {
%(body)s
}
'''.lstrip()

# TODO: Maximize processor usage instead of thread usage!
_kernel_body = r'''
%(declarations)s

%(call)s
'''.strip()

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
        funcargs = []
        for arg in args:
            kind = types[arg][1]
            if kind.endswith('Array'):
                funcargs.append('%s *%s' % (types[arg][3], arg))
                funcargs.extend('int32 __%s_shape%d' % (arg, dim) for dim in range(3))
            else:
                funcargs.append('%s %s' % (kind, arg))

        data = {
            'args': ', '.join(funcargs),
            'name': name,
            'func_type': func_type,
            'return_type': types.get('return', (0, 'void'))[1],
        }
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

    @property
    def func_name(self):
        return getattr(self, '_func_name', None)

    def parse(self, data, rule='grammar', *args):
        # print data, rule, [(getattr(item, 'lineno', 0), getattr(item, 'col_offset', 0)) for item in (data if isinstance(data, (tuple, list)) else [data])]
        if not isinstance(data, (tuple, list)):
            data = (data,)
        try:
            grammar = self.__class__(data)
            grammar._func_name = self.func_name
            result, error = grammar.apply(rule, *args)
        except ParseError:
            self.raise_parse_error(data, None, 'Unsupported node type')
        try:
            head = grammar.input.head()
        except EOFError:
            pass
        else:
            self.raise_parse_error(head[0], error)
        return result

    def gen_subscript(self, name, indices, assigning):
        dims = len(indices)
        if '->' in name:
            assert dims == 1, 'Attribute values can only be one-dimensional.'
            name, attr = name.rsplit('->', 1)
            if attr == 'offset':
                info = _gpu_funcs[self.func_name]
                blockshapes = info['blockshapes']
                overlapping = info['overlapping']
                threadmemory = info['threadmemory']
                center_on_origin = info['center_on_origin']
                if name in threadmemory:
                    return '0'
                shape = blockshapes.get(name)
                if shape:
                    try:
                        dim = int(indices[0])
                    except:
                        raise ValueError('Offset must be an integer')
                    return self.get_block_init(name, dim, shape[dim],
                        name in overlapping, center_on_origin)[2]
                else:
                    raise ValueError('%s is not an array' % name)
            elif attr == 'shape':
                shape = '__%s_shape2' % name
                for dim in reversed(range(2)):
                    shape = '(%s == %d ? __%s_shape%d : %s)' % (indices[0], dim, name, dim, shape)
                return shape
            else:
                return '%s->%s[%s]' % (name, attr, indices[0])
        access = []
        shifted_indices = []
        for dim, index in enumerate(indices):
            shifted_indices.append(index)
            access.append(' * '.join(['__%s_shape%d' % (name, subdim)
                                      for subdim in range(dim+1, dims)] + [index]))
        subscript = '%s[%s]' % (name, ' + '.join(access))
        return subscript

    def gen_func(self, func, level):
        name = func.name
        info = _gpu_funcs[name]

        # Store function name, so it can be reused in other parser rules
        self._func_name = name
        info['funcnode'] = func

        types = info['types']
        threadmemory = info['threadmemory']
        args = set(arg.id for arg in func.args.args)
        vars = set(types.keys()).symmetric_difference(
            args).symmetric_difference(threadmemory.keys())
        vars = '\n'.join('%s %s;' % (types[var][3], var) for var in vars
                         if var != 'return')

        # Calculate register requirements before parsing the function body
        maxthreads = None
        # TODO: actually calculate the real number of unused registers
        unused_registers = 8192 - 600
        needed_registers = 0
        for var, shape in threadmemory.items():
            size = numpy.array(shape).prod()
            needed_registers += size
            vars += '\n%s %s[%d];' % (types[var][3], var, size)
        if needed_registers:
            threads = unused_registers / needed_registers
            # If there aren't enough registers we fall back to local memory
            # and use a relatively high number of threads to compensate for
            # the slower memory access
            if threads < 96:
                threads = 64
            x_threads = int(sqrt(threads))
            maxthreads = (threads // x_threads, x_threads, 1)
        info['maxthreads'] = maxthreads

        if vars:
            vars += '\n\n'
        data = {
            'func': make_prototype(func, '__device__', name, info),
            'body': indent_source(level+1, vars) + self.parse(func.body, 'body', level+1),
        }
        source = _func_template % data

        # Functions with return values can't be called from the CPU
        if types.get('return'):
            self._func_name = None
            return source

        # Generate kernel that shifts block by offset and calls device function
        blockinit = []
        blockshapes = info['blockshapes']
        overlapping = info['overlapping']
        center_on_origin = info['center_on_origin']

        args = []
        for arg in func.args.args:
            origarg = arg = arg.id
            kind = types[arg][1]

            if kind.endswith('Array') and arg in blockshapes:
                arg = '__array_' + arg

            shape = blockshapes.get(origarg)
            if shape:
                offsetinit = []
                blockinit.append('%s *%s;' % (types[origarg][3], arg))
                blockinit.append('if (%s != NULL) {' % origarg)
                for dim, dimlength in enumerate(shape):
                    block, limit, shift = self.get_block_init(origarg, dim,
                        dimlength, origarg in overlapping, center_on_origin)
                    blockinit.append('    if (%s >= %s)\n        return;' % (block, limit))
                    if dim == len(shape) - 1:
                        offsetinit.append(shift)
                    else:
                        offsetinit.append('%s * %s' % (' * '.join('__%s_shape%d' % (origarg, subdim) for subdim in range(dim+1, len(shape))), shift))
                blockinit.append('    %s = %s + %s;' % (
                    arg, origarg, ' + '.join(offsetinit)))
                blockinit.append('} else {')
                blockinit.append('    %s = %s;' % (arg, origarg))
                blockinit.append('}')

            args.append(arg)

            if kind.endswith('Array'):
                args.extend('__%s_shape%d' % (origarg, dim) for dim in range(3))

        bodydata = {
            'declarations': '%s' % '\n'.join(blockinit),
            'call': '%s(%s);' % (func.name, ', '.join(args)),
        }
        data['func'] = make_prototype(func, '__global__', '__kernel_' + name, info)
        data['body'] = indent_source(level+1, _kernel_body % bodydata)
        source += _func_template % data

        # Reset function context
        self._func_name = None

        return source

    def get_block_init(self, name, dim, dimlength, overlapping, center_on_origin):
        block = 'BLOCK(%d)' % dim
        if overlapping:
            if center_on_origin:
                limit = '__%s_shape%d' % (name, dim)
                shift = '(%s - %s/2)' % (block, dimlength)
            else:
                limit = '__%s_shape%d - (%s - 1)' % (name, dim, dimlength)
                shift = block
        else:
            limit = '__%s_shape%d/%s' % (name, dim, dimlength)
            shift = '%s * %s' % (block, dimlength)
        return block, limit, shift

    def gen_call(self, call):
        assert call.starargs is None
        assert call.kwargs is None
        info = _gpu_funcs[self.func_name]
        types = info['types']
        name = self.parse(call.func, 'varaccess')
        args = []
        for arg in [self.parse(arg, 'op') for arg in call.args]:
            args.append(arg)
            typeinfo = types.get(arg)
            if typeinfo and typeinfo[1].endswith('Array'):
                args.extend(self._get_dim_args(arg))
            elif arg == 'NULL':
                args.extend(3 * ('0',))
        if '->' in name:
            arg, name = name.split('->', 1)
            if info['types'][arg][2].name == 'float32':
                name = 'f' + name
            shape = info['blockshapes'].get(arg) or info['threadmemory'].get(arg)
            name += '%dd' % len(shape)
            dimargs = ', '.join(self._get_dim_args(arg))
            args.insert(0, '%s, %s' % (arg, dimargs))
            args.extend(str(arg) for arg in shape)
            name = '__array_' + name
        elif name in ('int', 'float', 'min', 'max', 'sqrt', 'log', 'abs'):
            # These functions are specialized via templates
            name = '__py_' + name
        return '%s(%s)' % (name, ', '.join(args))

    def _get_dim_args(self, arg):
        info = _gpu_funcs[self.func_name]
        threadmemory = info['threadmemory']
        shape = threadmemory.get(arg)
        if shape:
            shape += (3 - len(shape)) * (1,)
            return ('%d' % dim for dim in shape)
        return ('__%s_shape%d' % (arg, dim) for dim in range(3))

    def gen_for(self, node, level):
        assert not node.orelse, 'else clause not supported in for loops'
        rangespec = tuple(self.parse(node.iter, 'range'))
        length = len(rangespec)
        assert length in range(1, 4), 'range() must get 1-3 parameters'
        if length == 3:
            start, stop, step = rangespec
        elif length == 2:
            start, stop = rangespec
            step = 1
        else:
            start, step = 0, 1
            stop = rangespec[0]
        return _for_loop % {
            'name': self.parse(node.target, 'name'),
            'body': self.parse(node.body, 'body', level+1),
            'indent': indent(level),
            'start': start, 'stop': stop, 'step': step,
        }

    def gen_while(self, node, level):
        assert not node.orelse, 'else clause not supported in for loops'
        return _while_loop % {
            'test': self.parse(node.test, 'op'),
            'body': self.parse(node.body, 'body', level+1),
            'indent': indent(level),
        }
