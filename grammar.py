"""
Python ast to C++ converter
"""
import ast
from pymeta.grammar import OMeta
from pymeta.runtime import ParseError, EOFError

# Globals for storing type information about registered functions
_gpu_funcs = {}
_no_bounds_check = object()

class Py2GPUParseError(ValueError):
    pass

py2gpu_grammar = r'''
node :name = :n ?(n.__class__.__name__ == name) -> n

attribute = node('Attribute'):n -> '%s->%s' % (self.parse(n.value, 'name'), n.attr)
name = node('Name'):n -> name_mapper.get(n.id, n.id)
varaccess = attribute | name
num = node('Num'):n -> str(n.n)
anyvar = varaccess | num

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
   | varaccess
   | num
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

def p(*x):
    print x

name_mapper = {
    'None': 'null',
    'True': 'true',
    'False': 'false',
}

vars = {
    'indent': indent,
    'p': p,
    'name_mapper': name_mapper,
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

_array_def = r'''
%(type)sStruct __array_%(name)s;
__array_%(name)s.data = %(name)s;
for (int i=0; i < NDIMS; i++) {
    __array_%(name)s.dim[i] = __dim_%(name)s[i];
    __array_%(name)s.offset[i] = 0;
}
'''.lstrip()

_offset_template = r'''
__numblocks = (%(dims)s) / (%(size)s);
__blockpos = __rest / __numblocks;
%(name)s.offset[%(dim)d] = __blockpos * %(dimlength)d;
__rest -= __blockpos * __numblocks;
'''.strip();

# TODO: Maximize processor usage instead of thread usage!
_kernel_body = r'''
int __gpu_thread_index = CPU_INDEX * THREAD_COUNT + THREAD_INDEX;
int __block, __blocks_per_thread, __end, __rest, __blockpos, __numblocks;

%(declarations)s

__blocks_per_thread = (__count + __total_threads - 1) / __total_threads;
__block = __gpu_thread_index * __blocks_per_thread;
__end = __block + __blocks_per_thread;

if (__end > __count)
    __end = __count;

%(threadmemory)s

for (; __block < __end; __block++) {
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
        funcargs = []
        for arg in args:
            kind = types[arg][1]
            if func_type == '__global__' and kind.endswith('Array'):
                funcargs.append('%s *%s' % (types[arg][3], arg))
                funcargs.append('int *__dim_%s' % arg)
            else:
                funcargs.append('%s %s' % (kind, arg))

        data = {
            'args': ', '.join(funcargs),
            'name': name,
            'func_type': func_type,
            'return_type': types.get('return', (0, 'void'))[1],
        }
        if func_type == '__global__':
             data['args'] += ', int __count, int __total_threads'
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
            return '%s[%s]' % (name, indices[0])
        access = []
        shifted_indices = []
        for dim, index in enumerate(indices):
            index = '(%s->offset[%d] + %s)' % (name, dim, index)
            shifted_indices.append(index)
            access.append(' * '.join(['%s->dim[%d]' % (name, subdim)
                                      for subdim in range(dim+1, dims)] + [index]))
        subscript = '%s->data[%s]' % (name, ' + '.join(access))

        # Check bounds, if necessary
        info = _gpu_funcs[self.func_name]
        default = info['out_of_bounds_value']
        if default is not _no_bounds_check:
            blockshape = info['blockshapes'].get(name)
            if blockshape is not None and blockshape != (1,) * len(blockshape):
                bounds_check = ' && '.join('%s %s' % (index, check)
                                           for check in ('>= 0', '< %s->dim[%d]' % (name, dim))
                                           for dim, index in enumerate(shifted_indices))
                if assigning:
                    subscript = 'if (%s) %s' % (bounds_check, subscript)
                else:
                    subscript = '(%s ? %s : %s)' % (bounds_check, subscript, default)
        return subscript

    def gen_func(self, func, level):
        name = func.name
        info = _gpu_funcs[name]

        # Store function name, so it can be reused in other parser rules
        self._func_name = name
        info['funcnode'] = func

        types = info['types']
        args = set(arg.id for arg in func.args.args)
        vars = set(types.keys()).symmetric_difference(args)
        vars = '\n'.join('%s %s;' % (types[var][1], var) for var in vars
                         if var != 'return')
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
        offsetinit = []
        threadmeminit = []
        blockshapes = info['blockshapes']
        overlapping = info['overlapping']
        threadmemory = info['threadmemory']
        center_as_origin = info['center_as_origin']
        args = []
        for arg in func.args.args:
            origarg = arg = arg.id
            kind = types[arg][1]
            if kind.endswith('Array'):
                blockinit.append(_array_def % {'name': arg, 'type': kind})
                origarg = arg
                arg = '__array_' + arg

            shape = threadmemory.get(origarg)
            if shape:
                threadmeminit.append('%s.data += __gpu_thread_index * %s;' % (
                                     arg, ' * '.join(str(dim) for dim in shape)))
                threadmeminit.append(' = '.join('%s.offset[%d]' % (arg, dim)
                                                for dim in range(len(shape)))
                                     + ' = 0;')
                threadmeminit.extend('%s.dim[%d] = %s;' % (arg, dim, dimlength)
                                     for dim, dimlength in enumerate(shape))

            shape = blockshapes.get(origarg)
            if shape:
                offsetinit.append('__rest = __block;')
                for dim, dimlength in enumerate(shape):
                    if origarg in overlapping:
                        dimlength = 1
                    if dim == len(shape) - 1:
                        # TODO: instead of having to add offset to calculate
                        # block coordinates we could manipulate the data
                        # pointer directly and only use the offset for
                        # bounds checking (if needed, at all)
                        offsetinit.append('%s.offset[%d] = __rest * %d;\n' % (
                            arg, dim, dimlength))
                        break
                    if origarg in overlapping and not center_as_origin:
                        dims = ' * '.join('(%s.dim[%d] - (%s - 1))' % (arg, subdim, shape[subdim])
                                          for subdim in range(dim+1, len(shape)))
                    else:
                        dims = ' * '.join('%s.dim[%d]' % (arg, subdim)
                                          for subdim in range(dim+1, len(shape)))
                    if origarg in overlapping:
                        size = '1'
                    else:
                        size = ' * '.join('%d' % shape[subdim]
                                          for subdim in range(dim+1, len(shape)))
                    offsetinit.append(_offset_template % {'name': arg, 'dim': dim,
                                                          'dims': dims, 'size': size,
                                                          'dimlength': dimlength})

            if kind.endswith('Array'):
                args.append('&' + arg)
                continue
            args.append(arg)
        bodydata = {
            'declarations': '%s' % ''.join(blockinit),
            'offset': indent_source(level + 1, '\n'.join(offsetinit)).lstrip(),
            'threadmemory': indent_source(level, '\n'.join(threadmeminit)).lstrip(),
            'call': '%s(%s);' % (func.name, ', '.join(args)),
        }
        data['func'] = make_prototype(func, '__global__', '__kernel_' + name, info)
        data['body'] = indent_source(level+1, _kernel_body % bodydata)
        source += _func_template % data

        # Reset function context
        self._func_name = None

        return source

    def gen_call(self, call):
        name = self.parse(call.func, 'varaccess')
        args = [self.parse(arg, 'op') for arg in call.args]
        if '->' in name:
            arg, name = name.split('->', 1)
            info = _gpu_funcs[self.func_name]
            shape = info['blockshapes'].get(arg) or info['threadmemory'].get(arg)
            if info['types'][arg][2].name == 'float32':
                name = 'f' + name
            name += '%dd' % len(shape)
            args.insert(0, arg)
            args.extend(str(arg) for arg in shape)
            name = '__array_' + name
        elif name in ('int', 'float'):
            name = '__' + name
        elif name in ('min', 'max'):
            name = 'f' + name
        assert call.starargs is None
        assert call.kwargs is None
        assert call.keywords == []
        return '%s(%s)' % (name, ', '.join(args))

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
