"""
Python ast to C++ converter
"""
import ast
from pymeta.grammar import OMeta
from pymeta.runtime import ParseError, EOFError

# Globals for storing type information about registered functions
_blockwise_funcs = {}

class Py2GPUParseError(ValueError):
    pass

py2gpu_grammar = r'''
node :name = :n ?(n.__class__.__name__ == name) -> n

add = node('Add') -> '+'
binop = node('BinOp'):n -> '(%s %s %s)' % (parse(n.left, 'op'), parse(n.op, 'op'), parse(n.right, 'op'))
div = node('Div') -> '/'
mult = node('Mult') -> '*'
name = node('Name'):n -> n.id
num = node('Num'):n -> str(n.n)
sub = node('Sub') -> '-'

subscript = node('Subscript'):n !(parse(n.value, 'name')):name
    -> '%s->data[%s]' % (name, self.gen_subscript(name, parse(n.slice, 'index')))
index = node('Index'):n -> parse(n.value, 'subscriptindex')
subscriptindex = op:n -> [n]
               | tupleslice
tupleslice = node('Tuple'):n -> [parse(index, 'op') for index in n.elts]

op = add
   | binop
   | div
   | mult
   | name
   | num
   | sub
   | subscript

assign = node('Assign'):n ?(len(n.targets) == 1) -> '%s = %s' % (parse(n.targets[0], 'op'), parse(n.value, 'op'))
expr = node('Expr'):n -> parse(n.value, 'op')
functiondef 0 = node('FunctionDef'):n -> self.gen_func(n, 0)

bodyitem :i = (assign
               | expr):n -> indent(i) + n + ';'
            | functiondef(i)
body :i = bodyitem(i)+:xs -> '\n'.join(xs) + '\n'

grammar = node('Module'):n -> parse(n.body, 'body', 0)
'''

def convert(code):
    if isinstance(code, ast.AST):
        tree = code
    else:
        tree = ast.parse(code)
    try:
        converted = parse(tree)
    except ParseError, e:
        lines = code.split('\n')
        start, stop = max(0, e.position-1), min(len(lines), e.position+2)
        snippet = '\n'.join(lines[start:stop])
        raise Py2GPUParseError('Parse error at line %d (%s):\n%s' % (e.position, str(e), snippet))
    return converted

def raise_parse_error(node, error, message=''):
    lineno = getattr(node, 'lineno', 1)
    col_offset = getattr(node, 'col_offset', 1)
    if message:
        message = ': ' + message
    raise ParseError(lineno, error, 'Parse error at line %d, col %d (node %s)%s' % (lineno, col_offset, node.__class__.__name__, message))

def parse(data, rule='grammar', *args):
    # print data, rule
    if not isinstance(data, (tuple, list)):
        data = (data,)
    try:
        grammar = Py2GPUGrammar(data)
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

def indent(level):
    return level * '    '

def p(*x):
    print x

vars = {
    'parse': parse,
    'indent': indent,
    'p': p,
}

class Py2GPUGrammar(OMeta.makeGrammar(py2gpu_grammar, vars, name="Py2CGrammar")):
    def gen_subscript(self, name, indices):
        dims = len(indices)
        access = []
        for dim, index in enumerate(indices):
            access.append(' * '.join(['%s->dims[%d]' % (name, subdim)
                                      for subdim in range(dim+1, dims)] + [index]))
        return ' + '.join(access)

    def gen_func(self, func, level):
        name = func.name
        args = [arg.id for arg in func.args.args]
        types = _blockwise_funcs[name]['types']
        assert func.args.vararg is None
        assert func.args.kwarg is None
        assert func.args.defaults == []
        return_type = 'void'
        typed_args = ['%s %s' % (types[arg], arg) for arg in args]
        body = '%s%s %s(%s) {\n%s%s}' % (indent(level), return_type, name,
            ', '.join(typed_args),
            parse(func.body, 'body', level+1),
            indent(level),
        )
        return body
