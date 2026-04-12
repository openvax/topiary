"""String parser for the DSL.

Grammar (lowest precedence first)::

    top      := or_expr
    or_expr  := and_expr ('|' and_expr)*
    and_expr := cmp_expr ('&' cmp_expr)*
    cmp_expr := arith (CMP arith)?
    arith    := term (('+'|'-') term)*
    term     := power (('*'|'/') power)*
    power    := unary ('**' power)?
    unary    := ('-' | '~') unary | postfix
    postfix  := atom ('.' IDENT call? | '[' STRING (',' STRING)? ']')*
    atom     := NUMBER | '(' top ')' | abs(expr) | agg(expr,...)
              | count(STR) | column(IDENT) | len
              | CONTEXT '.' scoped_atom | kind_ref | IDENT
"""

from __future__ import annotations

import operator
import re
from difflib import get_close_matches

from .nodes import (
    Affinity,
    BinOp,
    BoolOp,
    Column,
    Comparison,
    Const,
    Count,
    DSLNode,
    KindAccessor,
    Len,
    Presentation,
    Processing,
    Stability,
    _CONTEXT_KEYWORDS,
    _FIELD_ALIASES,
    _combine_bool,
    _resolve_qualified_kind,
    geomean,
    maximum,
    mean,
    median,
    minimum,
)

_AGGREGATION_FUNCS = {
    "mean": mean,
    "geomean": geomean,
    "minimum": minimum,
    "maximum": maximum,
    "median": median,
}

_TRANSFORM_NAMES = {
    "ascending_cdf", "descending_cdf", "norm", "logistic",
    "clip", "hinge", "log", "log2", "log10", "log1p", "exp", "sqrt",
}

_KIND_ACCESSOR_ALIASES = {
    "affinity": Affinity,
    "presentation": Presentation,
    "stability": Stability,
    "processing": Processing,
    "ba": Affinity,
    "aff": Affinity,
    "ic50": Affinity,
    "el": Presentation,
}


class _Tokenizer:
    """Tokenizer for the DSL."""

    def __init__(self, text):
        self.text = text
        self.pos = 0
        self.tokens = []
        self._tokenize()
        self._idx = 0

    def _tokenize(self):
        i = 0
        text = self.text
        while i < len(text):
            if text[i].isspace():
                i += 1
                continue
            m = re.match(r'(\d+\.?\d*([eE][+-]?\d+)?)', text[i:])
            if m:
                self.tokens.append(("NUMBER", float(m.group())))
                i += m.end()
                continue
            if text[i] in ('"', "'"):
                quote = text[i]
                j = i + 1
                while j < len(text) and text[j] != quote:
                    j += 1
                if j >= len(text):
                    raise ValueError(f"Unterminated string at position {i}")
                self.tokens.append(("STRING", text[i + 1:j]))
                i = j + 1
                continue
            if text[i].isalpha() or text[i] == '_':
                j = i
                while j < len(text) and (text[j].isalnum() or text[j] == '_'):
                    j += 1
                self.tokens.append(("IDENT", text[i:j]))
                i = j
                continue
            if i + 1 < len(text):
                two = text[i:i + 2]
                if two == '**':
                    self.tokens.append(("OP", "**"))
                    i += 2
                    continue
                if two in ('<=', '>=', '==', '!='):
                    self.tokens.append(("CMP", two))
                    i += 2
                    continue
            c = text[i]
            if c in '+-*/':
                self.tokens.append(("OP", c))
            elif c in '<>':
                self.tokens.append(("CMP", c))
            elif c in '|&':
                self.tokens.append(("BOOLOP", c))
            elif c == '~':
                self.tokens.append(("TILDE", c))
            elif c == '(':
                self.tokens.append(("LPAREN", c))
            elif c == ')':
                self.tokens.append(("RPAREN", c))
            elif c == '[':
                self.tokens.append(("LBRACKET", c))
            elif c == ']':
                self.tokens.append(("RBRACKET", c))
            elif c == '.':
                self.tokens.append(("DOT", c))
            elif c == ',':
                self.tokens.append(("COMMA", c))
            else:
                raise ValueError(
                    f"Unexpected character {c!r} at position {i} in {self.text!r}"
                )
            i += 1
        self.tokens.append(("EOF", None))

    def peek(self):
        return self.tokens[self._idx]

    def advance(self):
        tok = self.tokens[self._idx]
        self._idx += 1
        return tok

    def expect(self, ttype, value=None):
        tok = self.advance()
        if tok[0] != ttype:
            raise ValueError(
                f"Expected {ttype} but got {tok[0]} ({tok[1]!r}) in {self.text!r}"
            )
        if value is not None and tok[1] != value:
            raise ValueError(
                f"Expected {value!r} but got {tok[1]!r} in {self.text!r}"
            )
        return tok


def _parser_as_node(x):
    """Coerce a parser result to a DSLNode."""
    if isinstance(x, DSLNode):
        return x
    if isinstance(x, KindAccessor):
        return x.value
    if isinstance(x, (int, float, bool)):
        return Const(float(x))
    raise TypeError(f"Cannot convert {type(x).__name__} to DSLNode")


def _as_bool_node(node):
    """Coerce a parser node to a DSLNode suitable for BoolOp children."""
    if isinstance(node, KindAccessor):
        node = node.value
    if not isinstance(node, DSLNode):
        raise ValueError(
            f"Boolean operator applied to non-expression {type(node).__name__}"
        )
    return node


class _Parser:
    """Recursive-descent parser for the full DSL (arithmetic + booleans)."""

    def __init__(self, text):
        self.tokenizer = _Tokenizer(text)
        self.text = text

    def parse(self) -> DSLNode:
        node = self._or()
        tok = self.tokenizer.peek()
        if tok[0] != "EOF":
            raise ValueError(
                f"Unexpected token {tok[1]!r} after expression in {self.text!r}"
            )
        if isinstance(node, KindAccessor):
            return node.value
        return node

    def _or(self):
        left = self._and()
        while self.tokenizer.peek() == ("BOOLOP", "|"):
            self.tokenizer.advance()
            right = self._and()
            left = _combine_bool(operator.or_, _as_bool_node(left), _as_bool_node(right))
        return left

    def _and(self):
        left = self._cmp()
        while self.tokenizer.peek() == ("BOOLOP", "&"):
            self.tokenizer.advance()
            right = self._cmp()
            left = _combine_bool(operator.and_, _as_bool_node(left), _as_bool_node(right))
        return left

    def _cmp(self):
        left = self._arith()
        tok = self.tokenizer.peek()
        if tok[0] == "CMP":
            self.tokenizer.advance()
            right = self._arith()
            op_map = {
                "<=": operator.le, ">=": operator.ge,
                "<": operator.lt, ">": operator.gt,
                "==": operator.eq, "!=": operator.ne,
            }
            op = op_map[tok[1]]
            return Comparison(_parser_as_node(left), op, _parser_as_node(right))
        return left

    def _arith(self):
        left = self._term()
        while self.tokenizer.peek() in (("OP", "+"), ("OP", "-")):
            op_tok = self.tokenizer.advance()
            right = self._term()
            left_node = _parser_as_node(left)
            right_node = _parser_as_node(right)
            if op_tok[1] == "+":
                left = BinOp(left_node, right_node, operator.add)
            else:
                left = BinOp(left_node, right_node, operator.sub)
        return left

    def _term(self):
        left = self._power()
        while self.tokenizer.peek() in (("OP", "*"), ("OP", "/")):
            op_tok = self.tokenizer.advance()
            right = self._power()
            left_node = _parser_as_node(left)
            right_node = _parser_as_node(right)
            if op_tok[1] == "*":
                left = BinOp(left_node, right_node, operator.mul)
            else:
                left = BinOp(left_node, right_node, operator.truediv)
        return left

    def _power(self):
        base = self._unary()
        if self.tokenizer.peek() == ("OP", "**"):
            self.tokenizer.advance()
            exp = self._power()
            base = BinOp(_parser_as_node(base), _parser_as_node(exp), operator.pow)
        return base

    def _unary(self):
        tok = self.tokenizer.peek()
        if tok == ("OP", "-"):
            self.tokenizer.advance()
            inner = self._unary()
            return BinOp(Const(-1), _parser_as_node(inner), operator.mul)
        if tok[0] == "TILDE":
            self.tokenizer.advance()
            inner = self._unary()
            return BoolOp(operator.invert, [_as_bool_node(inner)])
        return self._postfix()

    def _postfix(self):
        node = self._atom()
        while True:
            tok = self.tokenizer.peek()
            if tok[0] == "DOT":
                self.tokenizer.advance()
                name_tok = self.tokenizer.expect("IDENT")
                name = name_tok[1]
                if self.tokenizer.peek()[0] == "LPAREN":
                    args = self._call_args()
                    node = self._apply_transform(node, name, args)
                else:
                    node = self._apply_field_access(node, name)
            elif tok[0] == "LBRACKET":
                self.tokenizer.advance()
                method_tok = self.tokenizer.expect("STRING")
                version = None
                if self.tokenizer.peek()[0] == "COMMA":
                    self.tokenizer.advance()
                    version_tok = self.tokenizer.expect("STRING")
                    version = version_tok[1]
                self.tokenizer.expect("RBRACKET")
                node = self._apply_bracket(node, method_tok[1], version)
            else:
                break
        return node

    def _atom(self):
        tok = self.tokenizer.peek()
        if tok[0] == "NUMBER":
            self.tokenizer.advance()
            return Const(tok[1])
        if tok[0] == "LPAREN":
            self.tokenizer.advance()
            expr = self._or()
            self.tokenizer.expect("RPAREN")
            return expr
        if tok[0] == "IDENT":
            name = tok[1].lower()
            if name == "abs":
                self.tokenizer.advance()
                self.tokenizer.expect("LPAREN")
                inner = self._or()
                self.tokenizer.expect("RPAREN")
                return abs(_parser_as_node(inner))
            if name in _AGGREGATION_FUNCS:
                self.tokenizer.advance()
                args = self._call_args()
                return _AGGREGATION_FUNCS[name](*args)
            if name == "len":
                self.tokenizer.advance()
                return Len()
            if name == "count":
                self.tokenizer.advance()
                self.tokenizer.expect("LPAREN")
                chars_tok = self.tokenizer.expect("STRING")
                self.tokenizer.expect("RPAREN")
                return Count(chars_tok[1])
            if name in _CONTEXT_KEYWORDS:
                self.tokenizer.advance()
                if self.tokenizer.peek()[0] == "DOT":
                    self.tokenizer.advance()
                    scope = name + "_"
                    return self._scoped_atom(scope)
                raise ValueError(
                    f"{name!r} is a reserved context keyword. "
                    f"Use '{name}.kind.field' syntax, e.g. '{name}.affinity.score'"
                )
            if name == "column":
                self.tokenizer.advance()
                self.tokenizer.expect("LPAREN")
                col_tok = self.tokenizer.expect("IDENT")
                self.tokenizer.expect("RPAREN")
                return Column(col_tok[1])
            if self._is_kind_name(name):
                return self._kind_accessor()
            self.tokenizer.advance()
            return Column(tok[1])
        raise ValueError(f"Unexpected token {tok!r} in expression {self.text!r}")

    def _scoped_atom(self, scope):
        tok = self.tokenizer.peek()
        if tok[0] != "IDENT":
            raise ValueError(
                f"Expected identifier after scope prefix in {self.text!r}"
            )
        name = tok[1].lower()
        if name == "len":
            self.tokenizer.advance()
            return Len(scope=scope)
        if name == "count":
            self.tokenizer.advance()
            self.tokenizer.expect("LPAREN")
            chars_tok = self.tokenizer.expect("STRING")
            self.tokenizer.expect("RPAREN")
            return Count(chars_tok[1], scope=scope)
        return self._kind_accessor(scope=scope)

    def _kind_accessor(self, scope=""):
        name_tok = self.tokenizer.expect("IDENT")
        name = name_tok[1].lower()
        if name in _CONTEXT_KEYWORDS:
            raise ValueError(
                f"{name!r} is a reserved context keyword and cannot be "
                f"used as a kind name in {self.text!r}"
            )
        if name not in _KIND_ACCESSOR_ALIASES:
            kind, method = _resolve_qualified_kind(name)
            accessor = KindAccessor(kind, method=method, scope=scope)
        else:
            accessor = KindAccessor(
                _KIND_ACCESSOR_ALIASES[name].kind, scope=scope,
            )
        if self.tokenizer.peek()[0] == "LBRACKET":
            self.tokenizer.advance()
            method_tok = self.tokenizer.expect("STRING")
            version = None
            if self.tokenizer.peek()[0] == "COMMA":
                self.tokenizer.advance()
                version_tok = self.tokenizer.expect("STRING")
                version = version_tok[1]
            self.tokenizer.expect("RBRACKET")
            accessor = accessor[method_tok[1], version] if version else accessor[method_tok[1]]
        return accessor

    def _call_args(self):
        self.tokenizer.expect("LPAREN")
        args = []
        if self.tokenizer.peek()[0] != "RPAREN":
            args.append(self._or())
            while self.tokenizer.peek()[0] == "COMMA":
                self.tokenizer.advance()
                args.append(self._or())
        self.tokenizer.expect("RPAREN")
        return args

    def _apply_transform(self, node, name, args):
        name_lower = name.lower()
        if name_lower not in _TRANSFORM_NAMES:
            available = sorted(_TRANSFORM_NAMES)
            close = get_close_matches(name_lower, available, n=3, cutoff=0.6)
            msg = f"Unknown transform {name!r}."
            if close:
                msg += f" Did you mean: {close}?"
            else:
                msg += f" Available: {available}"
            raise ValueError(msg)
        if isinstance(node, KindAccessor):
            method = getattr(node, name_lower)
            float_args = [a.val if isinstance(a, Const) else a for a in args]
            return method(*float_args)
        if not isinstance(node, DSLNode):
            raise ValueError(
                f"Cannot apply .{name}() to {type(node).__name__}"
            )
        method = getattr(node, name_lower, None)
        if method is None:
            raise ValueError(f"DSLNode has no method {name!r}")
        float_args = [a.val if isinstance(a, Const) else a for a in args]
        return method(*float_args)

    def _apply_field_access(self, node, name):
        name_lower = name.lower()
        if name_lower in _TRANSFORM_NAMES:
            return self._apply_transform(node, name, [])
        if isinstance(node, KindAccessor):
            field_name = _FIELD_ALIASES.get(name_lower)
            if field_name is not None:
                if field_name == "value":
                    return node.value
                if field_name == "percentile_rank":
                    return node.rank
                if field_name == "score":
                    return node.score
            raise ValueError(
                f"Unknown field {name!r}. Available: value, rank, score"
            )
        raise ValueError(f"Cannot access .{name} on {type(node).__name__}")

    def _apply_bracket(self, node, method, version):
        if isinstance(node, KindAccessor):
            if version is None:
                return node[method]
            return node[method, version]
        raise ValueError(f"Cannot use ['...'] on {type(node).__name__}")

    def _is_kind_name(self, name):
        if name in _KIND_ACCESSOR_ALIASES:
            return True
        try:
            _resolve_qualified_kind(name)
            return True
        except ValueError:
            return False


def parse(text: str) -> DSLNode:
    """Parse a DSL string into a :class:`DSLNode`.

    Supports the full grammar: arithmetic, comparisons, boolean
    combinators, transforms, aggregations, scoped fields.
    """
    return _Parser(text).parse()
