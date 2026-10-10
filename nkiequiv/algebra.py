"""Canonical forms for real-valued kernel expressions.

Every value a kernel computes is represented as a ``Poly``: a sum of
``coefficient * monomial`` terms, where a monomial is a product of atoms raised
to (possibly fractional or negative) exponents. Atoms are hash-consed, so two
structurally identical expressions are the same object and equality of two
``Poly`` values is a dictionary comparison.

The constructors below apply only rewrites that are valid identities over the
reals (on the domain where both sides are defined). The ones that matter for
kernel optimizations are:

* commutativity / associativity / distributivity (polynomial normal form), so
  reordered or tiled reductions compare equal;
* ``exp(a) * exp(b) = exp(a + b)``, and a shift normalization that factors
  ``exp(mean of exponents)`` out of sums of exponentials, so ``softmax`` with or
  without max-subtraction and online-softmax rescaling compare equal;
* ``max`` flattening, deduplication and shift / positive-scale extraction, so
  running maxima over tiles equal a single max; ``min`` and ``abs`` are
  expressed via ``max``;
* powers of a normalized base (``sqrt``, ``rsqrt``, ``reciprocal``) share one
  atom with rational exponents, so ``rsqrt(v) * sqrt(v) * x / v`` style rewrites
  cancel.

Two rules are approximations and are recorded in ``ASSUMPTIONS`` whenever they
fire: constants at or below ``-NEG_INF_THRESHOLD`` are treated as -infinity
inside ``max`` (they are dropped) and inside ``exp`` (the result is 0). Kernels
use such constants for masking and running-max initialization.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Iterable

NEG_INF_THRESHOLD = Fraction(1000)
# Stand-in for +-inf literals (larger than any float32).
BIG = Fraction(2) ** 128

ASSUMPTIONS: set[str] = set()


def set_neg_inf_threshold(value: float) -> None:
    global NEG_INF_THRESHOLD
    NEG_INF_THRESHOLD = Fraction(value)


def _num(c) -> int | Fraction:
    """Normalize a numeric coefficient to int when integral."""
    if isinstance(c, int):
        return c
    if isinstance(c, Fraction):
        return c.numerator if c.denominator == 1 else c
    if isinstance(c, float):
        if math.isinf(c):
            ASSUMPTIONS.add("infinite literals are modelled as +-2**128")
            return int(BIG) if c > 0 else -int(BIG)
        if math.isnan(c):
            raise ValueError("NaN literal in kernel")
        return _num(Fraction(c))
    if hasattr(c, "item"):  # numpy scalar
        return _num(c.item())
    return _num(Fraction(c))


# --------------------------------------------------------------------------
# Atoms
# --------------------------------------------------------------------------


class Atom:
    """A hash-consed non-polynomial building block.

    kinds:
      var    (name, index)            an input tensor element
      undef  (name, index, serial)    an uninitialized memory element
      exp    (Poly,)                  exponential, at most one per monomial
      base   (Poly,)                  a normalized multi-term base; its exponent
                                      in a monomial may be fractional/negative
      max    (frozenset[Poly],)       maximum of >= 2 arguments
      ind    (rel, Poly)              indicator [P rel 0], rel in gt/ge/eq
      log    (Poly,)
      fn     (name, tuple[Poly])      uninterpreted function application
    """

    __slots__ = ("kind", "args", "id", "has_undef", "__weakref__")

    def __init__(self, kind, args, ident):
        self.kind = kind
        self.args = args
        self.id = ident
        self.has_undef = kind == "undef" or any(
            p.has_undef for p in _atom_polys(kind, args)
        )

    def __repr__(self):
        return atom_str(self)


def _atom_polys(kind, args) -> Iterable["Poly"]:
    if kind in ("exp", "base", "log"):
        return (args[0],)
    if kind == "max":
        return args[0]
    if kind == "ind":
        return (args[1],)
    if kind == "fn":
        return args[1]
    return ()


_ATOMS: dict = {}
_next_id = 0


def _atom(kind, args) -> Atom:
    global _next_id
    key = (kind, args)
    a = _ATOMS.get(key)
    if a is None:
        a = Atom(kind, args, _next_id)
        _next_id += 1
        _ATOMS[key] = a
    return a


def reset_tables() -> None:
    """Drop all interned atoms (call between unrelated checks to free memory)."""
    global _next_id
    _ATOMS.clear()
    _next_id = 0
    ASSUMPTIONS.clear()


# --------------------------------------------------------------------------
# Poly
# --------------------------------------------------------------------------

# A monomial is a tuple of (Atom, exponent) sorted by atom id. The empty tuple
# is the constant monomial.
Mono = tuple


class Poly:
    __slots__ = ("terms", "_hash", "_undef")

    def __init__(self, terms: dict):
        self.terms = terms
        self._hash = None
        self._undef = None

    # -- identity --------------------------------------------------------
    def __hash__(self):
        h = self._hash
        if h is None:
            h = self._hash = hash(frozenset(self.terms.items()))
        return h

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, Poly):
            return NotImplemented
        return hash(self) == hash(other) and self.terms == other.terms

    @property
    def has_undef(self) -> bool:
        u = self._undef
        if u is None:
            u = self._undef = any(a.has_undef for m in self.terms for a, _ in m)
        return u

    # -- queries ---------------------------------------------------------
    def is_const(self) -> bool:
        t = self.terms
        return not t or (len(t) == 1 and () in t)

    def const_value(self):
        return self.terms.get((), 0)

    def atoms(self) -> set:
        out = set()
        stack = [self]
        seen = set()
        while stack:
            p = stack.pop()
            for m in p.terms:
                for a, _ in m:
                    if a.id in seen:
                        continue
                    seen.add(a.id)
                    out.add(a)
                    stack.extend(_atom_polys(a.kind, a.args))
        return out

    # -- operators (so numpy object arrays work) --------------------------
    def __add__(self, o):
        return add(self, lift(o))

    __radd__ = __add__

    def __sub__(self, o):
        return add(self, neg(lift(o)))

    def __rsub__(self, o):
        return add(lift(o), neg(self))

    def __mul__(self, o):
        return mul(self, lift(o))

    __rmul__ = __mul__

    def __truediv__(self, o):
        return mul(self, power(lift(o), -1))

    def __rtruediv__(self, o):
        return mul(lift(o), power(self, -1))

    def __neg__(self):
        return neg(self)

    def __repr__(self):
        return poly_str(self)


ZERO = Poly({})
ONE = Poly({(): 1})


def const(c) -> Poly:
    c = _num(c)
    if c == 0:
        return ZERO
    if c == 1:
        return ONE
    return Poly({(): c})


def lift(x) -> Poly:
    if isinstance(x, Poly):
        return x
    return const(x)


def _mono_poly(mono, coef=1) -> Poly:
    return Poly({mono: coef})


def var(name: str, index: tuple) -> Poly:
    return _mono_poly(((_atom("var", (name, tuple(index))), 1),))


_undef_serial = 0


def undef(name: str, index: tuple) -> Poly:
    global _undef_serial
    _undef_serial += 1
    return _mono_poly(((_atom("undef", (name, tuple(index), _undef_serial)), 1),))


def add(a: Poly, b: Poly) -> Poly:
    if not b.terms:
        return a
    if not a.terms:
        return b
    if len(a.terms) < len(b.terms):
        a, b = b, a
    t = dict(a.terms)
    for m, c in b.terms.items():
        v = t.get(m)
        if v is None:
            t[m] = c
        else:
            v = v + c
            if v == 0:
                del t[m]
            else:
                t[m] = _num(v) if isinstance(v, Fraction) else v
    return Poly(t)


def add_many(polys: Iterable[Poly]) -> Poly:
    t: dict = {}
    for p in polys:
        for m, c in p.terms.items():
            t[m] = t.get(m, 0) + c
    return Poly({m: _num(c) for m, c in t.items() if c != 0})


def neg(a: Poly) -> Poly:
    return Poly({m: -c for m, c in a.terms.items()})


def scale(a: Poly, c) -> Poly:
    c = _num(c)
    if c == 0:
        return ZERO
    if c == 1:
        return a
    return Poly({m: _num(v * c) for m, v in a.terms.items()})


def sub(a: Poly, b: Poly) -> Poly:
    return add(a, neg(b))


# -- monomial multiplication ---------------------------------------------

_SPECIAL = ("exp", "base", "ind")


def _mono_mul(m1: Mono, m2: Mono) -> Poly | tuple:
    """Multiply two monomials.

    Returns either a plain monomial (tuple) or a Poly when normalization of
    special atoms produced a sum (e.g. sqrt(P)^2 = P) or a scaling constant.
    """
    if not m1:
        return m2
    if not m2:
        return m1
    out = []
    i = j = 0
    special = False
    n_exp = 0
    n1, n2 = len(m1), len(m2)
    while i < n1 and j < n2:
        a1, e1 = m1[i]
        a2, e2 = m2[j]
        if a1.id == a2.id:
            e = e1 + e2
            if e != 0:
                out.append((a1, _num(e) if isinstance(e, Fraction) else e))
            if a1.kind in _SPECIAL:
                special = True
            i += 1
            j += 1
        elif a1.id < a2.id:
            out.append(m1[i])
            if a1.kind == "exp":
                n_exp += 1
            i += 1
        else:
            out.append(m2[j])
            if a2.kind == "exp":
                n_exp += 1
            j += 1
    for k in range(i, n1):
        out.append(m1[k])
        if m1[k][0].kind == "exp":
            n_exp += 1
    for k in range(j, n2):
        out.append(m2[k])
        if m2[k][0].kind == "exp":
            n_exp += 1
    if special or n_exp > 1:
        return _normalize_mono(out)
    return tuple(out)


def _normalize_mono(factors: list) -> Poly | tuple:
    """Apply exp merging, indicator idempotence and base-power expansion."""
    rest = []
    exp_arg = None
    expand = []  # (Poly, integer power) to multiply in
    for a, e in factors:
        k = a.kind
        if k == "exp":
            term = scale(a.args[0], e)
            exp_arg = term if exp_arg is None else add(exp_arg, term)
        elif k == "ind":
            rest.append((a, 1) if e > 0 else (a, e))
        elif k == "base":
            if e >= 1:
                whole = int(e)  # floor for positive
                frac = e - whole
                expand.append((a.args[0], whole))
                if frac != 0:
                    rest.append((a, _num(frac)))
            else:
                rest.append((a, e))
        else:
            rest.append((a, e))
    ex = exp(exp_arg) if exp_arg is not None else ONE
    if ex is not ONE and len(ex.terms) == 1:
        (em, ec), = ex.terms.items()
        if ec == 1 and len(em) == 1 and em[0][0].kind == "exp" and not expand:
            rest.append(em[0])
            ex = ONE
    rest.sort(key=lambda ae: ae[0].id)
    if ex is not ONE or expand:
        p = _mono_poly(tuple(rest)) if rest else ONE
        if ex is not ONE:
            p = mul(p, ex)
        for base_poly, k in expand:
            p = mul(p, pow_int(base_poly, k))
        return p
    return tuple(rest)


def mul(a: Poly, b: Poly) -> Poly:
    if not a.terms or not b.terms:
        return ZERO
    if a is ONE:
        return b
    if b is ONE:
        return a
    if a.is_const():
        return scale(b, a.const_value())
    if b.is_const():
        return scale(a, b.const_value())
    t: dict = {}
    extra = []
    for m1, c1 in a.terms.items():
        for m2, c2 in b.terms.items():
            r = _mono_mul(m1, m2)
            c = c1 * c2
            if isinstance(r, tuple):
                t[r] = t.get(r, 0) + c
            else:
                extra.append(scale(r, c))
    p = Poly({m: _num(c) for m, c in t.items() if c != 0})
    for e in extra:
        p = add(p, e)
    return p


def dot(xs: Iterable[Poly], ys: Iterable[Poly]) -> Poly:
    """sum_i xs[i] * ys[i] without building intermediate Polys."""
    t: dict = {}
    extra = []
    for a, b in zip(xs, ys):
        if not a.terms or not b.terms:
            continue
        for m1, c1 in a.terms.items():
            for m2, c2 in b.terms.items():
                r = _mono_mul(m1, m2)
                c = c1 * c2
                if isinstance(r, tuple):
                    t[r] = t.get(r, 0) + c
                else:
                    extra.append(scale(r, c))
    p = Poly({m: _num(c) for m, c in t.items() if c != 0})
    for e in extra:
        p = add(p, e)
    return p


def pow_int(p: Poly, k: int) -> Poly:
    if k == 0:
        return ONE
    if k < 0:
        return power(p, k)
    result = ONE
    base = p
    while k:
        if k & 1:
            result = mul(result, base)
        k >>= 1
        if k:
            base = mul(base, base)
    return result


# --------------------------------------------------------------------------
# Normalization helpers
# --------------------------------------------------------------------------


def _mono_key(m: Mono):
    return tuple((a.id, e) for a, e in m)


def _leading(p: Poly):
    """The (mono, coef) of the canonical leading term (smallest monomial key)."""
    m = min(p.terms, key=_mono_key)
    return m, p.terms[m]


def _content(p: Poly) -> Fraction:
    """Positive rational content: gcd(numerators) / lcm(denominators)."""
    g = 0
    l = 1
    for c in p.terms.values():
        c = Fraction(c)
        g = math.gcd(g, abs(c.numerator))
        l = l * c.denominator // math.gcd(l, c.denominator)
    return Fraction(g, l) if g else Fraction(1)


def _split_exp(m: Mono):
    """Split a monomial into (exp argument or None, monomial without exp)."""
    for idx, (a, e) in enumerate(m):
        if a.kind == "exp":
            return scale(a.args[0], e), m[:idx] + m[idx + 1 :]
    return None, m


def _exp_shift(p: Poly):
    """If every term carries an exp factor, return mean of their exponents."""
    args = []
    for m in p.terms:
        arg, _ = _split_exp(m)
        if arg is None:
            # Constant / non-exp terms count as exp(0).
            arg = ZERO
        args.append(arg)
    if all(a is ZERO for a in args):
        return None
    total = add_many(args)
    return scale(total, Fraction(1, len(args)))


def _normalize_base(p: Poly):
    """p = content * exp(shift) * q, with q a canonical multi-term base."""
    content = _content(p)
    q = scale(p, 1 / content) if content != 1 else p
    shift = _exp_shift(q)
    if shift is not None and shift.terms:
        q = mul(q, exp(neg(shift)))
    else:
        shift = None
    return content, shift, q


def _sqrt_rational(c: Fraction):
    """Return (r, s) with c = r^2 * s, s square-free-ish integer."""
    num, den = c.numerator, c.denominator
    # sqrt(n/d) = sqrt(n*d)/d
    n = num * den
    r = 1
    s = n
    if n < 10**12:
        f = 2
        while f * f <= s:
            while s % (f * f) == 0:
                s //= f * f
                r *= f
            f += 1
    return Fraction(r, den), s


def power(p: Poly, e) -> Poly:
    """p ** e for rational e (sqrt/rsqrt/reciprocal/integer powers)."""
    e = Fraction(e)
    if e.denominator == 1 and e >= 0:
        return pow_int(p, int(e))
    if not p.terms:
        ASSUMPTIONS.add("division by zero produced an undefined value")
        return _mono_poly(((_atom("fn", ("div_by_zero", ())), 1),))
    if e.denominator not in (1, 2):
        return _fn("pow", (p, const(e)))
    if len(p.terms) == 1:
        (m, c), = p.terms.items()
        if e.denominator == 1:
            # Integer negative power of a single term.
            cpart = const(Fraction(c) ** int(e))
            mono = [(a, _num(x * e)) for a, x in m]
            return mul(cpart, _mono_mul_list(mono))
        # Half power of a single term: only split off safe factors.
        if c < 0:
            return _base_power(p, e)
        exp_arg, rest = _split_exp(m)
        out = _const_power(Fraction(c), e)
        if exp_arg is not None:
            out = mul(out, exp(scale(exp_arg, e)))
        if rest:
            out = mul(out, _base_power(_mono_poly(rest), e))
        return out
    content, shift, q = _normalize_base(p)
    out = _const_power(content, e)
    if shift is not None:
        out = mul(out, exp(scale(shift, e)))
    if e.denominator == 1:
        mono, lc = _leading(q)
        if lc < 0:
            q = neg(q)
            if int(e) % 2:
                out = neg(out)
    return mul(out, _base_power(q, e))


def _mono_mul_list(factors: list) -> Poly:
    factors = [(a, x) for a, x in factors if x != 0]
    factors.sort(key=lambda ae: ae[0].id)
    r = _normalize_mono(factors)
    return r if isinstance(r, Poly) else _mono_poly(r)


def _base_power(q: Poly, e: Fraction) -> Poly:
    if q.is_const():
        return _const_power(Fraction(q.const_value()), e)
    return _mono_mul_list([(_atom("base", (q,)), _num(e))])


def _const_power(c: Fraction, e: Fraction) -> Poly:
    if c == 1:
        return ONE
    if e.denominator == 1:
        return const(c ** int(e))
    # c^(k + 1/2) = c^k * sqrt(c)
    k = math.floor(e)
    r, s = _sqrt_rational(c)
    out = const(c**k * r)
    if s != 1:
        out = mul(out, _mono_poly(((_atom("base", (const(s),)), Fraction(1, 2)),)))
    return out


# --------------------------------------------------------------------------
# Transcendentals and piecewise functions
# --------------------------------------------------------------------------


def exp(p: Poly) -> Poly:
    if not p.terms:
        return ONE
    c0 = p.terms.get((), 0)
    if c0 <= -NEG_INF_THRESHOLD:
        ASSUMPTIONS.add(
            f"exp(x) with a constant offset <= -{NEG_INF_THRESHOLD} is treated as 0 "
            "(masking constants act as -inf)"
        )
        return ZERO
    if len(p.terms) == 1:
        (m, c), = p.terms.items()
        if c == 1 and len(m) == 1 and m[0][0].kind == "log" and m[0][1] == 1:
            return m[0][0].args[0]
    return _mono_poly(((_atom("exp", (p,)), 1),))


def log(p: Poly) -> Poly:
    if p.is_const():
        c = Fraction(p.const_value())
        if c == 1:
            return ZERO
        return _mono_poly(((_atom("log", (p,)), 1),))
    if len(p.terms) == 1:
        (m, c), = p.terms.items()
        exp_arg, rest = _split_exp(m)
        if exp_arg is not None and c > 0:
            out = add(log(const(c)), exp_arg)
            if rest:
                out = add(out, _mono_poly(((_atom("log", (_mono_poly(rest),)), 1),)))
            return out
    content, shift, q = _normalize_base(p)
    out = log(const(content)) if content != 1 else ZERO
    if shift is not None:
        out = add(out, shift)
    return add(out, _mono_poly(((_atom("log", (q,)), 1),)))


def _max_single(p: Poly):
    """If p == t + g * max(T) (g > 0, max appears once), return (t, g, atom)."""
    found = None
    for m, c in p.terms.items():
        for a, e in m:
            if a.kind == "max":
                if found is not None or len(m) != 1 or e != 1 or c <= 0:
                    return None
                found = (m, c, a)
    if found is None:
        return None
    m, g, a = found
    t = Poly({mm: cc for mm, cc in p.terms.items() if mm != m})
    return t, g, a


def maximum(args: Iterable[Poly]) -> Poly:
    flat: list[Poly] = []
    work = list(args)
    while work:
        p = work.pop()
        ms = _max_single(p)
        if ms is not None:
            t, g, a = ms
            for q in a.args[0]:
                work.append(add(t, scale(q, g)))
        else:
            flat.append(p)
    # Drop "-inf" constants and keep, for args differing only by a constant,
    # the one with the largest constant.
    by_rest: dict = {}
    for p in flat:
        c = p.terms.get((), 0)
        rest = Poly({m: v for m, v in p.terms.items() if m != ()}) if c else p
        cur = by_rest.get(rest)
        if cur is None or c > cur:
            by_rest[rest] = c
    nonconst = {r: c for r, c in by_rest.items() if r.terms}
    consts = [c for r, c in by_rest.items() if not r.terms]
    if nonconst and consts:
        kept = [c for c in consts if c > -NEG_INF_THRESHOLD]
        if len(kept) != len(consts):
            ASSUMPTIONS.add(
                f"max() arguments that are constants <= -{NEG_INF_THRESHOLD} are "
                "treated as -inf and dropped"
            )
        consts = kept
    items = [add(r, const(c)) for r, c in nonconst.items()]
    if consts:
        items.append(const(max(consts)))
    if not items:
        raise ValueError("max of nothing")
    if len(items) == 1:
        return items[0]
    # Shift extraction: terms common to all arguments.
    common = dict(items[0].terms)
    for p in items[1:]:
        common = {m: c for m, c in common.items() if p.terms.get(m) == c}
        if not common:
            break
    t = Poly(common)
    if common:
        items = [sub(p, t) for p in items]
    # Positive scale extraction.
    g = Fraction(0)
    lcm = 1
    for p in items:
        for c in p.terms.values():
            c = Fraction(c)
            g = math.gcd(int(g), abs(c.numerator)) if g else abs(c.numerator)
            lcm = lcm * c.denominator // math.gcd(lcm, c.denominator)
    g = Fraction(g, lcm) if g else Fraction(1)
    if g != 1:
        items = [scale(p, 1 / g) for p in items]
    atom = _atom("max", (frozenset(items),))
    return add(t, _mono_poly(((atom, 1),), _num(g)))


def minimum(args: Iterable[Poly]) -> Poly:
    return neg(maximum([neg(a) for a in args]))


def absolute(p: Poly) -> Poly:
    return maximum([p, neg(p)])


def relu(p: Poly) -> Poly:
    return maximum([p, ZERO])


def compare(rel: str, a: Poly, b: Poly) -> Poly:
    """Indicator [a rel b] as a 0/1-valued Poly. rel in gt ge lt le eq ne."""
    if rel == "lt":
        return compare("gt", b, a)
    if rel == "le":
        return compare("ge", b, a)
    if rel == "ne":
        return sub(ONE, compare("eq", a, b))
    d = sub(a, b)
    if d.is_const():
        v = d.const_value()
        r = {"gt": v > 0, "ge": v >= 0, "eq": v == 0}[rel]
        return ONE if r else ZERO
    content, shift, q = _normalize_base(d)
    if rel == "eq":
        _, lc = _leading(q)
        if lc < 0:
            q = neg(q)
    return _mono_poly(((_atom("ind", (rel, q)), 1),))


def is_indicator(p: Poly) -> bool:
    if p is ONE or p is ZERO:
        return True
    if len(p.terms) == 1:
        (m, c), = p.terms.items()
        return c == 1 and len(m) == 1 and m[0][0].kind == "ind"
    if len(p.terms) == 2 and p.terms.get(()) == 1:
        return all(
            c == -1 and len(m) == 1 and m[0][0].kind == "ind"
            for m, c in p.terms.items()
            if m != ()
        )
    return False


def where(cond: Poly, x: Poly, y: Poly) -> Poly:
    if not is_indicator(cond):
        cond = compare("ne", cond, ZERO)
    if cond is ONE:
        return x
    if cond is ZERO:
        return y
    return add(mul(cond, x), mul(sub(ONE, cond), y))


def _fn(name: str, args: tuple) -> Poly:
    return _mono_poly(((_atom("fn", (name, tuple(args))), 1),))


def fn(name: str, *args: Poly) -> Poly:
    if all(a.is_const() for a in args):
        v = _eval_const_fn(name, [Fraction(a.const_value()) for a in args])
        if v is not None:
            return const(v)
    return _fn(name, args)


def _eval_const_fn(name, vals):
    try:
        if name == "floor":
            return math.floor(vals[0])
        if name == "ceil":
            return math.ceil(vals[0])
        if name == "trunc":
            return math.trunc(vals[0])
        if name == "fmod":
            a, b = vals
            return a - b * math.trunc(a / b)
        if name == "mod":
            a, b = vals
            return a - b * math.floor(a / b)
    except (ZeroDivisionError, OverflowError):
        return None
    return None


def sigmoid(p: Poly) -> Poly:
    return power(add(ONE, exp(neg(p))), -1)


def tanh(p: Poly) -> Poly:
    return sub(scale(sigmoid(scale(p, 2)), 2), ONE)


def silu(p: Poly) -> Poly:
    return mul(p, sigmoid(p))


def softplus(p: Poly) -> Poly:
    return log(add(ONE, exp(p)))


def sign(p: Poly) -> Poly:
    return sub(compare("gt", p, ZERO), compare("gt", ZERO, p))


# --------------------------------------------------------------------------
# Printing
# --------------------------------------------------------------------------


def atom_str(a: Atom, depth=0) -> str:
    k = a.kind
    if k == "var":
        name, idx = a.args
        return f"{name}[{','.join(map(str, idx))}]"
    if k == "undef":
        name, idx, _ = a.args
        return f"UNINIT<{name}[{','.join(map(str, idx))}]>"
    if depth > 4:
        return f"{k}(...)"
    if k == "exp":
        return f"exp({poly_str(a.args[0], depth + 1)})"
    if k == "log":
        return f"log({poly_str(a.args[0], depth + 1)})"
    if k == "base":
        return f"({poly_str(a.args[0], depth + 1)})"
    if k == "max":
        inner = ", ".join(sorted(poly_str(p, depth + 1) for p in a.args[0]))
        return f"max({inner})"
    if k == "ind":
        rel = {"gt": ">", "ge": ">=", "eq": "=="}[a.args[0]]
        return f"[{poly_str(a.args[1], depth + 1)} {rel} 0]"
    if k == "fn":
        name, args = a.args
        return f"{name}({', '.join(poly_str(p, depth + 1) for p in args)})"
    return f"{k}{a.args}"


def _coef_str(c) -> str:
    if isinstance(c, Fraction):
        f = float(c)
        return f"{f:.6g}"
    return str(c)


def poly_str(p: Poly, depth=0, max_terms=12) -> str:
    if not p.terms:
        return "0"
    parts = []
    items = sorted(p.terms.items(), key=lambda mc: _mono_key(mc[0]))
    for m, c in items[:max_terms]:
        if not m:
            parts.append(_coef_str(c))
            continue
        fs = []
        for a, e in m:
            s = atom_str(a, depth)
            if e != 1:
                s = f"{s}^{_coef_str(e)}"
            fs.append(s)
        body = "*".join(fs)
        if c == 1:
            parts.append(body)
        elif c == -1:
            parts.append(f"-{body}")
        else:
            parts.append(f"{_coef_str(c)}*{body}")
    s = " + ".join(parts).replace("+ -", "- ")
    if len(items) > max_terms:
        s += f" + ... ({len(items) - max_terms} more terms)"
    return s
