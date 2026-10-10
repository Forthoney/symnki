"""Discharge residual equalities with Z3 over the reals.

Used only for output elements whose canonical forms differ but where no
concrete counterexample was found (e.g. ``max(a,b) + min(a,b) = a + b``).
Transcendental functions are uninterpreted (with exp > 0), so ``unsat`` is a
proof for every interpretation of them and in particular the real one; a
``sat`` answer is only reported as a counterexample when the query contains no
uninterpreted functions.
"""

from __future__ import annotations

from fractions import Fraction

import z3

from . import algebra as A


class Translator:
    def __init__(self):
        self.vars = {}
        self.side = []
        self.cache = {}
        self.uninterpreted = False
        self._funcs = {}
        self._fresh = 0

    def _fresh_real(self, prefix):
        self._fresh += 1
        return z3.Real(f"{prefix}!{self._fresh}")

    def _func(self, name, arity):
        key = (name, arity)
        f = self._funcs.get(key)
        if f is None:
            f = z3.Function(name, *([z3.RealSort()] * (arity + 1)))
            self._funcs[key] = f
        return f

    def poly(self, p: A.Poly):
        key = ("p", id(p))
        hit = self.cache.get(key)
        if hit is not None and hit[0] is p:
            return hit[1]
        terms = []
        for m, c in p.terms.items():
            c = Fraction(c)
            coef = z3.RealVal(f"{c.numerator}/{c.denominator}")
            if not m:
                terms.append(coef)
            else:
                terms.append(coef * self.mono(m) if c != 1 else self.mono(m))
        e = z3.Sum(terms) if len(terms) > 1 else (terms[0] if terms else z3.RealVal(0))
        self.cache[key] = (p, e)
        return e

    def mono(self, m):
        fs = []
        for a, e in m:
            fs.append(self.power(a, Fraction(e)))
        out = fs[0]
        for f in fs[1:]:
            out = out * f
        return out

    def power(self, a: A.Atom, e: Fraction):
        if a.kind == "base" and e.denominator == 2:
            s = self.sqrt(a)
            return self._ipow(s, int(e * 2), positive=True)
        if e.denominator != 1:
            raise ValueError("unsupported fractional exponent")
        x = self.atom(a)
        return self._ipow(x, int(e), positive=False)

    def _ipow(self, x, k, positive):
        if k == 0:
            return z3.RealVal(1)
        base = x
        for _ in range(abs(k) - 1):
            base = base * x
        if k < 0:
            self.side.append(x > 0 if positive else x != 0)
            return 1 / base
        return base

    def sqrt(self, a: A.Atom):
        key = ("sqrt", a.id)
        if key in self.cache:
            return self.cache[key]
        inner = self.poly(a.args[0])
        s = self._fresh_real("sqrt")
        self.side += [s >= 0, s * s == inner]
        self.cache[key] = s
        return s

    def atom(self, a: A.Atom):
        key = ("a", a.id)
        if key in self.cache:
            return self.cache[key]
        k = a.kind
        if k in ("var", "undef"):
            name, idx = a.args[0], a.args[1]
            r = z3.Real(f"{name}[{','.join(map(str, idx))}]")
        elif k == "base":
            r = self.poly(a.args[0])
        elif k == "exp":
            self.uninterpreted = True
            r = self._func("exp", 1)(self.poly(a.args[0]))
            self.side.append(r > 0)
        elif k == "log":
            self.uninterpreted = True
            r = self._func("log", 1)(self.poly(a.args[0]))
        elif k == "max":
            args = [self.poly(p) for p in a.args[0]]
            r = self._fresh_real("max")
            self.side += [r >= x for x in args]
            self.side.append(z3.Or([r == x for x in args]))
        elif k == "ind":
            rel, p = a.args
            x = self.poly(p)
            cond = {"gt": x > 0, "ge": x >= 0, "eq": x == 0}[rel]
            r = z3.If(cond, z3.RealVal(1), z3.RealVal(0))
        elif k == "fn":
            self.uninterpreted = True
            name, args = a.args
            r = self._func(name, len(args))(*[self.poly(p) for p in args])
        else:
            raise ValueError(f"cannot translate atom kind {k}")
        self.cache[key] = r
        return r


def prove_equal(lhs: A.Poly, rhs: A.Poly, timeout_ms: int = 5000):
    """Returns ("proved" | "refuted" | "unknown", model-or-None)."""
    t = Translator()
    try:
        diff = t.poly(lhs) - t.poly(rhs)
    except ValueError as e:
        return "unknown", str(e)
    s = z3.Solver()
    s.set("timeout", timeout_ms)
    s.add(*t.side)
    s.add(diff != 0)
    r = s.check()
    if r == z3.unsat:
        return "proved", None
    if r == z3.sat and not t.uninterpreted:
        m = s.model()
        return "refuted", {str(d): str(m[d]) for d in m.decls() if "!" not in str(d)}
    return "unknown", None
