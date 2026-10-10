"""Value domains: the same kernel interpreter runs over symbolic or concrete values.

``SymbolicDomain`` stores ``algebra.Poly`` objects in numpy object arrays and
computes canonical real-valued expressions.

``ConcreteDomain`` stores float64 numpy arrays. In ``real`` mode it computes in
float64 and does not round on stores (an approximation of real semantics used
to search for counterexamples); in ``hw`` mode it mirrors the NKI simulator:
engines compute in float32 and every store rounds to the destination dtype.
"""

from __future__ import annotations

import math
from fractions import Fraction

import ml_dtypes
import numpy as np

from . import algebra as A

FLOAT_DTYPES = {
    "float32": np.float32,
    "float16": np.float16,
    "bfloat16": ml_dtypes.bfloat16,
    "tfloat32": np.float32,  # rounded separately below
    "float8_e4m3": ml_dtypes.float8_e4m3fn,
    "float8_e4m3fn": ml_dtypes.float8_e4m3fn,
    "float8_e5m2": ml_dtypes.float8_e5m2,
}
INT_DTYPES = {
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "bool_": np.bool_,
}


def is_int_dtype(dtype: str) -> bool:
    return dtype in INT_DTYPES


def is_float_dtype(dtype: str) -> bool:
    return dtype in FLOAT_DTYPES


def dtype_bits(dtype: str) -> int:
    if dtype in ("float32", "tfloat32", "int32", "uint32"):
        return 32
    if dtype in ("float16", "bfloat16", "int16", "uint16"):
        return 16
    return 8


class Unsupported(Exception):
    """Raised for NKI features the validator does not model."""


# --------------------------------------------------------------------------
# Symbolic
# --------------------------------------------------------------------------


def _vec(f, nin):
    u = np.frompyfunc(f, nin, 1)

    def g(*args):
        out = u(*args)
        if not isinstance(out, np.ndarray):
            arr = np.empty((), dtype=object)
            arr[()] = out
            return arr
        return out

    return g


def _abs_max(x, y):
    # abs(x) > abs(y) ? x : y
    return A.where(A.compare("gt", A.absolute(x), A.absolute(y)), x, y)


def _abs_min(x, y):
    return A.where(A.compare("lt", A.absolute(x), A.absolute(y)), x, y)


def _power(x, y):
    if y.is_const():
        e = Fraction(y.const_value())
        if e.denominator in (1, 2):
            return A.power(x, e)
    return A.fn("pow", x, y)


def _logical(name):
    def f(x, y):
        bx = A.compare("ne", x, A.ZERO)
        by = A.compare("ne", y, A.ZERO)
        if name == "logical_and":
            return A.mul(bx, by)
        if name == "logical_or":
            return A.sub(A.add(bx, by), A.mul(bx, by))
        return A.sub(A.add(bx, by), A.scale(A.mul(bx, by), 2))

    return f


_SYM_BINARY = {
    "add": A.add,
    "subtract": A.sub,
    "multiply": A.mul,
    "divide": lambda x, y: A.mul(x, A.power(y, -1)),
    "maximum": lambda x, y: A.maximum([x, y]),
    "minimum": lambda x, y: A.minimum([x, y]),
    "abs_max": _abs_max,
    "abs_min": _abs_min,
    "power": _power,
    "equal": lambda x, y: A.compare("eq", x, y),
    "not_equal": lambda x, y: A.compare("ne", x, y),
    "less": lambda x, y: A.compare("lt", x, y),
    "less_equal": lambda x, y: A.compare("le", x, y),
    "greater": lambda x, y: A.compare("gt", x, y),
    "greater_equal": lambda x, y: A.compare("ge", x, y),
    "logical_and": _logical("logical_and"),
    "logical_or": _logical("logical_or"),
    "logical_xor": _logical("logical_xor"),
    "fmod": lambda x, y: A.fn("fmod", x, y),
    "mod": lambda x, y: A.fn("mod", x, y),
}
for _b in ("bitwise_and", "bitwise_or", "bitwise_xor", "left_shift", "right_shift"):
    _SYM_BINARY[_b] = (lambda name: lambda x, y: A.fn(name, x, y))(_b)

_SQRT_2_OVER_PI = None  # irrational: gelu stays an uninterpreted function

_SYM_UNARY = {
    "exp": A.exp,
    "log": A.log,
    "sqrt": lambda x: A.power(x, Fraction(1, 2)),
    "rsqrt": lambda x: A.power(x, Fraction(-1, 2)),
    "reciprocal": lambda x: A.power(x, -1),
    "tanh": A.tanh,
    "sigmoid": A.sigmoid,
    "relu": A.relu,
    "silu": A.silu,
    "silu_dx": lambda x: A.mul(A.sigmoid(x), A.add(A.ONE, A.mul(x, A.sub(A.ONE, A.sigmoid(x))))),
    "softplus": A.softplus,
    "mish": lambda x: A.mul(x, A.tanh(A.softplus(x))),
    "gelu_apprx_sigmoid": lambda x: A.mul(x, A.sigmoid(A.scale(x, Fraction("1.702")))),
    "copy": lambda x: x,
    "square": lambda x: A.mul(x, x),
    "abs": A.absolute,
    "negative": A.neg,
    "sign": A.sign,
    "logical_not": lambda x: A.compare("eq", x, A.ZERO),
}
for _u in (
    "gelu", "gelu_apprx_tanh", "gelu_dx", "gelu_apprx_sigmoid_dx", "erf", "erf_dx",
    "sin", "cos", "tan", "arctan", "ceil", "floor", "trunc", "invert",
):
    _SYM_UNARY[_u] = (lambda name: lambda x: A.fn(name, x))(_u)


class SymbolicDomain:
    kind = "symbolic"
    obj = True

    def __init__(self):
        self._bin = {k: _vec(f, 2) for k, f in _SYM_BINARY.items()}
        self._un = {k: _vec(f, 1) for k, f in _SYM_UNARY.items()}

    # storage --------------------------------------------------------------
    def empty(self, size: int, name: str):
        arr = np.empty(size, dtype=object)
        for i in range(size):
            arr[i] = A.undef(name, (i,))
        return arr

    def inputs(self, name: str, shape: tuple, rng=None):
        arr = np.empty(shape, dtype=object)
        for idx in np.ndindex(*shape):
            arr[idx] = A.var(name, idx)
        return arr

    def full(self, shape, value):
        arr = np.empty(shape, dtype=object)
        p = A.lift(value) if not isinstance(value, A.Poly) else value
        arr.fill(p)
        return arr

    def scalar(self, value):
        return A.lift(value)

    def from_numpy(self, arr):
        out = np.empty(arr.shape, dtype=object)
        flat = arr.reshape(-1)
        of = out.reshape(-1)
        for i, v in enumerate(flat):
            of[i] = A.lift(v.item() if hasattr(v, "item") else v)
        return out

    # compute --------------------------------------------------------------
    def to_compute(self, x):
        return x

    def binary(self, op: str, x, y):
        f = self._bin.get(op)
        if f is None:
            raise Unsupported(f"binary operator {op}")
        return f(x, y)

    def unary(self, op: str, x):
        f = self._un.get(op)
        if f is None:
            raise Unsupported(f"activation/unary operator {op}")
        return f(x)

    def prelu(self, x, alpha):
        return _vec(lambda v, a: A.where(A.compare("ge", v, A.ZERO), v, A.mul(a, v)), 2)(x, alpha)

    def reduce(self, op: str, x, axes: tuple, keepdims=False):
        x = np.asarray(x, dtype=object)
        axes = tuple(sorted(a % x.ndim for a in axes))
        keep = [d for d in range(x.ndim) if d not in axes]
        moved = np.transpose(x, keep + list(axes))
        out_shape = tuple(x.shape[d] for d in keep)
        rows = moved.reshape(int(np.prod(out_shape, dtype=np.int64)), -1)
        out = np.empty(rows.shape[0], dtype=object)
        for i in range(rows.shape[0]):
            out[i] = self._reduce_list(op, list(rows[i]))
        out = out.reshape(out_shape)
        if keepdims:
            for a in axes:
                out = np.expand_dims(out, a)
        return out

    def _reduce_list(self, op, items):
        if op == "add":
            return A.add_many(items)
        if op == "maximum":
            return A.maximum(items)
        if op == "minimum":
            return A.minimum(items)
        if op == "multiply":
            r = A.ONE
            for it in items:
                r = A.mul(r, it)
            return r
        if op in ("abs_max", "abs_min"):
            f = _abs_max if op == "abs_max" else _abs_min
            r = items[0]
            for it in items[1:]:
                r = f(it, r)
            return r
        if op in ("logical_and", "logical_or"):
            f = _logical(op)
            r = items[0]
            for it in items[1:]:
                r = f(r, it)
            return r
        raise Unsupported(f"reduction operator {op}")

    def matmul_T(self, stat, mov):
        """stat.T @ mov for stat (K, M), mov (K, N)."""
        K, M = stat.shape
        _, N = mov.shape
        out = np.empty((M, N), dtype=object)
        cols_s = [list(stat[:, m]) for m in range(M)]
        cols_m = [list(mov[:, n]) for n in range(N)]
        for m in range(M):
            cs = cols_s[m]
            for n in range(N):
                out[m, n] = A.dot(cs, cols_m[n])
        return out

    def where(self, cond, x, y):
        return _vec(A.where, 3)(cond, x, y)

    def store_cast(self, values, dtype: str):
        """Values written into a tensor of ``dtype``."""
        if is_int_dtype(dtype):
            return _vec(lambda v: _int_cast(v, dtype), 1)(values)
        return values

    def rounding_note(self, src_dtype, dst_dtype):
        return None

    def is_poison(self, arr):
        return np.vectorize(lambda p: p.has_undef, otypes=[bool])(arr)


def _int_cast(p, dtype):
    if p.is_const():
        v = Fraction(p.const_value())
        iv = math.trunc(v)
        info = np.iinfo(INT_DTYPES[dtype]) if dtype != "bool_" else None
        if info is not None:
            span = info.max - info.min + 1
            iv = (iv - info.min) % span + info.min
        else:
            iv = 1 if iv else 0
        return A.const(iv)
    if A.is_indicator(p):
        return p
    return A.fn(f"to_{dtype}", p)


# --------------------------------------------------------------------------
# Concrete
# --------------------------------------------------------------------------


def _erf(x):
    return np.vectorize(math.erf, otypes=[np.float64])(x)


def _sigmoid(x):
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-x))


_C_BINARY = {
    "add": np.add,
    "subtract": np.subtract,
    "multiply": np.multiply,
    "divide": np.divide,
    "maximum": np.maximum,
    "minimum": np.minimum,
    "abs_max": lambda x, y: np.where(np.abs(x) > np.abs(y), x, y),
    "abs_min": lambda x, y: np.where(np.abs(x) < np.abs(y), x, y),
    "power": np.power,
    "equal": lambda x, y: np.equal(x, y).astype(np.float64),
    "not_equal": lambda x, y: np.not_equal(x, y).astype(np.float64),
    "less": lambda x, y: np.less(x, y).astype(np.float64),
    "less_equal": lambda x, y: np.less_equal(x, y).astype(np.float64),
    "greater": lambda x, y: np.greater(x, y).astype(np.float64),
    "greater_equal": lambda x, y: np.greater_equal(x, y).astype(np.float64),
    "logical_and": lambda x, y: np.logical_and(x, y).astype(np.float64),
    "logical_or": lambda x, y: np.logical_or(x, y).astype(np.float64),
    "logical_xor": lambda x, y: np.logical_xor(x, y).astype(np.float64),
    "fmod": np.fmod,
    "mod": np.mod,
}

_C_UNARY = {
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "rsqrt": lambda x: 1.0 / np.sqrt(x),
    "reciprocal": lambda x: 1.0 / x,
    "tanh": np.tanh,
    "sigmoid": _sigmoid,
    "relu": lambda x: np.maximum(0, x),
    "silu": lambda x: x * _sigmoid(x),
    "silu_dx": lambda x: _sigmoid(x) * (1 + x * (1 - _sigmoid(x))),
    "softplus": lambda x: np.log(1.0 + np.exp(x)),
    "mish": lambda x: x * np.tanh(np.log(1.0 + np.exp(x))),
    "gelu": lambda x: 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3))),
    "gelu_apprx_tanh": lambda x: 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3))),
    "gelu_apprx_sigmoid": lambda x: x * _sigmoid(1.702 * x),
    "gelu_dx": lambda x: 0.5 * (1 + _erf(x / np.sqrt(2.0))) + x * np.exp(-(x**2) / 2) / np.sqrt(2.0 * math.pi),
    "gelu_apprx_sigmoid_dx": lambda x: _sigmoid(1.702 * x) + 1.702 * x * _sigmoid(1.702 * x) * (1 - _sigmoid(1.702 * x)),
    "erf": _erf,
    "erf_dx": lambda x: (2.0 / np.sqrt(math.pi)) * np.exp(-(x**2)),
    "copy": lambda x: np.array(x, copy=True),
    "square": np.square,
    "abs": np.abs,
    "negative": np.negative,
    "sign": np.sign,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "arctan": np.arctan,
    "ceil": np.ceil,
    "floor": np.floor,
    "trunc": np.trunc,
    "logical_not": lambda x: np.logical_not(x).astype(np.float64),
}

_C_REDUCE = {
    "add": np.sum,
    "multiply": np.prod,
    "maximum": np.max,
    "minimum": np.min,
    "logical_and": lambda x, axis, keepdims: np.all(x, axis=axis, keepdims=keepdims).astype(np.float64),
    "logical_or": lambda x, axis, keepdims: np.any(x, axis=axis, keepdims=keepdims).astype(np.float64),
}


def round_to_dtype(values: np.ndarray, dtype: str) -> np.ndarray:
    """Round float64 values to the nearest value representable in dtype."""
    if is_int_dtype(dtype):
        if dtype == "bool_":
            return (values != 0).astype(np.float64)
        info = np.iinfo(INT_DTYPES[dtype])
        t = np.trunc(np.nan_to_num(values, nan=0.0, posinf=info.max, neginf=info.min))
        return np.clip(t, info.min, info.max)
    if dtype == "tfloat32":
        f32 = values.astype(np.float32)
        bits = f32.view(np.uint32).astype(np.uint64)
        # round-to-nearest-even to a 10-bit mantissa
        rounding = ((bits >> 13) & 1) + 0x0FFF
        bits = ((bits + rounding) >> 13) << 13
        return bits.astype(np.uint32).view(np.float32).astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        return values.astype(FLOAT_DTYPES[dtype]).astype(np.float64)


class ConcreteDomain:
    kind = "concrete"
    obj = False

    def __init__(self, mode: str = "real"):
        assert mode in ("real", "hw")
        self.mode = mode

    def empty(self, size: int, name: str):
        return np.full(size, np.nan)

    def inputs(self, name, shape, rng=None):
        raise RuntimeError("concrete inputs are supplied by the caller")

    def full(self, shape, value):
        return np.full(shape, float(value))

    def scalar(self, value):
        return float(value)

    def from_numpy(self, arr):
        return np.asarray(arr, dtype=np.float64)

    def to_compute(self, x):
        """Engines compute in float32 on hardware."""
        if self.mode == "hw":
            return np.asarray(x, dtype=np.float64).astype(np.float32).astype(np.float64)
        return np.asarray(x, dtype=np.float64)

    def _fin(self, x):
        if self.mode == "hw":
            with np.errstate(over="ignore", invalid="ignore"):
                return np.asarray(x, dtype=np.float64).astype(np.float32).astype(np.float64)
        return np.asarray(x, dtype=np.float64)

    def binary(self, op, x, y):
        f = _C_BINARY.get(op)
        if f is None:
            if op in ("bitwise_and", "bitwise_or", "bitwise_xor", "left_shift", "right_shift"):
                fn = getattr(np, op)
                return fn(np.asarray(x).astype(np.int64), np.asarray(y).astype(np.int64)).astype(np.float64)
            raise Unsupported(f"binary operator {op}")
        with np.errstate(all="ignore"):
            return self._fin(f(self.to_compute(x), self.to_compute(y)))

    def unary(self, op, x):
        f = _C_UNARY.get(op)
        if f is None:
            if op == "invert":
                return np.invert(np.asarray(x).astype(np.int64)).astype(np.float64)
            raise Unsupported(f"activation/unary operator {op}")
        with np.errstate(all="ignore"):
            return self._fin(f(self.to_compute(x)))

    def prelu(self, x, alpha):
        x = self.to_compute(x)
        return self._fin(np.where(x >= 0, x, alpha * x))

    def reduce(self, op, x, axes, keepdims=False):
        f = _C_REDUCE.get(op)
        x = self.to_compute(x)
        axes = tuple(a % x.ndim for a in axes)
        if f is None:
            if op in ("abs_max", "abs_min"):
                moved = np.moveaxis(x, axes, tuple(range(-len(axes), 0)))
                flat = moved.reshape(moved.shape[: x.ndim - len(axes)] + (-1,))
                idx = (np.argmax if op == "abs_max" else np.argmin)(np.abs(flat), axis=-1)
                r = np.take_along_axis(flat, idx[..., None], axis=-1)[..., 0]
                if keepdims:
                    for a in sorted(axes):
                        r = np.expand_dims(r, a)
                return self._fin(r)
            raise Unsupported(f"reduction operator {op}")
        with np.errstate(all="ignore"):
            return self._fin(f(x, axis=axes, keepdims=keepdims))

    def matmul_T(self, stat, mov):
        with np.errstate(all="ignore"):
            return self._fin(self.to_compute(stat).T @ self.to_compute(mov))

    def where(self, cond, x, y):
        return np.where(np.asarray(cond) != 0, x, y)

    def store_cast(self, values, dtype):
        values = np.asarray(values, dtype=np.float64)
        if is_int_dtype(dtype):
            return round_to_dtype(values, dtype)
        if self.mode == "hw":
            return round_to_dtype(values, dtype)
        return values

    def is_poison(self, arr):
        return np.isnan(np.asarray(arr, dtype=np.float64))
