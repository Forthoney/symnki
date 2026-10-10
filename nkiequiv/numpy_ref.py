"""NumPy functions as references.

A NumPy reference runs unmodified on ``TracedArray``, an array type that
intercepts NumPy's dispatch protocols (``__array_ufunc__`` /
``__array_function__``) and computes in one of the validator's value domains:

* symbolic: canonical real-valued expressions (for the proof),
* concrete ``real``: float64 without rounding (for counterexample search and as
  ground truth for numerics).

In both, ``astype`` to a float dtype keeps the value and only changes the
recorded dtype (casts are value-preserving over the reals); the dtype matters
for the interface check. For the hardware-precision numerics stage the
function runs on plain numpy arrays of the real input dtypes, i.e. exactly as
written, including its float32/float64 intermediates and output rounding.

Supported: arithmetic operators and ufuncs, comparisons, ``where``, reductions
(sum/mean/max/min/prod, with axis/keepdims), ``@``/matmul/dot, transpose/.T,
reshape, broadcasting, basic slicing, astype. Anything else raises
``Unsupported`` naming the NumPy function.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from .domains import FLOAT_DTYPES, INT_DTYPES, Unsupported, is_int_dtype
from .runner import TraceResult, load_kernel

_UFUNC_BINARY = {
    "add": "add", "subtract": "subtract", "multiply": "multiply", "true_divide": "divide",
    "divide": "divide", "maximum": "maximum", "minimum": "minimum", "fmax": "maximum",
    "fmin": "minimum", "power": "power", "float_power": "power", "greater": "greater",
    "greater_equal": "greater_equal", "less": "less", "less_equal": "less_equal",
    "equal": "equal", "not_equal": "not_equal", "logical_and": "logical_and",
    "logical_or": "logical_or", "logical_xor": "logical_xor", "fmod": "fmod", "remainder": "mod",
}
_UFUNC_UNARY = {
    "exp": "exp", "log": "log", "sqrt": "sqrt", "tanh": "tanh", "negative": "negative",
    "absolute": "abs", "fabs": "abs", "square": "square", "reciprocal": "reciprocal",
    "sign": "sign", "sin": "sin", "cos": "cos", "tan": "tan", "arctan": "arctan",
    "floor": "floor", "ceil": "ceil", "trunc": "trunc", "logical_not": "logical_not",
    "positive": "copy",
}
_REDUCE = {"add": "add", "multiply": "multiply", "maximum": "maximum", "minimum": "minimum"}


def _dtype_name(dt) -> str:
    return np.dtype(dt).name


class TracedArray:
    __array_priority__ = 1000

    def __init__(self, vals, dtype, dom):
        self.v = vals if isinstance(vals, np.ndarray) else np.asarray(vals, dtype=object if dom.obj else np.float64)
        self.dtype = np.dtype(dtype)
        self.dom = dom

    # -- metadata ------------------------------------------------------------
    shape = property(lambda self: self.v.shape)
    ndim = property(lambda self: self.v.ndim)
    size = property(lambda self: self.v.size)

    def __len__(self):
        return self.v.shape[0]

    def __repr__(self):
        return f"TracedArray(shape={self.shape}, dtype={self.dtype})"

    def __bool__(self):
        raise Unsupported("data-dependent Python control flow in the NumPy reference")

    def _wrap(self, vals, dtype=None):
        return TracedArray(vals, self.dtype if dtype is None else dtype, self.dom)

    def _lift(self, x):
        """(values, dtype or None for weak python scalars)"""
        if isinstance(x, TracedArray):
            return x.v, x.dtype
        if isinstance(x, (bool, int, float)):
            return self.dom.from_numpy(np.array(float(x))), None
        arr = np.asarray(x)
        if arr.dtype == object:
            raise Unsupported("object arrays in the NumPy reference")
        return self.dom.from_numpy(arr.astype(np.float64)), arr.dtype

    # -- views ---------------------------------------------------------------
    @property
    def T(self):
        return self._wrap(self.v.T)

    def transpose(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = axes[0]
        return self._wrap(np.transpose(self.v, axes or None))

    def reshape(self, *shape, **kw):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return self._wrap(self.v.reshape(shape))

    def __getitem__(self, key):
        if isinstance(key, TracedArray) or (isinstance(key, tuple) and any(isinstance(k, TracedArray) for k in key)):
            raise Unsupported("indexing with traced (data-dependent) arrays")
        return self._wrap(self.v[key])

    def copy(self):
        return self._wrap(self.v.copy())

    def astype(self, dtype, copy=True, **kw):
        name = _dtype_name(dtype)
        if is_int_dtype(name):
            return self._wrap(self.dom.store_cast(self.v, name), dtype)
        if name not in FLOAT_DTYPES and name not in ("float64", "bool"):
            raise Unsupported(f"astype({name})")
        return self._wrap(self.v, dtype)

    # -- arithmetic ------------------------------------------------------------
    def _binary(self, op, a, b):
        av, ad = self._lift(a)
        bv, bd = self._lift(b)
        dts = [d for d in (ad, bd) if d is not None]
        dtype = np.result_type(*dts) if dts else self.dtype
        if op in ("greater", "greater_equal", "less", "less_equal", "equal", "not_equal",
                  "logical_and", "logical_or", "logical_xor"):
            dtype = np.dtype(bool)
        if op == "divide" and np.issubdtype(dtype, np.integer):
            dtype = np.dtype(np.float64)
        return TracedArray(self.dom.binary(op, av, bv), dtype, self.dom)

    def __add__(self, o): return self._binary("add", self, o)
    def __radd__(self, o): return self._binary("add", o, self)
    def __sub__(self, o): return self._binary("subtract", self, o)
    def __rsub__(self, o): return self._binary("subtract", o, self)
    def __mul__(self, o): return self._binary("multiply", self, o)
    def __rmul__(self, o): return self._binary("multiply", o, self)
    def __truediv__(self, o): return self._binary("divide", self, o)
    def __rtruediv__(self, o): return self._binary("divide", o, self)
    def __pow__(self, o): return self._binary("power", self, o)
    def __gt__(self, o): return self._binary("greater", self, o)
    def __ge__(self, o): return self._binary("greater_equal", self, o)
    def __lt__(self, o): return self._binary("less", self, o)
    def __le__(self, o): return self._binary("less_equal", self, o)
    def __neg__(self): return self._wrap(self.dom.unary("negative", self.v))
    def __abs__(self): return self._wrap(self.dom.unary("abs", self.v))
    def __matmul__(self, o): return _matmul(self, o)
    def __rmatmul__(self, o): return _matmul(o, self)

    # -- reductions ------------------------------------------------------------
    def _reduce(self, op, axis=None, keepdims=False, dtype=None, out=None, **kw):
        if out is not None or kw.get("where") is not None:
            raise Unsupported("reductions with out= or where=")
        axes = tuple(range(self.ndim)) if axis is None else (axis if isinstance(axis, tuple) else (axis,))
        axes = tuple(a % self.ndim for a in axes)
        vals = self.dom.reduce(op, self.v, axes, keepdims=keepdims)
        return self._wrap(vals, dtype or self.dtype)

    def sum(self, axis=None, dtype=None, keepdims=False, **kw):
        return self._reduce("add", axis, keepdims, dtype, **kw)

    def prod(self, axis=None, dtype=None, keepdims=False, **kw):
        return self._reduce("multiply", axis, keepdims, dtype, **kw)

    def max(self, axis=None, keepdims=False, **kw):
        return self._reduce("maximum", axis, keepdims, **kw)

    def min(self, axis=None, keepdims=False, **kw):
        return self._reduce("minimum", axis, keepdims, **kw)

    def mean(self, axis=None, dtype=None, keepdims=False, **kw):
        s = self._reduce("add", axis, keepdims, dtype, **kw)
        axes = range(self.ndim) if axis is None else (axis if isinstance(axis, tuple) else (axis,))
        n = int(np.prod([self.shape[a] for a in axes]))
        r = s * (1.0 / n)
        if np.issubdtype(r.dtype, np.integer):
            r.dtype = np.dtype(np.float64)
        return r

    # -- numpy protocols -----------------------------------------------------------
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if kwargs.get("out") is not None or kwargs.get("where", True) is not True:
            raise Unsupported(f"np.{ufunc.__name__} with out= or where=")
        name = ufunc.__name__
        if method == "__call__":
            if name in _UFUNC_UNARY and len(inputs) == 1:
                x = inputs[0]
                return x._wrap(x.dom.unary(_UFUNC_UNARY[name], x.v))
            if name in _UFUNC_BINARY and len(inputs) == 2:
                return self._binary(_UFUNC_BINARY[name], *inputs)
            if name == "matmul":
                return _matmul(*inputs)
        if method == "reduce" and name in _REDUCE:
            x = inputs[0]
            return x._reduce(_REDUCE[name], kwargs.get("axis", 0), kwargs.get("keepdims", False))
        raise Unsupported(f"np.{name}.{method}" if method != "__call__" else f"np.{name}")

    def __array_function__(self, func, types, args, kwargs):
        impl = _FUNCTIONS.get(func.__name__)
        if impl is None:
            raise Unsupported(f"np.{func.__name__}")
        return impl(*args, **kwargs)

    def __array__(self, *a, **k):
        raise Unsupported("converting a traced array to a concrete numpy array")


def _as_traced(x, like):
    if isinstance(x, TracedArray):
        return x
    v, d = like._lift(x)
    return TracedArray(v, d if d is not None else like.dtype, like.dom)


def _matmul(a, b):
    like = a if isinstance(a, TracedArray) else b
    a, b = _as_traced(a, like), _as_traced(b, like)
    dom = like.dom
    av, bv = a.v, b.v
    squeeze = []
    if av.ndim == 1:
        av, squeeze = av[None, :], squeeze + [-2]
    if bv.ndim == 1:
        bv, squeeze = bv[:, None], squeeze + [-1]
    if av.shape[-1] != bv.shape[-2]:
        raise ValueError(f"matmul: shapes {a.shape} and {b.shape} not aligned")
    batch = np.broadcast_shapes(av.shape[:-2], bv.shape[:-2])
    av = np.broadcast_to(av, batch + av.shape[-2:])
    bv = np.broadcast_to(bv, batch + bv.shape[-2:])
    out = np.empty(batch + (av.shape[-2], bv.shape[-1]), dtype=object if dom.obj else np.float64)
    for idx in np.ndindex(*batch):
        out[idx] = dom.matmul_T(np.ascontiguousarray(av[idx].T), bv[idx])
    for ax in squeeze:
        out = np.squeeze(out, axis=ax)
    return TracedArray(out, np.result_type(a.dtype, b.dtype), dom)


def _where(cond, x, y):
    like = next(t for t in (cond, x, y) if isinstance(t, TracedArray))
    c, x, y = (_as_traced(t, like) for t in (cond, x, y))
    vals = like.dom.where(*np.broadcast_arrays(c.v, x.v, y.v))
    return TracedArray(vals, np.result_type(x.dtype, y.dtype), like.dom)


def _method(name):
    def f(a, *args, **kwargs):
        return getattr(_as_traced(a, a), name)(*args, **kwargs)
    return f


_FUNCTIONS = {
    "sum": _method("sum"), "mean": _method("mean"), "max": _method("max"), "amax": _method("max"),
    "min": _method("min"), "amin": _method("min"), "prod": _method("prod"),
    "transpose": lambda a, axes=None: a.transpose(axes) if axes is not None else a.transpose(),
    "reshape": lambda a, shape=None, **k: a.reshape(shape if shape is not None else k["newshape"]),
    "matmul": _matmul, "dot": _matmul, "where": _where,
    "broadcast_to": lambda a, shape: a._wrap(np.broadcast_to(a.v, shape)),
    "expand_dims": lambda a, axis: a._wrap(np.expand_dims(a.v, axis)),
    "squeeze": lambda a, axis=None: a._wrap(np.squeeze(a.v, axis)),
    "square": lambda a: a._wrap(a.dom.unary("square", a.v)),
}


# --------------------------------------------------------------------------


@dataclass
class NumpyRef:
    """A NumPy function used as the reference: 'numpy:path/to/file.py::fn'."""

    path: str
    func: str
    kwargs: dict = field(default_factory=dict)
    grid: tuple = (1,)

    @classmethod
    def parse(cls, text: str, **kw) -> "NumpyRef":
        text = text[len("numpy:"):] if text.startswith("numpy:") else text
        path, func = text.split("::", 1) if "::" in text else text.rsplit(":", 1)
        kw.pop("grid", None)
        return cls(os.path.abspath(path), func, **kw)

    def label(self):
        return f"numpy:{os.path.basename(self.path)}::{self.func}"


def run_numpy(ref: NumpyRef, specs, domain, input_values=None) -> TraceResult:
    fn = load_kernel(ref)
    res = TraceResult(ok=False)
    try:
        if domain.kind == "concrete" and domain.mode == "hw":
            # Run exactly as written on arrays of the declared dtypes.
            args = []
            for s in specs:
                np_dt = FLOAT_DTYPES.get(s.dtype) or INT_DTYPES.get(s.dtype)
                args.append(np.asarray(input_values[s.name]).astype(np_dt))
            out = fn(*args, **ref.kwargs)
            outs = out if isinstance(out, (tuple, list)) else [out]
            for k, o in enumerate(outs):
                o = np.asarray(o)
                res.outputs.append(o.astype(np.float64))
                res.out_meta.append((f"out{k}", o.shape, o.dtype.name))
                res.unwritten.append(0)
        else:
            args = []
            for s in specs:
                vals = domain.inputs(s.name, tuple(s.shape)) if domain.obj else np.asarray(
                    input_values[s.name], dtype=np.float64)
                args.append(TracedArray(vals, FLOAT_DTYPES.get(s.dtype) or INT_DTYPES[s.dtype], domain))
            with np.errstate(all="ignore"):
                out = fn(*args, **ref.kwargs)
            outs = out if isinstance(out, (tuple, list)) else [out]
            for k, o in enumerate(outs):
                if not isinstance(o, TracedArray):
                    raise Unsupported(f"reference output {k} is {type(o).__name__}, not derived from the inputs")
                res.outputs.append(o.v)
                res.out_meta.append((f"out{k}", o.shape, o.dtype.name))
                res.unwritten.append(0)
        res.ok = True
    except Unsupported as e:
        res.error_kind, res.error = "unsupported", f"unsupported in NumPy reference: {e}"
    except Exception as e:  # noqa: BLE001
        res.error_kind, res.error = "exception", f"{type(e).__name__}: {e}"
    return res
