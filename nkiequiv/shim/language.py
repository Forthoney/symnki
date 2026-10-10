"""Shim for ``nki.language``."""

from __future__ import annotations

import contextlib

import numpy as np

from ..domains import Unsupported
from . import isa as _isa
from . import runtime as R
from .runtime import DynSlice, KernelError, NkiTensor, ctx, new_tensor

# --------------------------------------------------------------------------
# dtypes and buffers
# --------------------------------------------------------------------------

bool_ = "bool_"
int8 = "int8"
int16 = "int16"
int32 = "int32"
uint8 = "uint8"
uint16 = "uint16"
uint32 = "uint32"
float16 = "float16"
float32 = "float32"
bfloat16 = "bfloat16"
tfloat32 = "tfloat32"
float8_e4m3 = "float8_e4m3"
float8_e4m3fn = "float8_e4m3fn"
float8_e5m2 = "float8_e5m2"

sbuf = "sbuf"
psum = "psum"
hbm = "hbm"
private_hbm = "private_hbm"
shared_hbm = "shared_hbm"


class MemoryRegion:
    sbuf = sbuf
    psum = psum
    hbm = hbm
    private_hbm = private_hbm
    shared_hbm = shared_hbm


def is_hbm(buffer):
    return R._buf(buffer) in R.HBM_BUFFERS


def is_on_chip(buffer):
    return R._buf(buffer) in R.ON_CHIP


def is_psum(buffer):
    return R._buf(buffer) == "psum"


def is_sbuf(buffer):
    return R._buf(buffer) == "sbuf"


NKIObject = object


# --------------------------------------------------------------------------
# Operators: opcode sentinels that are also callable high-level ops
# --------------------------------------------------------------------------


class _Op:
    def __init__(self, op_name, kind):
        self.op_name = op_name
        self.kind = kind  # "binary" | "unary" | "marker"

    def __repr__(self):
        return f"nl.{self.op_name}"

    def __call__(self, *args, **kwargs):
        if self.kind == "binary":
            return _binary_op(args[0], args[1], self, kwargs.get("dtype", args[2] if len(args) > 2 else None))
        if self.kind == "unary":
            return _unary_op(args[0], self, kwargs.get("dtype", args[1] if len(args) > 1 else None))
        raise KernelError(f"nl.{self.op_name} is not callable")


_BINARY = (
    "add subtract multiply divide maximum minimum abs_max abs_min power equal not_equal less "
    "less_equal greater greater_equal bitwise_and bitwise_or bitwise_xor left_shift right_shift "
    "logical_and logical_or logical_xor fmod mod"
).split()
_UNARY = (
    "exp log sqrt rsqrt tanh sigmoid relu gelu gelu_apprx_sigmoid gelu_apprx_tanh gelu_dx "
    "gelu_apprx_sigmoid_dx silu silu_dx softplus mish erf erf_dx reciprocal copy sin cos tan arctan "
    "ceil floor trunc negative sign abs square logical_not invert"
).split()

for _n in _BINARY:
    globals()[_n] = _Op(_n, "binary")
for _n in _UNARY:
    globals()[_n] = _Op(_n, "unary")
prelu = _Op("prelu", "marker")
bypass = _Op("bypass", "marker")
average = _Op("average", "marker")


def _is_vector_operand(t):
    return isinstance(t, NkiTensor) and len(t.shape) == 2 and t.shape[1] == 1


def _binary_op(x, y, op, dtype=None):
    xt = isinstance(x, NkiTensor)
    yt = isinstance(y, NkiTensor)
    if not (xt or yt):
        raise KernelError(f"nl.{op.op_name} needs at least one tensor input")
    if xt and (not yt or _is_vector_operand(y)):
        out = ndarray(x.shape, dtype or x.dtype, x.buffer if x.on_chip else sbuf)
        _isa.tensor_scalar(out, x, op0=op, operand0=y)
        return out
    if yt and (not xt or _is_vector_operand(x)):
        out = ndarray(y.shape, dtype or y.dtype, y.buffer if y.on_chip else sbuf)
        _isa.tensor_scalar(out, y, op0=op, operand0=x, reverse0=True)
        return out
    out = ndarray(x.shape, dtype or x.dtype, x.buffer if x.on_chip else sbuf)
    _isa.tensor_tensor(out, x, y, op=op)
    return out


def _unary_op(x, op, dtype=None):
    out = ndarray(x.shape, dtype or x.dtype, x.buffer if x.on_chip else sbuf)
    _isa.activation(out, op, x)
    return out


# --------------------------------------------------------------------------
# Allocation
# --------------------------------------------------------------------------


def ndarray(shape, dtype, buffer=sbuf, name="", address=None):
    ctx().count("alloc")
    return new_tensor(shape, dtype, buffer, name)


def zeros(shape, dtype, buffer=sbuf, name=""):
    return full(shape, 0.0, dtype, buffer, name)


def ones(shape, dtype, buffer=sbuf, name=""):
    return full(shape, 1.0, dtype, buffer, name)


def full(shape, fill_value, dtype, buffer=sbuf, name=""):
    t = ndarray(shape, dtype, buffer, name)
    t.write(ctx().domain.full(t.shape, fill_value))
    return t


def zeros_like(x, dtype=None, buffer=None, name=""):
    return zeros(x.shape, dtype or x.dtype, buffer or x.buffer, name)


def empty_like(x, dtype=None, buffer=None, name=""):
    return ndarray(x.shape, dtype or x.dtype, buffer or x.buffer, name)


def shared_identity_matrix(n, dtype=uint8, dst=None):
    t = dst if dst is not None else ndarray((n, n), dtype, sbuf)
    t.write(ctx().domain.from_numpy(np.eye(n)))
    return t


def shared_constant(constant):
    arr = np.asarray(constant)
    t = new_tensor(arr.shape, str(arr.dtype) if str(arr.dtype) in _DT else float32, shared_hbm)
    t.write(ctx().domain.from_numpy(arr.astype(np.float64)))
    return t


_DT = {"float32", "float16", "int32", "int8", "uint8", "int16", "uint16", "uint32"}


def ds(start, size):
    return DynSlice(start, size)


# --------------------------------------------------------------------------
# Loops
# --------------------------------------------------------------------------


def _range(start, stop, step):
    if stop is None:
        start, stop = 0, start
    for v in (start, stop, step):
        if not isinstance(v, (int, np.integer)):
            raise Unsupported("loop bounds that are not compile-time integers")
    return range(int(start), int(stop), int(step))


def affine_range(start, stop=None, step=1):
    c = ctx()
    r = _range(start, stop, step)
    site = R.current_site()

    def iterate():
        c.loop_instances += 1
        frame = [c.loop_instances, None, site]
        c.loop_stack.append(frame)
        try:
            for i in r:
                frame[1] = i
                yield i
        finally:
            c.loop_stack.remove(frame)

    return iterate()


def sequential_range(start, stop=None, step=1):
    return _range(start, stop, step)


def static_range(start, stop=None, step=1):
    return _range(start, stop, step)


def dynamic_range(start, stop=None, step=1):
    return _range(start, stop, step)


def fori_loop(lower, upper, body_fun, step=1, init_val=None):
    val = init_val
    for i in _range(lower, upper, step):
        val = body_fun(i, val) if init_val is not None else body_fun(i)
    return val


@contextlib.contextmanager
def no_reorder():
    yield


def device_print(print_prefix, tensor):
    return None


def program_id(axis=0):
    return ctx().program_id[axis]


def num_programs(axes=0):
    g = ctx().grid
    if isinstance(axes, (tuple, list)):
        return int(np.prod([g[a] for a in axes]))
    return g[axes]


def program_ndim():
    return len(ctx().grid)


# --------------------------------------------------------------------------
# High-level ops
# --------------------------------------------------------------------------


def load(src, dtype=None):
    out = ndarray(src.shape if len(src.shape) > 1 else (src.shape[0], 1), dtype or src.dtype, sbuf)
    _isa.dma_copy(out, src)
    return out


def store(dst, value):
    if isinstance(value, NkiTensor) and value.buffer == "psum":
        tmp = ndarray(value.shape, value.dtype, sbuf)
        _isa.tensor_copy(tmp, value)
        value = tmp
    if not isinstance(value, NkiTensor):
        tmp = full(dst.shape, value, dst.dtype)
        value = tmp
    _isa.dma_copy(dst, value)


def _reduce(x, axis, op, dtype, keepdims):
    axes = (axis,) if isinstance(axis, (int, np.integer)) else tuple(axis)
    axes = tuple(int(a) % x.ndim for a in axes)
    out_shape = tuple(1 if d in axes else s for d, s in enumerate(x.shape)) if keepdims else tuple(
        s for d, s in enumerate(x.shape) if d not in axes
    )
    if len(out_shape) == 1:
        out_shape = out_shape + (1,)
    out = ndarray(out_shape, dtype or x.dtype, sbuf)
    _isa.tensor_reduce(out, op, x, axes, keepdims=keepdims)
    return out


def sum(x, axis, dtype=None, keepdims=False):
    return _reduce(x, axis, add, dtype, keepdims)


def max(x, axis, dtype=None, keepdims=False):
    return _reduce(x, axis, maximum, dtype, keepdims)


def min(x, axis, dtype=None, keepdims=False):
    return _reduce(x, axis, minimum, dtype, keepdims)


def prod(x, axis, dtype=None, keepdims=False):
    return _reduce(x, axis, multiply, dtype, keepdims)


def all(x, axis, dtype=None):
    return _reduce(x, axis, logical_and, dtype, False)


def mean(x, axis, dtype=None, keepdims=False):
    s = sum(x, axis, dtype=float32, keepdims=keepdims)
    axes = (axis,) if isinstance(axis, (int, np.integer)) else tuple(axis)
    n = int(np.prod([x.shape[a] for a in axes]))
    return _binary_op(s, 1.0 / n, multiply, dtype or x.dtype)


def var(x, axis, dtype=None, keepdims=False):
    m = mean(x, axis, dtype=float32, keepdims=True)
    centered = ndarray(x.shape, float32, sbuf)
    _isa.tensor_scalar(centered, x, op0=subtract, operand0=m)
    sq = _unary_op(centered, square)
    return mean(sq, axis, dtype or x.dtype, keepdims)


def softmax(x, axis=-1, dtype=None):
    if int(axis) % x.ndim != x.ndim - 1:
        raise Unsupported("nl.softmax along a non-final axis")
    m = max(x, axis, dtype=float32, keepdims=True)
    shifted = ndarray(x.shape, float32, sbuf)
    _isa.tensor_scalar(shifted, x, op0=subtract, operand0=m)
    e = _unary_op(shifted, exp)
    s = sum(e, axis, dtype=float32, keepdims=True)
    r = _unary_op(s, reciprocal)
    out = ndarray(x.shape, dtype or x.dtype, sbuf)
    _isa.tensor_scalar(out, e, op0=multiply, operand0=r)
    return out


def where(condition, x, y, dtype=None):
    c = ctx()
    d = c.domain
    ref = condition if isinstance(condition, NkiTensor) else (x if isinstance(x, NkiTensor) else y)
    shape = ref.shape
    out = ndarray(shape, dtype or (x.dtype if isinstance(x, NkiTensor) else (y.dtype if isinstance(y, NkiTensor) else float32)), sbuf)
    cond = _isa._operand(condition, shape, "nl.where", "condition")
    xv = _isa._operand(x, shape, "nl.where", "x")
    yv = _isa._operand(y, shape, "nl.where", "y")
    c.count("where")
    out.write(d.where(cond, xv, yv))
    return out


def transpose(x, dtype=None):
    if x.ndim != 2:
        raise KernelError("nl.transpose expects a 2D tile")
    tmp = ndarray((x.shape[1], x.shape[0]), x.dtype, sbuf)
    _isa.nc_transpose(tmp, x)
    if (dtype or x.dtype) == x.dtype:
        return tmp
    out = ndarray(tmp.shape, dtype, sbuf)
    _isa.tensor_copy(out, tmp)
    return out


def matmul(x, y, transpose_x=False):
    if not transpose_x:
        xt = ndarray((x.shape[1], x.shape[0]), x.dtype, sbuf)
        _isa.nc_transpose(xt, x)
        x = xt
    out = ndarray((x.shape[1], y.shape[1]), float32, psum)
    _isa.nc_matmul(out, x, y)
    return out


def broadcast_to(x, shape, dtype=None):
    out = ndarray(shape, dtype or x.dtype, sbuf)
    vals = x.read()
    try:
        b = np.broadcast_to(vals, tuple(shape))
    except ValueError:
        raise KernelError(f"nl.broadcast_to: cannot broadcast {x.shape} to {tuple(shape)}")
    out.write(np.array(b))
    return out


def expand_dims(x, axis):
    return x.expand_dim(axis)


def load_transpose2d(src, dtype=None):
    out = ndarray((src.shape[1], src.shape[0]), dtype or src.dtype, sbuf)
    _isa.dma_transpose(out, src)
    return out


# --------------------------------------------------------------------------
# Module-level dynamic attributes
# --------------------------------------------------------------------------

_UNSUPPORTED = ("rms_norm", "dropout", "random_seed", "rand", "gather_flattened", "while_loop")


def __getattr__(name):
    if name == "tile_size":
        return ctx().profile
    if name in _UNSUPPORTED or not name.startswith("_"):
        raise Unsupported(f"nl.{name}")
    raise AttributeError(name)
