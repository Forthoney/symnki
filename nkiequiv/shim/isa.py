"""Shim for ``nki.isa``: instruction semantics over the active value domain.

Semantics follow the NKI 0.6 CPU simulator: engines compute in float32 (the
concrete hw domain models that; the symbolic domain computes over the reals),
results are cast to the destination dtype on store, PSUM matmul accumulation
follows the per-element written / accumulate rules, and the Scalar / Vector
engine reduce registers persist across instructions.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from ..domains import Unsupported
from . import runtime as R
from .runtime import KernelError, NkiTensor, ctx

# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------


class engine(Enum):
    unknown = 0
    tensor = 1
    vector = 2
    scalar = 3
    gpsimd = 4
    activation = 3
    sync = 5
    dma = 6


tensor_engine = engine.tensor
vector_engine = engine.vector
scalar_engine = engine.scalar
gpsimd_engine = engine.gpsimd
unknown_engine = engine.unknown


class reduce_cmd(Enum):
    idle = 0
    reset = 1
    reduce = 2
    reset_reduce = 3
    load_reduce = 4


class dge_mode(Enum):
    unknown = 0
    none = 1
    swdge = 2
    hwdge = 3


class oob_mode(Enum):
    error = 0
    skip = 1


class matmul_perf_mode(Enum):
    none = 0
    double_row = 1
    double_row_gen3 = 1


class nc_version(Enum):
    gen2 = 2
    gen3 = 3
    gen4 = 4


class dma_engine(Enum):
    dma = 0


class NkiValidationError(Exception):
    pass


def get_nc_version():
    t = ctx().target
    return {"trn1": nc_version.gen2, "trn2": nc_version.gen3, "trn3": nc_version.gen4}.get(t, nc_version.gen3)


def get_nc_sub_version():
    return 0


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _opname(op) -> str:
    name = getattr(op, "op_name", None)
    if name is None:
        if isinstance(op, str):
            return op
        raise KernelError(f"expected an nl operator (e.g. nl.add), got {op!r}")
    return name


def _need_tensor(t, what):
    if not isinstance(t, NkiTensor):
        raise KernelError(f"{what} must be an NKI tensor, got {type(t).__name__}")
    return t


def _on_chip(t, what, op):
    _need_tensor(t, f"{op}: {what}")
    if not t.on_chip:
        raise KernelError(f"{op}: {what} must be in SBUF or PSUM, got {t.buffer} (use nisa.dma_copy)")
    return t


def _shaped(vals, shape, op, what):
    """Reshape operand values to shape when the element counts agree."""
    if vals.shape == tuple(shape):
        return vals
    if vals.size == int(np.prod(shape)):
        return vals.reshape(shape)
    raise KernelError(f"{op}: {what} has shape {vals.shape}, incompatible with {tuple(shape)}")


def _check_partitions(op, dst, *srcs):
    for name, s in srcs:
        if s is not None and isinstance(s, NkiTensor) and s.on_chip and dst.on_chip:
            if s.shape[0] != dst.shape[0]:
                raise KernelError(
                    f"{op}: partition dimension of {name} ({s.shape[0]}) does not match dst ({dst.shape[0]})"
                )


def _operand(x, like_shape, op, what):
    """Scalar, [P,1] vector or full-tile operand, broadcast to like_shape."""
    c = ctx()
    if isinstance(x, NkiTensor):
        _on_chip(x, what, op)
        v = c.domain.to_compute(x.read())
        if v.shape == tuple(like_shape):
            return v
        P = like_shape[0]
        if v.size == P:
            return np.broadcast_to(v.reshape((P,) + (1,) * (len(like_shape) - 1)), like_shape)
        if v.size == int(np.prod(like_shape)):
            return v.reshape(like_shape)
        try:
            return np.broadcast_to(v, like_shape)
        except ValueError:
            raise KernelError(f"{op}: {what} of shape {v.shape} cannot broadcast to {tuple(like_shape)}")
    if x is None:
        raise KernelError(f"{op}: {what} is required")
    return c.domain.full(like_shape, float(x) if not c.domain.obj else x)


def _read(t, op, what, shape=None):
    _on_chip(t, what, op)
    v = ctx().domain.to_compute(t.read())
    return _shaped(v, shape, op, what) if shape is not None else v


def _store(dst, values, op):
    dst.write(_shaped(np.asarray(values), dst.shape, op, "result"))


def _psum_dtype_check(dst, op):
    if dst.buffer == "psum" and dst.dtype not in ("float32", "bfloat16", "int32"):
        raise KernelError(f"{op}: PSUM destination must be float32/bfloat16, got {dst.dtype}")


def _reduce_reset_value(op_name):
    if op_name == "maximum":
        return float("-inf")
    if op_name == "minimum":
        return float("inf")
    if op_name == "multiply":
        return 1.0
    return 0.0


def _regs_update(regs_name, partitions, result, cmd, reduce_op, reduce_res, reduce_init=None):
    """Shared Scalar/Vector engine accumulate-register semantics."""
    c = ctx()
    d = c.domain
    regs = getattr(c, regs_name)
    if isinstance(cmd, int):
        cmd = reduce_cmd(cmd)
    if cmd == reduce_cmd.idle and reduce_op is not None and regs_name == "act_regs":
        raise KernelError("reduce_op cannot be specified when reduce_cmd is idle")
    rname = _opname(reduce_op) if reduce_op is not None else "add"
    if cmd in (reduce_cmd.reset, reduce_cmd.reset_reduce):
        regs[:partitions] = d.full((partitions,), _reduce_reset_value(rname) if regs_name == "dve_regs" else 0.0)
    elif cmd == reduce_cmd.load_reduce:
        if isinstance(reduce_init, NkiTensor):
            regs[:partitions] = d.to_compute(reduce_init.read()).reshape(-1)[:partitions]
        else:
            regs[:partitions] = d.full((partitions,), reduce_init or 0.0)
    if cmd in (reduce_cmd.reduce, reduce_cmd.reset_reduce, reduce_cmd.load_reduce):
        if reduce_op is None and regs_name == "act_regs":
            raise KernelError(f"reduce_op must be specified when reduce_cmd is {cmd.name}")
        axes = tuple(range(1, result.ndim))
        red = d.reduce(rname, result, axes) if axes else result
        red = np.asarray(red).reshape(-1)
        cur = regs[:partitions]
        if rname == "add" or regs_name == "act_regs":
            regs[:partitions] = d.binary("add", cur, red)
        else:
            regs[:partitions] = d.binary(rname, cur, red)
    if reduce_res is not None:
        _on_chip(reduce_res, "reduce_res", "reduce_res")
        vals = regs[:partitions].copy()
        reduce_res.write(_shaped(vals.reshape(partitions, 1), reduce_res.shape, "reduce_res", "reduce_res"))


# --------------------------------------------------------------------------
# Data movement
# --------------------------------------------------------------------------


def dma_copy(dst, src, priority=None, oob_mode=oob_mode.error, dge_mode=dge_mode.unknown, engine=engine.unknown, name=None):
    c = ctx()
    c.count("dma_copy")
    _need_tensor(dst, "dma_copy: dst")
    _need_tensor(src, "dma_copy: src")
    if dst.buffer == "psum" or src.buffer == "psum":
        raise KernelError("dma_copy cannot access PSUM; copy PSUM to SBUF with nisa.tensor_copy first")
    vals = src.read()
    if vals.size != dst.size:
        raise KernelError(f"dma_copy: src {src.shape} and dst {dst.shape} have different sizes")
    dst.write(vals.reshape(dst.shape))


def dma_transpose(dst, src, axes=None, priority=None, dge_mode=dge_mode.unknown, oob_mode=oob_mode.error, name=None):
    c = ctx()
    c.count("dma_transpose")
    vals = src.read()
    if axes is None:
        axes = tuple(reversed(range(vals.ndim)))
    t = np.transpose(vals, axes)
    if t.size != dst.size:
        raise KernelError(f"dma_transpose: result {t.shape} does not fit dst {dst.shape}")
    dst.write(t.reshape(dst.shape))


def tensor_copy(dst, src, engine=engine.unknown, name=None):
    c = ctx()
    c.count("tensor_copy")
    _on_chip(dst, "dst", "tensor_copy")
    vals = _read(src, "tensor_copy", "src")
    _check_partitions("tensor_copy", dst, ("src", src))
    _store(dst, vals, "tensor_copy")


def memset(dst, value, engine=engine.unknown, name=None):
    c = ctx()
    c.count("memset")
    _on_chip(dst, "dst", "memset")
    dst.write(c.domain.full(dst.shape, value))


def tensor_copy_predicated(dst, src, predicate, reverse_pred=False, name=None):
    c = ctx()
    c.count("tensor_copy_predicated")
    d = c.domain
    _on_chip(dst, "dst", "tensor_copy_predicated")
    pred = _read(predicate, "tensor_copy_predicated", "predicate", dst.shape)
    if reverse_pred:
        pred = d.unary("logical_not", pred)
    new = _operand(src, dst.shape, "tensor_copy_predicated", "src")
    old = d.to_compute(dst.read())
    _store(dst, d.where(pred, new, old), "tensor_copy_predicated")


# --------------------------------------------------------------------------
# Tensor engine
# --------------------------------------------------------------------------


def _flat2(vals):
    return vals.reshape(vals.shape[0], -1)


def nc_matmul(dst, stationary, moving, is_stationary_onezero=False, is_moving_onezero=False,
              is_transpose=False, accumulate=None, tile_position=(), tile_size=(),
              perf_mode=matmul_perf_mode.none, name=None):
    c = ctx()
    c.count("nc_matmul")
    d = c.domain
    p = c.profile
    _on_chip(stationary, "stationary", "nc_matmul")
    _on_chip(moving, "moving", "nc_matmul")
    _need_tensor(dst, "nc_matmul: dst")
    if dst.buffer != "psum":
        raise KernelError(f"nc_matmul: dst must be in PSUM, got {dst.buffer}")
    if stationary.buffer != "sbuf" or moving.buffer != "sbuf":
        raise KernelError("nc_matmul: stationary and moving must be in SBUF")
    _psum_dtype_check(dst, "nc_matmul")
    if perf_mode not in (None, matmul_perf_mode.none, "none"):
        raise Unsupported(f"nc_matmul perf_mode={perf_mode}")
    s = _flat2(d.to_compute(stationary.read()))
    m = _flat2(d.to_compute(moving.read()))
    K, M = s.shape
    K2, N = m.shape
    if K != K2:
        raise KernelError(f"nc_matmul: contraction dims differ (stationary {K}, moving {K2})")
    if K > p.pmax:
        raise KernelError(f"nc_matmul: contraction dim {K} exceeds pmax={p.pmax}")
    if M > p.gemm_stationary_fmax:
        raise KernelError(f"nc_matmul: stationary free size {M} exceeds gemm_stationary_fmax={p.gemm_stationary_fmax}")
    if N > p.gemm_moving_fmax:
        raise KernelError(f"nc_matmul: moving free size {N} exceeds gemm_moving_fmax={p.gemm_moving_fmax}")
    if dst.size != M * N:
        raise KernelError(f"nc_matmul: dst {dst.shape} does not hold a [{M}, {N}] result")
    if dst.shape[0] != M:
        raise KernelError(f"nc_matmul: dst partition dim {dst.shape[0]} != stationary free size {M}")
    if N > p.psum_bank_fmax:
        raise KernelError(f"nc_matmul: result free size {N} exceeds a PSUM bank ({p.psum_bank_fmax} fp32)")
    result = d.matmul_T(s, m).reshape(dst.shape)

    st = dst._storage
    idx = dst._idx
    if accumulate is True:
        if not st.written[idx].all():
            c.warn("nc_matmul(accumulate=True) without a prior write to the PSUM destination (undefined behavior)")
        elif (st.written[idx] & ~st.mm[idx]).any() and c.target in ("trn1", "trn2"):
            c.warn("nc_matmul(accumulate=True) onto PSUM written by a non-matmul instruction (UB on trn1/trn2)")
        mask = np.ones(dst.shape, dtype=bool)
    elif accumulate is False:
        st.written[:] = False
        st.mm[:] = False
        mask = np.zeros(dst.shape, dtype=bool)
    elif is_transpose:
        mask = np.zeros(dst.shape, dtype=bool)
    elif accumulate is None:
        mask = st.written[idx].copy()
        non_mm = mask & ~st.mm[idx]
        if non_mm.any() and c.target in ("trn1", "trn2"):
            c.warn("nc_matmul accumulates onto PSUM written by a non-matmul instruction (wrong on trn1/trn2 hardware)")
    else:
        raise Unsupported("nc_matmul with a dynamic accumulate flag")
    old = d.to_compute(dst.read(accumulate=True))
    out = result.copy() if hasattr(result, "copy") else result
    if mask.any():
        out[mask] = d.binary("add", old[mask], result[mask])
        if dst.dtype != "float32":
            c.note_precision(f"nc_matmul accumulates into a {dst.dtype} PSUM tile")
    dst.write(out, accumulate=True)
    st.mm[idx] = True


def nc_transpose(dst, data, engine=engine.unknown, name=None):
    c = ctx()
    c.count("nc_transpose")
    vals = _flat2(_read(data, "nc_transpose", "data"))
    if data.dtype != dst.dtype:
        raise KernelError(f"nc_transpose: dst dtype {dst.dtype} must equal data dtype {data.dtype}")
    t = vals.T
    if t.shape[0] > c.profile.pmax:
        raise KernelError(f"nc_transpose: result partition dim {t.shape[0]} exceeds pmax")
    _on_chip(dst, "dst", "nc_transpose")
    _store(dst, t, "nc_transpose")


# --------------------------------------------------------------------------
# Vector / scalar / gpsimd engines
# --------------------------------------------------------------------------


def tensor_tensor(dst, data1, data2, op, engine=engine.unknown, name=None):
    c = ctx()
    c.count("tensor_tensor")
    d = c.domain
    _on_chip(dst, "dst", "tensor_tensor")
    if data1.buffer == "psum" and data2.buffer == "psum":
        c.warn("tensor_tensor: both inputs in PSUM is not supported on hardware")
    a = _read(data1, "tensor_tensor", "data1", dst.shape)
    b = _read(data2, "tensor_tensor", "data2", dst.shape)
    _check_partitions("tensor_tensor", dst, ("data1", data1), ("data2", data2))
    if _opname(op) in ("add", "subtract") and dst.dtype not in ("float32", "int32") and (
        dst._storage is data1._storage or dst._storage is data2._storage
    ):
        c.note_precision(f"tensor_tensor accumulates in place into a {dst.dtype} tile")
    _store(dst, d.binary(_opname(op), a, b), "tensor_tensor")


_TS_UNARY = ("abs", "rsqrt", "square", "relu")


def _apply_scalar_op(d, op, x, operand, reverse, shape, opname):
    name = _opname(op)
    if name in _TS_UNARY:
        return d.unary(name, x)
    v = _operand(operand, shape, opname, "operand")
    return d.binary(name, v, x) if reverse else d.binary(name, x, v)


def tensor_scalar(dst, data, op0, operand0=None, reverse0=False, op1=None, operand1=None,
                  reverse1=False, engine=engine.unknown, name=None):
    c = ctx()
    c.count("tensor_scalar")
    d = c.domain
    _on_chip(dst, "dst", "tensor_scalar")
    x = _read(data, "tensor_scalar", "data", dst.shape)
    _check_partitions("tensor_scalar", dst, ("data", data))
    r = _apply_scalar_op(d, op0, x, operand0, reverse0, dst.shape, "tensor_scalar")
    if op1 is not None and (operand1 is not None or _opname(op1) in _TS_UNARY):
        r = _apply_scalar_op(d, op1, r, operand1, reverse1, dst.shape, "tensor_scalar")
    _store(dst, r, "tensor_scalar")


def scalar_tensor_tensor(dst, data, op0, operand0, op1, operand1, reverse0=False, reverse1=False, name=None):
    c = ctx()
    c.count("scalar_tensor_tensor")
    d = c.domain
    _on_chip(dst, "dst", "scalar_tensor_tensor")
    x = _read(data, "scalar_tensor_tensor", "data", dst.shape)
    r = _apply_scalar_op(d, op0, x, operand0, reverse0, dst.shape, "scalar_tensor_tensor")
    other = _read(operand1, "scalar_tensor_tensor", "operand1", dst.shape)
    n1 = _opname(op1)
    r = d.binary(n1, other, r) if reverse1 else d.binary(n1, r, other)
    _store(dst, r, "scalar_tensor_tensor")


def _norm_axes(axis, ndim, op):
    axes = (axis,) if isinstance(axis, (int, np.integer)) else tuple(axis)
    axes = tuple(sorted(int(a) % ndim for a in axes))
    if 0 in axes:
        raise KernelError(f"{op}: cannot reduce along the partition axis (0)")
    if axes != tuple(range(ndim - len(axes), ndim)):
        raise KernelError(f"{op}: reduction axes {axes} must be the last contiguous free dims")
    return axes


def tensor_reduce(dst, op, data, axis, negate=False, keepdims=False, name=None):
    c = ctx()
    c.count("tensor_reduce")
    d = c.domain
    _on_chip(dst, "dst", "tensor_reduce")
    x = _read(data, "tensor_reduce", "data")
    axes = _norm_axes(axis, x.ndim, "tensor_reduce")
    r = d.reduce(_opname(op), x, axes, keepdims=keepdims)
    if negate:
        r = d.unary("negative", r)
    _check_partitions("tensor_reduce", dst, ("data", data))
    _store(dst, r, "tensor_reduce")


def tensor_partition_reduce(dst, op, data, name=None):
    c = ctx()
    c.count("tensor_partition_reduce")
    d = c.domain
    x = _read(data, "tensor_partition_reduce", "data")
    r = d.reduce(_opname(op), x, (0,), keepdims=True)
    _on_chip(dst, "dst", "tensor_partition_reduce")
    _store(dst, r, "tensor_partition_reduce")


def _activation_core(op, x, bias, scale, shape, opname, relu_param=None):
    c = ctx()
    d = c.domain
    if isinstance(scale, NkiTensor) or scale != 1.0:
        x = d.binary("multiply", x, _operand(scale, shape, opname, "scale"))
    if bias is not None:
        x = d.binary("add", x, _operand(bias, shape, opname, "bias"))
    name = _opname(op)
    if name == "prelu":
        return d.prelu(x, _operand(relu_param or 0.0, shape, opname, "relu_param"))
    if name == "bypass":
        return x
    return d.unary(name, x)


def activation(dst, op, data, bias=None, scale=1.0, reduce_op=None, reduce_res=None,
               reduce_cmd=reduce_cmd.idle, name=None):
    c = ctx()
    c.count("activation")
    _on_chip(dst, "dst", "activation")
    x = _read(data, "activation", "data", dst.shape)
    _check_partitions("activation", dst, ("data", data))
    if reduce_op is not None and _opname(reduce_op) != "add":
        raise KernelError("activation: reduce_op must be nl.add")
    if reduce_res is not None and reduce_cmd == globals()["reduce_cmd"].idle and reduce_op is not None:
        reduce_cmd = globals()["reduce_cmd"].reset_reduce
    result = _activation_core(op, x, bias, scale, dst.shape, "activation")
    _store(dst, result, "activation")
    _regs_update("act_regs", dst.shape[0], np.asarray(result), reduce_cmd, reduce_op, reduce_res)


def activation_reduce(dst, op, data, reduce_op, reduce_res, bias=None, scale=1.0, name=None):
    activation(dst, op, data, bias=bias, scale=scale, reduce_op=reduce_op, reduce_res=reduce_res,
               reduce_cmd=reduce_cmd.reset_reduce, name=name)


def activate2(dst, op, data, imm0, imm1, op0, op1, relu_param=0.0, reverse0=False, reverse1=False,
              reduce_op=None, reduce_res=None, reduce_cmd=reduce_cmd.idle, name=None):
    c = ctx()
    c.count("activate2")
    d = c.domain
    _on_chip(dst, "dst", "activate2")
    x = _read(data, "activate2", "data", dst.shape)
    for opi, imm, rev in ((op0, imm0, reverse0), (op1, imm1, reverse1)):
        if opi is not None and _opname(opi) != "bypass":
            v = _operand(imm, dst.shape, "activate2", "imm")
            x = d.binary(_opname(opi), v, x) if rev else d.binary(_opname(opi), x, v)
    name_ = _opname(op)
    if name_ == "prelu":
        result = d.prelu(x, _operand(relu_param, dst.shape, "activate2", "relu_param"))
    elif name_ == "bypass":
        result = x
    else:
        result = d.unary(name_, x)
    _store(dst, result, "activate2")
    _regs_update("act_regs", dst.shape[0], np.asarray(result), reduce_cmd, reduce_op, reduce_res)


def exponential(dst, src, max_value=0.0, reduce_res=None, reduce_cmd=reduce_cmd.idle, reduce_init=0.0, name=None):
    c = ctx()
    c.count("exponential")
    d = c.domain
    _on_chip(dst, "dst", "exponential")
    x = _read(src, "exponential", "src", dst.shape)
    mv = _operand(max_value, dst.shape, "exponential", "max_value")
    result = d.unary("exp", d.binary("subtract", x, mv))
    _store(dst, result, "exponential")
    if reduce_res is not None and reduce_cmd == globals()["reduce_cmd"].idle:
        reduce_cmd = globals()["reduce_cmd"].reset_reduce
    from . import language as nl

    _regs_update("dve_regs", dst.shape[0], np.asarray(result), reduce_cmd, nl.add, reduce_res, reduce_init)


def reciprocal(dst, data, name=None):
    c = ctx()
    c.count("reciprocal")
    _on_chip(dst, "dst", "reciprocal")
    x = _read(data, "reciprocal", "data", dst.shape)
    _store(dst, c.domain.unary("reciprocal", x), "reciprocal")


def tensor_scalar_reduce(dst, data, op0, operand0, reduce_op, reduce_res=None, reverse0=False,
                         reduce_cmd=reduce_cmd.reset_reduce, reduce_init=None, name=None):
    c = ctx()
    c.count("tensor_scalar_reduce")
    d = c.domain
    _on_chip(dst, "dst", "tensor_scalar_reduce")
    x = _read(data, "tensor_scalar_reduce", "data", dst.shape)
    r = _apply_scalar_op(d, op0, x, operand0, reverse0, dst.shape, "tensor_scalar_reduce")
    _store(dst, r, "tensor_scalar_reduce")
    _regs_update("dve_regs", dst.shape[0], np.asarray(r), reduce_cmd, reduce_op, reduce_res, reduce_init)


def select_reduce(dst, predicate, on_true, on_false, reduce_res=None, reduce_cmd=reduce_cmd.idle,
                  reduce_op=None, reverse_pred=False, name=None):
    c = ctx()
    c.count("select_reduce")
    d = c.domain
    _on_chip(dst, "dst", "select_reduce")
    pred = _read(predicate, "select_reduce", "predicate", dst.shape)
    if reverse_pred:
        pred = d.unary("logical_not", pred)
    t = _read(on_true, "select_reduce", "on_true", dst.shape)
    f = _operand(on_false, dst.shape, "select_reduce", "on_false")
    r = d.where(pred, t, f)
    _store(dst, r, "select_reduce")
    from . import language as nl

    _regs_update("dve_regs", dst.shape[0], np.asarray(r), reduce_cmd, reduce_op or nl.maximum, reduce_res)


# --------------------------------------------------------------------------
# Pattern generation
# --------------------------------------------------------------------------


def _affine_values(pattern, offset, channel_multiplier, P, op):
    if not isinstance(offset, (int, np.integer)):
        raise Unsupported(f"{op} with a dynamic offset")
    counts = [int(n) for _, n in pattern]
    steps = [int(s) for s, _ in pattern]
    vals = np.full(counts, int(offset), dtype=np.int64)
    for dim, (s, n) in enumerate(zip(steps, counts)):
        shape = [1] * len(counts)
        shape[dim] = n
        vals = vals + (np.arange(n, dtype=np.int64) * s).reshape(shape)
    vals = vals.reshape(1, -1)
    ch = (np.arange(P, dtype=np.int64) * int(channel_multiplier)).reshape(-1, 1)
    return vals + ch


def iota(dst, pattern, offset=0, channel_multiplier=0, name=None):
    c = ctx()
    c.count("iota")
    _on_chip(dst, "dst", "iota")
    v = _affine_values(pattern, offset, channel_multiplier, dst.shape[0], "iota")
    if v.size != dst.size:
        raise KernelError(f"iota: pattern produces {v.shape[1]} values per partition, dst holds {dst.size // dst.shape[0]}")
    dst.write(c.domain.from_numpy(v.astype(np.float64)).reshape(dst.shape))


_CMP = {
    "equal": np.equal, "not_equal": np.not_equal, "less": np.less, "less_equal": np.less_equal,
    "greater": np.greater, "greater_equal": np.greater_equal,
}


def affine_select(dst, pattern, channel_multiplier, on_true_tile, on_false_value, cmp_op=None, offset=0, name=None):
    c = ctx()
    c.count("affine_select")
    d = c.domain
    _on_chip(dst, "dst", "affine_select")
    cmp_name = _opname(cmp_op) if cmp_op is not None else "equal"
    if cmp_name not in _CMP:
        raise KernelError(f"affine_select: unsupported comparison {cmp_name}")
    v = _affine_values(pattern, offset, channel_multiplier, dst.shape[0], "affine_select")
    pred = _CMP[cmp_name](v, 0).reshape(dst.shape)
    t = _read(on_true_tile, "affine_select", "on_true_tile", dst.shape)
    f = d.full(dst.shape, on_false_value)
    out = np.where(pred, t, f)
    _store(dst, out, "affine_select")


# --------------------------------------------------------------------------
# Everything else
# --------------------------------------------------------------------------


_UNSUPPORTED = (
    "nc_matmul_mx", "local_gather", "nc_n_gather", "bn_stats", "bn_aggr", "tensor_tensor_scan",
    "nc_stream_shuffle", "rng", "dropout", "set_rng_seed", "rand_set_state", "rand_get_state", "rand2",
    "max8", "sequence_bounds", "core_barrier", "dma_compute", "quantize_mx", "range_select", "sendrecv",
    "register_alloc", "register_move", "register_load", "register_store", "nc_find_index8",
    "nc_match_replace8", "nonzero_with_count", "topk", "tensor_scalar_cumulative",
)


def __getattr__(name):
    if name in _UNSUPPORTED or not name.startswith("_"):
        raise Unsupported(f"nisa.{name}")
    raise AttributeError(name)
