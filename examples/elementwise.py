"""Elementwise kernels whose equivalence needs case analysis (the Z3 path)."""

import nki
import nki.isa as nisa
import nki.language as nl


def _tiles(x):
    R, C = x.shape
    P = nl.tile_size.pmax
    return R, C, P


@nki.jit
def relu_ref(x):
    R, C, P = _tiles(x)
    out = nl.ndarray((R, C), dtype=x.dtype, buffer=nl.shared_hbm)
    for r in nl.affine_range(R // P):
        t = nl.ndarray((P, C), x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=x[r * P:(r + 1) * P, :])
        o = nl.ndarray((P, C), x.dtype, buffer=nl.sbuf)
        nisa.activation(dst=o, op=nl.relu, data=t)
        nisa.dma_copy(dst=out[r * P:(r + 1) * P, :], src=o)
    return out


@nki.jit
def relu_via_abs(x):
    """relu(x) = (x + |x|) / 2 -- equal, but only provable by case analysis."""
    R, C, P = _tiles(x)
    out = nl.ndarray((R, C), dtype=x.dtype, buffer=nl.shared_hbm)
    for r in nl.affine_range(R // P):
        t = nl.ndarray((P, C), x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=x[r * P:(r + 1) * P, :])
        a = nl.ndarray((P, C), nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=a, data=t, op0=nl.abs, operand0=0.0)
        s = nl.ndarray((P, C), nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=s, data1=t, data2=a, op=nl.add)
        o = nl.ndarray((P, C), x.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=o, data=s, op0=nl.multiply, operand0=0.5)
        nisa.dma_copy(dst=out[r * P:(r + 1) * P, :], src=o)
    return out


@nki.jit
def bug_relu_via_abs(x):
    """Wrong: (|x| - x) / 2 is relu(-x)."""
    R, C, P = _tiles(x)
    out = nl.ndarray((R, C), dtype=x.dtype, buffer=nl.shared_hbm)
    for r in nl.affine_range(R // P):
        t = nl.ndarray((P, C), x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=x[r * P:(r + 1) * P, :])
        a = nl.ndarray((P, C), nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=a, data=t, op0=nl.abs, operand0=0.0)
        s = nl.ndarray((P, C), nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=s, data1=a, data2=t, op=nl.subtract)
        o = nl.ndarray((P, C), x.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=o, data=s, op0=nl.multiply, operand0=0.5)
        nisa.dma_copy(dst=out[r * P:(r + 1) * P, :], src=o)
    return out


@nki.jit
def clamp_ref(x):
    """min(max(x, -1), 1)"""
    R, C, P = _tiles(x)
    out = nl.ndarray((R, C), dtype=x.dtype, buffer=nl.shared_hbm)
    for r in nl.affine_range(R // P):
        t = nl.ndarray((P, C), x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=x[r * P:(r + 1) * P, :])
        o = nl.ndarray((P, C), x.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=o, data=t, op0=nl.maximum, operand0=-1.0, op1=nl.minimum, operand1=1.0)
        nisa.dma_copy(dst=out[r * P:(r + 1) * P, :], src=o)
    return out


@nki.jit
def clamp_swapped(x):
    """max(min(x, 1), -1): same function, different nesting."""
    R, C, P = _tiles(x)
    out = nl.ndarray((R, C), dtype=x.dtype, buffer=nl.shared_hbm)
    for r in nl.affine_range(R // P):
        t = nl.ndarray((P, C), x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=x[r * P:(r + 1) * P, :])
        o = nl.ndarray((P, C), x.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=o, data=t, op0=nl.minimum, operand0=1.0, op1=nl.maximum, operand1=-1.0)
        nisa.dma_copy(dst=out[r * P:(r + 1) * P, :], src=o)
    return out


def args(ts):
    return [("x", (2 * ts.pmax, ts.psum_bank_fmax), "float32")]


@nki.jit
def relu_spmd(x):
    """Rows split across SPMD programs (launch with a grid of 2)."""
    R, C, P = _tiles(x)
    n_prog = nl.num_programs(0)
    pid = nl.program_id(0)
    rows = R // n_prog
    out = nl.ndarray((R, C), dtype=x.dtype, buffer=nl.shared_hbm)
    for r in nl.affine_range(rows // P):
        r0 = pid * rows + r * P
        t = nl.ndarray((P, C), x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=x[r0:r0 + P, :])
        o = nl.ndarray((P, C), x.dtype, buffer=nl.sbuf)
        nisa.activation(dst=o, op=nl.relu, data=t)
        nisa.dma_copy(dst=out[r0:r0 + P, :], src=o)
    return out
