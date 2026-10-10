"""Row softmax kernels: rows on partitions, softmax along the free dimension."""

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def softmax_ref(x):
    R, C = x.shape
    P = nl.tile_size.pmax
    out = nl.ndarray((R, C), dtype=x.dtype, buffer=nl.shared_hbm)
    for r in nl.affine_range(R // P):
        t = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=x[r * P:(r + 1) * P, :])
        mx = nl.ndarray((P, 1), nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=mx, op=nl.maximum, data=t, axis=1)
        sh = nl.ndarray((P, C), nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=sh, data=t, op0=nl.subtract, operand0=mx)
        e = nl.ndarray((P, C), nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=e, op=nl.exp, data=sh)
        s = nl.ndarray((P, 1), nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=s, op=nl.add, data=e, axis=1)
        rs = nl.ndarray((P, 1), nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=rs, data=s)
        o = nl.ndarray((P, C), x.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=o, data=e, op0=nl.multiply, operand0=rs)
        nisa.dma_copy(dst=out[r * P:(r + 1) * P, :], src=o)
    return out


@nki.jit
def softmax_highlevel(x):
    """Same thing through the nl.* convenience API."""
    R, C = x.shape
    P = nl.tile_size.pmax
    out = nl.ndarray((R, C), dtype=x.dtype, buffer=nl.shared_hbm)
    for r in nl.affine_range(R // P):
        t = nl.load(x[r * P:(r + 1) * P, :])
        nl.store(out[r * P:(r + 1) * P, :], nl.softmax(t, axis=1))
    return out


@nki.jit
def softmax_fused(x):
    """exp, bias (-max) and row sum fused into one Scalar Engine instruction."""
    R, C = x.shape
    P = nl.tile_size.pmax
    out = nl.ndarray((R, C), dtype=x.dtype, buffer=nl.shared_hbm)
    for r in nl.affine_range(R // P):
        t = nl.ndarray((P, C), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=x[r * P:(r + 1) * P, :])
        neg_max = nl.ndarray((P, 1), nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=neg_max, op=nl.maximum, data=t, axis=1, negate=True)
        e = nl.ndarray((P, C), nl.float32, buffer=nl.sbuf)
        s = nl.ndarray((P, 1), nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=e, op=nl.exp, data=t, bias=neg_max, reduce_op=nl.add, reduce_res=s,
                        reduce_cmd=nisa.reduce_cmd.reset_reduce)
        o = nl.ndarray((P, C), x.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=o, data=e, op0=nl.divide, operand0=s)
        nisa.dma_copy(dst=out[r * P:(r + 1) * P, :], src=o)
    return out


def _online_stats(x, r, P, C, W, rescale=True):
    """Running max / running sum over column chunks of width W (flash-style)."""
    m = nl.ndarray((P, 1), nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=m, value=-3.0e38)
    l = nl.ndarray((P, 1), nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=l, value=0.0)
    for j in nl.sequential_range(C // W):
        chunk = nl.ndarray((P, W), nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=chunk, src=x[r * P:(r + 1) * P, j * W:(j + 1) * W])
        cm = nl.ndarray((P, 1), nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=cm, op=nl.maximum, data=chunk, axis=1)
        m_new = nl.ndarray((P, 1), nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=m_new, data1=m, data2=cm, op=nl.maximum)
        neg_m_new = nl.ndarray((P, 1), nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=neg_m_new, data=m_new, op0=nl.multiply, operand0=-1.0)
        e = nl.ndarray((P, W), nl.float32, buffer=nl.sbuf)
        cs = nl.ndarray((P, 1), nl.float32, buffer=nl.sbuf)
        nisa.activation_reduce(dst=e, op=nl.exp, data=chunk, bias=neg_m_new, reduce_op=nl.add, reduce_res=cs)
        if rescale:
            # alpha = exp(m_old - m_new); l = l * alpha + chunk_sum
            alpha = nl.ndarray((P, 1), nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=alpha, op=nl.exp, data=m, bias=neg_m_new)
            nisa.scalar_tensor_tensor(dst=l, data=l, op0=nl.multiply, operand0=alpha, op1=nl.add, operand1=cs)
        else:
            nisa.tensor_tensor(dst=l, data1=l, data2=cs, op=nl.add)
        nisa.tensor_copy(dst=m, src=m_new)
    return m, l


@nki.jit
def softmax_online(x):
    """Two-pass online softmax over column chunks (the flash-attention recurrence)."""
    R, C = x.shape
    P = nl.tile_size.pmax
    W = C // 4
    out = nl.ndarray((R, C), dtype=x.dtype, buffer=nl.shared_hbm)
    for r in nl.affine_range(R // P):
        m, l = _online_stats(x, r, P, C, W)
        neg_m = nl.ndarray((P, 1), nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=neg_m, data=m, op0=nl.multiply, operand0=-1.0)
        inv_l = nl.ndarray((P, 1), nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=inv_l, data=l)
        for j in nl.affine_range(C // W):
            chunk = nl.ndarray((P, W), nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=chunk, src=x[r * P:(r + 1) * P, j * W:(j + 1) * W])
            e = nl.ndarray((P, W), nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=e, op=nl.exp, data=chunk, bias=neg_m)
            o = nl.ndarray((P, W), x.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=o, data=e, op0=nl.multiply, operand0=inv_l)
            nisa.dma_copy(dst=out[r * P:(r + 1) * P, j * W:(j + 1) * W], src=o)
    return out


@nki.jit
def bug_online_no_rescale(x):
    """Online softmax that forgets to rescale the running sum when the max grows."""
    R, C = x.shape
    P = nl.tile_size.pmax
    W = C // 4
    out = nl.ndarray((R, C), dtype=x.dtype, buffer=nl.shared_hbm)
    for r in nl.affine_range(R // P):
        m, l = _online_stats(x, r, P, C, W, rescale=False)
        neg_m = nl.ndarray((P, 1), nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=neg_m, data=m, op0=nl.multiply, operand0=-1.0)
        for j in nl.affine_range(C // W):
            chunk = nl.ndarray((P, W), nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=chunk, src=x[r * P:(r + 1) * P, j * W:(j + 1) * W])
            e = nl.ndarray((P, W), nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=e, op=nl.exp, data=chunk, bias=neg_m)
            o = nl.ndarray((P, W), x.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=o, data=e, op0=nl.divide, operand0=l)
            nisa.dma_copy(dst=out[r * P:(r + 1) * P, j * W:(j + 1) * W], src=o)
    return out


@nki.jit
def risky_first_chunk_max(x):
    """Shifts by the max of the first chunk only.

    Exactly equal to softmax over the reals (softmax is shift invariant), but
    exp overflows when a later chunk is much larger -- only the numerics stage
    can see that.
    """
    R, C = x.shape
    P = nl.tile_size.pmax
    W = C // 4
    out = nl.ndarray((R, C), dtype=x.dtype, buffer=nl.shared_hbm)
    for r in nl.affine_range(R // P):
        t = nl.ndarray((P, C), nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=x[r * P:(r + 1) * P, :])
        neg_m = nl.ndarray((P, 1), nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=neg_m, op=nl.maximum, data=t[:, 0:W], axis=1, negate=True)
        e = nl.ndarray((P, C), nl.float32, buffer=nl.sbuf)
        s = nl.ndarray((P, 1), nl.float32, buffer=nl.sbuf)
        nisa.activation_reduce(dst=e, op=nl.exp, data=t, bias=neg_m, reduce_op=nl.add, reduce_res=s)
        o = nl.ndarray((P, C), x.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=o, data=e, op0=nl.divide, operand0=s)
        nisa.dma_copy(dst=out[r * P:(r + 1) * P, :], src=o)
    return out


def args(ts):
    return [("x", (2 * ts.pmax, 2 * ts.psum_bank_fmax), "float32")]
