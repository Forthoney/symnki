"""Single-head causal attention: reference vs. a flash-attention style rewrite.

Inputs are pre-transposed so the head dim sits on partitions:
  qT, kT: (d, S)   v: (S, d)   ->  out: (S, d)
"""

import math

import nki
import nki.isa as nisa
import nki.language as nl

MASK = -9984.0


def _scores(qT, kT, q0, k0, P, B, scale):
    """(P, B) scaled scores for query rows q0.. and key columns k0.."""
    d = qT.shape[0]
    q_t = nl.ndarray((d, P), qT.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=q_t, src=qT[:, q0:q0 + P])
    k_t = nl.ndarray((d, B), kT.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=k_t, src=kT[:, k0:k0 + B])
    s_ps = nl.ndarray((P, B), nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=s_ps, stationary=q_t, moving=k_t)
    s = nl.ndarray((P, B), nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=s, data=s_ps, op0=nl.multiply, operand0=scale)
    return s


def _causal(s, q0, k0, B, strict=False):
    """Keep s[p, j] where (q0 + p) - (k0 + j) >= 0."""
    masked = nl.ndarray(s.shape, nl.float32, buffer=nl.sbuf)
    nisa.affine_select(dst=masked, pattern=[[-1, B]], channel_multiplier=1, on_true_tile=s,
                       on_false_value=MASK, cmp_op=nl.greater if strict else nl.greater_equal,
                       offset=q0 - k0)
    return masked


@nki.jit
def attention_ref(qT, kT, v):
    d, S = qT.shape
    P = nl.tile_size.pmax
    scale = 1.0 / math.sqrt(d)
    out = nl.ndarray((S, d), dtype=v.dtype, buffer=nl.shared_hbm)
    for qi in nl.affine_range(S // P):
        q0 = qi * P
        s = _causal(_scores(qT, kT, q0, 0, P, S, scale), q0, 0, S)
        # softmax over the full row
        neg_m = nl.ndarray((P, 1), nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=neg_m, op=nl.maximum, data=s, axis=1, negate=True)
        p = nl.ndarray((P, S), nl.float32, buffer=nl.sbuf)
        l = nl.ndarray((P, 1), nl.float32, buffer=nl.sbuf)
        nisa.activation_reduce(dst=p, op=nl.exp, data=s, bias=neg_m, reduce_op=nl.add, reduce_res=l)
        inv_l = nl.ndarray((P, 1), nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=inv_l, data=l)
        probs = nl.ndarray((P, S), nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=probs, data=p, op0=nl.multiply, operand0=inv_l)
        # out = probs @ v, contracting over S in chunks of P
        o_ps = nl.ndarray((P, d), nl.float32, buffer=nl.psum)
        for kc in nl.affine_range(S // P):
            pT_ps = nl.ndarray((P, P), nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=pT_ps, data=probs[:, kc * P:(kc + 1) * P])
            pT = nl.ndarray((P, P), nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=pT, src=pT_ps)
            v_t = nl.ndarray((P, d), nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=v_t, src=v[kc * P:(kc + 1) * P, :])
            nisa.nc_matmul(dst=o_ps, stationary=pT, moving=v_t)
        o = nl.ndarray((P, d), v.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=o, src=o_ps)
        nisa.dma_copy(dst=out[q0:q0 + P, :], src=o)
    return out


def _flash(qT, kT, v, strict):
    d, S = qT.shape
    P = nl.tile_size.pmax
    B = P
    scale = 1.0 / math.sqrt(d)
    out = nl.ndarray((S, d), dtype=v.dtype, buffer=nl.shared_hbm)
    for qi in nl.affine_range(S // P):
        q0 = qi * P
        m = nl.ndarray((P, 1), nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=m, value=-3.0e38)
        l = nl.ndarray((P, 1), nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=l, value=0.0)
        o = nl.ndarray((P, d), nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=o, value=0.0)
        # Causal block skipping: key blocks entirely above the diagonal are never visited.
        for kc in nl.sequential_range(qi + 1):
            k0 = kc * B
            s = _scores(qT, kT, q0, k0, P, B, scale)
            if k0 + B > q0:  # block intersects the diagonal
                s = _causal(s, q0, k0, B, strict)
            bm = nl.ndarray((P, 1), nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=bm, op=nl.maximum, data=s, axis=1)
            m_new = nl.ndarray((P, 1), nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=m_new, data1=m, data2=bm, op=nl.maximum)
            neg_m_new = nl.ndarray((P, 1), nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=neg_m_new, data=m_new, op0=nl.multiply, operand0=-1.0)
            alpha = nl.ndarray((P, 1), nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=alpha, op=nl.exp, data=m, bias=neg_m_new)
            p = nl.ndarray((P, B), nl.float32, buffer=nl.sbuf)
            rs = nl.ndarray((P, 1), nl.float32, buffer=nl.sbuf)
            nisa.activation_reduce(dst=p, op=nl.exp, data=s, bias=neg_m_new, reduce_op=nl.add, reduce_res=rs)
            nisa.scalar_tensor_tensor(dst=l, data=l, op0=nl.multiply, operand0=alpha, op1=nl.add, operand1=rs)
            # o = o * alpha + p @ v_block
            pT_ps = nl.ndarray((B, P), nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=pT_ps, data=p)
            pT = nl.ndarray((B, P), nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=pT, src=pT_ps)
            v_t = nl.ndarray((B, d), nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=v_t, src=v[k0:k0 + B, :])
            pv = nl.ndarray((P, d), nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=pv, stationary=pT, moving=v_t)
            nisa.scalar_tensor_tensor(dst=o, data=o, op0=nl.multiply, operand0=alpha, op1=nl.add, operand1=pv)
            nisa.tensor_copy(dst=m, src=m_new)
        res = nl.ndarray((P, d), v.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=res, data=o, op0=nl.divide, operand0=l)
        nisa.dma_copy(dst=out[q0:q0 + P, :], src=res)
    return out


@nki.jit
def attention_flash(qT, kT, v):
    return _flash(qT, kT, v, strict=False)


@nki.jit
def bug_flash_strict_mask(qT, kT, v):
    """Off-by-one causal mask: a query cannot attend to its own key."""
    return _flash(qT, kT, v, strict=True)


def args(ts):
    d = ts.pmax
    S = 2 * ts.pmax
    return [("qT", (d, S), "float32"), ("kT", (d, S), "float32"), ("v", (S, d), "float32")]
