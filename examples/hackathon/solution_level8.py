"""Level 8: single-head attention, softmax(Q K^T / sqrt(d)) V, with q, k, v all [seq, dim].

Everything stays on chip: Q and K are transposed on the Tensor Engine so the head dim is the
contraction (partition) axis, the scores never leave SBUF/PSUM, and the probabilities are
transposed again so the key axis is the contraction axis of the second matmul.
Assumes seq and dim each fit in one tile (the harness shapes do).
"""

import math

import nki
import nki.isa as nisa
import nki.language as nl


def _transposed(x):
  """[rows, cols] in HBM -> [cols, rows] in SBUF via an on-chip transpose."""
  rows, cols = x.shape
  t = nl.ndarray((rows, cols), x.dtype, buffer=nl.sbuf)
  nisa.dma_copy(dst=t, src=x)
  t_ps = nl.ndarray((cols, rows), x.dtype, buffer=nl.psum)
  nisa.nc_transpose(dst=t_ps, data=t)
  out = nl.ndarray((cols, rows), x.dtype, buffer=nl.sbuf)
  nisa.tensor_copy(dst=out, src=t_ps)
  return out


def _attention(q, k, v, subtract_max=True, scale=True, transpose_probs=True):
  seq, d = q.shape
  assert seq <= nl.tile_size.pmax and d <= nl.tile_size.pmax
  out = nl.ndarray((seq, d), dtype=q.dtype, buffer=nl.shared_hbm)

  qT = _transposed(q)  # [d, seq]
  kT = _transposed(k)  # [d, seq]
  s_ps = nl.ndarray((seq, seq), nl.float32, buffer=nl.psum)
  nisa.nc_matmul(dst=s_ps, stationary=qT, moving=kT)  # q @ k.T
  s = nl.ndarray((seq, seq), nl.float32, buffer=nl.sbuf)
  nisa.tensor_scalar(dst=s, data=s_ps, op0=nl.multiply, operand0=1.0 / math.sqrt(d) if scale else 1.0)

  # Numerically stable softmax along the key axis.
  e = nl.ndarray((seq, seq), nl.float32, buffer=nl.sbuf)
  l = nl.ndarray((seq, 1), nl.float32, buffer=nl.sbuf)
  if subtract_max:
    neg_m = nl.ndarray((seq, 1), nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=neg_m, op=nl.maximum, data=s, axis=1, negate=True)
    nisa.activation_reduce(dst=e, op=nl.exp, data=s, bias=neg_m, reduce_op=nl.add, reduce_res=l)
  else:
    nisa.activation_reduce(dst=e, op=nl.exp, data=s, reduce_op=nl.add, reduce_res=l)

  # (e @ v) / l, contracting over keys: the stationary operand must be e transposed.
  if transpose_probs:
    eT_ps = nl.ndarray((seq, seq), nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=eT_ps, data=e)
    eT = nl.ndarray((seq, seq), nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=eT, src=eT_ps)
  else:
    eT = e
  v_t = nl.ndarray((seq, d), v.dtype, buffer=nl.sbuf)
  nisa.dma_copy(dst=v_t, src=v)
  o_ps = nl.ndarray((seq, d), nl.float32, buffer=nl.psum)
  nisa.nc_matmul(dst=o_ps, stationary=eT, moving=v_t)
  o = nl.ndarray((seq, d), q.dtype, buffer=nl.sbuf)
  nisa.tensor_scalar(dst=o, data=o_ps, op0=nl.divide, operand0=l)
  nisa.dma_copy(dst=out, src=o)
  return out


@nki.jit
def nki_attention_(q, k, v):
  return _attention(q, k, v)
