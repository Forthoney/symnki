"""Wrong answers for level 2 (transpose of the free axes). Each must be rejected."""

import nki
import nki.isa as nisa
import nki.language as nl


def _transpose(in_tensor, shape2D, dst_col, src_col, rows=None):
  out_tensor = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype, buffer=nl.shared_hbm)
  sz_p, _ = in_tensor.shape
  in_tile = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype, buffer=nl.sbuf)
  nisa.dma_copy(dst=in_tile, src=in_tensor)
  sz_f1, sz_f2 = shape2D
  out_tile = nl.ndarray(shape=(sz_p, sz_f2 * sz_f1), dtype=in_tensor.dtype, buffer=nl.sbuf)
  for i_f1 in nl.affine_range(sz_f1 if rows is None else rows):
    for i_f2 in nl.affine_range(sz_f2):
      nisa.tensor_copy(dst=out_tile[:, nl.ds(dst_col(i_f1, i_f2, sz_f1, sz_f2), 1)],
                       src=in_tile[:, nl.ds(src_col(i_f1, i_f2, sz_f1, sz_f2), 1)])
  nisa.dma_copy(dst=out_tensor, src=out_tile)
  return out_tensor


@nki.jit
def inverse_permutation(in_tensor, shape2D):
  """Index formulas swapped: applies the inverse permutation. Identical to the right
  answer when the matrices are square (the 8x8 harness shape), wrong otherwise."""
  return _transpose(in_tensor, shape2D,
                    dst_col=lambda a, b, f1, f2: a * f2 + b,
                    src_col=lambda a, b, f1, f2: b * f1 + a)


@nki.jit
def uses_wrong_stride(in_tensor, shape2D):
  """Strides by F2 on both sides: a plain copy, not a transpose."""
  return _transpose(in_tensor, shape2D,
                    dst_col=lambda a, b, f1, f2: a * f2 + b,
                    src_col=lambda a, b, f1, f2: a * f2 + b)


@nki.jit
def misses_last_row(in_tensor, shape2D):
  """Off-by-one loop bound: the last source row is never copied, so part of the
  output is whatever was in SBUF."""
  return _transpose(in_tensor, shape2D,
                    dst_col=lambda a, b, f1, f2: b * f1 + a,
                    src_col=lambda a, b, f1, f2: a * f2 + b,
                    rows=shape2D[0] - 1)
