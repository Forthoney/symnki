"""Wrong answers for level 1 (average pooling 2D). Each must be rejected."""

import nki
import nki.isa as nisa
import nki.language as nl


def _pool_view(in_tile, sz_p, sz_hin, sz_win, sz_pool, inner_row_stride=None):
  return in_tile.ap([
    [sz_hin * sz_win, sz_p],
    [sz_pool * sz_win, sz_hin // sz_pool],
    [sz_pool, sz_win // sz_pool],
    [inner_row_stride if inner_row_stride is not None else sz_win, sz_pool],
    [1, sz_pool],
  ])


@nki.jit
def divides_by_pool_not_area(in_tensor, pool_size):
  """Divides the window sum by p instead of p*p."""
  sz_cin, sz_hin, sz_win = in_tensor.shape
  out_tensor = nl.ndarray((sz_cin, sz_hin // pool_size, sz_win // pool_size), dtype=in_tensor.dtype,
                          buffer=nl.shared_hbm)
  in_tile = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype, buffer=nl.sbuf)
  nisa.dma_copy(dst=in_tile, src=in_tensor)
  sum_tile = nl.sum(_pool_view(in_tile, sz_cin, sz_hin, sz_win, pool_size), axis=[3, 4])
  out_tile = nl.ndarray(sum_tile.shape, dtype=sum_tile.dtype, buffer=nl.sbuf)
  nisa.tensor_scalar(dst=out_tile, data=sum_tile, op0=nl.multiply, operand0=1.0 / pool_size)
  nisa.dma_copy(dst=out_tensor, src=out_tile)
  return out_tensor


@nki.jit
def wrong_window_row_stride(in_tensor, pool_size):
  """Steps through the window's rows by p elements instead of a full image row:
  averages a 1 x p*p horizontal strip instead of a p x p square."""
  sz_cin, sz_hin, sz_win = in_tensor.shape
  out_tensor = nl.ndarray((sz_cin, sz_hin // pool_size, sz_win // pool_size), dtype=in_tensor.dtype,
                          buffer=nl.shared_hbm)
  in_tile = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype, buffer=nl.sbuf)
  nisa.dma_copy(dst=in_tile, src=in_tensor)
  view = _pool_view(in_tile, sz_cin, sz_hin, sz_win, pool_size, inner_row_stride=pool_size)
  sum_tile = nl.sum(view, axis=[3, 4])
  out_tile = nl.ndarray(sum_tile.shape, dtype=sum_tile.dtype, buffer=nl.sbuf)
  nisa.tensor_scalar(dst=out_tile, data=sum_tile, op0=nl.multiply, operand0=1.0 / (pool_size * pool_size))
  nisa.dma_copy(dst=out_tensor, src=out_tile)
  return out_tensor


@nki.jit
def max_pool_instead(in_tensor, pool_size):
  """Max pooling: right shape, plausible values, wrong operation."""
  sz_cin, sz_hin, sz_win = in_tensor.shape
  out_tensor = nl.ndarray((sz_cin, sz_hin // pool_size, sz_win // pool_size), dtype=in_tensor.dtype,
                          buffer=nl.shared_hbm)
  in_tile = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype, buffer=nl.sbuf)
  nisa.dma_copy(dst=in_tile, src=in_tensor)
  max_tile = nl.max(_pool_view(in_tile, sz_cin, sz_hin, sz_win, pool_size), axis=[3, 4])
  nisa.dma_copy(dst=out_tensor, src=max_tile)
  return out_tensor
