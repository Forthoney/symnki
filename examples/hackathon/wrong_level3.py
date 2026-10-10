"""Wrong answers for level 3 (single-tile matmul). Each must be rejected."""

import nki
import nki.isa as nisa
import nki.language as nl


def _load(lhsT, rhs):
  lhs_tile = nl.ndarray(lhsT.shape, dtype=lhsT.dtype, buffer=nl.sbuf)
  rhs_tile = nl.ndarray(rhs.shape, dtype=rhs.dtype, buffer=nl.sbuf)
  nisa.dma_copy(dst=lhs_tile, src=lhsT)
  nisa.dma_copy(dst=rhs_tile, src=rhs)
  return lhs_tile, rhs_tile


@nki.jit
def dma_straight_from_psum(lhsT, rhs):
  """Skips the PSUM -> SBUF copy and DMAs the accumulator straight to HBM."""
  K, M = lhsT.shape
  K_, N = rhs.shape
  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
  lhs_tile, rhs_tile = _load(lhsT, rhs)
  result_psum = nl.ndarray(result.shape, dtype=nl.float32, buffer=nl.psum)
  nisa.nc_matmul(result_psum, lhs_tile, rhs_tile)
  nisa.dma_copy(dst=result, src=result_psum)
  return result


@nki.jit
def accumulates_onto_garbage(lhsT, rhs):
  """accumulate=True on a fresh PSUM tile adds the product to whatever was there."""
  K, M = lhsT.shape
  K_, N = rhs.shape
  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
  lhs_tile, rhs_tile = _load(lhsT, rhs)
  result_psum = nl.ndarray(result.shape, dtype=nl.float32, buffer=nl.psum)
  nisa.nc_matmul(result_psum, lhs_tile, rhs_tile, accumulate=True)
  result_sbuf = nl.ndarray(result_psum.shape, dtype=result.dtype, buffer=nl.sbuf)
  nisa.tensor_copy(dst=result_sbuf, src=result_psum)
  nisa.dma_copy(dst=result, src=result_sbuf)
  return result


@nki.jit
def half_contraction(lhsT, rhs):
  """Contracts over only the first half of K (a 64-row tile where 128 were needed)."""
  K, M = lhsT.shape
  K_, N = rhs.shape
  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
  lhs_tile, rhs_tile = _load(lhsT, rhs)
  result_psum = nl.ndarray(result.shape, dtype=nl.float32, buffer=nl.psum)
  nisa.nc_matmul(result_psum, lhs_tile[0:K // 2, :], rhs_tile[0:K // 2, :])
  result_sbuf = nl.ndarray(result_psum.shape, dtype=result.dtype, buffer=nl.sbuf)
  nisa.tensor_copy(dst=result_sbuf, src=result_psum)
  nisa.dma_copy(dst=result, src=result_sbuf)
  return result
