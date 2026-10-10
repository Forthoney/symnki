"""Wrong answers for level 4 (tiled matmul). Each must be rejected."""

import nki
import nki.isa as nisa
import nki.language as nl


def _tiled(lhsT, rhs, k_tiles=None, rhs_block=None, out_block=None, psum_per_row=False):
  K, M = lhsT.shape
  K_, N = rhs.shape
  TILE_M = nl.tile_size.gemm_stationary_fmax
  TILE_K = nl.tile_size.pmax
  TILE_N = nl.tile_size.gemm_moving_fmax
  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
  for m in nl.affine_range(M // TILE_M):
    if psum_per_row:
      res_psum = nl.ndarray((TILE_M, TILE_N), nl.float32, buffer=nl.psum)
    for n in nl.affine_range(N // TILE_N):
      if not psum_per_row:
        res_psum = nl.ndarray((TILE_M, TILE_N), nl.float32, buffer=nl.psum)
      for k in nl.affine_range(k_tiles(K // TILE_K) if k_tiles else K // TILE_K):
        lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
        rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=lhsT_tile, src=lhsT[k * TILE_K:(k + 1) * TILE_K, m * TILE_M:(m + 1) * TILE_M])
        nb = rhs_block(m, n) if rhs_block else n
        nisa.dma_copy(dst=rhs_tile, src=rhs[k * TILE_K:(k + 1) * TILE_K, nb * TILE_N:(nb + 1) * TILE_N])
        nisa.nc_matmul(dst=res_psum, stationary=lhsT_tile, moving=rhs_tile)
      res_sb = nl.ndarray(res_psum.shape, dtype=result.dtype, buffer=nl.sbuf)
      nisa.tensor_copy(dst=res_sb, src=res_psum)
      om, on = out_block(m, n) if out_block else (m, n)
      nisa.dma_copy(dst=result[om * TILE_M:(om + 1) * TILE_M, on * TILE_N:(on + 1) * TILE_N], src=res_sb)
  return result


@nki.jit
def skips_last_k_tile(lhsT, rhs):
  """range(K // TILE_K - 1): the last contraction tile is never accumulated."""
  return _tiled(lhsT, rhs, k_tiles=lambda n: n - 1)


@nki.jit
def psum_shared_across_n(lhsT, rhs):
  """PSUM accumulator hoisted out of the n loop, so partial sums leak between tiles."""
  return _tiled(lhsT, rhs, psum_per_row=True)


@nki.jit
def rhs_block_indexed_by_m(lhsT, rhs):
  """Selects the rhs column block with the wrong loop variable (min(m, n))."""
  return _tiled(lhsT, rhs, rhs_block=lambda m, n: min(m, n))


@nki.jit
def output_block_swapped(lhsT, rhs):
  """Writes tile (m, n) to block (n % M_tiles, m % N_tiles)."""
  def blk(m, n):
    M_t = lhsT.shape[1] // nl.tile_size.gemm_stationary_fmax
    N_t = rhs.shape[1] // nl.tile_size.gemm_moving_fmax
    return n % M_t, m % N_t
  return _tiled(lhsT, rhs, out_block=blk)
