"""Level 5: tiled matmul with the operand loads hoisted out of the innermost loop.

Based on the NKI tutorial's nki_matmul_hoist_load_, with the loops swapped: the tutorial
keeps one row of lhsT tiles resident and re-reads all of rhs for every m, which on the
harness's K=256 M=512 N=1024 shape moves 1.86x the byte floor (the level-5 bar is 1.6x).
rhs is the larger operand there, so this keeps a column of rhs tiles resident instead and
re-reads only lhsT: 1.14x.
"""

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def nki_matmul_hoist_load_(lhsT, rhs):
  K, M = lhsT.shape
  K_, N = rhs.shape
  assert K == K_, "lhsT and rhs must have the same contraction dimension"
  TILE_M = nl.tile_size.gemm_stationary_fmax
  TILE_K = nl.tile_size.pmax
  TILE_N = nl.tile_size.gemm_moving_fmax

  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
  for n in nl.affine_range(N // TILE_N):
    # One column of rhs tiles, loaded once and reused for every m.
    rhs_tiles = []
    for k in nl.affine_range(K // TILE_K):
      rhs_tile = nl.ndarray(shape=(TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)
      nisa.dma_copy(dst=rhs_tile, src=rhs[k * TILE_K:(k + 1) * TILE_K, n * TILE_N:(n + 1) * TILE_N])
      rhs_tiles.append(rhs_tile)
    for m in nl.affine_range(M // TILE_M):
      lhsT_tiles = []
      for k in nl.affine_range(K // TILE_K):
        lhsT_tile = nl.ndarray(shape=(TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=lhsT_tile, src=lhsT[k * TILE_K:(k + 1) * TILE_K, m * TILE_M:(m + 1) * TILE_M])
        lhsT_tiles.append(lhsT_tile)
      res_psum = nl.ndarray(shape=(TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
      for k in nl.affine_range(K // TILE_K):
        nisa.nc_matmul(dst=res_psum, stationary=lhsT_tiles[k], moving=rhs_tiles[k])
      res_sb = nl.ndarray(shape=(TILE_M, TILE_N), dtype=result.dtype, buffer=nl.sbuf)
      nisa.tensor_copy(dst=res_sb, src=res_psum)
      nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M, n * TILE_N:(n + 1) * TILE_N], src=res_sb)
  return result
