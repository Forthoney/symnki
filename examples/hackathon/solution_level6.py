"""Level 6: tiled matmul with the M and N free dimensions blocked.

From the NKI matrix-multiplication tutorial (nki_matmul_block_free_dimension_). The tutorial
fixes 2 x 2 tiles per block, which does not divide the harness's single-tile shapes and, on
K=256 M=512 N=1024, re-reads rhs once per M block (1.29x the byte floor; the level-6 bar is
1.25x). Blocks here are up to 4 M tiles by 2 N tiles, capped at the tiles available.
"""

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def nki_matmul_block_free_dimension_(lhsT, rhs):
  K, M = lhsT.shape
  K_, N = rhs.shape
  assert K == K_, "lhsT and rhs must have the same contraction dimension"
  TILE_M = nl.tile_size.gemm_stationary_fmax
  TILE_K = nl.tile_size.pmax
  TILE_N = nl.tile_size.gemm_moving_fmax
  TILES_IN_BLOCK_M = min(4, M // TILE_M)
  TILES_IN_BLOCK_N = min(2, N // TILE_N)
  BLOCK_M = TILE_M * TILES_IN_BLOCK_M
  BLOCK_N = TILE_N * TILES_IN_BLOCK_N
  assert M % BLOCK_M == 0 and N % BLOCK_N == 0

  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
  for m in nl.affine_range(M // BLOCK_M):
    lhsT_tiles = []
    for bm in nl.affine_range(TILES_IN_BLOCK_M):
      row = []
      for k in nl.affine_range(K // TILE_K):
        t = nl.ndarray(shape=(TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
        c0 = (m * TILES_IN_BLOCK_M + bm) * TILE_M
        nisa.dma_copy(dst=t, src=lhsT[k * TILE_K:(k + 1) * TILE_K, c0:c0 + TILE_M])
        row.append(t)
      lhsT_tiles.append(row)
    for n in nl.affine_range(N // BLOCK_N):
      rhs_tiles = []
      for bn in nl.affine_range(TILES_IN_BLOCK_N):
        row = []
        for k in nl.affine_range(K // TILE_K):
          t = nl.ndarray(shape=(TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)
          c0 = (n * TILES_IN_BLOCK_N + bn) * TILE_N
          nisa.dma_copy(dst=t, src=rhs[k * TILE_K:(k + 1) * TILE_K, c0:c0 + TILE_N])
          row.append(t)
        rhs_tiles.append(row)
      for bm in nl.affine_range(TILES_IN_BLOCK_M):
        for bn in nl.affine_range(TILES_IN_BLOCK_N):
          acc = nl.ndarray(shape=(TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
          for k in nl.affine_range(K // TILE_K):
            nisa.nc_matmul(dst=acc, stationary=lhsT_tiles[bm][k], moving=rhs_tiles[bn][k])
          tmp = nl.ndarray(shape=acc.shape, dtype=result.dtype, buffer=nl.sbuf)
          nisa.tensor_copy(dst=tmp, src=acc)
          r0 = (m * TILES_IN_BLOCK_M + bm) * TILE_M
          c0 = (n * TILES_IN_BLOCK_N + bn) * TILE_N
          nisa.dma_copy(dst=result[r0:r0 + TILE_M, c0:c0 + TILE_N], src=tmp)
  return result
