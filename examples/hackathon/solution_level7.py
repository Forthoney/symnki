"""Level 7: tiled matmul with M, N and K all blocked.

From the NKI matrix-multiplication tutorial (nki_matmul_fully_optimized_). The tutorial's
default block sizes (16 x 2 x 8 tiles) need M >= 2048; the harness shapes are far smaller, so
each block is capped at the number of tiles in that dimension.
"""

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def nki_matmul_fully_optimized_(lhsT, rhs, TILES_IN_BLOCK_M=16, TILES_IN_BLOCK_N=2, TILES_IN_BLOCK_K=8):
  K, M = lhsT.shape
  K_, N = rhs.shape
  assert K == K_, "lhsT and rhs must have the same contraction dimension"
  TILE_M = nl.tile_size.gemm_stationary_fmax
  TILE_K = nl.tile_size.pmax
  TILE_N = nl.tile_size.gemm_moving_fmax
  TILES_IN_BLOCK_M = min(TILES_IN_BLOCK_M, M // TILE_M)
  TILES_IN_BLOCK_N = min(TILES_IN_BLOCK_N, N // TILE_N)
  TILES_IN_BLOCK_K = min(TILES_IN_BLOCK_K, K // TILE_K)
  BLOCK_M = TILE_M * TILES_IN_BLOCK_M
  BLOCK_N = TILE_N * TILES_IN_BLOCK_N
  BLOCK_K = TILE_K * TILES_IN_BLOCK_K
  assert M % BLOCK_M == 0 and N % BLOCK_N == 0 and K % BLOCK_K == 0

  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
  NUM_BLOCK_M = M // BLOCK_M
  NUM_BLOCK_N = N // BLOCK_N
  NUM_BLOCK_K = K // BLOCK_K

  for n in nl.affine_range(NUM_BLOCK_N):
    n_start = n * BLOCK_N
    result_m_tiles = []
    for m in nl.affine_range(NUM_BLOCK_M):
      for m_tile in nl.affine_range(TILES_IN_BLOCK_M):
        t = nl.ndarray(shape=(TILE_M, TILES_IN_BLOCK_N, TILE_N), dtype=result.dtype, buffer=nl.sbuf)
        nisa.memset(dst=t, value=0.0)
        result_m_tiles.append(t)

    for k in nl.sequential_range(NUM_BLOCK_K):
      k_block_tile_start = k * TILES_IN_BLOCK_K
      rhs_tiles = nl.ndarray(shape=(TILE_K, TILES_IN_BLOCK_K, BLOCK_N), dtype=rhs.dtype, buffer=nl.sbuf)
      for k_tile in nl.affine_range(TILES_IN_BLOCK_K):
        ks = (k_block_tile_start + k_tile) * TILE_K
        nisa.dma_copy(dst=rhs_tiles[0:TILE_K, k_tile, 0:BLOCK_N], src=rhs[ks:ks + TILE_K, n_start:n_start + BLOCK_N])

      for m in nl.affine_range(NUM_BLOCK_M):
        lhsT_tiles = nl.ndarray(shape=(TILE_K, TILES_IN_BLOCK_K, BLOCK_M), dtype=lhsT.dtype, buffer=nl.sbuf)
        m_start = m * BLOCK_M
        for k_tile in nl.affine_range(TILES_IN_BLOCK_K):
          ks = (k_block_tile_start + k_tile) * TILE_K
          nisa.dma_copy(dst=lhsT_tiles[0:TILE_K, k_tile, 0:BLOCK_M], src=lhsT[ks:ks + TILE_K, m_start:m_start + BLOCK_M])
        for n_tile in nl.affine_range(TILES_IN_BLOCK_N):
          for m_tile in nl.affine_range(TILES_IN_BLOCK_M):
            result_tile = nl.ndarray(shape=(TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
            for k_tile in nl.affine_range(TILES_IN_BLOCK_K):
              ms = m_tile * TILE_M
              ns = n_tile * TILE_N
              nisa.nc_matmul(dst=result_tile, stationary=lhsT_tiles[0:TILE_K, k_tile, ms:ms + TILE_M],
                             moving=rhs_tiles[0:TILE_K, k_tile, ns:ns + TILE_N])
            acc = result_m_tiles[m * TILES_IN_BLOCK_M + m_tile]
            nisa.tensor_tensor(dst=acc[0:TILE_M, n_tile, 0:TILE_N], data1=acc[0:TILE_M, n_tile, 0:TILE_N],
                               data2=result_tile, op=nl.add)

    for m in nl.affine_range(NUM_BLOCK_M):
      for m_tile in nl.affine_range(TILES_IN_BLOCK_M):
        idx = m * TILES_IN_BLOCK_M + m_tile
        block = result_m_tiles[idx].reshape((TILE_M, BLOCK_N))
        ms = idx * TILE_M
        nisa.dma_copy(dst=result[ms:ms + TILE_M, n_start:n_start + BLOCK_N], src=block[0:TILE_M, 0:BLOCK_N])
  return result
