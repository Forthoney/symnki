"""Wrong answers for levels 5-8. Each must be rejected."""

import nki
import nki.isa as nisa
import nki.language as nl

from solution_level8 import _attention


# ---------------------------------------------------------------- level 5


@nki.jit
def hoists_lhsT_out_of_m_loop(lhsT, rhs):
  """Hoists the lhsT row one loop too far: loaded once for m = 0 and reused for every m."""
  K, M = lhsT.shape
  K_, N = rhs.shape
  TILE_M = nl.tile_size.gemm_stationary_fmax
  TILE_K = nl.tile_size.pmax
  TILE_N = nl.tile_size.gemm_moving_fmax
  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
  lhsT_tiles = []
  for k in nl.affine_range(K // TILE_K):
    t = nl.ndarray(shape=(TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=lhsT[k * TILE_K:(k + 1) * TILE_K, 0:TILE_M])
    lhsT_tiles.append(t)
  for m in nl.affine_range(M // TILE_M):
    for n in nl.affine_range(N // TILE_N):
      res_psum = nl.ndarray(shape=(TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
      for k in nl.affine_range(K // TILE_K):
        r = nl.ndarray(shape=(TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=r, src=rhs[k * TILE_K:(k + 1) * TILE_K, n * TILE_N:(n + 1) * TILE_N])
        nisa.nc_matmul(dst=res_psum, stationary=lhsT_tiles[k], moving=r)
      res_sb = nl.ndarray(shape=(TILE_M, TILE_N), dtype=result.dtype, buffer=nl.sbuf)
      nisa.tensor_copy(dst=res_sb, src=res_psum)
      nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M, n * TILE_N:(n + 1) * TILE_N], src=res_sb)
  return result


# ---------------------------------------------------------------- level 6


def _blocked(lhsT, rhs, drop_n=False, drop_m=False):
  K, M = lhsT.shape
  K_, N = rhs.shape
  TILE_M = nl.tile_size.gemm_stationary_fmax
  TILE_K = nl.tile_size.pmax
  TILE_N = nl.tile_size.gemm_moving_fmax
  TBM = min(2, M // TILE_M)
  TBN = min(2, N // TILE_N)
  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
  for m in nl.affine_range(M // (TILE_M * TBM)):
    for n in nl.affine_range(N // (TILE_N * TBN)):
      for bm in nl.affine_range(TBM):
        for bn in nl.affine_range(TBN):
          acc = nl.ndarray(shape=(TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
          for k in nl.affine_range(K // TILE_K):
            a = nl.ndarray(shape=(TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
            r0 = (m * TBM + bm) * TILE_M
            ra = (bm if drop_m else m * TBM + bm) * TILE_M
            nisa.dma_copy(dst=a, src=lhsT[k * TILE_K:(k + 1) * TILE_K, ra:ra + TILE_M])
            b = nl.ndarray(shape=(TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)
            cb = (bn if drop_n else n * TBN + bn) * TILE_N
            nisa.dma_copy(dst=b, src=rhs[k * TILE_K:(k + 1) * TILE_K, cb:cb + TILE_N])
            nisa.nc_matmul(dst=acc, stationary=a, moving=b)
          tmp = nl.ndarray(shape=acc.shape, dtype=result.dtype, buffer=nl.sbuf)
          nisa.tensor_copy(dst=tmp, src=acc)
          c0 = (n * TBN + bn) * TILE_N
          nisa.dma_copy(dst=result[r0:r0 + TILE_M, c0:c0 + TILE_N], src=tmp)
  return result


@nki.jit
def lhsT_offset_drops_block_index(lhsT, rhs):
  """Row offset of an lhsT tile uses only the tile-within-block index bm, dropping m:
  every M block multiplies the first block's rows."""
  return _blocked(lhsT, rhs, drop_m=True)


@nki.jit
def rhs_offset_drops_block_index(lhsT, rhs):
  """The same bug on the N side. Every nkibench shape has a single N block (N <= 1024 with
  2 tiles of 512 per block), so on the harness shapes this kernel IS equal to the
  reference -- nkibench and the proof both accept it. Only a shape with N >= 2048 exposes it.
  """
  return _blocked(lhsT, rhs, drop_n=True)


# ---------------------------------------------------------------- level 7


def _fully_optimized(lhsT, rhs, zero_init=True, k_offset_bug=False):
  K, M = lhsT.shape
  K_, N = rhs.shape
  TILE_M = nl.tile_size.gemm_stationary_fmax
  TILE_K = nl.tile_size.pmax
  TILE_N = nl.tile_size.gemm_moving_fmax
  TBM, TBN, TBK = min(16, M // TILE_M), min(2, N // TILE_N), min(2, K // TILE_K)
  BLOCK_M, BLOCK_N, BLOCK_K = TILE_M * TBM, TILE_N * TBN, TILE_K * TBK
  result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
  for n in nl.affine_range(N // BLOCK_N):
    n_start = n * BLOCK_N
    acc_tiles = []
    for i in nl.affine_range(M // TILE_M):
      t = nl.ndarray(shape=(TILE_M, TBN, TILE_N), dtype=result.dtype, buffer=nl.sbuf)
      if zero_init:
        nisa.memset(dst=t, value=0.0)
      acc_tiles.append(t)
    for k in nl.sequential_range(K // BLOCK_K):
      for m in nl.affine_range(M // BLOCK_M):
        for n_tile in nl.affine_range(TBN):
          for m_tile in nl.affine_range(TBM):
            ps = nl.ndarray(shape=(TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
            for k_tile in nl.affine_range(TBK):
              # BUG (when enabled): forgets the block start, so every K block re-reads block 0.
              ks = (k_tile if k_offset_bug else k * TBK + k_tile) * TILE_K
              a = nl.ndarray(shape=(TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
              ms = m * BLOCK_M + m_tile * TILE_M
              nisa.dma_copy(dst=a, src=lhsT[ks:ks + TILE_K, ms:ms + TILE_M])
              b = nl.ndarray(shape=(TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)
              ns = n_start + n_tile * TILE_N
              nisa.dma_copy(dst=b, src=rhs[ks:ks + TILE_K, ns:ns + TILE_N])
              nisa.nc_matmul(dst=ps, stationary=a, moving=b)
            acc = acc_tiles[m * TBM + m_tile]
            nisa.tensor_tensor(dst=acc[0:TILE_M, n_tile, 0:TILE_N], data1=acc[0:TILE_M, n_tile, 0:TILE_N],
                               data2=ps, op=nl.add)
    for i in nl.affine_range(M // TILE_M):
      blk = acc_tiles[i].reshape((TILE_M, BLOCK_N))
      nisa.dma_copy(dst=result[i * TILE_M:(i + 1) * TILE_M, n_start:n_start + BLOCK_N], src=blk)
  return result


@nki.jit
def accumulator_never_zeroed(lhsT, rhs):
  """The SBUF accumulators are allocated but never memset before += partial products."""
  return _fully_optimized(lhsT, rhs, zero_init=False)


@nki.jit
def k_block_offset_dropped(lhsT, rhs):
  """Every K block reads the first block's rows (offset uses k_tile, not k * TBK + k_tile)."""
  return _fully_optimized(lhsT, rhs, k_offset_bug=True)


# ---------------------------------------------------------------- level 8


@nki.jit
def attention_no_max_subtraction(q, k, v):
  """The trap the harness warns about: exact over the reals, overflows exp() on real data."""
  return _attention(q, k, v, subtract_max=False)


@nki.jit
def attention_unscaled(q, k, v):
  """Forgets the 1/sqrt(d) scaling of the scores."""
  return _attention(q, k, v, scale=False)


@nki.jit
def attention_probs_not_transposed(q, k, v):
  """Uses the probabilities untransposed as the stationary operand: computes e.T @ v.
  Same output shape (the score matrix is square), wrong values."""
  return _attention(q, k, v, transpose_probs=False)
