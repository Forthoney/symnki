"""Matmul kernels (NKI 0.6 style), adapted from the NKI matrix-multiplication tutorial.

``matmul_tiled`` is the reference; the others are optimized rewrites. The
``bug_*`` kernels are wrong in ways an optimizing agent might plausibly be.
All tile sizes come from ``nl.tile_size`` so the kernels run at any profile.
"""

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def matmul_tiled(lhsT, rhs):
    K, M = lhsT.shape
    K_, N = rhs.shape
    assert K == K_
    TILE_M = nl.tile_size.gemm_stationary_fmax
    TILE_K = nl.tile_size.pmax
    TILE_N = nl.tile_size.gemm_moving_fmax
    assert M % TILE_M == 0 and N % TILE_N == 0 and K % TILE_K == 0

    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
    for m in nl.affine_range(M // TILE_M):
        for n in nl.affine_range(N // TILE_N):
            res_psum = nl.ndarray((TILE_M, TILE_N), nl.float32, buffer=nl.psum)
            for k in nl.affine_range(K // TILE_K):
                lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
                rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=lhsT_tile, src=lhsT[k * TILE_K:(k + 1) * TILE_K, m * TILE_M:(m + 1) * TILE_M])
                nisa.dma_copy(dst=rhs_tile, src=rhs[k * TILE_K:(k + 1) * TILE_K, n * TILE_N:(n + 1) * TILE_N])
                nisa.nc_matmul(dst=res_psum, stationary=lhsT_tile, moving=rhs_tile)
            res_sb = nl.ndarray(res_psum.shape, dtype=result.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=res_sb, src=res_psum)
            nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M, n * TILE_N:(n + 1) * TILE_N], src=res_sb)
    return result


@nki.jit
def matmul_hoist_load(lhsT, rhs):
    K, M = lhsT.shape
    K_, N = rhs.shape
    TILE_M = nl.tile_size.gemm_stationary_fmax
    TILE_K = nl.tile_size.pmax
    TILE_N = nl.tile_size.gemm_moving_fmax

    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
    for m in nl.affine_range(M // TILE_M):
        lhsT_tiles = []
        for k in nl.affine_range(K // TILE_K):
            lhsT_tile = nl.ndarray(shape=(TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=lhsT_tile, src=lhsT[k * TILE_K:(k + 1) * TILE_K, m * TILE_M:(m + 1) * TILE_M])
            lhsT_tiles.append(lhsT_tile)
        for n in nl.affine_range(N // TILE_N):
            rhs_tiles = []
            for k in nl.affine_range(K // TILE_K):
                rhs_tile = nl.ndarray(shape=(TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=rhs_tile, src=rhs[k * TILE_K:(k + 1) * TILE_K, n * TILE_N:(n + 1) * TILE_N])
                rhs_tiles.append(rhs_tile)
            res_psum = nl.ndarray(shape=(TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
            for k in nl.affine_range(K // TILE_K):
                nisa.nc_matmul(dst=res_psum, stationary=lhsT_tiles[k], moving=rhs_tiles[k])
            res_sb = nl.ndarray(shape=(TILE_M, TILE_N), dtype=result.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=res_sb, src=res_psum)
            nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M, n * TILE_N:(n + 1) * TILE_N], src=res_sb)
    return result


@nki.jit
def matmul_block_free(lhsT, rhs):
    K, M = lhsT.shape
    K_, N = rhs.shape
    TILE_M = nl.tile_size.gemm_stationary_fmax
    TILE_K = nl.tile_size.pmax
    TILE_N = nl.tile_size.gemm_moving_fmax
    TILES_IN_BLOCK_M = 2
    TILES_IN_BLOCK_N = 2
    BLOCK_M = TILE_M * TILES_IN_BLOCK_M
    BLOCK_N = TILE_N * TILES_IN_BLOCK_N
    assert M % BLOCK_M == 0 and N % BLOCK_N == 0

    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
    for m in nl.affine_range(M // BLOCK_M):
        lhsT_tiles = []
        for bm in nl.affine_range(TILES_IN_BLOCK_M):
            inner = []
            for k in nl.affine_range(K // TILE_K):
                t = nl.ndarray(shape=(TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
                c0 = (m * TILES_IN_BLOCK_M + bm) * TILE_M
                nisa.dma_copy(dst=t, src=lhsT[k * TILE_K:(k + 1) * TILE_K, c0:c0 + TILE_M])
                inner.append(t)
            lhsT_tiles.append(inner)
        for n in nl.affine_range(N // BLOCK_N):
            rhs_tiles = []
            for bn in nl.affine_range(TILES_IN_BLOCK_N):
                inner = []
                for k in nl.affine_range(K // TILE_K):
                    t = nl.ndarray(shape=(TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)
                    c0 = (n * TILES_IN_BLOCK_N + bn) * TILE_N
                    nisa.dma_copy(dst=t, src=rhs[k * TILE_K:(k + 1) * TILE_K, c0:c0 + TILE_N])
                    inner.append(t)
                rhs_tiles.append(inner)
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


@nki.jit
def matmul_fully_optimized(lhsT, rhs, TILES_IN_BLOCK_M=16, TILES_IN_BLOCK_N=2, TILES_IN_BLOCK_K=8):
    K, M = lhsT.shape
    K_, N = rhs.shape
    TILE_M = nl.tile_size.gemm_stationary_fmax
    TILE_K = nl.tile_size.pmax
    TILE_N = nl.tile_size.gemm_moving_fmax
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
        n_end = n_start + BLOCK_N
        result_m_tiles = []
        for m in nl.affine_range(NUM_BLOCK_M):
            for m_tile in nl.affine_range(TILES_IN_BLOCK_M):
                t = nl.ndarray(shape=(TILE_M, TILES_IN_BLOCK_N, TILE_N), dtype=result.dtype, buffer=nl.sbuf)
                nisa.memset(dst=t, value=0.0)
                result_m_tiles.append(t)

        for k in nl.sequential_range(NUM_BLOCK_K):
            k_block_tile_start = k * TILES_IN_BLOCK_K
            rhs_tiles = nl.ndarray(shape=(TILE_K, TILES_IN_BLOCK_K, BLOCK_N), dtype=rhs.dtype, buffer=nl.sbuf)
            for k_tile in range(TILES_IN_BLOCK_K):
                ks = (k_block_tile_start + k_tile) * TILE_K
                nisa.dma_copy(dst=rhs_tiles[0:TILE_K, k_tile, 0:BLOCK_N], src=rhs[ks:ks + TILE_K, n_start:n_end])

            for m in nl.affine_range(NUM_BLOCK_M):
                lhsT_tiles = nl.ndarray(shape=(TILE_K, TILES_IN_BLOCK_K, BLOCK_M), dtype=lhsT.dtype, buffer=nl.sbuf)
                m_start = m * BLOCK_M
                m_end = m_start + BLOCK_M
                for k_tile in nl.affine_range(TILES_IN_BLOCK_K):
                    ks = (k_block_tile_start + k_tile) * TILE_K
                    nisa.dma_copy(dst=lhsT_tiles[0:TILE_K, k_tile, 0:BLOCK_M], src=lhsT[ks:ks + TILE_K, m_start:m_end])

                m_block_tile_start = m * TILES_IN_BLOCK_M
                for n_tile in nl.affine_range(TILES_IN_BLOCK_N):
                    for m_tile in nl.affine_range(TILES_IN_BLOCK_M):
                        result_tile = nl.ndarray(shape=(TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
                        for k_tile in nl.affine_range(TILES_IN_BLOCK_K):
                            ms = m_tile * TILE_M
                            ns = n_tile * TILE_N
                            nisa.nc_matmul(
                                dst=result_tile,
                                stationary=lhsT_tiles[0:TILE_K, k_tile, ms:ms + TILE_M],
                                moving=rhs_tiles[0:TILE_K, k_tile, ns:ns + TILE_N],
                            )
                        result_m_tile = result_m_tiles[m_block_tile_start + m_tile]
                        nisa.tensor_tensor(
                            dst=result_m_tile[0:TILE_M, n_tile, 0:TILE_N],
                            data1=result_m_tile[0:TILE_M, n_tile, 0:TILE_N],
                            data2=result_tile,
                            op=nl.add,
                        )

        for m in nl.affine_range(NUM_BLOCK_M):
            for m_tile in nl.affine_range(TILES_IN_BLOCK_M):
                idx = m * TILES_IN_BLOCK_M + m_tile
                block = result_m_tiles[idx].reshape((TILE_M, BLOCK_N))
                ms = idx * TILE_M
                nisa.dma_copy(dst=result[ms:ms + TILE_M, n_start:n_end], src=block[0:TILE_M, 0:BLOCK_N])
    return result


# ---------------------------------------------------------------------------
# Broken variants
# ---------------------------------------------------------------------------


@nki.jit
def bug_psum_outside_loop(lhsT, rhs):
    """PSUM accumulator hoisted out of the n loop: partial sums leak across tiles."""
    K, M = lhsT.shape
    K_, N = rhs.shape
    TILE_M = nl.tile_size.gemm_stationary_fmax
    TILE_K = nl.tile_size.pmax
    TILE_N = nl.tile_size.gemm_moving_fmax
    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
    for m in nl.sequential_range(M // TILE_M):
        res_psum = nl.ndarray((TILE_M, TILE_N), nl.float32, buffer=nl.psum)
        for n in nl.sequential_range(N // TILE_N):
            for k in nl.affine_range(K // TILE_K):
                a = nl.ndarray((TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
                b = nl.ndarray((TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=a, src=lhsT[k * TILE_K:(k + 1) * TILE_K, m * TILE_M:(m + 1) * TILE_M])
                nisa.dma_copy(dst=b, src=rhs[k * TILE_K:(k + 1) * TILE_K, n * TILE_N:(n + 1) * TILE_N])
                nisa.nc_matmul(dst=res_psum, stationary=a, moving=b)
            res_sb = nl.ndarray(res_psum.shape, dtype=result.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=res_sb, src=res_psum)
            nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M, n * TILE_N:(n + 1) * TILE_N], src=res_sb)
    return result


@nki.jit
def legacy_sbuf_accumulate_affine(lhsT, rhs):
    """Accumulates K partial products in SBUF inside an affine_range.

    Fine in NKI 0.6 (affine_range is an alias of range() and the compiler
    respects data dependencies); a race under the older affine_range contract,
    which --strict-affine enforces.
    """
    K, M = lhsT.shape
    K_, N = rhs.shape
    TILE_M = nl.tile_size.gemm_stationary_fmax
    TILE_K = nl.tile_size.pmax
    TILE_N = nl.tile_size.gemm_moving_fmax
    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
    for m in nl.affine_range(M // TILE_M):
        for n in nl.affine_range(N // TILE_N):
            acc = nl.ndarray((TILE_M, TILE_N), nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=acc, value=0.0)
            for k in nl.affine_range(K // TILE_K):
                a = nl.ndarray((TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
                b = nl.ndarray((TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=a, src=lhsT[k * TILE_K:(k + 1) * TILE_K, m * TILE_M:(m + 1) * TILE_M])
                nisa.dma_copy(dst=b, src=rhs[k * TILE_K:(k + 1) * TILE_K, n * TILE_N:(n + 1) * TILE_N])
                part = nl.ndarray((TILE_M, TILE_N), nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=part, stationary=a, moving=b)
                nisa.tensor_tensor(dst=acc, data1=acc, data2=part, op=nl.add)
            out = nl.ndarray((TILE_M, TILE_N), dtype=result.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=out, src=acc)
            nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M, n * TILE_N:(n + 1) * TILE_N], src=out)
    return result


@nki.jit
def bug_missing_memset(lhsT, rhs):
    """Same as the sequential SBUF accumulation, but the accumulator is never zeroed."""
    K, M = lhsT.shape
    K_, N = rhs.shape
    TILE_M = nl.tile_size.gemm_stationary_fmax
    TILE_K = nl.tile_size.pmax
    TILE_N = nl.tile_size.gemm_moving_fmax
    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
    for m in nl.affine_range(M // TILE_M):
        for n in nl.affine_range(N // TILE_N):
            acc = nl.ndarray((TILE_M, TILE_N), nl.float32, buffer=nl.sbuf)
            for k in nl.sequential_range(K // TILE_K):
                a = nl.ndarray((TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
                b = nl.ndarray((TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=a, src=lhsT[k * TILE_K:(k + 1) * TILE_K, m * TILE_M:(m + 1) * TILE_M])
                nisa.dma_copy(dst=b, src=rhs[k * TILE_K:(k + 1) * TILE_K, n * TILE_N:(n + 1) * TILE_N])
                part = nl.ndarray((TILE_M, TILE_N), nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=part, stationary=a, moving=b)
                nisa.tensor_tensor(dst=acc, data1=acc, data2=part, op=nl.add)
            out = nl.ndarray((TILE_M, TILE_N), dtype=result.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=out, src=acc)
            nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M, n * TILE_N:(n + 1) * TILE_N], src=out)
    return result


@nki.jit
def bug_skip_last_k(lhsT, rhs):
    """Off-by-one: the last K tile is never accumulated."""
    K, M = lhsT.shape
    K_, N = rhs.shape
    TILE_M = nl.tile_size.gemm_stationary_fmax
    TILE_K = nl.tile_size.pmax
    TILE_N = nl.tile_size.gemm_moving_fmax
    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
    for m in nl.affine_range(M // TILE_M):
        for n in nl.affine_range(N // TILE_N):
            res_psum = nl.ndarray((TILE_M, TILE_N), nl.float32, buffer=nl.psum)
            for k in nl.affine_range(K // TILE_K - 1):
                a = nl.ndarray((TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
                b = nl.ndarray((TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=a, src=lhsT[k * TILE_K:(k + 1) * TILE_K, m * TILE_M:(m + 1) * TILE_M])
                nisa.dma_copy(dst=b, src=rhs[k * TILE_K:(k + 1) * TILE_K, n * TILE_N:(n + 1) * TILE_N])
                nisa.nc_matmul(dst=res_psum, stationary=a, moving=b)
            res_sb = nl.ndarray(res_psum.shape, dtype=result.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=res_sb, src=res_psum)
            nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M, n * TILE_N:(n + 1) * TILE_N], src=res_sb)
    return result


@nki.jit
def bug_transposed_tile(lhsT, rhs):
    """Stores the last output tile's rows in the wrong column block."""
    K, M = lhsT.shape
    K_, N = rhs.shape
    TILE_M = nl.tile_size.gemm_stationary_fmax
    TILE_K = nl.tile_size.pmax
    TILE_N = nl.tile_size.gemm_moving_fmax
    NN = N // TILE_N
    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
    for m in nl.affine_range(M // TILE_M):
        for n in nl.affine_range(NN):
            res_psum = nl.ndarray((TILE_M, TILE_N), nl.float32, buffer=nl.psum)
            for k in nl.affine_range(K // TILE_K):
                a = nl.ndarray((TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
                b = nl.ndarray((TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=a, src=lhsT[k * TILE_K:(k + 1) * TILE_K, m * TILE_M:(m + 1) * TILE_M])
                # BUG: the rhs block is selected with the (wrapped) next n index.
                nn = (n + 1) % NN if m == M // TILE_M - 1 else n
                nisa.dma_copy(dst=b, src=rhs[k * TILE_K:(k + 1) * TILE_K, nn * TILE_N:(nn + 1) * TILE_N])
                nisa.nc_matmul(dst=res_psum, stationary=a, moving=b)
            res_sb = nl.ndarray(res_psum.shape, dtype=result.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=res_sb, src=res_psum)
            nisa.dma_copy(dst=result[m * TILE_M:(m + 1) * TILE_M, n * TILE_N:(n + 1) * TILE_N], src=res_sb)
    return result


def args(ts):
    """Input spec: 4 tiles along every dimension."""
    return [
        ("lhsT", (4 * ts.pmax, 4 * ts.gemm_stationary_fmax), "bfloat16"),
        ("rhs", (4 * ts.pmax, 4 * ts.gemm_moving_fmax), "bfloat16"),
    ]
