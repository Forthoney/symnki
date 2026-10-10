"""The nkibench ladder (trainium-agent-labs, project 2) as nkiequiv checks.

For every level: the NumPy reference nkibench grades against, the harness's own test shapes,
one correct kernel that must be accepted, and wrong kernels that must be rejected.

Shapes. Levels 1-3 are proved at exactly the harness shapes (level 3's kernel hard-codes
them). The matmul levels 4-7 and attention (8) are proved at the harness shapes scaled down
tile-for-tile (same number of tiles in every dimension, smaller tiles), and their numerics are
checked at the real harness shapes on full-size Trainium tiles.
"""

import os
from dataclasses import dataclass, field

from nkiequiv import ArgSpec

HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Case:
    label: str
    specs: list
    kwargs: dict = field(default_factory=dict)
    profile: str = "full"
    numerics_specs: list | None = None  # at the full profile; None = same as specs


def _f32(name, shape):
    return ArgSpec(name, tuple(shape), "float32")


def _pool_cases():
    shapes = [((32, 32, 32), 2), ((128, 16, 16), 4), ((8, 24, 24), 3), ((64, 8, 8), 2)]
    return [Case(f"C,H,W={s} pool={p}", [_f32("x", s)], {"pool_size": p}) for s, p in shapes]


def _transpose_cases():
    shapes = [((32, 12), (3, 4)), ((128, 64), (8, 8)), ((64, 128), (4, 32)), ((8, 35), (5, 7))]
    return [Case(f"shape={s} as {f[0]}x{f[1]}", [_f32("x", s)], {"shape2D": f}) for s, f in shapes]


# nkibench's _MM_SHAPES, as (K, M, N).
MM_SHAPES = [(128, 128, 512), (256, 256, 1024), (512, 128, 512), (256, 512, 1024)]


def _mm_cases(shapes=None):
    out = []
    for K, M, N in shapes or MM_SHAPES:
        kt, mt, nt = K // 128, M // 128, N // 512
        small = [_f32("lhsT", (4 * kt, 4 * mt)), _f32("rhs", (4 * kt, 8 * nt))]
        full = [_f32("lhsT", (K, M)), _f32("rhs", (K, N))]
        out.append(Case(f"K={K} M={M} N={N}", small, profile="small", numerics_specs=full))
    return out


def _attention_cases():
    out = []
    for seq, dim in [(128, 64), (64, 128), (96, 32)]:
        s, d = seq * 8 // 128, dim * 8 // 128
        small = [_f32(n, (s, d)) for n in ("q", "k", "v")]
        full = [_f32(n, (seq, dim)) for n in ("q", "k", "v")]
        out.append(Case(f"seq={seq} dim={dim}", small, profile="medium", numerics_specs=full))
    return out


LEVELS = {
    1: dict(op="average pooling 2D", ref="ref_avgpool2d", cases=_pool_cases,
            correct=("reference_level1.py", "tensor_avgpool_kernel"),
            wrong=[("wrong_level1.py", n) for n in
                   ("divides_by_pool_not_area", "wrong_window_row_stride", "max_pool_instead")]),
    2: dict(op="2D transpose", ref="ref_transpose2d", cases=_transpose_cases,
            correct=("reference_level2.py", "tensor_transpose2D_kernel_"),
            wrong=[("wrong_level2.py", n) for n in
                   ("inverse_permutation", "uses_wrong_stride", "misses_last_row")]),
    3: dict(op="matmul, single tile", ref="ref_matmul",
            cases=lambda: [Case("K=128 M=64 N=512", [_f32("lhsT", (128, 64)), _f32("rhs", (128, 512))])],
            correct=("reference_level3.py", "nki_matmul_basic_"),
            wrong=[("wrong_level3.py", n) for n in
                   ("dma_straight_from_psum", "accumulates_onto_garbage", "half_contraction")]),
    4: dict(op="matmul, tiled", ref="ref_matmul", cases=_mm_cases,
            correct=("reference_level4.py", "nki_matmul_tiled_"),
            wrong=[("wrong_level4.py", n) for n in
                   ("skips_last_k_tile", "psum_shared_across_n", "rhs_block_indexed_by_m",
                    "output_block_swapped")]),
    5: dict(op="matmul, loads hoisted", ref="ref_matmul", cases=_mm_cases,
            correct=("solution_level5.py", "nki_matmul_hoist_load_"),
            wrong=[("wrong_level5to8.py", "hoists_lhsT_out_of_m_loop")]),
    6: dict(op="matmul, M and N blocked", ref="ref_matmul", cases=_mm_cases,
            correct=("solution_level6.py", "nki_matmul_block_free_dimension_"),
            wrong=[("wrong_level5to8.py", "lhsT_offset_drops_block_index")],
            # Wrong in general, equal to the reference on every harness shape.
            blind_spots=[("wrong_level5to8.py", "rhs_offset_drops_block_index")],
            extra_cases=lambda: _mm_cases([(128, 128, 2048)])),
    7: dict(op="matmul, M, N and K blocked", ref="ref_matmul", cases=_mm_cases,
            correct=("solution_level7.py", "nki_matmul_fully_optimized_"),
            wrong=[("wrong_level5to8.py", n) for n in ("accumulator_never_zeroed", "k_block_offset_dropped")]),
    8: dict(op="single-head attention", ref="ref_attention", cases=_attention_cases,
            correct=("solution_level8.py", "nki_attention_"),
            wrong=[("wrong_level5to8.py", n) for n in
                   ("attention_no_max_subtraction", "attention_unscaled", "attention_probs_not_transposed")]),
}


def candidates(level):
    """[(file, function, should_be_accepted)]"""
    spec = LEVELS[level]
    return [(*spec["correct"], True)] + [(f, fn, False) for f, fn in spec["wrong"]]
