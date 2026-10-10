"""NumPy reference functions (the specs NKI kernels are proved against)."""

import numpy as np


def ref_matmul(lhsT, rhs):
    """lhsT: [K, M] (left operand arrives TRANSPOSED), rhs: [K, N] -> [M, N]

    The transposed left operand is not a quirk of this harness. nc_matmul consumes its
    stationary operand down the partition axis, so K has to be the partition dimension of
    both inputs.
    """
    return (lhsT.astype(np.float32).T @ rhs.astype(np.float32)).astype(lhsT.dtype)


def matmul_args(ts):
    """Two tiles along every dimension, so tile-boundary indexing is exercised."""
    return [
        ("lhsT", (2 * ts.pmax, 2 * ts.gemm_stationary_fmax), "bfloat16"),
        ("rhs", (2 * ts.pmax, 2 * ts.gemm_moving_fmax), "bfloat16"),
    ]


args = matmul_args


def ref_softmax(x):
    m = x.max(axis=-1, keepdims=True)
    e = np.exp((x - m).astype(np.float64))
    return (e / e.sum(axis=-1, keepdims=True)).astype(x.dtype)


def softmax_args(ts):
    return [("x", (2 * ts.pmax, 2 * ts.psum_bank_fmax), "float32")]


def ref_sorted_rows(x):
    """Not traceable: sorting is outside the supported NumPy subset."""
    return np.sort(x, axis=-1)
