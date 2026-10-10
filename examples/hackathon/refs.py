"""The NumPy references of the nkibench ladder (trainium-agent-labs, project 2).

Copied verbatim from projects/02-kernel-agent/nkibench.py; tests/test_hackathon_ladder.py
fails if they drift from the originals when that repo is present.
"""

import numpy as np

def ref_avgpool2d(x, pool_size):
    """x: [C, H, W] -> [C, H//p, W//p]"""
    C, H, W = x.shape
    p = pool_size
    return x[:, :H // p * p, :W // p * p].reshape(
        C, H // p, p, W // p, p).mean(axis=(2, 4)).astype(x.dtype)

def ref_transpose2d(x, shape2D):
    """Transpose the two FREE axes inside each partition.

    Matches the tutorial's signature: x is [P, F1*F2] and shape2D is (F1, F2), so each
    partition row holds an F1-by-F2 matrix laid out row-major and comes back column-major.
    The partition axis is untouched -- that is the point, since it is the constrained one.
    """
    P, F = x.shape
    F1, F2 = shape2D
    assert F1 * F2 == F, f"shape2D {shape2D} does not match the free size {F}"
    return np.ascontiguousarray(
        x.reshape(P, F1, F2).transpose(0, 2, 1)).reshape(P, F)

def ref_attention(q, k, v):
    """Single-head attention: softmax(Q Kt / sqrt(d)) V, with q, k, v all [seq, dim].

    This level exists to show that the harness is not about matmul. It was added by writing this
    reference and one input builder -- nothing else in the file changed. Do the same for whatever you
    care about.

    Note the trap, which is the whole reason attention is interesting on a chip: a naive exp()
    overflows. The reference subtracts the row max first, and a kernel that does not will return
    plausible numbers on friendly data and NaN on real data.
    """
    d = q.shape[1]
    scores = (q.astype(np.float32) @ k.astype(np.float32).T) / np.sqrt(d)
    scores = scores - scores.max(axis=-1, keepdims=True)
    e = np.exp(scores)
    return ((e / e.sum(axis=-1, keepdims=True)) @ v.astype(np.float32)).astype(q.dtype)

def ref_matmul(lhsT, rhs):
    """lhsT: [K, M] (left operand arrives TRANSPOSED), rhs: [K, N] -> [M, N]

    The transposed left operand is not a quirk of this harness. nc_matmul consumes its
    stationary operand down the partition axis, so K has to be the partition dimension of
    both inputs.
    """
    return (lhsT.astype(np.float32).T @ rhs.astype(np.float32)).astype(lhsT.dtype)
