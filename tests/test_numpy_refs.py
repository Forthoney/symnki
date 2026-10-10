import os

import pytest

from nkiequiv import ArgSpec, KernelRef, Options, check
from nkiequiv.numpy_ref import NumpyRef
from nkiequiv.shim.runtime import parse_profile

EX = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples"))
REFS = os.path.join(EX, "numpy_refs.py")


def matmul_specs(dtype, profile="small"):
    ts = parse_profile(profile)
    return [ArgSpec("lhsT", (2 * ts.pmax, 2 * ts.gemm_stationary_fmax), dtype),
            ArgSpec("rhs", (2 * ts.pmax, 2 * ts.gemm_moving_fmax), dtype)]


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_level4_matmul_equals_numpy_reference(dtype):
    rep = check(NumpyRef(REFS, "ref_matmul"), KernelRef(os.path.join(EX, "level4_matmul.py"), "nki_matmul_tiled_"),
                matmul_specs(dtype))
    assert rep.verdict == "EQUIVALENT", rep.reason
    assert rep.accept, rep.numerics


def test_buggy_matmul_rejected_against_numpy_reference():
    rep = check(NumpyRef(REFS, "ref_matmul"), KernelRef(os.path.join(EX, "matmul.py"), "bug_skip_last_k"),
                matmul_specs("bfloat16"))
    assert rep.verdict == "NOT_EQUIVALENT"
    assert rep.counterexample is not None


@pytest.mark.parametrize("kernel", ["softmax_ref", "softmax_fused", "softmax_online"])
def test_softmax_kernels_equal_numpy_softmax(kernel):
    ts = parse_profile("small")
    specs = [ArgSpec("x", (2 * ts.pmax, 2 * ts.psum_bank_fmax), "float32")]
    rep = check(NumpyRef(REFS, "ref_softmax"), KernelRef(os.path.join(EX, "softmax.py"), kernel), specs)
    assert rep.verdict == "EQUIVALENT", rep.reason
    assert rep.accept, rep.numerics


def test_unsupported_numpy_function_is_named():
    specs = [ArgSpec("x", (4, 8), "float32")]
    rep = check(NumpyRef(REFS, "ref_sorted_rows"), KernelRef(os.path.join(EX, "elementwise.py"), "relu_ref"),
                specs, Options(numerics=False))
    assert rep.verdict == "REFERENCE_ERROR"
    assert "np.sort" in rep.reason
