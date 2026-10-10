import importlib.util
import os

import pytest

from nkiequiv import ArgSpec, KernelRef, Options, check
from nkiequiv.runner import shim_modules
from nkiequiv.shim.runtime import parse_profile

EX = os.path.join(os.path.dirname(__file__), "..", "examples")


def _specs(file, profile="small"):
    path = os.path.join(EX, file)
    spec = importlib.util.spec_from_file_location("_spec_" + file[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    with shim_modules():
        spec.loader.exec_module(mod)
    return [ArgSpec(n, tuple(s), d) for n, s, d in mod.args(parse_profile(profile))]


def _ref(file, fn, **kw):
    return KernelRef(os.path.abspath(os.path.join(EX, file)), fn, **kw)


SMALL_BLOCKS = {"TILES_IN_BLOCK_M": 2, "TILES_IN_BLOCK_N": 2, "TILES_IN_BLOCK_K": 2}

CASES = [
    # file, reference, candidate, candidate kwargs, expected verdict, expected accept
    ("matmul.py", "matmul_tiled", "matmul_hoist_load", {}, "EQUIVALENT", True),
    ("matmul.py", "matmul_tiled", "matmul_block_free", {}, "EQUIVALENT", True),
    ("matmul.py", "matmul_tiled", "matmul_fully_optimized", SMALL_BLOCKS, "EQUIVALENT", True),
    ("matmul.py", "matmul_tiled", "bug_psum_outside_loop", {}, "NOT_EQUIVALENT", False),
    ("matmul.py", "matmul_tiled", "legacy_sbuf_accumulate_affine", {}, "EQUIVALENT", True),
    ("matmul.py", "matmul_tiled", "bug_missing_memset", {}, "INVALID", False),
    ("matmul.py", "matmul_tiled", "bug_skip_last_k", {}, "NOT_EQUIVALENT", False),
    ("matmul.py", "matmul_tiled", "bug_transposed_tile", {}, "NOT_EQUIVALENT", False),
    ("softmax.py", "softmax_ref", "softmax_highlevel", {}, "EQUIVALENT", True),
    ("softmax.py", "softmax_ref", "softmax_fused", {}, "EQUIVALENT", True),
    ("softmax.py", "softmax_ref", "softmax_online", {}, "EQUIVALENT", True),
    ("softmax.py", "softmax_ref", "bug_online_no_rescale", {}, "NOT_EQUIVALENT", False),
    # Equal over the reals, but overflows in float32: rejected by the numerics stage.
    ("softmax.py", "softmax_ref", "risky_first_chunk_max", {}, "EQUIVALENT", False),
    ("elementwise.py", "relu_ref", "relu_via_abs", {}, "EQUIVALENT", True),
    ("elementwise.py", "relu_ref", "bug_relu_via_abs", {}, "NOT_EQUIVALENT", False),
    ("elementwise.py", "clamp_ref", "clamp_swapped", {}, "EQUIVALENT", True),
    ("attention.py", "attention_ref", "attention_flash", {}, "EQUIVALENT", True),
    ("attention.py", "attention_ref", "bug_flash_strict_mask", {}, "NOT_EQUIVALENT", False),
]


@pytest.mark.parametrize("file,ref,cand,kwargs,verdict,accept", CASES, ids=[c[2] for c in CASES])
def test_example(file, ref, cand, kwargs, verdict, accept):
    rep = check(_ref(file, ref), _ref(file, cand, kwargs=kwargs), _specs(file), Options())
    assert rep.verdict == verdict, rep.reason
    assert rep.accept == accept, (rep.reason, rep.numerics)


def test_spmd_grid():
    specs = _specs("elementwise.py")
    rep = check(_ref("elementwise.py", "relu_ref"), _ref("elementwise.py", "relu_spmd", grid=(2,)), specs)
    assert rep.verdict == "EQUIVALENT", rep.reason


def test_counterexample_is_reported():
    rep = check(_ref("matmul.py", "matmul_tiled"), _ref("matmul.py", "bug_skip_last_k"), _specs("matmul.py"))
    ce = rep.counterexample
    assert ce is not None and ce["output"] == 0 and len(ce["index"]) == 2
    assert ce["reference_value"] != ce["candidate_value"]


def test_strict_affine_flags_loop_carried_dependency():
    rep = check(_ref("matmul.py", "matmul_tiled"), _ref("matmul.py", "legacy_sbuf_accumulate_affine"),
                _specs("matmul.py"), Options(numerics=False, strict_affine=True))
    assert rep.verdict == "INVALID"
    assert "affine_range loop at matmul.py:" in rep.reason
    assert "sequential_range" in rep.reason
