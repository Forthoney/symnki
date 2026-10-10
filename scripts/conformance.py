"""Check the shim's semantics against the real NKI CPU simulator.

The real ``nki`` wheel only exists for CPython <= 3.13, so this runs in two steps:

  # 1. in an environment with the real nki package installed
  python -I scripts/conformance.py record OUT_DIR

  # 2. in the validator's environment
  python scripts/conformance.py compare OUT_DIR

``record`` runs every case through ``nki.simulate`` at full hardware tile sizes
and saves inputs and outputs; ``compare`` replays the same inputs through the
shim's hardware-precision interpreter and reports the differences.
"""

import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(HERE, "..", "examples")
BLOCKS = {"TILES_IN_BLOCK_M": 2, "TILES_IN_BLOCK_N": 2, "TILES_IN_BLOCK_K": 2}

CASES = [
    ("matmul.py", "matmul_tiled", {}),
    ("matmul.py", "matmul_hoist_load", {}),
    ("matmul.py", "matmul_block_free", {}),
    ("matmul.py", "matmul_fully_optimized", BLOCKS),
    ("matmul.py", "bug_psum_outside_loop", {}),
    ("matmul.py", "bug_skip_last_k", {}),
    ("softmax.py", "softmax_ref", {}),
    ("softmax.py", "softmax_highlevel", {}),
    ("softmax.py", "softmax_fused", {}),
    ("softmax.py", "softmax_online", {}),
    ("softmax.py", "bug_online_no_rescale", {}),
    ("softmax.py", "risky_first_chunk_max", {}),
    ("elementwise.py", "relu_ref", {}),
    ("elementwise.py", "relu_via_abs", {}),
    ("elementwise.py", "clamp_ref", {}),
    ("elementwise.py", "clamp_swapped", {}),
    ("attention.py", "attention_ref", {}),
    ("attention.py", "attention_flash", {}),
]


def _inputs(args, seed=0):
    import ml_dtypes

    dt = {"float32": np.float32, "bfloat16": ml_dtypes.bfloat16}
    rng = np.random.default_rng(seed)
    return {n: rng.standard_normal(s).astype(dt[d]) for n, s, d in args}


def record(out_dir):
    import nki
    import nki.language as nl

    os.makedirs(out_dir, exist_ok=True)
    mods = {}
    for file, fn, kwargs in CASES:
        if file not in mods:
            spec = importlib.util.spec_from_file_location(file[:-3], os.path.join(EXAMPLES, file))
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            mods[file] = m
        m = mods[file]
        args = m.args(nl.tile_size)
        inputs = _inputs(args)
        try:
            out = np.asarray(nki.simulate(getattr(m, fn))(*[inputs[n] for n, _, _ in args], **kwargs))
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {file}::{fn}: {type(e).__name__}: {e}")
            continue
        np.savez(os.path.join(out_dir, f"{file[:-3]}__{fn}.npz"), __out__=out.astype(np.float64),
                 __meta__=json.dumps({"args": args, "kwargs": kwargs, "out_dtype": str(out.dtype)}),
                 **{n: v.astype(np.float64) for n, v in inputs.items()})
        print(f"recorded {file}::{fn}")


def compare(out_dir):
    sys.path.insert(0, os.path.join(HERE, ".."))
    from nkiequiv.domains import ConcreteDomain, round_to_dtype
    from nkiequiv.runner import ArgSpec, KernelRef, run_kernel
    from nkiequiv.shim.runtime import parse_profile

    prof = parse_profile("full")
    worst = 0.0
    for file, fn, _ in CASES:
        path = os.path.join(out_dir, f"{file[:-3]}__{fn}.npz")
        if not os.path.exists(path):
            print(f"{file}::{fn:28s} (not recorded)")
            continue
        data = np.load(path)
        meta = json.loads(str(data["__meta__"]))
        specs = [ArgSpec(n, tuple(s), d) for n, s, d in meta["args"]]
        ref = KernelRef(os.path.abspath(os.path.join(EXAMPLES, file)), fn, kwargs=meta["kwargs"])
        tr = run_kernel(ref, specs, ConcreteDomain("hw"), prof, {s.name: data[s.name] for s in specs},
                        track_deps=False)
        if not tr.ok:
            print(f"{file}::{fn:28s} shim error: {tr.error}")
            continue
        real = data["__out__"]
        ours = tr.outputs[0]
        out_dtype = tr.out_meta[0][2]
        exact = float(np.mean(round_to_dtype(ours, out_dtype) == real))
        scale = max(float(np.max(np.abs(real))), 1e-30)
        diff = float(np.max(np.abs(ours - real))) / scale
        worst = max(worst, diff)
        print(f"{file}::{fn:28s} bit-exact {exact:7.2%}   max |diff| / max|out| = {diff:.2e}")
    print(f"worst relative difference: {worst:.2e}")


if __name__ == "__main__":
    mode, out_dir = sys.argv[1], sys.argv[2]
    {"record": record, "compare": compare}[mode](out_dir)
