"""Command line: python -m nkiequiv check REF CAND [options]

REF / CAND are 'path/to/kernels.py::function'. REF may also be a NumPy function:
'numpy:path/to/refs.py::function'.

Input shapes come from either repeated --arg flags or a --spec file:

    # spec.py -- shapes may depend on the hardware profile so the same spec
    # works for the small symbolic profile and the full numerics profile.
    def args(ts):
        return [
            ("lhsT", (2 * ts.pmax, 2 * ts.gemm_stationary_fmax), "bfloat16"),
            ("rhs",  (2 * ts.pmax, 2 * ts.gemm_moving_fmax), "bfloat16"),
        ]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys

from .check import Options, check
from .numpy_ref import NumpyRef
from .runner import ArgSpec, KernelRef, shim_modules
from .shim.runtime import parse_profile


def _parse_arg(text: str) -> ArgSpec:
    name, rest = text.split("=", 1)
    shape, _, dtype = rest.partition(":")
    dims = tuple(int(x) for x in shape.lower().split("x"))
    return ArgSpec(name, dims, dtype or "float32")


def _load_spec(path: str):
    spec = importlib.util.spec_from_file_location("_nkiequiv_spec", path)
    mod = importlib.util.module_from_spec(spec)
    with shim_modules():
        spec.loader.exec_module(mod)
    return mod


def _specs_from(mod, profile_name):
    ts = parse_profile(profile_name)
    return [ArgSpec(n, tuple(s), d) for n, s, d in mod.args(ts)]


def format_report(rep) -> str:
    lines = [f"verdict: {rep.verdict}" + ("  (accept)" if rep.accept else ""), f"  {rep.reason}"]
    if rep.counterexample:
        ce = rep.counterexample
        lines.append("  counterexample:")
        for k in ("inputs", "output", "index", "reference_value", "candidate_value", "z3_model"):
            if k in ce:
                lines.append(f"    {k}: {ce[k]}")
        for k in ("reference_expr", "candidate_expr"):
            if k in ce:
                lines.append(f"    {k}: {ce[k][:400]}")
    for ex in rep.details.get("unproved_examples", []):
        lines.append(f"  unproved output {ex['output']}{ex['index']}:")
        lines.append(f"    reference: {ex['reference_expr'][:300]}")
        lines.append(f"    candidate: {ex['candidate_expr'][:300]}")
    if rep.numerics:
        n = rep.numerics
        if "error" in n:
            lines.append(f"numerics: FAIL ({n['error']})")
        else:
            lines.append(
                f"numerics: {'pass' if n['pass'] else 'FAIL'}  cand rel err {n['cand_max_rel_err']:.3g}, "
                f"ref rel err {n['ref_max_rel_err']:.3g}, limit {n['limit']:.3g}"
            )
            if not n["pass"]:
                for run in n["runs"]:
                    if run["cand_rel_err"] > n["limit"]:
                        lines.append(
                            f"  on inputs {run['inputs']}: cand rel err {run['cand_rel_err']:.3g} "
                            f"(ref {run['ref_rel_err']:.3g})"
                        )
    for key in ("candidate_precision_notes",):
        for note in rep.details.get(key, []):
            lines.append(f"precision: {note}")
    for w in rep.warnings:
        lines.append(f"warning: {w}")
    for a in rep.assumptions:
        lines.append(f"assumption: {a}")
    if rep.stats:
        s = {k: v for k, v in rep.stats.items() if not k.endswith("instructions")}
        lines.append(f"stats: {s}")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(prog="nkiequiv")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="check a candidate kernel against a reference")
    c.add_argument("reference")
    c.add_argument("candidate")
    c.add_argument("--arg", action="append", default=[], help="name=AxBxC:dtype (repeatable, in order)")
    c.add_argument("--spec", help="python file defining args(tile_size) -> [(name, shape, dtype)]")
    c.add_argument("--profile", default="small", help="hardware profile for the symbolic check (small|medium|full[:k=v,...])")
    c.add_argument("--numerics-profile", default=None, help="profile for the hw-precision run (needs --spec to rescale shapes)")
    c.add_argument("--no-numerics", action="store_true")
    c.add_argument("--ref-kwargs", default="{}", help="JSON kwargs for the reference kernel")
    c.add_argument("--cand-kwargs", default="{}", help="JSON kwargs for the candidate kernel")
    c.add_argument("--ref-grid", default=None, help="SPMD launch grid for the reference, e.g. 2 or 2,2")
    c.add_argument("--cand-grid", default=None, help="SPMD launch grid for the candidate")
    c.add_argument("--target", default="trn2", choices=["trn1", "trn2", "trn3"])
    c.add_argument("--z3-budget", type=int, default=256)
    c.add_argument("--neg-inf-threshold", type=float, default=1000.0)
    c.add_argument("--strict-affine", action="store_true",
                   help="reject loop-carried dependencies in nl.affine_range (pre-0.6 NKI semantics)")
    c.add_argument("--json", action="store_true", help="print the full JSON report")
    a = p.parse_args(argv)

    def grid(text):
        return tuple(int(x) for x in text.split(",")) if text else (1,)

    ref_cls = NumpyRef if a.reference.startswith("numpy:") else KernelRef
    ref = ref_cls.parse(a.reference, kwargs=json.loads(a.ref_kwargs), grid=grid(a.ref_grid))
    cand = KernelRef.parse(a.candidate, kwargs=json.loads(a.cand_kwargs), grid=grid(a.cand_grid))
    opts = Options(profile=a.profile, numerics=not a.no_numerics, target=a.target,
                   z3_budget=a.z3_budget, neg_inf_threshold=a.neg_inf_threshold,
                   strict_affine=a.strict_affine)
    if a.spec:
        mod = _load_spec(a.spec)
        specs = _specs_from(mod, a.profile)
        if a.numerics_profile:
            opts.numerics_profile = a.numerics_profile
            opts.numerics_specs = _specs_from(mod, a.numerics_profile)
    else:
        if not a.arg:
            p.error("give input shapes with --arg or --spec")
        specs = [_parse_arg(x) for x in a.arg]
        if a.numerics_profile:
            p.error("--numerics-profile needs --spec (shapes must be rescaled)")
    rep = check(ref, cand, specs, opts)
    if a.json:
        print(json.dumps(rep.to_dict(), indent=2, default=str))
    else:
        print(format_report(rep))
    return 0 if rep.accept else 1


if __name__ == "__main__":
    sys.exit(main())
