"""The validator pipeline: is a candidate kernel equivalent to a reference?

Verdicts:
  EQUIVALENT       every output element is equal over the reals (canonical form
                   or Z3 proof) at the traced shapes, and the candidate passed
                   the validity checks
  NOT_EQUIVALENT   a concrete counterexample (or a structural mismatch:
                   output count/shape/dtype)
  INVALID          the candidate is broken independent of the reference:
                   raised an error, reads uninitialized memory, leaves output
                   elements unwritten, or (with strict_affine) has an
                   affine_range loop-carried dependency
  UNSUPPORTED      the candidate uses an NKI feature the validator does not model
  UNKNOWN          no counterexample found and not every element was proved
  REFERENCE_ERROR  the reference kernel itself failed to trace
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from . import algebra as A
from . import z3check
from .domains import ConcreteDomain, SymbolicDomain, round_to_dtype
from .numpy_ref import NumpyRef, run_numpy
from .runner import ArgSpec, KernelRef, TraceResult, run_kernel, undef_sources
from .shim.runtime import TileSize, parse_profile


@dataclass
class Options:
    profile: str = "small"
    z3_budget: int = 256  # max mismatching elements sent to Z3
    z3_timeout_ms: int = 5000
    z3_total_s: float = 120.0
    numerics: bool = True
    numerics_profile: str | None = None  # defaults to the symbolic profile
    numerics_specs: list | None = None  # shapes for the numerics run (defaults to specs)
    numerics_rtol_factor: float = 4.0  # cand error may be this many times ref error
    numerics_atol: float = 1e-6
    neg_inf_threshold: float = 1000.0
    target: str = "trn2"
    seeds: int = 2
    # NKI <= 0.3 semantics: affine_range iterations must be independent. In NKI
    # 0.6 affine_range is an alias of range() and the compiler orders
    # instructions by their data dependencies, so this is off by default.
    strict_affine: bool = False


@dataclass
class Report:
    verdict: str
    reason: str = ""
    reference: str = ""
    candidate: str = ""
    profile: str = ""
    details: dict = field(default_factory=dict)
    counterexample: dict | None = None
    warnings: list = field(default_factory=list)
    assumptions: list = field(default_factory=list)
    numerics: dict | None = None
    stats: dict = field(default_factory=dict)

    @property
    def accept(self) -> bool:
        ok = self.verdict == "EQUIVALENT"
        if self.numerics is not None:
            ok = ok and self.numerics.get("pass", False)
        return ok

    def to_dict(self):
        d = asdict(self)
        d["accept"] = self.accept
        return d


# --------------------------------------------------------------------------


def _run(ref, specs, domain, prof, inputs=None, opts=None, track_deps=False) -> TraceResult:
    """Execute a reference (NKI kernel or NumPy function) or a candidate kernel."""
    if isinstance(ref, NumpyRef):
        return run_numpy(ref, specs, domain, inputs)
    return run_kernel(ref, specs, domain, prof, inputs, opts.target, track_deps=track_deps)


# Relative error a correct kernel may show regardless of the reference's own
# error (used for NumPy references, which typically compute in float32/64).
DTYPE_TOLERANCE = {"float32": 1e-5, "tfloat32": 2e-3, "float16": 2e-3, "bfloat16": 1.6e-2,
                   "float8_e4m3": 0.13, "float8_e4m3fn": 0.13, "float8_e5m2": 0.25}


def _fmt_idx(idx):
    return "[" + ",".join(str(int(i)) for i in idx) + "]"


def _validity(tr: TraceResult, who: str):
    """Problems that make a kernel invalid on its own. Returns (reason, details) or None."""
    if not tr.ok:
        return tr.error, {"error_kind": tr.error_kind}
    if tr.dep_errors:
        return tr.dep_errors[0], {"all": tr.dep_errors}
    for k, ((name, shape, dtype), n) in enumerate(zip(tr.out_meta, tr.unwritten)):
        if n:
            return (f"{who} output {k} ('{name}') has {n} of {int(np.prod(shape))} elements never written", {})
    return None


def _uninit(tr: TraceResult, who: str):
    for k, arr in enumerate(tr.outputs):
        flat = arr.reshape(-1)
        for i, p in enumerate(flat):
            if p.has_undef:
                idx = np.unravel_index(i, arr.shape)
                srcs = undef_sources(p)
                where = ", ".join(f"'{s}' (allocated at {tr.storage_sites.get(s) or '?'})" for s in srcs)
                return (
                    f"{who} output {k}{_fmt_idx(idx)} depends on uninitialized memory of {where}",
                    {"expression": A.poly_str(p)},
                )
    return None


def _input_sets(specs: list[ArgSpec], seeds: int):
    """Input distributions for the concrete counterexample search."""
    dists = [
        ("normal", lambda r, s: r.standard_normal(s)),
        ("small_ints", lambda r, s: r.integers(-3, 4, size=s).astype(np.float64)),
        ("positive", lambda r, s: r.uniform(0.5, 2.0, size=s)),
        ("wide", lambda r, s: 30.0 * r.standard_normal(s)),
        # Monotone ramp over [-100, 100]: large spreads within a row expose
        # overflow in kernels that are only shift-invariant over the reals.
        ("ramp", lambda r, s: np.broadcast_to(np.linspace(-100.0, 100.0, s[-1]), s)
         + r.standard_normal(s)),
    ]
    for name, gen in dists:
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            yield f"{name}/seed={seed}", {s.name: gen(rng, tuple(s.shape)) for s in specs}


def _close(a, b, rtol=1e-7, atol=1e-9):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        bad = ~(np.abs(a - b) <= atol + rtol * np.maximum(np.abs(a), np.abs(b)))
    # Only count elements where the reference is finite.
    return bad & np.isfinite(a)


def _search_counterexample(ref, cand, specs, prof, opts, mismatch_mask):
    dom = ConcreteDomain("real")
    for label, inputs in _input_sets(specs, opts.seeds):
        rr = _run(ref, specs, dom, prof, inputs, opts)
        cr = _run(cand, specs, dom, prof, inputs, opts)
        if not (rr.ok and cr.ok):
            continue
        for k, (ra, ca) in enumerate(zip(rr.outputs, cr.outputs)):
            bad = _close(ra, ca)
            if mismatch_mask is not None:
                bad &= mismatch_mask[k]
            if bad.any():
                flat = int(np.flatnonzero(bad.reshape(-1))[0])
                idx = np.unravel_index(flat, ra.shape)
                ce = {
                    "inputs": label,
                    "output": k,
                    "index": [int(i) for i in idx],
                    "reference_value": float(ra.reshape(-1)[flat]),
                    "candidate_value": float(ca.reshape(-1)[flat]),
                    "num_mismatching_elements": int(bad.sum()),
                }
                small = all(int(np.prod(s.shape)) <= 64 for s in specs)
                if small:
                    ce["input_values"] = {n: v.tolist() for n, v in inputs.items()}
                return ce
    return None


def _numerics(ref, cand, specs, prof: TileSize, opts):
    """Compare candidate vs reference vs float64 ground truth under hw precision."""
    worst = {"ref_err": 0.0, "cand_err": 0.0}
    per_dist = []
    for label, inputs in _input_sets(specs, 1):
        # Inputs are representable in their dtype.
        inputs = {s.name: round_to_dtype(inputs[s.name], s.dtype) for s in specs}
        truth = _run(ref, specs, ConcreteDomain("real"), prof, inputs, opts)
        rh = _run(ref, specs, ConcreteDomain("hw"), prof, inputs, opts)
        ch = _run(cand, specs, ConcreteDomain("hw"), prof, inputs, opts)
        if not (truth.ok and rh.ok and ch.ok):
            return {"pass": False, "error": (truth.error or rh.error or ch.error)}
        ref_err = cand_err = 0.0
        for t, r, c in zip(truth.outputs, rh.outputs, ch.outputs):
            scale = max(1e-30, float(np.nanmax(np.abs(t))) if t.size else 1.0)
            fin = np.isfinite(t)
            if not np.all(np.isfinite(c)[fin]):
                cand_err = float("inf")
                continue
            ref_err = max(ref_err, float(np.max(np.abs(r - t)[fin], initial=0.0)) / scale)
            cand_err = max(cand_err, float(np.max(np.abs(c - t)[fin], initial=0.0)) / scale)
        per_dist.append({"inputs": label, "ref_rel_err": ref_err, "cand_rel_err": cand_err})
        worst["ref_err"] = max(worst["ref_err"], ref_err)
        worst["cand_err"] = max(worst["cand_err"], cand_err)
    limit = opts.numerics_rtol_factor * worst["ref_err"] + opts.numerics_atol
    if isinstance(ref, NumpyRef):
        out_dtypes = [m[2] for m in ch.out_meta]
        limit = max(limit, max(DTYPE_TOLERANCE.get(d, 1e-5) for d in out_dtypes))
    return {
        "pass": worst["cand_err"] <= limit,
        "ref_max_rel_err": worst["ref_err"],
        "cand_max_rel_err": worst["cand_err"],
        "limit": limit,
        "runs": per_dist,
    }


def check(ref: KernelRef | NumpyRef, cand: KernelRef, specs: list[ArgSpec], opts: Options | None = None) -> Report:
    opts = opts or Options()
    t0 = time.time()
    A.reset_tables()
    A.set_neg_inf_threshold(opts.neg_inf_threshold)
    prof = parse_profile(opts.profile)
    rep = Report(verdict="UNKNOWN", reference=ref.label(), candidate=cand.label(), profile=opts.profile)

    sym = SymbolicDomain()
    rt = _run(ref, specs, sym, prof, opts=opts, track_deps=opts.strict_affine)
    t_ref = time.time()
    if not rt.ok:
        rep.verdict, rep.reason = "REFERENCE_ERROR", rt.error
        return rep
    ct = run_kernel(cand, specs, sym, prof, target=opts.target, track_deps=opts.strict_affine)
    t_cand = time.time()
    rep.warnings = list(ct.warnings)
    rep.stats = {"trace_ref_s": round(t_ref - t0, 3), "trace_cand_s": round(t_cand - t_ref, 3),
                 "cand_instructions": ct.op_counts, "ref_instructions": rt.op_counts}
    if ct.precision_notes:
        rep.details["candidate_precision_notes"] = ct.precision_notes
    if rt.precision_notes:
        rep.details["reference_precision_notes"] = rt.precision_notes

    if not ct.ok and ct.error_kind == "unsupported":
        rep.verdict, rep.reason = "UNSUPPORTED", ct.error
        return rep
    v = _validity(ct, "candidate")
    if v:
        rep.verdict, rep.reason = "INVALID", v[0]
        rep.details.update(v[1])
        return rep
    v = _validity(rt, "reference") or _uninit(rt, "reference")
    if v:
        rep.verdict, rep.reason = "REFERENCE_ERROR", v[0]
        return rep
    v = _uninit(ct, "candidate")
    if v:
        rep.verdict, rep.reason = "INVALID", v[0]
        rep.details.update(v[1])
        return rep

    # Interface.
    if len(rt.outputs) != len(ct.outputs):
        rep.verdict = "NOT_EQUIVALENT"
        rep.reason = f"reference returns {len(rt.outputs)} outputs, candidate returns {len(ct.outputs)}"
        return rep
    for k, ((_, rs, rd), (_, cs, cd)) in enumerate(zip(rt.out_meta, ct.out_meta)):
        if tuple(rs) != tuple(cs):
            rep.verdict, rep.reason = "NOT_EQUIVALENT", f"output {k} shape {cs} != reference {rs}"
            return rep
        if rd != cd:
            rep.verdict, rep.reason = "NOT_EQUIVALENT", f"output {k} dtype {cd} != reference {rd}"
            return rep

    # Canonical-form comparison.
    eq = np.frompyfunc(lambda a, b: a == b, 2, 1)
    mismatch = [~eq(r, c).astype(bool) for r, c in zip(rt.outputs, ct.outputs)]
    n_total = sum(int(m.size) for m in mismatch)
    n_mis = sum(int(m.sum()) for m in mismatch)
    rep.stats.update({"elements": n_total, "canonical_equal": n_total - n_mis})

    if n_mis:
        ce = _search_counterexample(ref, cand, specs, prof, opts, mismatch)
        if ce is not None:
            k = ce["output"]
            idx = tuple(ce["index"])
            ce["reference_expr"] = A.poly_str(rt.outputs[k][idx])
            ce["candidate_expr"] = A.poly_str(ct.outputs[k][idx])
            rep.verdict = "NOT_EQUIVALENT"
            rep.reason = (
                f"output {k}{_fmt_idx(idx)} differs: reference {ce['reference_value']:.6g}, "
                f"candidate {ce['candidate_value']:.6g} ({ce['num_mismatching_elements']} elements differ "
                f"on inputs {ce['inputs']})"
            )
            rep.counterexample = ce
            rep.assumptions = sorted(A.ASSUMPTIONS)
            return rep

        # Z3 on the residual elements.
        proved = refuted = unknown = 0
        deadline = time.time() + opts.z3_total_s
        examples = []
        budget = opts.z3_budget
        for k, m in enumerate(mismatch):
            for flat in np.flatnonzero(m.reshape(-1)):
                if budget <= 0 or time.time() > deadline:
                    break
                budget -= 1
                idx = np.unravel_index(int(flat), m.shape)
                status, model = z3check.prove_equal(rt.outputs[k][idx], ct.outputs[k][idx], opts.z3_timeout_ms)
                if status == "proved":
                    proved += 1
                elif status == "refuted":
                    refuted += 1
                    rep.verdict = "NOT_EQUIVALENT"
                    rep.reason = f"output {k}{_fmt_idx(idx)} differs (Z3 counterexample)"
                    rep.counterexample = {
                        "output": k, "index": [int(i) for i in idx], "z3_model": model,
                        "reference_expr": A.poly_str(rt.outputs[k][idx]),
                        "candidate_expr": A.poly_str(ct.outputs[k][idx]),
                    }
                    rep.stats.update({"z3_proved": proved})
                    rep.assumptions = sorted(A.ASSUMPTIONS)
                    return rep
                else:
                    unknown += 1
                    if len(examples) < 3:
                        examples.append({
                            "output": k, "index": [int(i) for i in idx],
                            "reference_expr": A.poly_str(rt.outputs[k][idx]),
                            "candidate_expr": A.poly_str(ct.outputs[k][idx]),
                        })
        rep.stats.update({"z3_proved": proved, "z3_unknown": unknown,
                          "unchecked": n_mis - proved - unknown - refuted})
        if proved < n_mis:
            rep.verdict = "UNKNOWN"
            rep.reason = (
                f"{n_mis} of {n_total} elements differ in canonical form; no counterexample found; "
                f"Z3 proved {proved}, {unknown} unknown, {n_mis - proved - unknown} not attempted"
            )
            rep.details["unproved_examples"] = examples
            rep.assumptions = sorted(A.ASSUMPTIONS)
            if opts.numerics:
                rep.numerics = _numerics_run(ref, cand, specs, prof, opts)
            return rep

    rep.verdict = "EQUIVALENT"
    rep.reason = (
        f"all {n_total} output elements equal over the reals at profile '{opts.profile}'"
        + (f" ({rep.stats.get('z3_proved', 0)} via Z3)" if n_mis else "")
    )
    rep.assumptions = sorted(A.ASSUMPTIONS)
    if opts.numerics:
        rep.numerics = _numerics_run(ref, cand, specs, prof, opts)
    rep.stats["total_s"] = round(time.time() - t0, 3)
    return rep


def _numerics_run(ref, cand, specs, prof, opts):
    nprof = parse_profile(opts.numerics_profile) if opts.numerics_profile else prof
    nspecs = opts.numerics_specs or specs
    return _numerics(ref, cand, nspecs, nprof, opts)
