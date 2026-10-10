# nkiequiv: an equivalence checker for NKI kernels

`nkiequiv` decides whether a candidate NKI kernel computes the same function as a
reference kernel. It is meant to sit in an optimization loop, where agents
propose faster rewrites and every candidate must pass the checker before it is
accepted. Agents do not write proofs. The checker is fully automatic and returns
a verdict plus a concrete counterexample or diagnostic the agent can act on.

It targets the NKI 0.6 API (`import nki`, `nki.language as nl`, `nki.isa as nisa`).
It does not need the `nki` package or Trainium hardware: kernels run unmodified
on a built-in interpreter of the NKI instruction set.

```
$ nkiequiv check examples/attention.py::attention_ref examples/attention.py::attention_flash \
      --spec examples/attention.py
verdict: EQUIVALENT  (accept)
  all 32 output elements equal over the reals at profile 'small'
numerics: pass  cand rel err 8.45e-07, ref rel err 7.75e-07, limit 4.1e-06
assumption: exp(x) with a constant offset <= -1000 is treated as 0 (masking constants act as -inf)
assumption: max() arguments that are constants <= -1000 are treated as -inf and dropped
```

## How it works

```
reference.py::k   candidate.py::k
        │                │      run on a drop-in `nki` shim whose tensors hold
        ▼                ▼      symbolic expressions (one variable per input element)
   symbolic trace   symbolic trace
        │                │
        ├── validity ────┤   errors, out-of-bounds slices, hardware limits (pmax,
        │                │   PSUM bank, matmul tile sizes), uninitialized reads,
        │                │   unwritten output elements
        ▼                ▼
   canonical form == canonical form ?  (per output element, exact rational arithmetic)
        │ no
        ▼
   concrete counterexample search      normal / small-integer (ties) / positive / wide /
        │ none found                   ramp inputs, both kernels run in float64
        ▼
   Z3 over the reals                   max/min/abs/comparisons exact; exp/log/... are
        │                              uninterpreted, so `unsat` is a proof
        ▼
   numerics                            both kernels in emulated hardware precision
                                       (fp32 engines, bf16/fp8 stores) vs float64 truth
```

The core is `nkiequiv/algebra.py`, a normal form for real-valued expressions.
Polynomials are kept in expanded form with exact rational coefficients, so
reordered, tiled or blocked reductions compare equal. On top of that:

- `exp(a)·exp(b) = exp(a+b)`, and sums of exponentials are normalized by factoring
  out `exp(mean exponent)`. Softmax with or without max-subtraction, and the
  online-softmax / flash-attention rescaling recurrence, all reduce to the same
  form.
- `max` is flattened, deduplicated and has common shifts and positive scales
  pulled out, so a running max over tiles equals a single max. `min`, `abs` and
  `relu` are expressed through `max`.
- `sqrt`, `rsqrt` and `reciprocal` of the same expression share one atom with
  rational exponents, so they cancel.
- `sigmoid`, `tanh`, `silu` and `softplus` are rewritten into `exp`/`log`.

Whatever the normal form cannot settle goes to Z3. For example,
`relu(x) = (x + |x|)/2` and reordered clamps are proved that way.

## Verdicts

| verdict | meaning |
|---|---|
| `EQUIVALENT` | every output element is equal over the reals at the traced shapes |
| `NOT_EQUIVALENT` | concrete counterexample (input distribution and seed, output index, both values, both symbolic expressions), or an output count/shape/dtype mismatch |
| `INVALID` | candidate is broken on its own: exception, hardware-limit violation, output depends on uninitialized memory (names the tile and its allocation line), unwritten output elements |
| `UNSUPPORTED` | candidate uses an NKI feature the interpreter does not model (named in the report) |
| `UNKNOWN` | no counterexample, but not every element was proved (budget, or a rewrite outside the normal form) |
| `REFERENCE_ERROR` | the reference itself failed |

`accept` = `EQUIVALENT` **and** numerics pass. Numerics are separate because
some rewrites are exact over the reals but unsafe in floating point.
`risky_first_chunk_max` in `examples/softmax.py` normalizes by the max of the
first chunk only. That is algebraically exact, but `exp` overflows on hardware
when a later chunk is larger. The checker reports it as `EQUIVALENT`, with
`numerics: FAIL`, and `accept` is false.

## Usage

```bash
# shapes from a spec function (scales with the hardware profile)
nkiequiv check ref.py::kernel cand.py::kernel --spec spec.py
# or explicit shapes, in argument order
nkiequiv check ref.py::kernel cand.py::kernel --arg lhsT=16x16:bfloat16 --arg rhs=16x32:bfloat16
# prove at small tiles, measure numerics at real Trainium2 tile sizes
nkiequiv check ref.py::k cand.py::k --spec spec.py --numerics-profile full
# meta-parameters, SPMD launch grids, JSON for agents
nkiequiv check ref.py::k cand.py::k --spec spec.py --cand-kwargs '{"TILES_IN_BLOCK_M": 2}' \
    --cand-grid 2 --json
```

A spec file defines `args(ts)` and returns `[(name, shape, dtype), ...]`. Here
`ts` is the hardware profile (`ts.pmax`, `ts.gemm_moving_fmax`, ...). The example
files carry their own `args`.

Python API, for an agent loop:

```python
from nkiequiv import check, KernelRef, ArgSpec, Options

rep = check(KernelRef("ref.py", "kernel"), KernelRef("cand.py", "kernel"),
            [ArgSpec("x", (8, 16), "float32")], Options(profile="small"))
if not rep.accept:
    feedback = rep.to_dict()  # verdict, reason, counterexample, numerics, warnings
```

Profiles: `small` (pmax 4, 4×8 matmul tiles), `medium` (8, 8×16) and `full`
(Trainium2: 128, 128×512). You can override fields, e.g.
`--profile small:pmax=8,gemm_moving_fmax=16`.

## What `EQUIVALENT` does and does not mean

- **Equal over the reals.** The check is not bit-exact. Floating-point behaviour
  is covered by the numerics stage, which is testing, not proof.
- **At the traced shapes and kwargs.** The symbolic trace runs at concrete
  shapes, usually a scaled-down profile. Kernels that size tiles from
  `nl.tile_size` keep the same structure at any profile. Use shapes that span at
  least two tiles in every dimension; the examples use four. Hard-coded `128`
  sizes need `--profile full`, which works but is slow: a 256×1024 online softmax
  took about 160 s.
- **Under the listed assumptions.** Constants at or below `-1000` act as `-inf`
  in `max` and `exp` (masking and running-max initialization). Each report lists
  the assumptions that were actually used. Change the threshold with
  `--neg-inf-threshold`.
- **On the common domain.** Where one kernel divides by zero or takes `sqrt` of
  a negative, no claim is made.
- **Relative to the interpreter's model of NKI.** See conformance below.

## Conformance with the real NKI simulator

`scripts/conformance.py` runs the examples through the real `nki.simulate`
(NKI 0.6.0, which needs Python ≤ 3.13) at full Trainium2 tile sizes. It then
replays the same inputs through the interpreter in hardware-precision mode.

| kernels | agreement |
|---|---|
| relu / clamp variants | bit-exact |
| matmul (4 optimized variants, 2 buggy) | 99.99% bit-exact; the rest differ by 1 bf16 ulp (fp32 summation order) |
| softmax (6 variants) | max diff 5.6e-8 relative (fp32 `exp` implementation) |
| causal attention (reference, flash) | max diff 2.5e-7 relative |

Re-run it whenever NKI changes:

```bash
python -I scripts/conformance.py record OUT   # in an env with the real nki package
python scripts/conformance.py compare OUT     # in this env
```

## Examples

| file | kernels |
|---|---|
| `examples/matmul.py` | NKI tutorial matmuls (tiled, hoisted loads, free-dim blocking, fully optimized) plus bugs: PSUM tile hoisted out of the loop, missing memset, skipped K tile, wrong rhs block |
| `examples/softmax.py` | reference, `nl.softmax`, fused activation+reduce, two-pass online softmax; bugs: missing rescale, first-chunk max (overflow) |
| `examples/attention.py` | causal attention reference vs flash-style (online softmax, causal block skipping, `nc_transpose`); bug: strict mask |
| `examples/elementwise.py` | relu via abs, clamp reorderings (Z3 path), SPMD `program_id` split |

`pytest tests` runs all of them (about 1 s).

## Limitations and next steps

- **Not modelled:** indirect/dynamic addressing (`vector_select`, tensor-valued
  indices, `.indirect`), dtype reinterpretation (`.view`), `bn_stats`/`bn_aggr`,
  scans, RNG/dropout, top-k/max8, collectives, MX matmul, perf modes. These
  return `UNSUPPORTED`; add semantics in `nkiequiv/shim/isa.py` as agents need
  them. LayerNorm kernels that use `bn_stats` are the obvious next addition.
- **Integer and bitwise math on data** is uninterpreted: equal only if computed
  identically. Index math from `iota`/`affine_select` is exact.
- **Control flow** must not depend on tensor values (`dynamic_range` bounds and
  `nl.ds` starts must be Python ints).
- **No proof for all shapes.** A proof at one shape does not prove another. A
  parametric proof would need symbolic indices; that is the natural next step if
  agents start specializing on shape.
- **Compile-time legality is only partly checked.** The interpreter is lenient
  about some layout rules the real compiler enforces, such as `reshape` of
  non-contiguous views and engine/buffer placement. Keep running the real
  compiler or simulator.
- **`affine_range`.** In NKI 0.6, `affine_range` and `sequential_range` are both
  aliases of `range()`, and the compiler orders instructions by their data
  dependencies. Loop-carried dependencies are therefore legal and not flagged by
  default. `--strict-affine` enforces the older contract (iterations
  independent).
