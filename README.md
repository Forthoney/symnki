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

## Setup

The NKI SDK ships wheels for CPython 3.11–3.13 only, so the project is pinned to 3.13
(`.python-version`) and the validator runs in the same environment as the real `nki`
package. The `nki` extra installs it from AWS's Neuron index (PyPI has an unrelated
placeholder of the same name, which `pyproject.toml` rules out):

```bash
uv venv --python 3.13
uv pip install -e ".[test,nki]"
```

While tracing a kernel the validator swaps its own `nki` module into `sys.modules` and
restores the real one afterwards, so both can be used in one process.

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

### NumPy functions as the reference

The reference can be a plain NumPy function instead of an NKI kernel: prefix it with `numpy:`.

```
$ nkiequiv check numpy:examples/numpy_refs.py::ref_matmul \
      examples/level4_matmul.py::nki_matmul_tiled_ --spec examples/numpy_refs.py
verdict: EQUIVALENT  (accept)
  all 128 output elements equal over the reals at profile 'small'
numerics: pass  cand rel err 0.00323, ref rel err 0.00323, limit 0.016
```

The function runs unmodified on a traced array type (`nkiequiv/numpy_ref.py`)
that intercepts NumPy's ufunc and array-function dispatch.

- **Casts.** `astype` to a float dtype keeps the value over the reals; only the
  recorded dtype changes, and output dtypes must match the kernel's.
- **Numerics.** The function runs exactly as written on arrays of the declared
  input dtypes. The tolerance is the larger of 4× the reference's own error and
  a per-output-dtype floor (float32 1e-5, bf16 1.6e-2, ...).
- **Supported subset.** Arithmetic and comparisons, ufuncs, `where`, reductions
  (`sum`/`mean`/`max`/`min`/`prod` with `axis`/`keepdims`), `@`/`matmul`/`dot`,
  transpose/reshape/broadcasting/slicing, and `astype`. Anything else (e.g.
  `np.sort`, boolean-mask indexing, data-dependent `if`) is reported by name.

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
(NKI 0.6.0) at full Trainium2 tile sizes. It then
replays the same inputs through the interpreter in hardware-precision mode.

| kernels | agreement |
|---|---|
| relu / clamp variants | bit-exact |
| matmul (4 optimized variants, 2 buggy) | 99.99% bit-exact; the rest differ by 1 bf16 ulp (fp32 summation order) |
| softmax (6 variants) | max diff 5.6e-8 relative (fp32 `exp` implementation) |
| causal attention (reference, flash) | max diff 2.5e-7 relative |

Re-run it whenever NKI changes:

```bash
python scripts/conformance.py record OUT    # runs the real nki.simulate
python scripts/conformance.py compare OUT   # replays the same inputs through the shim
```

## Examples

| file | kernels |
|---|---|
| `examples/matmul.py` | NKI tutorial matmuls (tiled, hoisted loads, free-dim blocking, fully optimized) plus bugs: PSUM tile hoisted out of the loop, missing memset, skipped K tile, wrong rhs block |
| `examples/softmax.py` | reference, `nl.softmax`, fused activation+reduce, two-pass online softmax; bugs: missing rescale, first-chunk max (overflow) |
| `examples/attention.py` | causal attention reference vs flash-style (online softmax, causal block skipping, `nc_transpose`); bug: strict mask |
| `examples/elementwise.py` | relu via abs, clamp reorderings (Z3 path), SPMD `program_id` split |

`pytest tests` runs all of them (about 1 s).

## The nkibench ladder (trainium-agent-labs, project 2)

`examples/hackathon/` checks every level of the hackathon's kernel ladder against the NumPy
reference nkibench grades with (`refs.py`, a verbatim copy that the tests keep in sync). Each
level has one correct kernel that must be accepted and plausible wrong answers that must be
rejected:

```bash
python scripts/check_ladder.py        # all 8 levels, ~1 min (level 3 is proved at 128x64x512)
python scripts/check_ladder.py 4 -v   # one level, every case and verdict
```

| level | correct kernel | wrong answers (all rejected) |
|---|---|---|
| 1 avgpool | `reference_level1.py` | divide by p not p², wrong window stride, max-pool |
| 2 transpose | `reference_level2.py` | inverse permutation (right only on square shapes), plain copy, missed last row (uninitialized) |
| 3 matmul, 1 tile | `reference_level3.py` | DMA straight from PSUM, `accumulate=True` on a fresh tile, half the contraction |
| 4 tiled | `reference_level4.py` | skipped K tile, PSUM shared across n, wrong rhs block, swapped output block |
| 5 hoisted | `solution_level5.py` | lhsT hoisted one loop too far |
| 6 M/N blocked | `solution_level6.py` | lhsT block offset drops the block index |
| 7 M/N/K blocked | `solution_level7.py` | accumulator never zeroed, K-block offset dropped |
| 8 attention | `solution_level8.py` | no max subtraction (exact over the reals, overflows: caught by numerics), no 1/√d, probabilities not transposed |

Levels 1–3 are proved at the exact harness shapes. Levels 4–8 are proved at the harness
shapes scaled tile-for-tile and checked numerically at the real ones. Each level 5–7 solution
also meets that level's HBM-traffic bar. Cross-checked against the real `nki.simulate` with
nkibench's own checking code: every correct kernel passes all harness shapes, and every wrong
one fails at least one, except `attention_no_max_subtraction`, which nkibench's numerical
check accepts on its shapes.

**A blind spot of the harness shapes.** `rhs_offset_drops_block_index` (level 6) drops the N
block index from an offset. Every harness shape has a single N block, so on those shapes it
*is* the reference: both the proof and nkibench accept it. A shape with N = 2048 exposes it.
A proof covers only the shapes it is given, so the shapes need at least two blocks in every
dimension a kernel blocks.

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
