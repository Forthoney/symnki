# Task: add a `--prove` gate to the project-2 kernel agent

## Goal

Project 2's agent (`/workspace/projects/02-kernel-agent/agent.py`) grades each candidate NKI
kernel in `grade(source, level)`. Grading runs a rule scan, then `nki.simulate` against
nkibench's NumPy reference on random inputs (tolerance 2e-2), then a traffic bar. Add an
opt-in `--prove` flag. With it, a kernel counts as correct only if it **also** passes the
symbolic equivalence checker in `/workspace/symnki` (package `nkiequiv`).

This is worth doing because the random-input test has holes the checker closes:

- **A known wrong answer it accepts.** `attention_no_max_subtraction` (no row-max
  subtraction before `exp`) passes nkibench 3/3 but overflows on real data. The checker
  rejects it in its numerics stage.
- **Weak feedback.** A wrong kernel gets "worst error 4.2 of the RMS". The checker gives a
  counterexample (output index, expected value, actual value, both symbolic expressions), or
  names the uninitialized tile and the line where it was allocated. That is better input
  for `repair_prompt`.

The experiment the flag enables: **does proof-based grading and feedback change the solve
rate?** Measure it. Don't assume it.

## Environment (seat 2, already set up)

- **The checker:** `/workspace/symnki` is a clone of `github.com/Forthoney/symnki`,
  installed with `pip install --no-deps -e /workspace/symnki` into `/opt/conda` (Python
  3.13). If `import nkiequiv` fails after a container restart, re-run that line. Do **not**
  install the `nki` extra; the image's own `nki` 0.6.0 is the right one.
- **The entry point to call:** `check_candidate(level, path, entry)` in
  `/workspace/symnki/scripts/check_ladder.py`.
  - It checks one kernel file against nkibench's NumPy reference at the harness shapes for
    that level. The level definitions live in `/workspace/symnki/examples/hackathon/ladder.py`.
  - It returns `(accepted, rows)`, one row per harness case:
    `(case_label, verdict, numerics_ok, reason, case_accepted)`.
  - `accepted` is true only if every case is `EQUIVALENT` and passes numerics.
  - Read `check_candidate` and the verdict list at the top of `nkiequiv/check.py` before
    starting.
- **Validated:** the checker is validated on this machine against the real `nki.simulate`.
  `python scripts/check_ladder.py` prints `ALL AS EXPECTED` and `pytest tests` passes 68/68.
  Run both once before you start, so you know the environment is good.
- **Other runs share this pod.** It may be running `agent.py` and vLLM. Run heavy checks
  with `nice -n 10`, and start long runs with `nohup ... > log 2>&1 < /dev/null &` as the
  repo README says.

## What to build

### 1. A flag, plumbed into `grade`

- Add `--prove` to `main()`'s argparse (default off).
- `grade(source, level)` doesn't receive the parsed arguments. Pass the setting in
  explicitly, e.g. `grade(source, level, prove=False)`, and update the caller in `solve()`.
  Without `--prove`, behaviour must be **byte-for-byte unchanged**.
- At startup with `--prove`, import the checker once. Make its location configurable with
  an env var, e.g. `SYMNKI_ROOT`, defaulting to `/workspace/symnki`. Put
  `$SYMNKI_ROOT/scripts` on `sys.path` and import `check_candidate`. If the import fails,
  exit with a clear message. Never silently grade without the proof when the flag is on.

### 2. Where the gate goes

Run the proof only after nkibench's own checks pass on every shape, i.e. at the point where
`grade()` is about to set `parts["correct"] = True` (currently agent.py:157). The proof is
the expensive step: about 15 s per candidate on level 3, under 1 s elsewhere. Kernels that
already fail nkibench don't need it.

Call `check_candidate(level, path, spec["entry"])`, where `path` is the file `grade()`
already wrote.

### 3. What each outcome does

| checker result | grading | feedback to the model |
|---|---|---|
| `accepted` | unchanged: correct, full reward | unchanged, plus one line: "Proved equivalent to the reference on every harness shape." |
| a case is `NOT_EQUIVALENT` or `INVALID` | **not** correct: drop the `correct` weight (0.5); keep parses/rules/runs credit; record `parts["proved"] = False` | the first failing case's label and `reason`. Include the counterexample index and expected/actual values; if the reason names an uninitialized tile, keep the line number. One short paragraph, phrased as an instruction, like the rest of `grade()`'s messages. |
| `EQUIVALENT` but numerics fail | not correct, as above | say the kernel is mathematically right but not numerically safe, and quote which input distribution broke it (e.g. "overflows on wide-range inputs: subtract the row max before exp"). |
| `UNSUPPORTED`, `UNKNOWN` or `REFERENCE_ERROR` | **don't penalize**: keep nkibench's verdict; record `parts["proved"] = None` | nothing extra for the model; log the reason |

`UNSUPPORTED` means the checker can't model an NKI feature the kernel uses. It says
nothing about whether the kernel is correct.

### 4. Logging

- Every `attempts.jsonl` record should show what the proof decided. Add `parts["proved"]`
  (`True`/`False`/`None`) and a `prove` field with the verdict of the first non-accepted
  case and the checker's wall time.
- The whole point is comparing solve rates with and without the flag. Make sure the
  existing `--repeat` summary can be produced for both settings from the logs.

### 5. Two pitfalls to handle

- **Stop parallel runs from overwriting each other's candidate.** `grade()` writes every
  candidate to the fixed path `/tmp/_agent_level{level}.py`. Two `agent.py` processes on
  the same pod (seat 2 currently has two) overwrite each other's file between writing and
  grading. Give each process its own path, e.g. include `os.getpid()`, or use
  `tempfile.NamedTemporaryFile(dir=..., suffix=".py", delete=False)`. The function name
  inside the file must stay the level's entry name.
- **Don't re-prove identical candidates.** Samples within a round are often identical, and
  repair rounds often resubmit the same kernel. Memoize the checker result on
  `(level, sha256(source))`. The checker already loads kernels by content hash, so there's
  no stale-module risk.

## Acceptance checks

Put these in a small script under `projects/02-kernel-agent/`, e.g.
`test_prove_gate.py`. It needs no model; it drives `grade()` directly.

1. **No change without the flag.** Grade the four shipped `reference_level{1..4}.py` with
   `prove=False` and `prove=True`. Same rewards, and `parts["proved"] is True` with the flag.
2. **Correct solutions pass.** `/workspace/symnki/examples/hackathon/solution_level{5..8}.py`
   get full reward with `prove=True`.
3. **The gate catches what nkibench misses.** Take `attention_no_max_subtraction` from
   `/workspace/symnki/examples/hackathon/wrong_level5to8.py`.
   - Copy it into a standalone file with the function renamed to the level-8 entry
     `nki_attention_`.
   - It imports `_attention` from `solution_level8.py`, so put that file next to the copy,
     or inline the helper.
   - Expected: without `--prove` it gets full reward (nkibench accepts it); with `--prove`
     it is not correct, and the feedback mentions numerics/overflow.
4. **Wrong answers carry useful feedback.** For each wrong answer in
   `examples/hackathon/wrong_level*.py`, renamed to its level's entry point (the level of
   each is in `ladder.py`'s `LEVELS[n]["wrong"]`), `grade(..., prove=True)` doesn't mark it
   correct. Its feedback contains either an index and two values, or "uninitialized".
   Most of these already fail nkibench before the proof runs; that's fine, just check the
   ones that reach it.
5. **The blind spot stays a blind spot.** `rhs_offset_drops_block_index`, renamed to level
   6's entry, is accepted with and without `--prove`. It's a real bug, but it's invisible
   on every harness shape. Don't "fix" this in the gate. It's a gap in nkibench's shapes,
   documented in `/workspace/symnki/README.md`.
6. **Concurrency.** Two `grade()` calls from different processes at the same moment, with
   different kernels for the same level, each get their own kernel's grade.

## The experiment

Once the checks pass, run the same configuration with and without the flag, e.g.

```bash
nohup python agent.py --all --rounds 8 --samples 4 --context 8192 --repeat 5 --log base.jsonl  > base.log  2>&1 < /dev/null &
nohup python agent.py --all --rounds 8 --samples 4 --context 8192 --repeat 5 --log prove.jsonl --prove > prove.log 2>&1 < /dev/null &
```

Run them one after the other if the pod's CPU or the single vLLM server is the bottleneck.
Running them concurrently also shares the model server, which skews latency but not
correctness.

Report the solve rate per level (how many of the 5 repeats solved it), with best/mean
reward, for both settings.

Separately, count rounds where the proof changed the verdict:
- kernels nkibench accepted and the proof rejected (in this case, nkibench's grade was too
  lenient);
- whether the next round's repair succeeded more often after proof feedback than after
  nkibench feedback.

The project README warns that a reasonable-looking change made every level worse once. A
negative result is a result; report it as such.

## Out of scope

- **Changing nkibench itself:** its tolerance, shapes or rules. The level-6 blind spot is
  a finding to report, not something to patch here.
- **Making the proof the default**, or replacing nkibench's simulation with it. Both run
  under `--prove`; the proof adds a check, it doesn't swap one out.
- **Proving traffic bars or roofline numbers.** The checker checks correctness only;
  levels 5–7 still need nkibench's byte counts.
