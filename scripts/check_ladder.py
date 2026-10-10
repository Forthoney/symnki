"""Validate every level of the nkibench ladder: correct kernels accepted, wrong ones rejected.

    python scripts/check_ladder.py            # all levels
    python scripts/check_ladder.py 4 8        # some levels
    python scripts/check_ladder.py -v         # print the reason for every verdict
"""

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "examples", "hackathon"))

from ladder import HERE as LADDER_DIR, LEVELS, candidates  # noqa: E402

from nkiequiv import KernelRef, Options, check  # noqa: E402
from nkiequiv.numpy_ref import NumpyRef  # noqa: E402


def check_candidate(level, file, fn, cases=None):
    """Accepted only if every harness case is EQUIVALENT with passing numerics.
    Returns (accepted, [(case label, verdict, numerics ok, reason)])."""
    spec = LEVELS[level]
    ref_path = os.path.join(LADDER_DIR, "refs.py")
    rows = []
    for case in cases if cases is not None else spec["cases"]():
        ref = NumpyRef(ref_path, spec["ref"], kwargs=case.kwargs)
        cand = KernelRef(os.path.join(LADDER_DIR, file), fn, kwargs=case.kwargs)
        opts = Options(profile=case.profile, numerics_profile="full" if case.numerics_specs else None,
                       numerics_specs=case.numerics_specs)
        rep = check(ref, cand, case.specs, opts)
        num_ok = rep.numerics.get("pass") if rep.numerics else None
        reason = rep.reason
        if rep.verdict == "EQUIVALENT" and num_ok is False:
            bad = [r for r in rep.numerics.get("runs", []) if r["cand_rel_err"] > rep.numerics["limit"]]
            reason = (f"numerics FAIL: cand rel err {rep.numerics.get('cand_max_rel_err', float('nan')):.3g} "
                      f"> limit {rep.numerics.get('limit', float('nan')):.3g}"
                      + (f" on inputs {bad[0]['inputs']}" if bad else ""))
        rows.append((case.label, rep.verdict, num_ok, reason, rep.accept))
    return all(r[4] for r in rows), rows


def main(argv):
    verbose = "-v" in argv
    levels = [int(a) for a in argv if a.isdigit()] or sorted(LEVELS)
    failures = 0
    for level in levels:
        print(f"\nlevel {level}: {LEVELS[level]['op']}  (reference: {LEVELS[level]['ref']})")
        for file, fn, should_accept in candidates(level):
            t0 = time.time()
            accepted, rows = check_candidate(level, file, fn)
            ok = accepted == should_accept
            failures += not ok
            status = "accepted" if accepted else "rejected"
            expect = "correct kernel" if should_accept else "wrong answer"
            print(f"  [{'ok' if ok else 'UNEXPECTED'}] {expect:14s} {fn:36s} {status}  ({time.time() - t0:.1f}s)")
            if verbose or not ok or not accepted:
                for label, verdict, num_ok, reason, acc in rows:
                    if verbose or not acc:
                        first = reason.splitlines()[0][:150]
                        print(f"        {label:24s} {verdict:15s} {first}")
                        if not verbose:
                            break  # the first failing case is enough to explain a rejection
        for file, fn in LEVELS[level].get("blind_spots", []):
            extra = LEVELS[level]["extra_cases"]()
            on_harness, _ = check_candidate(level, file, fn)
            on_extra, rows = check_candidate(level, file, fn, extra)
            ok = on_harness and not on_extra
            failures += not ok
            print(f"  [{'ok' if ok else 'UNEXPECTED'}] blind spot     {fn:36s} "
                  f"{'accepted' if on_harness else 'rejected'} on the harness shapes, "
                  f"{'accepted' if on_extra else 'rejected'} on {extra[0].label}")
            for label, verdict, num_ok, reason, acc in rows:
                if not acc:
                    print(f"        {label:24s} {verdict:15s} {reason.splitlines()[0][:150]}")
                    break
    print(f"\n{'ALL AS EXPECTED' if not failures else f'{failures} UNEXPECTED RESULT(S)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
