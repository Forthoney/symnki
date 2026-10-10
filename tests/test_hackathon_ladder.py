"""Every level of the nkibench ladder: the correct kernel is accepted, every wrong one rejected."""

import inspect
import os
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "examples", "hackathon"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import ladder  # noqa: E402
from check_ladder import check_candidate  # noqa: E402

NKIBENCH = os.environ.get(
    "NKIBENCH", os.path.expanduser("~/repos/trainium-agent-labs/projects/02-kernel-agent/nkibench.py"))

CASES = [(lv, f, fn, ok) for lv in ladder.LEVELS for f, fn, ok in ladder.candidates(lv)]


@pytest.mark.parametrize("level,file,fn,should_accept", CASES, ids=[f"L{c[0]}-{c[2]}" for c in CASES])
def test_ladder_candidate(level, file, fn, should_accept):
    accepted, rows = check_candidate(level, file, fn)
    assert accepted == should_accept, rows


def test_level6_blind_spot_needs_a_wider_shape():
    """Wrong in general but equal to the reference on every harness shape."""
    (file, fn), = ladder.LEVELS[6]["blind_spots"]
    assert check_candidate(6, file, fn)[0]
    assert not check_candidate(6, file, fn, ladder.LEVELS[6]["extra_cases"]())[0]


@pytest.mark.skipif(not os.path.exists(NKIBENCH), reason="trainium-agent-labs not checked out")
def test_references_match_nkibench():
    """examples/hackathon/refs.py must be a verbatim copy of nkibench's references."""
    import importlib.util

    import refs

    spec = importlib.util.spec_from_file_location("nkibench", NKIBENCH)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    for name in ("ref_avgpool2d", "ref_transpose2d", "ref_matmul", "ref_attention"):
        assert inspect.getsource(getattr(refs, name)) == inspect.getsource(getattr(bench, name)), name
    for lv, spec_ in ladder.LEVELS.items():
        assert spec_["ref"] == bench.LEVELS[lv]["ref"].__name__
    # The proved shapes are the harness's shapes (scaled tile-for-tile for levels 4-8).
    assert [(s["shape"], s["pool_size"]) for s in bench.LEVELS[1]["shapes"]] == \
        [(c.specs[0].shape, c.kwargs["pool_size"]) for c in ladder.LEVELS[1]["cases"]()]
    assert [(s["shape"], s["shape2D"]) for s in bench.LEVELS[2]["shapes"]] == \
        [(c.specs[0].shape, c.kwargs["shape2D"]) for c in ladder.LEVELS[2]["cases"]()]
    for lv in (3, 4, 5, 6, 7):
        harness = [(s["K"], s["M"], s["N"]) for s in bench.LEVELS[lv]["shapes"]]
        ours = [(c.numerics_specs or c.specs)[0].shape + (c.numerics_specs or c.specs)[1].shape[1:]
                for c in ladder.LEVELS[lv]["cases"]()]
        assert harness == ours, lv
    assert [(s["seq"], s["dim"]) for s in bench.LEVELS[8]["shapes"]] == \
        [c.numerics_specs[0].shape for c in ladder.LEVELS[8]["cases"]()]
