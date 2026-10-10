"""Load NKI kernel files against the shim and execute them in a domain."""

from __future__ import annotations

import contextlib
import importlib.util
import itertools
import os
import sys
import traceback
from dataclasses import dataclass, field

import numpy as np

from . import algebra as A
from . import shim
from .domains import Unsupported
from .shim import runtime as R


@dataclass
class ArgSpec:
    name: str
    shape: tuple
    dtype: str = "float32"


@dataclass
class KernelRef:
    path: str
    func: str
    kwargs: dict = field(default_factory=dict)
    grid: tuple = (1,)

    @classmethod
    def parse(cls, text: str, **kw) -> "KernelRef":
        """'path/to/file.py::func' (or 'path:func')."""
        if "::" in text:
            path, func = text.split("::", 1)
        else:
            path, func = text.rsplit(":", 1)
        return cls(os.path.abspath(path), func, **kw)

    def label(self):
        return f"{os.path.basename(self.path)}::{self.func}"


@contextlib.contextmanager
def shim_modules():
    names = ("nki", "nki.language", "nki.isa", "nki.typing")
    saved = {n: sys.modules.get(n) for n in names}
    sys.modules["nki"] = shim
    sys.modules["nki.language"] = shim.language
    sys.modules["nki.isa"] = shim.isa
    sys.modules["nki.typing"] = shim.typing
    try:
        yield
    finally:
        for n, m in saved.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m


_LOADED: dict = {}


def load_kernel(ref: KernelRef):
    key = (ref.path, os.path.getmtime(ref.path))
    mod = _LOADED.get(key)
    if mod is None:
        modname = f"_nkiequiv_kernel_{abs(hash(key))}"
        spec = importlib.util.spec_from_file_location(modname, ref.path)
        mod = importlib.util.module_from_spec(spec)
        with shim_modules():
            spec.loader.exec_module(mod)
        _LOADED[key] = mod
    fn = getattr(mod, ref.func, None)
    if fn is None:
        raise AttributeError(f"{ref.path} has no kernel named {ref.func!r}")
    return fn


@dataclass
class TraceResult:
    ok: bool
    error_kind: str | None = None  # invalid | unsupported | exception
    error: str | None = None
    outputs: list = field(default_factory=list)  # value arrays
    out_meta: list = field(default_factory=list)  # (name, shape, dtype)
    unwritten: list = field(default_factory=list)  # per output: count of never-written elements
    warnings: list = field(default_factory=list)
    dep_errors: list = field(default_factory=list)
    precision_notes: list = field(default_factory=list)
    op_counts: dict = field(default_factory=dict)
    storage_sites: dict = field(default_factory=dict)


def _kernel_traceback(exc, kernel_files) -> str:
    frames = [
        f"{os.path.basename(fs.filename)}:{fs.lineno} in {fs.name}: {fs.line}"
        for fs in traceback.extract_tb(exc.__traceback__)
        if fs.filename in kernel_files
    ]
    msg = f"{type(exc).__name__}: {exc}"
    if frames:
        msg += "\n  " + "\n  ".join(frames[-4:])
    return msg


def run_kernel(ref: KernelRef, specs: list[ArgSpec], domain, profile, input_values=None,
               target="trn2", track_deps=True) -> TraceResult:
    """Execute a kernel. input_values: dict name -> float64 array (concrete domains)."""
    fn = load_kernel(ref)
    grid = tuple(ref.grid)
    if isinstance(fn, shim.Kernel):
        grid = ref.grid if ref.grid != (1,) else fn.grid
        fn = fn.fn
    c = R.TraceContext(domain=domain, profile=profile, kernel_files={ref.path}, target=target,
                       track_deps=track_deps, grid=tuple(grid))
    R.set_ctx(c)
    res = TraceResult(ok=False)
    try:
        args = []
        for s in specs:
            if domain.obj:
                vals = domain.inputs(s.name, tuple(s.shape))
            else:
                vals = np.asarray(input_values[s.name], dtype=np.float64)
            t = R.new_tensor(tuple(s.shape), s.dtype, "hbm", s.name)
            t.write(vals.reshape(t.shape))
            args.append(t)
        ret = None
        for pid in itertools.product(*[range(g) for g in grid]):
            c.program_id = pid
            c.shared_counter = 0
            with shim_modules():
                r = fn(*args, **ref.kwargs)
            if ret is None:
                ret = r
        outs = ret if isinstance(ret, (tuple, list)) else [ret]
        for i, o in enumerate(outs):
            if not isinstance(o, R.NkiTensor):
                raise R.KernelError(f"kernel output {i} is {type(o).__name__}, expected an HBM tensor", site="")
            if o.on_chip:
                raise R.KernelError(f"kernel output {i} ({o.name}) is on-chip; outputs must be HBM tensors", site="")
            res.outputs.append(np.array(o._storage.vals[o._idx]))
            res.out_meta.append((o.name, o.shape, o.dtype))
            res.unwritten.append(int((~o._storage.written[o._idx]).sum()))
        res.ok = True
    except R.KernelError as e:
        res.error_kind, res.error = "invalid", str(e)
    except Unsupported as e:
        res.error_kind, res.error = "unsupported", f"unsupported NKI feature: {e} (at {_site_of(e, ref.path)})"
    except RecursionError as e:
        res.error_kind, res.error = "exception", f"RecursionError: {e}"
    except Exception as e:  # kernel's own asserts, Python errors
        res.error_kind, res.error = "exception", _kernel_traceback(e, {ref.path})
    finally:
        R.set_ctx(None)
    res.warnings = c.warnings
    res.dep_errors = c.dep_errors
    res.precision_notes = c.precision_notes
    res.op_counts = c.op_counts
    res.storage_sites = {s.name: s.site for s in c.storages}
    return res


def _site_of(exc, path) -> str:
    for fs in reversed(traceback.extract_tb(exc.__traceback__)):
        if fs.filename == path:
            return f"{os.path.basename(fs.filename)}:{fs.lineno}"
    return "?"


def undef_sources(poly: A.Poly) -> list[str]:
    out = []
    for a in poly.atoms():
        if a.kind == "undef":
            name, idx, _ = a.args
            out.append(name)
    return sorted(set(out))
