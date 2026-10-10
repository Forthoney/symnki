"""A drop-in ``nki`` package used while tracing kernels.

Kernels written against NKI 0.6 (``import nki``, ``import nki.language as nl``,
``import nki.isa as nisa``) run unmodified on top of this shim; values are
whatever the active domain stores (symbolic expressions or numbers).
"""

from __future__ import annotations

import functools

from . import isa, language
from .runtime import KernelError, NkiTensor


class Kernel:
    def __init__(self, fn, grid=(1,), **options):
        self.fn = fn
        self.grid = grid
        self.options = options
        functools.update_wrapper(self, fn)

    def __getitem__(self, grid):
        if isinstance(grid, int):
            grid = (grid,)
        return Kernel(self.fn, tuple(grid), **self.options)

    def __call__(self, *args, **kwargs):
        # Grid launches are driven by the runner; a direct call runs one program.
        return self.fn(*args, **kwargs)


def jit(fn=None, **kwargs):
    if fn is None:
        return lambda f: Kernel(f, **kwargs)
    if isinstance(fn, Kernel):
        return fn
    return Kernel(fn, **kwargs)


def simulate(kernel):
    return kernel


def baremetal(fn=None, **kwargs):
    return jit(fn, **kwargs)


benchmark = profile = baremetal

__all__ = ["jit", "simulate", "language", "isa", "Kernel", "KernelError", "NkiTensor"]
