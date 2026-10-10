"""Execution state for the NKI shim: storages, tensor views, trace context."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np

from ..domains import Unsupported


class KernelError(Exception):
    """The kernel did something invalid (would fail to compile or is UB)."""

    def __init__(self, message: str, site: str | None = None):
        self.site = site if site is not None else current_site()
        super().__init__(f"{message}" + (f" (at {self.site})" if self.site else ""))
        self.raw = message


# --------------------------------------------------------------------------
# Hardware profiles
# --------------------------------------------------------------------------


@dataclass
class TileSize:
    pmax: int
    psum_bank_fmax: int
    gemm_stationary_fmax: int
    gemm_moving_fmax: int
    bn_stats_fmax: int
    psum_num_banks: int = 8
    sbuf_fmax_bytes: int = 192 * 1024
    psum_min_align: int = 4
    sbuf_min_align: int = 4

    @property
    def psum_fmax(self):
        return self.psum_bank_fmax

    @property
    def psum_bank_fmax_bytes(self):
        return self.psum_bank_fmax * 4

    @property
    def sbuf_fmax(self):
        return self.sbuf_fmax_bytes // 4

    @property
    def sbuf_size_bytes(self):
        return self.sbuf_fmax_bytes * self.pmax

    @property
    def total_available_sbuf_size(self):
        return self.sbuf_fmax_bytes


PROFILES = {
    # Trainium2 / NeuronCore-v3 values.
    "full": TileSize(128, 512, 128, 512, 512),
    # Same structure, shrunk so symbolic traces stay small. Kernels that size
    # their tiles from nl.tile_size scale down automatically.
    "small": TileSize(4, 8, 4, 8, 8, sbuf_fmax_bytes=4096),
    "medium": TileSize(8, 16, 8, 16, 16, sbuf_fmax_bytes=16384),
}


def parse_profile(spec: str) -> TileSize:
    """'small', 'full', or 'small:pmax=8,gemm_moving_fmax=16'."""
    name, _, overrides = spec.partition(":")
    base = PROFILES[name]
    kw = dict(base.__dict__)
    if overrides:
        for item in overrides.split(","):
            k, v = item.split("=")
            if k not in kw:
                raise ValueError(f"unknown tile size field {k}")
            kw[k] = int(v)
    return TileSize(**kw)


# --------------------------------------------------------------------------
# Context
# --------------------------------------------------------------------------

HBM_BUFFERS = ("hbm", "shared_hbm", "private_hbm")
ON_CHIP = ("sbuf", "psum")


@dataclass
class TraceContext:
    domain: object
    profile: TileSize
    kernel_files: set = field(default_factory=set)
    target: str = "trn2"
    track_deps: bool = True
    warnings: list = field(default_factory=list)
    storages: list = field(default_factory=list)
    op_counts: dict = field(default_factory=dict)
    program_id: tuple = (0,)
    grid: tuple = (1,)
    # Active affine_range loops: list of [instance id, iteration, site].
    loop_stack: list = field(default_factory=list)
    loop_instances: int = 0
    dep_errors: list = field(default_factory=list)
    precision_notes: list = field(default_factory=list)
    # SPMD: the i-th shared_hbm allocation of every program aliases program 0's
    shared_allocs: list = field(default_factory=list)
    shared_counter: int = 0
    act_regs: object = None
    dve_regs: object = None

    def __post_init__(self):
        self.act_regs = self.domain.full((128,), 0.0) if self.domain.obj else np.zeros(128)
        self.dve_regs = self.domain.full((128,), 0.0) if self.domain.obj else np.zeros(128)

    def warn(self, msg: str):
        site = current_site()
        entry = f"{msg}" + (f" (at {site})" if site else "")
        if entry not in self.warnings:
            self.warnings.append(entry)

    def count(self, name: str):
        self.op_counts[name] = self.op_counts.get(name, 0) + 1

    def signature(self):
        if not self.loop_stack:
            return None
        return tuple((f[0], f[1]) for f in self.loop_stack)

    def loop_site(self, inst):
        for f in self.loop_stack:
            if f[0] == inst:
                return f[2]
        return None

    def note_precision(self, msg: str):
        site = current_site()
        entry = f"{msg}" + (f" (at {site})" if site else "")
        if entry not in self.precision_notes:
            self.precision_notes.append(entry)


_CTX: TraceContext | None = None


def ctx() -> TraceContext:
    if _CTX is None:
        raise RuntimeError("NKI shim used outside of a validator trace")
    return _CTX


def set_ctx(c: TraceContext | None):
    global _CTX
    _CTX = c


def current_site() -> str | None:
    c = _CTX
    if c is None or not c.kernel_files:
        return None
    f = sys._getframe(1)
    while f is not None:
        fn = f.f_code.co_filename
        if fn in c.kernel_files:
            return f"{os.path.basename(fn)}:{f.f_lineno}"
        f = f.f_back
    return None


# --------------------------------------------------------------------------
# Storage and views
# --------------------------------------------------------------------------


class Storage:
    _serial = 0

    def __init__(self, shape, dtype, buffer, name, values=None):
        Storage._serial += 1
        self.sid = Storage._serial
        self.shape = tuple(int(s) for s in shape)
        self.dtype = dtype
        self.buffer = buffer
        self.site = current_site()
        self.name = name or (f"tile@{self.site}" if self.site else f"t{self.sid}")
        size = int(np.prod(self.shape, dtype=np.int64))
        c = ctx()
        self.vals = values if values is not None else c.domain.empty(size, self.name)
        self.written = np.zeros(size, dtype=bool)
        self.mm = np.zeros(size, dtype=bool)
        # Loop signature of the last write of every element (dependency tracking).
        self.wsig = np.full(size, None, dtype=object) if c.track_deps else None
        c.storages.append(self)

    @property
    def size(self):
        return self.vals.shape[0]


def _norm_index(i, n, what):
    if not isinstance(i, (int, np.integer)):
        raise Unsupported(f"dynamic (tensor-valued) index {type(i).__name__} in {what}")
    i = int(i)
    if i < 0:
        i += n
    if not 0 <= i < n:
        raise KernelError(f"index {i} out of bounds for dimension of size {n} in {what}")
    return i


def _norm_slice(s: slice, n, what):
    step = 1 if s.step is None else int(s.step)
    if step <= 0:
        raise KernelError(f"slice step must be positive in {what}")
    start = 0 if s.start is None else int(s.start)
    stop = n if s.stop is None else int(s.stop)
    if start < 0:
        start += n
    if stop < 0:
        stop += n
    if not (0 <= start <= n and 0 <= stop <= n):
        raise KernelError(
            f"slice {start}:{stop} out of bounds for dimension of size {n} in {what}"
        )
    if stop <= start:
        raise KernelError(f"empty slice {start}:{stop} in {what}")
    return slice(start, stop, step)


class DynSlice:
    def __init__(self, start, size):
        self.start = start
        self.size = size


class NkiTensor:
    """A view: an integer array of flat storage indices with the view's shape."""

    def __init__(self, storage: Storage, idx: np.ndarray):
        self._storage = storage
        self._idx = idx

    # -- metadata ----------------------------------------------------------
    @property
    def shape(self):
        return tuple(int(s) for s in self._idx.shape)

    @property
    def ndim(self):
        return self._idx.ndim

    @property
    def size(self):
        return int(self._idx.size)

    @property
    def dtype(self):
        return self._storage.dtype

    @property
    def buffer(self):
        return self._storage.buffer

    @property
    def name(self):
        return self._storage.name

    @property
    def offset(self):
        return int(self._idx.reshape(-1)[0])

    @property
    def strides(self):
        out = []
        for d in range(self.ndim):
            if self._idx.shape[d] > 1:
                a = np.take(self._idx, 0, axis=d).reshape(-1)[0]
                b = np.take(self._idx, 1, axis=d).reshape(-1)[0]
                out.append(int(b - a))
            else:
                out.append(0)
        return tuple(out)

    def get_pattern(self):
        return [[s, n] for s, n in zip(self.strides, self.shape)]

    def is_contiguous(self):
        flat = self._idx
        if self.on_chip:
            flat = flat.reshape(flat.shape[0], -1)
            d = np.diff(flat, axis=1)
            return bool(np.all(d == 1)) if d.size else True
        d = np.diff(flat.reshape(-1))
        return bool(np.all(d == 1)) if d.size else True

    def is_indirect(self):
        return False

    @property
    def on_chip(self):
        return self._storage.buffer in ON_CHIP

    def __repr__(self):
        return f"NkiTensor({self.name}, shape={self.shape}, dtype={self.dtype}, buffer={self.buffer})"

    def __len__(self):
        return self.shape[0]

    def __iter__(self):
        raise TypeError("NKI tensors are not iterable; index them explicitly")

    def __bool__(self):
        raise KernelError("NKI tensors have no truth value (data-dependent Python control flow)")

    # -- data access (used by the shim ops) --------------------------------
    def read(self, accumulate=False):
        c = ctx()
        if c.track_deps and c.loop_stack:
            _check_deps(self, c, "read", accumulate)
        vals = self._storage.vals[self._idx]
        if not isinstance(vals, np.ndarray):
            # A 0-d view (e.g. x[p, i]) of an object array indexes to the bare element (a Poly),
            # not a 0-d array; callers rely on .size and .reshape.
            holder = np.empty((), dtype=self._storage.vals.dtype)
            holder[()] = vals
            vals = holder
        return vals

    def write(self, values, accumulate=False):
        c = ctx()
        values = c.domain.store_cast(values, self.dtype)
        if hasattr(values, "shape") and values.shape != self._idx.shape:
            values = np.broadcast_to(values, self._idx.shape)
        if isinstance(values, np.ndarray) and values.ndim == 0:
            # A 0-d view (e.g. out[c, h, w]): assigning a 0-d object array into one cell would
            # store the array itself, not its element.
            values = values[()]
        if c.track_deps and c.loop_stack:
            _check_deps(self, c, "write", accumulate)
        self._storage.vals[self._idx] = values
        self._storage.written[self._idx] = True
        if self._storage.wsig is not None:
            holder = np.empty((), dtype=object)
            holder[()] = c.signature()
            self._storage.wsig[self._idx] = holder

    # -- views -------------------------------------------------------------
    def _view(self, idx):
        return NkiTensor(self._storage, idx)

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        nd = self.ndim
        if any(k is Ellipsis for k in key):
            pos = next(i for i, k in enumerate(key) if k is Ellipsis)
            fill = nd - (len(key) - 1)
            key = key[:pos] + (slice(None),) * fill + key[pos + 1 :]
        if len(key) > nd:
            raise KernelError(f"too many indices ({len(key)}) for tensor of rank {nd}")
        key = key + (slice(None),) * (nd - len(key))
        norm = []
        for d, k in enumerate(key):
            n = self._idx.shape[d]
            what = f"{self.name}{list(self.shape)}"
            if isinstance(k, DynSlice):
                k = slice(k.start, k.start + k.size) if isinstance(k.start, (int, np.integer)) else k
                if isinstance(k, DynSlice):
                    raise Unsupported("nl.ds with a dynamic start")
            if isinstance(k, slice):
                norm.append(_norm_slice(k, n, what))
            elif isinstance(k, NkiTensor):
                raise Unsupported("tensor-valued (indirect) indexing")
            else:
                i = _norm_index(k, n, what)
                if d == 0 and self.on_chip:
                    norm.append(slice(i, i + 1))
                else:
                    norm.append(i)
        return self._view(self._idx[tuple(norm)])

    def __setitem__(self, key, value):
        dst = self[key]
        _assign(dst, value)

    def slice(self, dim, start, end, step=1):
        key = [slice(None)] * self.ndim
        key[dim] = slice(start, end, step)
        return self[tuple(key)]

    def select(self, dim, index):
        if isinstance(index, NkiTensor):
            raise Unsupported("dynamic select with a tensor index")
        key = [slice(None)] * self.ndim
        n = self.shape[dim]
        key[dim] = _norm_index(index, n, f"{self.name}.select")
        return self._view(self._idx[tuple(key)])

    def vector_select(self, dim, vector_offset):
        raise Unsupported("vector_select (indirect addressing)")

    def indirect(self, index, num_elem=None):
        raise Unsupported("tensor indirection (.indirect)")

    def permute(self, dims):
        dims = tuple(dims)
        if sorted(dims) != list(range(self.ndim)):
            raise KernelError(f"permute dims {dims} is not a permutation")
        if self.on_chip and dims[0] != 0:
            raise KernelError("permute cannot move the partition dimension of an on-chip tensor")
        return self._view(np.transpose(self._idx, dims))

    def broadcast(self, dim, size):
        if self.shape[dim] != 1:
            raise KernelError(f"broadcast: dimension {dim} has size {self.shape[dim]}, not 1")
        if self.on_chip and dim == 0:
            raise KernelError("cannot broadcast the partition dimension of an on-chip tensor")
        shape = list(self.shape)
        shape[dim] = size
        return self._view(np.broadcast_to(self._idx, tuple(shape)))

    def expand_dim(self, dim):
        if self.on_chip and dim == 0:
            raise KernelError("cannot insert a dimension before the partition dimension")
        return self._view(np.expand_dims(self._idx, dim))

    def squeeze_dim(self, dim):
        if self.shape[dim] != 1:
            raise KernelError(f"squeeze_dim: dimension {dim} has size {self.shape[dim]}")
        if self.on_chip and dim == 0:
            raise KernelError("cannot squeeze the partition dimension of an on-chip tensor")
        return self._view(np.squeeze(self._idx, dim))

    def reshape_dim(self, dim, shape):
        shape = list(shape)
        n = self.shape[dim]
        if -1 in shape:
            known = int(np.prod([s for s in shape if s != -1]))
            shape[shape.index(-1)] = n // known
        if int(np.prod(shape)) != n:
            raise KernelError(f"reshape_dim: {shape} does not match size {n}")
        new = self.shape[:dim] + tuple(shape) + self.shape[dim + 1 :]
        return self._view(self._idx.reshape(new))

    def flatten_dims(self, start_dim, end_dim):
        if self.on_chip and start_dim == 0:
            raise KernelError("flatten_dims cannot merge the partition dimension")
        new = (
            self.shape[:start_dim]
            + (int(np.prod(self.shape[start_dim : end_dim + 1])),)
            + self.shape[end_dim + 1 :]
        )
        return self._view(self._idx.reshape(new))

    def reshape(self, shape):
        shape = tuple(int(s) for s in (shape if isinstance(shape, (tuple, list)) else (shape,)))
        if -1 in shape:
            known = int(np.prod([s for s in shape if s != -1]))
            shape = tuple(self.size // known if s == -1 else s for s in shape)
        if int(np.prod(shape)) != self.size:
            raise KernelError(f"cannot reshape {self.shape} to {shape}")
        if self.on_chip and shape[0] != self.shape[0]:
            raise KernelError(
                f"reshape {self.shape} -> {shape} changes the partition dimension of an on-chip tensor"
            )
        return self._view(self._idx.reshape(shape))

    def rearrange(self, src_pattern, dst_pattern, fixed_sizes=None):
        sizes = dict(fixed_sizes or {})
        names = []
        split_shape = []
        for d, item in enumerate(src_pattern):
            group = item if isinstance(item, tuple) else (item,)
            n = self.shape[d]
            unknown = [g for g in group if g not in sizes]
            known = int(np.prod([sizes[g] for g in group if g in sizes])) if group else 1
            if len(unknown) > 1:
                raise KernelError(f"rearrange: cannot infer sizes of {unknown}")
            if unknown:
                sizes[unknown[0]] = n // known
            for g in group:
                names.append(g)
                split_shape.append(sizes[g])
        idx = self._idx.reshape(split_shape)
        order = []
        out_shape = []
        for item in dst_pattern:
            group = item if isinstance(item, tuple) else (item,)
            for g in group:
                order.append(names.index(g))
            out_shape.append(int(np.prod([sizes[g] for g in group])))
        return self._view(np.transpose(idx, order).reshape(out_shape))

    def ap(self, pattern, offset=None, scalar_offset=None, vector_offset=None, indirect_dim=0, dtype=None):
        if scalar_offset is not None or vector_offset is not None:
            raise Unsupported("indirect access patterns (scalar_offset/vector_offset)")
        if dtype is not None and dtype != self.dtype:
            raise Unsupported("dtype reinterpretation in .ap()")
        base = self.offset + (int(offset) if offset is not None else 0)
        strides = [int(s) for s, _ in pattern]
        counts = [int(n) for _, n in pattern]
        idx = np.full(counts, base, dtype=np.int64)
        for d, (s, n) in enumerate(zip(strides, counts)):
            shape = [1] * len(counts)
            shape[d] = n
            idx = idx + (np.arange(n, dtype=np.int64) * s).reshape(shape)
        if idx.size and (idx.min() < 0 or idx.max() >= self._storage.size):
            raise KernelError(f"access pattern {pattern} (offset {base}) is out of bounds of {self.name}")
        return self._view(idx)

    def view(self, dtype):
        if dtype == self.dtype:
            return self
        raise Unsupported("bit-level dtype reinterpretation (.view / reinterpret_cast)")

    reinterpret_cast = view

    # Python operators on tiles are not part of NKI 0.6 kernels.
    def _no_op(self, *a):
        raise Unsupported("Python arithmetic operators on NKI tensors; use nl.* or nisa.* ops")

    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = _no_op
    __truediv__ = __rtruediv__ = __neg__ = __pow__ = _no_op


def _check_deps(t: NkiTensor, c: TraceContext, kind: str, accumulate: bool):
    """Flag accesses that depend on another iteration of an enclosing affine_range."""
    st = t._storage
    if st.wsig is None:
        return
    cur = dict(f[:2] for f in c.loop_stack)
    sel = st.wsig[t._idx].ravel()
    mm = st.mm[t._idx].ravel() if accumulate else None
    seen = set()
    for i, sig in enumerate(sel):
        if sig is None or sig in seen:
            continue
        seen.add(sig)
        for inst, it in sig:
            if inst in cur and cur[inst] != it:
                if accumulate and mm[i]:
                    break  # PSUM matmul accumulation across iterations is allowed
                loop = c.loop_site(inst)
                if kind == "read":
                    msg = (
                        f"loop-carried dependency in affine_range loop at {loop}: tensor "
                        f"'{st.name}' is written in iteration {it} and read in iteration {cur[inst]}; "
                        "results depend on iteration order (use nl.sequential_range, or "
                        "allocate the tile inside the loop)"
                    )
                    site = current_site()
                    entry = msg + (f" (read at {site})" if site else "")
                    if entry not in c.dep_errors:
                        c.dep_errors.append(entry)
                else:
                    c.warn(
                        f"tile '{st.name}' is overwritten in several iterations of the affine_range "
                        f"loop at {loop}; iterations may run in parallel (allocate it inside the loop)"
                    )
                break


def _assign(dst: NkiTensor, value):
    """t[...] = value: copy (with cast) from a tensor or fill with a scalar."""
    c = ctx()
    if isinstance(value, NkiTensor):
        vals = value.read()
        if vals.size != dst.size:
            raise KernelError(f"assignment of {value.shape} into {dst.shape}")
        dst.write(vals.reshape(dst.shape))
    else:
        dst.write(c.domain.full(dst.shape, value))


def new_tensor(shape, dtype, buffer="sbuf", name="", values=None) -> NkiTensor:
    if isinstance(shape, (int, np.integer)):
        shape = (int(shape),)
    shape = tuple(int(s) for s in shape)
    buffer = _buf(buffer)
    c = ctx()
    if dtype is None:
        raise KernelError("tensor allocation requires a dtype")
    if any(s <= 0 for s in shape):
        raise KernelError(f"invalid tensor shape {shape}")
    if buffer in ON_CHIP:
        if len(shape) < 1 or shape[0] > c.profile.pmax:
            raise KernelError(
                f"on-chip tensor {name or ''}{list(shape)}: partition dimension {shape[0]} exceeds "
                f"pmax={c.profile.pmax}"
            )
        free_bytes = int(np.prod(shape[1:], dtype=np.int64)) * _bytes(dtype)
        if buffer == "psum":
            limit = c.profile.psum_bank_fmax * 4 * c.profile.psum_num_banks
            if free_bytes > limit:
                raise KernelError(f"PSUM tensor {list(shape)} needs {free_bytes} B/partition > {limit}")
        elif free_bytes > c.profile.sbuf_fmax_bytes:
            raise KernelError(
                f"SBUF tensor {list(shape)} needs {free_bytes} B/partition > {c.profile.sbuf_fmax_bytes}"
            )
    if buffer == "shared_hbm" and values is None and c.grid != (1,):
        # SPMD programs share shared_hbm allocations by allocation order.
        i = c.shared_counter
        c.shared_counter += 1
        if c.program_id != tuple(0 for _ in c.grid):
            st = c.shared_allocs[i]
            if st.shape != shape or st.dtype != dtype:
                raise KernelError("SPMD programs allocate different shared_hbm tensors")
            return NkiTensor(st, np.arange(st.size).reshape(shape))
        st = Storage(shape, dtype, buffer, name, values)
        c.shared_allocs.append(st)
        return NkiTensor(st, np.arange(st.size).reshape(shape))
    st = Storage(shape, dtype, buffer, name, values)
    return NkiTensor(st, np.arange(st.size).reshape(shape))


def _bytes(dtype):
    from ..domains import dtype_bits

    return dtype_bits(dtype) // 8


def _buf(buffer) -> str:
    if isinstance(buffer, str):
        b = buffer
    else:
        b = getattr(buffer, "value", None) or getattr(buffer, "name", None) or str(buffer)
    if b == "hbm" or b in HBM_BUFFERS or b in ON_CHIP:
        return b
    raise KernelError(f"unknown buffer {buffer!r}")
