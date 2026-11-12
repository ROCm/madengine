#!/usr/bin/env python3
"""Catch the first NaN-producing op on every rank and dump replay capsules.

Single process:

    from op_nan_capture import OpNanCapture
    with OpNanCapture(out="/path/to/scratch/nan_capsule.pt"):
        main()

Distributed (one capsule per rank -- {rank} is filled in automatically; passing a
DIRECTORY instead writes rank<N>_<uuid>.pt into it, which is usually what you want):

    cap = OpNanCapture(out="/path/to/scratch/nan_capsule.rank{rank}.pt")
    with cap:
        for step in range(num_steps):
            train_step()
            cap.sync_stop()        # <-- once per step, on EVERY rank, see below

Then:  compare_nan_capsules.py  ->  replay_nan_op_dist.py

The mode watches every aten op, so the first non-finite output is often not the
one you are hunting.  Narrow it with the op filters -- each takes a comma-
separated list:

    OpNanCapture(out=..., include_only_op_classes="gemm")
    OpNanCapture(out=..., include_only_ops="aten.mm.default, aten.addmm.default")
    OpNanCapture(out=..., include_only_ops="aiter")        # a whole namespace
    OpNanCapture(out=..., exclude_ops="aten.div.Tensor")

--------------------------------------------------------------------------------
Why sync_stop() exists

Raising from inside __torch_dispatch__ is safe in one process and fatal in eight.
Only the rank that saw the NaN unwinds; the others walk into the next all-reduce
and RCCL blocks until the collective timeout -- by which point the job is killed
and you have one capsule instead of eight.  So in a distributed run the capture
never raises on its own: it writes the capsule, sets a flag, and returns.

sync_stop() is the coordinated exit.  Every rank calls it at the same point in
the loop, it all-reduces the flag, and every rank raises together (or none do).
It must be called OUTSIDE any other collective and on every rank unconditionally
-- guarding it with `if something:` reintroduces exactly the hang it prevents.

If you cannot place a per-step hook, leave raise_on_nan=False and let the run's
own NaN check end the job.  The capsules are already on disk by then.

--------------------------------------------------------------------------------
What the capsule holds, and why each part is load-bearing

  values   the exact input bits.  The NaN is data-dependent -- if random data of
           the same shape reproduced it, TunableOp's numerical check would have
           caught it already, so we cannot regenerate the inputs.  Recorded only
           with record_data=True (NAN_CAPTURE_DATA=1); the default writes shapes
           and per-tensor statistics only (capsule["data"] == "shapes"; integer
           tensors such as cu_seqlens and indices keep their values), which
           replay --synthesize turns into look-alike operands -- enough for a
           fault that does not depend on the values, nothing more.

  strides  hipBLASLt is column major, PyTorch is row major.  A row-major (M, K)
           tensor IS a column-major (K, M) array with lda=K, so transA/transB and
           lda/ldb are decided purely by the operands' strides -- see the mapping
           table in replay_hipblaslt_log.py.  A .clone() or .contiguous() here
           would flip transA and hand you a different kernel.

  offset   pointer alignment feeds hipBLASLt's vector-width choice.  The caching
           allocator hands out 512B-aligned blocks, so we keep the original
           storage_offset modulo 512B and re-create it at replay time.

  sha      a digest per operand, so compare_nan_capsules.py can tell which inputs
           agreed across ranks (replicated weights should; activations should not)
           without loading gigabytes.

  identity op + call index + stack digest + geometry.  Two ranks hit "the same
           op" when these match; their DATA will differ, because each rank sees
           a different microbatch.  Comparing data is how you tell a replicated
           weight from an activation, not how you tell whether it is the same op.

  dist     rank / world_size / backend, so the replay can be launched at the same
           width as the original run.

We do NOT snapshot inputs before the call: aten mm/addmm/bmm/baddbmm/_scaled_mm are
all out-of-place with const operands, so args are still pristine after the call.
That keeps the non-failing path free of copies -- the only cost is the isfinite sync.

--------------------------------------------------------------------------------
Mutating ops (AITER and friends)

That last paragraph is a statement about aten's matmul family, not a law.  Fused
libraries -- AITER's `torch.ops.aiter.*` above all -- are full of out-param
kernels: they write the result into a caller-allocated buffer and return None, or
return that same buffer.  Such an op breaks the assumption twice.  Its result is
not in the return value, so scanning the return value alone misses the NaN; and
its operands are no longer pristine after the call, so packing them then yields a
capsule that replays something other than what failed.

So for any op whose schema says it writes to an argument (`Tensor(a!)`), and only
for such an op, the capture:

  * copies the written arguments BEFORE the call, keeping size/stride/alignment
    (a plain .clone() would flip transA -- see the strides note above), and packs
    those copies instead of the post-call operands;
  * scans the written arguments as outputs alongside the return value;
  * keeps written arguments out of the `inputs_finite=False` verdict, because
    before the call they hold allocator garbage that is often NaN already -- and
    reports inputs_finite=None rather than guessing from operands the kernel has
    already overwritten, or from operands it may never have read.

That last point is the subtle one.  A `Tensor(a!)` annotation says an argument is
WRITTEN; it does not say whether it is also READ, and the two cases are
indistinguishable in the schema.  An in-place `self` is read; a positional
out-param -- `dq`/`dk`/`dv` in flash-attn's backward, and most fused-library
kernels -- is not.  Only the classic kwarg-only `out=` is self-evident.  So a
dirty written operand yields `inputs_finite=None`, never `False`: scoring it
`False` would claim "the NaN arrived on the inputs" for every out-param op whose
buffers came from `empty_like`, and send you hunting a producer that does not
exist.  `nonfinite_operands` records which operands were dirty and which
population they belong to, so the ambiguity is visible instead of guessed at.

The copy costs a device-to-device clone per call of such an op, which is why it
is worth aiming the filter (include_only_ops="aiter") rather than watching
everything.  Set snapshot_mutated=False to trade capsule fidelity for speed; the
capsule then records args_are_pre_call=False so the replay knows not to trust it.

--------------------------------------------------------------------------------
Settle mode (settle=True; capture_from_env's default)

Not every NaN is a bug.  torch's own sinc_backward computes (pi x cos pi x -
sin pi x) / (pi x^2), which is 0/0 wherever x == 0, and masks it with a where()
three ops later; a model that calls torch.sinc makes the capture fire on every
step at an op that did nothing wrong.  Excluding aten.div and aten.mul to get
past it blinds the capture to every division and product in the model.

So in settle mode the first NaN an op creates becomes a PENDING candidate -- its
capsule is built at once, because only then do the operands exist -- and is
written only if the NaN outlives its step:

  persistent write   an op writes a non-finite value into a tensor that existed
                     before the step began (parameters, optimizer state, buffers)
  persistent read    an op reads a non-finite tensor that existed before the step
  escape             a non-finite tensor is handed to a collective or read back
                     to the host (.item(), .cpu(), print)

None of those by the end of the step after the candidate's (settle_steps=1) and
the NaN was masked: it is logged as MASKED, the site is remembered, and the run
goes on.  "Existed before the step" is judged by storage: every storage a non-view
op allocated during the step is "born" this step, and a view of a parameter
shares the parameter's storage, so it does not count as new.

Each candidate also gets a nan_origin, from re-running a pointwise op in fp64 on
the same operands: `math` (fp64 is non-finite there too: 0/0, sqrt(-1), log(0)
-- every correct kernel agrees), `overflow` (fp64 finite but beyond the dtype's
range), `unexplained` (fp64 finite and in range: a NaN no correct kernel makes),
`propagated` (an input was already non-finite) or `no_reference` (not pointwise,
seeded, or the re-run failed).  A masked `math` site is only counted after its
first settle (re-verified as `math` each time it fires); a masked site of any
other origin is written to <capsule>.masked-<n> -- no `.pt` suffix, so nothing
treats it as a capsule -- because "masked" is not "benign": a model's own
nan_to_num masks a real kernel NaN just as well.

What it cannot do: tell WHICH pending candidate a surviving NaN came from.  When
several are pending, the earliest non-`math` one is written; the capsule's
`settle` record names the op where survival was seen, so a misattribution is
visible.  And ops outside the dispatcher (compiled regions) are invisible to the
born-this-step bookkeeping, so their outputs look persistent -- which errs
towards writing a capsule, i.e. towards the behaviour without settle mode.

A caveat the filter cannot fix: only ops that go through the dispatcher are
visible here.  AITER entry points that are plain pybind calls into a compiled
extension, and anything inside a torch.compile region or a replayed CUDA graph,
never reach __torch_dispatch__ at all.  Check what is reachable with
`[f"aiter.{n}" for n in dir(torch.ops.aiter) if not n.startswith("_")]`.
"""

import atexit
import multiprocessing
import multiprocessing.util
import collections
import contextlib
import hashlib
import math
import os
import pickle
import re
import sys
import traceback

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves, tree_map

try:
    from torch.utils._python_dispatch import _disable_current_modes
except ImportError:                    # older torch: settle-time work may be seen by the mode
    _disable_current_modes = contextlib.nullcontext

try:
    # Identity-keyed and weak: the caching allocator reuses an address the instant
    # a tensor is freed, so a {data_ptr(): producer} map would attribute a fresh
    # tensor to whatever used to live at that address.  Keying on the storage
    # OBJECT cannot make that mistake.
    from torch.utils.weak import WeakIdKeyDictionary
except ImportError:                    # pre-2.1 torch: chain capture refuses to arm
    WeakIdKeyDictionary = None

_TENSOR_TAG = "__op_nan_tensor__"
_CHAIN_REF = "__op_nan_ref__"       # {_CHAIN_REF: key} stands in for a chain tensor
# Capsules written before the tool was renamed from gemm_nan_capture carry these
# keys; upgrade_capsule() maps them onto the current ones at load.
_LEGACY_KEYS = {"__gemm_nan_tensor__": _TENSOR_TAG, "__gemm_nan_ref__": _CHAIN_REF,
                "gemm_call_index": "op_call_index"}
_ALIGN_BYTES = 512          # torch caching allocator block alignment


# --------------------------------------------------------------------------- #
# op filtering
#
# The mode sees EVERY aten op, not just GEMMs, so the first non-finite output it
# trips over may belong to an op you do not care about (an aten.div that divides
# by a zero norm, a softmax that saw a -inf mask).  The four filter options let
# you aim the capture: include_only_* restricts the population, exclude_* carves
# holes in it.
#
# Ops are named the way str(func) prints them -- "aten.addmm.default".  Matching
# walks three rungs, any of which a filter entry may name:
#
#     aten.addmm.default   the full overload
#     aten.addmm           the overload packet -- every overload of that op
#     aten                 the namespace -- every op registered by that library
#
# The namespace rung is what makes a fused library tractable: "aiter" selects
# every op AITER registered without naming them one by one, and survives AITER
# registering more of them later.  (An "aiter" op CLASS would not: classes are
# static sets, and AITER registers lazily as its submodules import, so a set
# snapshotted at construction time would miss whatever had not been imported.)
# --------------------------------------------------------------------------- #

# The matmul family.  Deliberately matmul-only: mv/addmv/dot route through BLAS
# too, but they are not GEMMs and hipBLASLt does not log them as such.
_GEMM_OPS = frozenset([
    "aten.mm",
    "aten.addmm",
    "aten._addmm_activation",
    "aten.bmm",
    "aten.baddbmm",
    "aten.addbmm",
    "aten.matmul",
    "aten.linear",
    "aten._scaled_mm",
    "aten._int_mm",
    "aten._grouped_mm",
])

_OP_CLASSES = {"gemm": _GEMM_OPS}


def _parse_name_list(value, what):
    """'a, b , c' -> frozenset({'a','b','c'}).  None/'' -> None (= unset)."""
    if value is None:
        return None
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
    else:
        raise TypeError("%s must be a comma-separated string or a sequence, got %r"
                        % (what, type(value).__name__))
    names = frozenset(s.strip() for s in items if str(s).strip())
    return names or None


def _expand_op_classes(value, what):
    """Resolve a list of op-class names to the set of op names they cover."""
    classes = _parse_name_list(value, what)
    if not classes:
        return None
    unknown = sorted(c for c in classes if c not in _OP_CLASSES)
    if unknown:
        raise ValueError("%s: unknown op class(es) %s; known classes are %s"
                         % (what, unknown, sorted(_OP_CLASSES)))
    out = set()
    for c in classes:
        out |= set(_OP_CLASSES[c])
    return frozenset(out)


_NAMES_CACHE = {}


def _op_names(func):
    """('aten.addmm.default', 'aten.addmm', 'aten') -- overload, packet, namespace.
    Memoised per OpOverload: str(func) is not free and this runs on every op."""
    n = _NAMES_CACHE.get(func)
    if n is None:
        full = str(func)
        parts = full.split(".")
        packet = ".".join(parts[:2]) if len(parts) > 2 else full
        n = _NAMES_CACHE[func] = (full, packet, parts[0])
    return n


# Collectives are never checked or chain-tracked.  A NaN coming out of an
# all_reduce was produced on SOME rank -- maybe another one -- so "inputs finite
# here" would wrongly read as "this op created it"; and their ProcessGroup
# arguments cannot be torch.save()d, so recording them breaks every later capsule
# whose chain passes through one.  The rank that created the NaN captures it.
_COLLECTIVE_NS = frozenset(("c10d", "_c10d_functional", "c10d_functional",
                            "_c10d_functional_autograd"))
_collective_cache = {}


def _is_collective(func):
    c = _collective_cache.get(func)
    if c is None:
        c = _collective_cache[func] = _op_names(func)[2] in _COLLECTIVE_NS
    return c


def _matches(names, wanted):
    """True if any of (overload, packet, namespace) is in the filter set."""
    return any(n in wanted for n in names)


# --------------------------------------------------------------------------- #
# cross-rank causality
#
# Collectives inside one process group run in the same order on every member rank,
# so (group, n-th collective of that group) names the same event on every rank.
# The capture counts them, remembers which storages each collective wrote, and when
# a NaN is first READ from such a storage records that it ARRIVED through that
# collective rather than being made on this rank.  At sync_stop() the ranks pool
# these records into <capsule dir>/manifest.json: the origin rank is one whose NaN
# did not arrive through an earlier collective.  Nothing is read from a collective's
# output when it is issued -- an async collective has not landed yet.
# --------------------------------------------------------------------------- #

try:
    from torch.distributed.distributed_c10d import ProcessGroup as _ProcessGroup
except Exception:                       # noqa: BLE001 - no distributed build
    _ProcessGroup = None


def _group_ranks(pg):
    try:
        import torch.distributed as dist
        return list(dist.get_process_group_ranks(pg))
    except Exception:                                   # noqa: BLE001
        return None


# Pure data movement: every output element is a copy of an input element (no
# arithmetic, no dtype narrowing), so these ops cannot CREATE a NaN.  Matched on the
# overload packet, or the whole `fsdp` namespace (FSDP2's copy-in/copy-out ops).
_MOVE_OPS = frozenset(("aten.copy_", "aten.clone", "aten.cat", "aten.stack",
                       "aten.split_with_sizes_copy", "aten.slice_copy", "aten.narrow_copy",
                       "aten.index_select", "aten.gather", "aten.index", "aten.unbind_copy",
                       "aten.as_strided_copy", "aten.permute_copy", "aten.t_copy"))
_MOVE_NS = frozenset(("fsdp",))


def _is_data_movement(func):
    names = _op_names(func)
    return names[1] in _MOVE_OPS or names[2] in _MOVE_NS


def _cpu_copy(obj):
    """Tensors -> detached CPU copies; lists/tuples/dicts recursed; ScriptObjects (process
    groups, reduce ops, Work) -> their repr; everything else as is.  For R3 records."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_cpu_copy(x) for x in obj) if not hasattr(obj, "_fields") \
            else [_cpu_copy(x) for x in obj]
    if isinstance(obj, dict):
        return dict((k, _cpu_copy(v)) for k, v in obj.items())
    if isinstance(obj, torch.ScriptObject):
        return "<%s>" % type(obj).__name__
    return obj


def _parse_record_collectives(spec):
    """'group:seq[:step],...' -> [(group, seq, step or None)]."""
    out = []
    for item in (spec or "").split(","):
        parts = item.strip().split(":")
        if len(parts) >= 2 and parts[0]:
            out.append((parts[0], int(parts[1]), int(parts[2]) if len(parts) > 2 and parts[2] else None))
    return out


def _collective_group(func, args, kwargs):
    """(group name, process group or None) for a collective call; (None, None) for
    one without a group argument (wait_tensor)."""
    for a in list(args) + list(kwargs.values()):
        if isinstance(a, str) and _op_names(func)[2] != "c10d":
            try:
                from torch.distributed.distributed_c10d import _resolve_process_group
                return a, _resolve_process_group(a)
            except Exception:                           # noqa: BLE001
                return a, None
        if _ProcessGroup is not None and isinstance(a, torch.ScriptObject):
            try:
                pg = _ProcessGroup.unbox(a)
            except Exception:                           # noqa: BLE001 - a ReduceOp, ...
                continue
            return str(pg.group_name), pg
    return None, None


# --------------------------------------------------------------------------- #
# tensor subclasses (DTensor, FSDP2 parameters, Float8 tensors, TwoTensor, ...)
#
# A dispatch mode runs BEFORE a subclass's own __torch_dispatch__, so it is handed
# the wrapper -- a DTensor whose data is not addressable as one tensor (`.tolist()`
# on its max-|value| raised and killed ranks 0-1 of the 2026-11-09 torchtitan
# FSDP+TP+EP capture) -- and when the mode then calls func, the ops the subclass runs
# on its local tensors happen with the mode popped, so the capture never sees the
# real arithmetic.  Returning NotImplemented makes the dispatcher call the subclass
# with the mode still active: every local op reaches the capture as a plain Tensor
# (verified on DTensor and TwoTensor).  "Subclass" means a class with its OWN
# __torch_dispatch__: nn.Parameter inherits Tensor's and is checked as usual.
#
# Fake and functional tensors are the exception: they carry no data, and DTensor's
# sharding propagation runs fake ops under the mode -- deferring those fails with
# "all __torch_dispatch__ handlers returned NotImplemented".  They pass through.
# --------------------------------------------------------------------------- #

# The allocator family: their outputs hold whatever bytes the caching allocator had,
# often NaN bit patterns.  Never checked (capture_from_env excludes them) and, in
# settle mode, tracked as "raw" storage until something writes all of it -- see
# _note_raw and the `uninitialized` origin.
_ALLOC_OPS = frozenset(("aten.empty.memory_format", "aten.empty_strided.default",
                        "aten.empty_like.default", "aten.new_empty.default",
                        "aten.new_empty_strided.default", "aten.empty_permuted.default"))


def _covers_storage(t):
    """True if `t` spans its whole storage, so a write into it leaves nothing raw."""
    try:
        return t.numel() * t.element_size() >= t.untyped_storage().nbytes()
    except Exception:                                   # noqa: BLE001
        return False


_PLAIN_DISPATCH = torch.Tensor.__torch_dispatch__
try:
    from torch._ops import HigherOrderOperator as _HOP
except ImportError:                     # pragma: no cover - very old torch
    class _HOP(object):
        pass
try:
    from torch._subclasses.fake_tensor import FakeTensor as _FakeTensor
except ImportError:                     # pragma: no cover - very old torch
    _FakeTensor = None
try:
    from torch._subclasses.functional_tensor import FunctionalTensor as _FunctionalTensor
except ImportError:                     # pragma: no cover
    _FunctionalTensor = None
_NO_DATA_TYPES = tuple(t for t in (_FakeTensor, _FunctionalTensor) if t is not None)
try:
    _FAKE_MODE_KEY = torch._C._TorchDispatchModeKey.FAKE
except AttributeError:                  # pragma: no cover
    _FAKE_MODE_KEY = None
_subclass_kind_cache = {}


def _subclass_kind(types):
    """'nodata' (fake/functional: pass through), 'subclass' (defer to it), or None."""
    kind = None
    for t in types:
        k = _subclass_kind_cache.get(t)
        if k is None:
            if t is torch.Tensor or getattr(t, "__torch_dispatch__", _PLAIN_DISPATCH) is _PLAIN_DISPATCH:
                k = ""
            elif _NO_DATA_TYPES and issubclass(t, _NO_DATA_TYPES):
                k = "nodata"
            else:
                k = "subclass"
            _subclass_kind_cache[t] = k
        if k == "nodata":
            return k
        if k:
            kind = k
    return kind


def _nodata(a):
    return (isinstance(a, _NO_DATA_TYPES) if _NO_DATA_TYPES else False) or \
        (isinstance(a, _Tensor) and a.is_meta)


def _has_nodata_arg(args, kwargs):
    """True if a top-level argument (or the first element of a top-level list) is a
    fake, functional or meta tensor -- no data to check.  `types` alone does not show
    them: the dispatcher hands a mode an empty `types` for most calls, fake ones
    included.  Meta: DTensor traces decompositions on meta tensors to find a sharding
    strategy (DecompShardingStrategy); checking those failed the trace and DTensor
    reported "Sharding propagation should have produced at least Replicate()"."""
    if _FAKE_MODE_KEY is not None and torch._C._get_dispatch_mode(_FAKE_MODE_KEY) is not None:
        return True             # a FakeTensorMode is active below us: every op is fake
    for a in args:
        if _nodata(a):
            return True
        if isinstance(a, (list, tuple)) and a and _nodata(a[0]):
            return True
    for a in kwargs.values():
        if _nodata(a):
            return True
    return False


# --------------------------------------------------------------------------- #
# which arguments does this op overwrite?
#
# Read straight off the schema's alias annotations: `Tensor(a!) out` is a written
# argument, `Tensor self` is not.  That is the authoritative answer -- no name
# heuristics, no per-library table, so it covers aiter, vllm, fbgemm and anything
# else registered through torch.library on the same footing as aten.
#
# Called on every considered op, so the answer is memoised per OpOverload.
# --------------------------------------------------------------------------- #

_NO_WRITES = {"any": False, "idx": frozenset(), "names": frozenset(),
              "outbuf_idx": frozenset(), "outbuf_names": frozenset()}

_WRITES_CACHE = {}


def _compute_op_writes(func):
    schema = getattr(func, "_schema", None)
    if schema is None:                 # not an OpOverload (shouldn't happen) -- assume pure
        return _NO_WRITES
    idx, names, oidx, onames = set(), set(), set(), set()
    for i, a in enumerate(schema.arguments):
        ai = getattr(a, "alias_info", None)
        if ai is None or not ai.is_write:
            continue
        # Record both: a schema position may be passed positionally or by name.
        idx.add(i)
        names.add(a.name)
        if a.kwarg_only:
            # The classic `out=` buffer: written, and NOT read as an input.  An
            # in-place op's positional `self` is written too but IS an input, so
            # it deliberately does not land here.
            oidx.add(i)
            onames.add(a.name)
    if not idx:
        return _NO_WRITES
    return {"any": True, "idx": frozenset(idx), "names": frozenset(names),
            "outbuf_idx": frozenset(oidx), "outbuf_names": frozenset(onames)}


def op_writes(func):
    """{'any', 'idx', 'names', 'outbuf_idx', 'outbuf_names'} for `func`."""
    w = _WRITES_CACHE.get(func)
    if w is None:
        w = _compute_op_writes(func)
        _WRITES_CACHE[func] = w
    return w


# --------------------------------------------------------------------------- #
# op introspection for the chain walk
#
# Same rule as op_writes above, and for the same reason: read it off the schema
# and the op's tags, never off a name list.  A hardcoded set of "view ops" or
# "random ops" is wrong the moment a model uses a library that registers its own,
# and this ladder is meant to run on whatever reaches the dispatcher.  Each
# answer is memoised per OpOverload because the chain tracker asks on every call.
# --------------------------------------------------------------------------- #

_VIEW_CACHE = {}
_TAGS_CACHE = {}


def _compute_is_view(func):
    """True if every return merely aliases an argument and writes nothing.

    aten.detach.default returns Tensor(a)  -> alias_info present, is_write False.
    aten.mm.default     returns Tensor     -> alias_info None.
    aten.add_.Tensor    returns Tensor(a!) -> aliases AND writes, so it is a
                                              mutation, not a view.

    A view computes nothing, so the walk traverses it without spending depth and
    without adding a node a replay would have to re-execute.  Without this,
    `chain_depth=2` on a model with two detaches between the op that matters and
    the op that failed would reach nothing at all.
    """
    schema = getattr(func, "_schema", None)
    if schema is None or not schema.returns:
        return False
    for r in schema.returns:
        ai = getattr(r, "alias_info", None)
        if ai is None or ai.is_write:
            return False
    return True


def is_view(func):
    v = _VIEW_CACHE.get(func)
    if v is None:
        v = _compute_is_view(func)
        _VIEW_CACHE[func] = v
    return v


def _tag_names(func):
    """{'core', 'pointwise', ...} -- the op's torch.Tag names, memoised."""
    n = _TAGS_CACHE.get(func)
    if n is None:
        n = frozenset(str(t).rsplit(".", 1)[-1] for t in getattr(func, "tags", ()))
        _TAGS_CACHE[func] = n
    return n


def needs_rng(func):
    """True if replaying `func` draws from the generator.

    torch tags these `nondeterministic_seeded`.  That is not only dropout:
    aten._scaled_dot_product_flash_attention carries it too, which is why a chain
    replay that did not restore the generator would be running different
    arithmetic than the run that failed -- silently, and only on the ops where it
    matters most.
    """
    return "nondeterministic_seeded" in _tag_names(func)


def breaks_chain(func):
    """Why the walk cannot pass through `func`, or None if it can.

    data_dependent_output covers .item()/nonzero: the value left the tensor world,
    so there is no edge left to follow.  A collective is reachable but not
    replayable standalone -- its result depends on another rank, which no capsule
    on this rank holds.
    """
    if "data_dependent_output" in _tag_names(func):
        return "data_dependent"
    if _op_names(func)[2] in ("c10d", "_c10d_functional", "c10d_functional"):
        return "collective"
    return None


# --------------------------------------------------------------------------- #
# finiteness, on dtypes torch.isfinite refuses
#
# torch.isfinite/isnan/isinf are not implemented for the float8 dtypes: calling
# one raises NotImplementedError.  Inside __torch_dispatch__ that is not a missing
# feature, it is a crash in the user's job -- the capture is watching, an op
# returns fp8, and the run dies in the instrumentation rather than at the bug.
#
# That is squarely on the path this filter is for.  AITER is a quantization
# library whose ops return fp8 routinely, and aten._scaled_mm -- already in the
# 'gemm' op class -- returns fp8 whenever out_dtype says so.
#
# So promote to float32 first for any dtype that needs it.  fp8 is a strict
# subset of fp32, so the promotion is exact and the verdict is the real one, not
# an approximation.  Which dtypes need it is probed once and cached, rather than
# hard-coded against a list of float8 types that grows with every torch release.
# --------------------------------------------------------------------------- #

_PROMOTE_CACHE = {}


def _needs_promote(dtype):
    p = _PROMOTE_CACHE.get(dtype)
    if p is None:
        try:
            torch.isfinite(torch.zeros(1, dtype=dtype))
            p = False
        except (NotImplementedError, RuntimeError):
            p = True
        _PROMOTE_CACHE[dtype] = p
    return p


def _scannable(t):
    """`t`, promoted to float32 if torch.isfinite cannot read its dtype."""
    return t.float() if _needs_promote(t.dtype) else t


# Large tensors are scanned in slices.  torch.isfinite(t).all() materialises a
# full-size bool mask first -- a quarter of t's bytes for fp32 -- and on a real
# model the biggest written operand is an optimizer buffer: fbgemm's rowwise-
# adagrad backward on the 1st NaN writes 52.45 GiB, whose mask is 13.11 GiB.
# At 2 trials per GPU that OOM'd the capture at step 2 (Lux smoke job 369405)
# with snapshot_mutated already off.  Slicing bounds the transient to one
# chunk's mask (plus its fp32 promotion for fp8) and keeps ONE device sync per
# tensor: per-chunk verdicts stay on the device and are reduced together.
_SCAN_CHUNK_ELEMS = int(os.environ.get("NAN_CAPTURE_SCAN_CHUNK_ELEMS", str(1 << 27)))


def _scan_chunks(t):
    """`t` as pieces of <= ~_SCAN_CHUNK_ELEMS elements, without copying it.

    Contiguous: flat slices of a view.  Otherwise: blocks of dim 0, which are
    views for any layout; a single dim-0 row larger than the budget is yielded
    whole rather than copied.
    """
    n = _SCAN_CHUNK_ELEMS
    if t.numel() <= n or t.dim() == 0:
        yield t
    elif t.is_contiguous():
        flat = t.view(-1)
        for i in range(0, flat.numel(), n):
            yield flat[i:i + n]
    else:
        rows = max(1, n // max(1, t.numel() // t.shape[0]))
        for block in t.split(rows, 0):
            yield block


def _finite_all(t):
    """torch.isfinite(t).all(), for every float dtype including fp8."""
    if t.numel() <= _SCAN_CHUNK_ELEMS:
        return bool(torch.isfinite(_scannable(t)).all())
    return bool(torch.stack([torch.isfinite(_scannable(c)).all()
                             for c in _scan_chunks(t)]).all())


# The per-op check, fused.  Under GPU co-tenancy every extra kernel waits behind
# the partner process's work (~1 ms each, sync or not), so the check's cost is
# set by how MANY kernels it launches.  isfinite(t).all() is two per tensor plus
# a host read per tensor; max|t| via vector_norm(t, inf) is one reduction per
# tensor, and an op's verdicts are read back together, once.  NaN propagates
# through the max and inf stays inf, so "max|t| is finite" == "t is finite".
# Measured cost of the old path: Lux profile job 369551, ~76% of capture time.
_NORM_OK = {}           # device type -> vector_norm(inf) propagates NaN here


def _norm_propagates_nan(device):
    ok = _NORM_OK.get(device.type)
    if ok is None:
        try:
            probe = torch.tensor([1.0, float("nan"), 2.0], device=device)
            ok = bool(torch.isnan(torch.linalg.vector_norm(probe, float("inf"))))
        except Exception:                                # noqa: BLE001
            ok = False
        _NORM_OK[device.type] = ok
    return ok


def _maxabs(t):
    """0-d max|t| on t's device (NaN propagates); sliced like _finite_all."""
    if t.numel() <= _SCAN_CHUNK_ELEMS:
        return torch.linalg.vector_norm(_scannable(t), float("inf"))
    return torch.stack([torch.linalg.vector_norm(_scannable(c), float("inf"))
                        for c in _scan_chunks(t)]).amax()


def _maxabs32(t):
    """_maxabs as a float32 0-d tensor, in ONE kernel where possible (the norm
    accumulates in fp32 itself; a separate .float() was a launch per check)."""
    if t.numel() > _SCAN_CHUNK_ELEMS or _needs_promote(t.dtype):
        return _maxabs(t).float()
    if t.dtype == torch.float32:
        return torch.linalg.vector_norm(t, float("inf"))
    return torch.linalg.vector_norm(t, float("inf"), dtype=torch.float32)


def _checkable(t):
    return (isinstance(t, torch.Tensor) and t.is_floating_point()
            and t.numel() > 0)


_Tensor = torch.Tensor


def _tensors(obj):
    """The tensors in `obj`, in tree_leaves order, for the hot path.

    An op's arguments and returns are tensors, scalars and None nested in
    tuples (incl. torch.return_types), lists and dicts; walking those directly
    is several times cheaper than torch.utils._pytree.tree_leaves, which was
    the largest single cost of the capture per op (it ran 5-7 times per op).
    Any other container goes through tree_leaves, so nothing is missed.
    """
    if isinstance(obj, _Tensor):
        return [obj]
    out = []
    stack = [obj]
    while stack:
        o = stack.pop()
        if isinstance(o, _Tensor):
            out.append(o)
        elif isinstance(o, (tuple, list)):
            stack.extend(reversed(o))
        elif isinstance(o, dict):
            stack.extend(reversed(list(o.values())))
        elif o is None or isinstance(o, (int, float, bool, str, torch.dtype,
                                         torch.device, torch.layout,
                                         torch.memory_format)):
            continue
        else:
            out.extend(t for t in tree_leaves(o) if isinstance(t, _Tensor))
    return out


def _first_bad(cands):
    """First (tensor, where) in `cands` holding a NaN/inf, or (None, None).

    One max-abs reduction per tensor and ONE device->host read for the lot.
    Falls back to the per-tensor isfinite path where the norm does not
    propagate NaN (probed once per device type).
    """
    if not cands:
        return None, None
    if not _norm_propagates_nan(cands[0][0].device):
        for t, where in cands:
            if not _finite_all(t):
                return t, where
        return None, None
    norms = [_maxabs(t).float().reshape(1) for t, _ in cands]
    vals = (norms[0] if len(norms) == 1 else torch.cat(norms)).tolist()
    for (t, where), v in zip(cands, vals):
        if not math.isfinite(v):
            return t, where
    return None, None


def _count_where(pred, t):
    """int(pred(t).sum()), sliced like _finite_all; pred is torch.isnan/isinf."""
    if t.numel() <= _SCAN_CHUNK_ELEMS:
        return int(pred(_scannable(t)).sum())
    return int(torch.stack([pred(_scannable(c)).sum()
                            for c in _scan_chunks(t)]).sum())


def _nonfinite(t):
    """True for a float tensor with at least one NaN/inf.  False for anything else."""
    return bool(isinstance(t, torch.Tensor) and t.is_floating_point()
                and t.numel() > 0 and not _finite_all(t))


def _nonfinite_report(items):
    """[{operand, shape, dtype, nan, inf}, ...] for the non-finite leaves of
    `items`, a sequence of (label, object) pairs.

    Only ever reached on the capture path, so the extra reductions cost nothing
    in steady state -- and naming WHICH operand was dirty is the difference
    between a verdict and a guess.  `inputs_finite=False` on its own sends you
    looking upstream; this says whether there is anything upstream to look at.
    """
    bad = []
    for label, obj in items:
        for leaf in tree_leaves([obj]):
            if not _nonfinite(leaf):
                continue
            bad.append({"operand": label,
                        "shape": tuple(leaf.shape),
                        "dtype": str(leaf.dtype).replace("torch.", ""),
                        "nan": _count_where(torch.isnan, leaf),
                        "inf": _count_where(torch.isinf, leaf)})
    return bad


# --------------------------------------------------------------------------- #
# settle mode: where did a NaN come from, and did it outlive its step?
# (see "Settle mode" in the module docstring)
# --------------------------------------------------------------------------- #

_BENIGN_ORIGINS = frozenset(("math",))
# Never the capsule while a candidate of any other origin is pending, and never
# written at exit on its own: allocator garbage that a NaN check happened to read.
# Unlike `math` it is still saved once per site (<capsule>.masked-<n>).
_SOFT_ORIGINS = _BENIGN_ORIGINS | frozenset(("uninitialized",))


def _nan_origin(func, args, kwargs, bad, where, inputs_finite):
    """{'class', 'nonfinite', 'math', 'overflow', 'unexplained', 'note'} for `bad`.

    `args`/`kwargs` must be the PRE-call operands.  Re-runs a pointwise op in fp64
    and sorts every non-finite position of `bad`: fp64 non-finite there too ->
    math; fp64 finite but beyond bad.dtype's range -> overflow; else unexplained.
    Never raises: anything it cannot judge is `no_reference`.
    """
    res = {"class": None, "nonfinite": _count_where(lambda x: ~torch.isfinite(x), bad),
           "math": None, "overflow": None, "unexplained": None, "note": None}
    if inputs_finite is False:
        res["class"], res["note"] = "propagated", "a read-only input was already non-finite"
        return res
    if "pointwise" not in _tag_names(func) or needs_rng(func):
        res["class"], res["note"] = "no_reference", "not a pointwise op (or seeded): no fp64 re-run"
        return res
    try:
        def up(x):
            if isinstance(x, torch.Tensor) and x.is_floating_point():
                return x.to(torch.float64, copy=True)   # copy: in-place ops write it
            return x
        a64, k64 = tree_map(up, tuple(args)), tree_map(up, dict(kwargs))
        r = func(*a64, **k64)
        if where == "return":
            ref = next((x for x in tree_leaves(r) if isinstance(x, torch.Tensor)
                        and x.shape == bad.shape), None)
        elif where.startswith("arg"):
            ref = a64[int(where[3:])]
        else:
            ref = k64.get(where.split(":", 1)[1])
        if not isinstance(ref, torch.Tensor) or ref.shape != bad.shape:
            raise ValueError("no fp64 output matching %s" % where)
        bad_m = ~torch.isfinite(_scannable(bad))
        ref = ref.to(bad.device)
        fin = torch.isfinite(ref)
        n_math = int((bad_m & ~fin).sum())
        n_over = int((bad_m & fin & (ref.abs() > torch.finfo(bad.dtype).max)).sum())
        n_unex = res["nonfinite"] - n_math - n_over
        res.update(math=n_math, overflow=n_over, unexplained=n_unex)
        res["class"] = "unexplained" if n_unex else "overflow" if n_over else "math"
    except Exception as e:                              # noqa: BLE001
        res["class"] = "no_reference"
        res["note"] = "fp64 re-run failed: %s: %s" % (type(e).__name__, str(e)[:120])
    return res


def _origin_text(o):
    """One line for a nan_origin dict."""
    if not o:
        return "?"
    if o["class"] in ("math", "overflow", "unexplained"):
        return ("%s  (fp64 re-run, of %d non-finite: %d also non-finite in fp64, "
                "%d beyond the dtype's range, %d unexplained)"
                % (o["class"], o["nonfinite"], o["math"], o["overflow"], o["unexplained"]))
    return "%s  (%s)" % (o["class"], o["note"])


# --------------------------------------------------------------------------- #
# wrong FINITE values (opt-in: verify_every / bounds; NAN_CAPTURE_VERIFY_EVERY,
# NAN_CAPTURE_BOUNDS)
#
# A NaN is often the rare tail of a kernel that writes wrong but finite values: the
# 2026-11-09 Meta #1 replay changed its aten.mm output on ~26x more runs than it went
# NaN, and the ROCm 7.14 MoE run spiked its grad norm to 1e5 on four steps before the
# first NaN.  Two checks catch the value itself, on the GEMM family:
#
#   bounds     |C_ij| <= ||A_i|| * ||B_j|| (+ |beta * bias|): Cauchy-Schwarz, true
#              for ANY input, so a violation is a wrong result -- no tolerance to
#              tune beyond rounding slack.  mm / addmm / bmm / baddbmm.
#   reference  recompute in higher precision (16-bit -> fp32, fp32 -> fp64) and flag
#              max|C - ref| / max|ref| above the site's tolerance: a dtype floor, or
#              50x the median error the site showed so far, whichever is larger.
#              Sampled: every verify_every-th call of each site.  _grouped_mm has a
#              per-group fallback where no fp32 kernel exists.
# --------------------------------------------------------------------------- #

_BOUND_SLACK = {torch.bfloat16: 0.02, torch.float16: 0.005, torch.float32: 0.01}
_REF_FLOOR = {torch.bfloat16: 3e-2, torch.float16: 5e-3, torch.float32: 1e-4}
# fp32 GEMMs computed in TF32 / xf32 (allow_tf32, float32_matmul_precision != highest):
# a 10-bit mantissa.  The 2026-11-09 Meta #1 wave (K = 15456, TF32 on) showed 1.1e-3 on
# correct calls -- 12 false captures at the fp32 floor.
_REF_FLOOR_TF32 = 1e-2


def _ref_floor(dtype):
    if dtype == torch.float32 and (torch.backends.cuda.matmul.allow_tf32
                                   or torch.get_float32_matmul_precision() != "highest"):
        return _REF_FLOOR_TF32
    return _REF_FLOOR[dtype]
_TINY = 1e-30


def _arg(args, kwargs, i, name, default=None):
    return args[i] if len(args) > i else kwargs.get(name, default)


def _gemm_bound_ratio(func, args, kwargs, c):
    """max over C of |C_ij| / (||A_i|| ||B_j||) (bias removed), as a float; <= 1 for
    a correct result up to rounding.  None for an op this does not cover."""
    op = _op_names(func)[1]
    with torch.no_grad():
        if op == "aten.mm":
            a, b, bias, beta, alpha = args[0], args[1], None, 1, 1
        elif op == "aten.addmm":
            bias, a, b = args[0], args[1], args[2]
            beta, alpha = _arg(args, kwargs, 3, "beta", 1), _arg(args, kwargs, 4, "alpha", 1)
        elif op == "aten.bmm":
            a, b, bias, beta, alpha = args[0], args[1], None, 1, 1
        elif op == "aten.baddbmm":
            bias, a, b = args[0], args[1], args[2]
            beta, alpha = _arg(args, kwargs, 3, "beta", 1), _arg(args, kwargs, 4, "alpha", 1)
        else:
            return None
        if not all(isinstance(t, torch.Tensor) for t in (a, b)) or c.dim() not in (2, 3):
            return None
        rn = torch.linalg.vector_norm(a.float(), dim=-1)          # (..., M)
        cn = torch.linalg.vector_norm(b.float(), dim=-2)          # (..., N)
        bound = abs(float(alpha)) * rn.unsqueeze(-1) * cn.unsqueeze(-2)
        num = c.float().abs()
        if bias is not None and float(beta) != 0:
            num = (num - abs(float(beta)) * bias.float().abs().expand_as(num)).clamp_min(0)
        return float((num / bound.clamp_min(_TINY)).amax())


def _grouped_mm_ref(args, kwargs, dt):
    """(reference, valid-region slice) for aten._grouped_mm in dtype `dt`, one plain
    matmul per group; None for a layout this does not cover."""
    a, b = args[0].to(dt), args[1].to(dt)
    offs = _arg(args, kwargs, 2, "offs")
    bias = _arg(args, kwargs, 3, "bias")
    bias = bias.to(dt) if isinstance(bias, torch.Tensor) else None
    if a.dim() == 3 and b.dim() == 3:
        r = torch.bmm(a, b)
        return (r + bias.unsqueeze(1) if bias is not None else r), (slice(None),)
    if offs is None:
        return None
    ends = [int(e) for e in offs.tolist()]
    starts = [0] + ends[:-1]
    if a.dim() == 2 and b.dim() == 3:                 # rows grouped: (T,K) x (G,K,N)
        r = torch.zeros(a.shape[0], b.shape[-1], dtype=dt, device=a.device)
        for g, (s, e) in enumerate(zip(starts, ends)):
            if e > s:
                r[s:e] = a[s:e] @ b[g] + (bias[g] if bias is not None else 0)
        return r, (slice(0, ends[-1] if ends else 0),)
    if a.dim() == 2 and b.dim() == 2:                 # K grouped: (M,Kt) x (Kt,N) -> (G,M,N)
        r = torch.stack([a[:, s:e] @ b[s:e, :] for s, e in zip(starts, ends)])
        return r, (slice(None),)
    if a.dim() == 3 and b.dim() == 2:                 # N grouped: (G,M,K) x (K,Nt) -> (M,Nt)
        r = torch.zeros(a.shape[1], b.shape[-1], dtype=dt, device=a.device)
        for g, (s, e) in enumerate(zip(starts, ends)):
            if e > s:
                r[:, s:e] = a[g] @ b[:, s:e]
        return r, (slice(None), slice(0, ends[-1] if ends else 0))
    return None


def _ref_rel_err(func, args, kwargs, c):
    """max|c - ref| / max|ref| against a higher-precision recompute, or None when
    there is no usable reference (unsupported layout, the reference failed or is
    itself non-finite)."""
    dt = torch.float64 if c.dtype == torch.float32 else torch.float32
    op = _op_names(func)[1]
    with torch.no_grad():
        try:
            if op == "aten._grouped_mm":
                got = _grouped_mm_ref(args, kwargs, dt)
                if got is None:
                    return None
                ref, region = got
            else:
                def up(x):
                    return x.to(dt) if isinstance(x, torch.Tensor) and x.is_floating_point() else x
                k = dict((n, up(v)) for n, v in kwargs.items() if n != "out_dtype")
                ref = func(*[up(x) for x in args], **k)
                if not isinstance(ref, torch.Tensor):
                    return None
                region = (slice(None),)
            ref = ref[region]
            got_c = c[region].to(dt)
            if ref.shape != got_c.shape:
                return None
            scale = float(ref.abs().amax()) if ref.numel() else 0.0
            if not math.isfinite(scale):
                return None
            return float((got_c - ref).abs().amax()) / max(scale, _TINY)
        except Exception:                               # noqa: BLE001
            return None


_HOST_READ_OPS = frozenset(("aten._local_scalar_dense.default",))


def _is_host_read(func, args, kwargs, out):
    """True if `func` hands device data to the host: .item(), or an op that
    returns CPU tensors from device inputs (.cpu(), .tolist(), print)."""
    if _op_names(func)[0] in _HOST_READ_OPS:
        return True
    if isinstance(out, _Tensor):
        if out.device.type != "cpu":
            return False
    else:
        outs = _tensors(out)
        if not outs or any(t.device.type != "cpu" for t in outs):
            return False
    return any(t.device.type != "cpu" for t in _tensors((args, kwargs)))


def _storage_ptr(t):
    """Base address of t's storage, or None for a tensor without one (sparse,
    nested, meta, fake).  Used ONLY for the born-this-step set, which is rebuilt
    every step: an address the allocator reuses within a step can make a tensor
    look born -- never persistent -- so the worst case is one missed survival
    (e.g. optimizer state created lazily on its first step), caught a step later.
    """
    try:
        return t.untyped_storage().data_ptr()
    except Exception:                                   # noqa: BLE001
        return None


_TORCH_DIR = os.path.dirname(os.path.abspath(torch.__file__))
_STDLIB_DIR = os.path.dirname(os.path.abspath(os.__file__))
_THIS_FILE = os.path.abspath(__file__)


_HAS_NODE = hasattr(torch._C, "_current_autograd_node")


def _node_name():
    """Name of the autograd node running now (backward), else None."""
    node = torch._C._current_autograd_node() if _HAS_NODE else None
    return node.name() if node is not None else None


def _grab_frames(f):
    """[(code, lineno)] from frame `f` outward -- the cheap form of a stack,
    recorded per op when checks are deferred; _frames_text renders it."""
    out = []
    while f is not None:
        out.append((f.f_code, f.f_lineno))
        f = f.f_back
    return out


def _frames_text(frames):
    """traceback.format_stack()-style text for _grab_frames output."""
    return "".join(traceback.format_list(
        [traceback.FrameSummary(c.co_filename, ln, c.co_name) for c, ln in reversed(frames)]))


def _call_site(func, where, ctx=None):
    """(op, place, where): the site key a NaN is remembered under.  `place` is the
    autograd node in backward -- the Python stack there is the engine's, the same
    for every backward op -- else the first user frame of the forward stack.
    `ctx` is a deferred-check entry: its node and frames are the op's own."""
    if ctx is not None:
        node, frames = ctx["node"], [(c.co_filename, ln) for c, ln in ctx["frames"]]
    else:
        node = _node_name()
        frames = None
    if node is not None:
        place = node
    else:
        place = "?"
        if frames is None:
            frames = [(fr.filename, fr.lineno) for fr in reversed(traceback.extract_stack())]
        for fn, ln in frames:
            f = os.path.abspath(fn)
            if not (f.startswith(_TORCH_DIR) or f.startswith(_STDLIB_DIR) or f == _THIS_FILE):
                place = "%s:%d" % (f, ln)
                break
    return (_op_names(func)[0], place, where)


def _site_text(site):
    op, place, where = site
    if ":" in place and os.sep in place:
        f, line = place.rsplit(":", 1)
        place = "%s:%s" % (os.path.basename(f), line)
    return "%s @ %s (%s)" % (op, place, where)


# --------------------------------------------------------------------------- #
# rank discovery
# --------------------------------------------------------------------------- #

def rank_info():
    """(rank, world_size, local_rank, backend).  Works before and after init."""
    rank = world = None
    backend = None
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            rank, world = dist.get_rank(), dist.get_world_size()
            backend = dist.get_backend()
    except Exception:
        pass
    if rank is None:
        rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
        world = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
    local = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    return rank, world, local, backend


#TODO(cluk): improve error messages
def _rank_path(outdir, rank):
    from pathlib import Path
    p = Path(outdir)
    assert p.exists()
    assert p.is_dir()
    assert os.access(outdir, os.W_OK)
    import uuid
    outpath = os.path.join(outdir, f"rank{rank}_{uuid.uuid4().hex}.pt") 
    return outpath

# --------------------------------------------------------------------------- #
# contention: the one thing a capsule used to lose completely
#
# A capsule records VALUES.  A race is carried by TIMING -- how many processes
# are issuing at this device at once -- and none of that survives a torch.save.
# Replayed alone, a capsule of a contention-gated fault is the single-process
# case, which for these bugs is the case that never fails.  Measured on the 2nd
# NaN, every other variable held fixed:
#
#     model --workers 1   ->      0 / 12000
#     model --workers 6   ->    105 / 72000
#     chain replay solo   ->      0 / 20000
#     chain replay + a real co-tenant -> 707 / 20000
#
# So a `0/N` from a solo replay is not evidence about the kernel; it is evidence
# that the experiment was not set up.  What makes this hard to spot is that the
# knob is usually INSIDE the model -- `--workers 6` here, a torchrun
# --nproc-per-node, a DataLoader's worker count, a launcher loop that starts
# more containers than there are GPUs.  Nothing about it is visible from the op
# that failed.
#
# So record it at capture time, when it is still observable, rather than hoping
# someone reconstructs it later from the source.
# --------------------------------------------------------------------------- #

def _cmdline(pid="self"):
    try:
        with open("/proc/%s/cmdline" % pid, "rb") as f:
            return [a.decode("utf-8", "replace")
                    for a in f.read().split(b"\0") if a]
    except Exception:                                  # noqa: BLE001
        return None


def _gpu_proc_scan():
    """[(pid, holds_kfd, cmdline)] for every process with the GPU open, by /proc fd
    scan, or None when nothing could be read.

    /dev/kfd is the ROCm compute device: only a process holding it can launch
    kernels.  /dev/dri/renderD* alone is what monitors and telemetry agents
    (amdsmi, rocm-smi) open -- they hold the device without loading it.

    Limits worth knowing: it sees only this PID namespace, so sibling CONTAINERS
    on the same GPU are invisible to it, and it needs permission to read other
    processes' fds.
    """
    try:
        pids = os.listdir("/proc")
    except Exception:                                  # noqa: BLE001
        return None
    found = []
    for pid in pids:
        if not pid.isdigit():
            continue
        d = "/proc/%s/fd" % pid
        kfd = render = False
        try:
            for fd in os.listdir(d):
                tgt = os.readlink(os.path.join(d, fd))
                kfd = kfd or tgt.startswith("/dev/kfd")
                render = render or tgt.startswith("/dev/dri/renderD")
        except Exception:                              # noqa: BLE001
            pass                                       # gone, or not ours to read
        if kfd or render:
            cmd = _cmdline(pid) or []
            found.append((int(pid), kfd, " ".join(cmd)[:160]))
    return found or None


_LAUNCHER_RE = re.compile(r"torchrun|torch\.distributed\.(?:run|launch)|deepspeed|accelerate|mpirun|srun")
# torch.compile's helper processes: inductor's compile-worker pool (one per core, up to
# ~32) opens /dev/kfd but only compiles -- it launches nothing on the device.  Counted,
# they read as a crowd of co-tenants (gpu_procs=34 beside one trainer, 2026-11-09).
_COMPILE_HELPER_RE = re.compile(r"torch/_inductor/compile_worker|torch\._inductor\.compile_worker")


def _is_compile_helper(cmdline):
    return bool(_COMPILE_HELPER_RE.search(cmdline or ""))


def _launcher_ancestors(scan=None):
    """pids of this process's ANCESTORS that hold the GPU and look like a launcher
    (torchrun & co.): such a parent opens a HIP context (~1-2 GB, /dev/kfd) but launches
    nothing, so counting it reads as a co-tenant and sets a replay one copy too high.
    A parent that is not a known launcher (an mp.spawn parent may compute) stays counted."""
    held = {p[0]: p for p in (scan or [])}
    out, pid = [], os.getpid()
    for _ in range(64):
        try:
            with open("/proc/%d/stat" % pid) as f:
                pid = int(f.read().rsplit(")", 1)[1].split()[1])
        except Exception:                              # noqa: BLE001
            break
        if pid <= 1:
            break
        if pid in held and _LAUNCHER_RE.search(held[pid][2] or ""):
            out.append(pid)
    return out


def _gpu_procs():
    """How many processes have the GPU open for COMPUTE, by /proc fd scan.

    Counts /dev/kfd holders: a render-node-only holder (a monitor) cannot launch
    kernels and is not contention.  Falls back to counting render-node holders
    when no /dev/kfd holder is visible at all (a device exposed differently).
    Deliberately not rocm-smi: no subprocess, no parsing, and it works inside a
    container, where rocm-smi reports the host's view and this reports ours.
    None means "could not tell", which is a different answer from 1.
    """
    scan = _gpu_proc_scan()
    if not scan:
        return None
    launchers = set(_launcher_ancestors(scan))           # a torchrun parent is not contention
    scan = [p for p in scan if p[0] not in launchers and not _is_compile_helper(p[2])] or scan
    kfd = [p for p in scan if p[1]]
    return len(kfd) if kfd else len(scan)


_CONTAINERS_PER_GPU_VAR = "NAN_CAPTURE_CONTAINERS_PER_GPU"


def _declared_containers_per_gpu():
    """Peer containers sharing this GPU, as declared by the launcher.

    The one piece of contention the process scan above CANNOT recover, and the
    reason it exists: `_gpu_procs()` reads /proc, /proc is per PID namespace, so
    a sibling CONTAINER on the same device is invisible to it however much of
    the device that sibling is using.  A launcher that puts two single-process
    containers on each GPU therefore captures `gpu_procs=1`, and a replay that
    believes it reconstructs the one-process case -- which for a contention-
    gated fault is the case that never fails.  The failure is silent: not an
    error, just a clean 0/N that reads like evidence about the kernel.

    So the launcher declares it, because only the launcher knows.  Set
    NAN_CAPTURE_CONTAINERS_PER_GPU in the trial loop to the number of copies of
    this container that share one device -- for a loop assigning
    `gpu=$((trial % 8))` over 16 trials, that is 2.

    This MULTIPLIES with the scan rather than replacing it: the scan sees
    fan-out INSIDE this container (mp.Process, --workers N, DataLoader workers),
    the variable sees peer containers outside it, and a model that does both has
    `gpu_procs * containers_per_gpu` processes on the device.  Neither number
    alone is the answer when both are > 1.

    None means undeclared -- which is a different answer from 1, and is left as
    None so the replay can say "nobody told me" rather than assume a solo
    capture.  A value that is not a positive integer is ignored with a warning;
    silently treating a typo as 1 would recreate exactly the wrong experiment
    this field exists to prevent.
    """
    raw = os.environ.get(_CONTAINERS_PER_GPU_VAR)
    if raw is None or not raw.strip():
        return None
    try:
        n = int(raw.strip())
        if n < 1:
            raise ValueError("must be >= 1")
    except (TypeError, ValueError) as e:
        print("[op-nan] WARNING: %s=%r ignored (%s). The capsule will record"
              " no peer-container count, and --contend-auto will see only this"
              " container's processes."
              % (_CONTAINERS_PER_GPU_VAR, raw, e), flush=True)
        return None
    return n


# ---------------------------------------------------------------------------
# Environment and build identity
#
# The capsule carries the environment the failing library saw so a replay can
# re-create it (replay_nan_op.py --restore-env).  Two things make that more
# than a convenience:
#
#   * hipBLASLt, rocBLAS and MIOpen read their environment when the .so
#     initialises, so a variable that differs cannot be fixed up afterwards --
#     the replay has to re-exec under it;
#   * several of these are amplifiers rather than settings.  HSA_DISABLE_CACHE
#     moved the 2nd NaN's rate ~70x and TORCH_ROCM_FA_PREFER_CK is what selects
#     the buggy backend at all, so a capsule that drops them restores an
#     environment that cannot reproduce the bug it is restoring for.
#
# The prefixes are deliberately wider than the set we know matters: carrying an
# irrelevant var costs nothing, dropping a relevant one costs a clean 0/N that
# reads like evidence.  ENV_NO_RESTORE and the not_captured list below close the
# two remaining gaps -- see each one's comment.
ENV_PREFIXES = (
    "HIPBLAS", "HIP_", "ROCBLAS_", "ROCR_", "ROCM_", "TENSILE_",
    "AMD_", "AMDGPU_", "HSA_", "MIOPEN_", "AITER_", "CK_", "TRITON_",
    "TORCH", "PYTORCH_", "NCCL_", "RCCL_", "OMP_", "GOMP_", "GPU_", "MKL_",
)
ENV_EXACT = ("LD_PRELOAD", "LD_LIBRARY_PATH")

# Recorded as evidence, never re-applied by a replay.  Two groups, two reasons,
# and in both cases the replay reports the difference rather than acting on it.
#
#   * device selection.  These name which slot the capturing process was given,
#     not how the library was configured.  Restoring HIP_VISIBLE_DEVICES out of
#     an 8-GPU model run repoints a replay the wrapper pinned to one device --
#     and on a mutation sweep it silently moves the trials off the device the
#     baseline was measured on, which is unreadable rather than merely wrong.
#   * loader paths.  These govern the whole process, including `import torch`
#     itself, which on a re-exec happens AFTER they are applied.  Restoring a
#     capture container's LD_LIBRARY_PATH into a replay container with a
#     different layout does not produce a differently-configured library, it
#     produces an ImportError -- or worse, a half-resolved one.  When they
#     differ, the two environments are different images and that is the finding.
#   * diagnostic logging.  A per-call library log is host work between launches
#     -- it changes the timing a race depends on -- and every replay copy would
#     append to the capture's own log file.  The debug-nan skill turns the
#     hipBLASLt log on for one capture trial; restored into 32 replay copies it
#     slowed them severalfold and wrote 341k replay calls into the model's log
#     (Lux job 386142).
ENV_NO_RESTORE = ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
                  "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL",
                  "LD_LIBRARY_PATH", "LD_PRELOAD",
                  "HIPBLASLT_LOG_MASK", "HIPBLASLT_LOG_LEVEL", "HIPBLASLT_LOG_FILE",
                  "ROCBLAS_LAYER", "ROCBLAS_LOG_TRACE_PATH", "ROCBLAS_LOG_BENCH_PATH",
                  "ROCBLAS_LOG_PROFILE_PATH", "AMD_LOG_LEVEL")


def _env_recorded(k):
    return k.startswith(ENV_PREFIXES) or k in ENV_EXACT


def env_info():
    """The environment the capsule carries, and the rule it carried it under."""
    return {
        # Names only, no values: this is what lets a replay say "47 vars were
        # set at capture and are outside this capsule's namespace" instead of
        # silently restoring a subset and reporting it as the environment.
        # Names rather than values so the capsule never becomes a place secrets
        # accumulate.
        "not_captured": sorted(k for k in os.environ if not _env_recorded(k)),
        # The rule itself, so a replay unsets under the SAME namespace the
        # capture recorded under.  Without it, widening ENV_PREFIXES later would
        # make a new replay unset vars an old capsule never had the chance to
        # record -- indistinguishable from the capture having run without them.
        "prefixes": list(ENV_PREFIXES),
        "exact": list(ENV_EXACT),
        "no_restore": list(ENV_NO_RESTORE),
    }


def _loaded_libs():
    """Identity of the ROCm libraries mapped into this process, by /proc scan.

    The ones actually loaded, not the ones on a search path: a replay that
    dispatches into a different hipBLASLt build is a different experiment, and
    nothing else in the capsule would show it.
    """
    want = ("libhipblaslt", "librocblas", "libamdhip64", "libMIOpen",
            "libtorch_hip", "libck", "libaiter")
    found = {}
    try:
        with open("/proc/self/maps") as fh:
            for line in fh:
                path = line.rstrip("\n").rpartition(" ")[2].strip()
                if not path.startswith("/") or path in found:
                    continue
                if not os.path.basename(path).startswith(want):
                    continue
                try:
                    st = os.stat(path)
                    found[path] = {"size": st.st_size, "mtime": int(st.st_mtime)}
                except OSError:
                    found[path] = None                 # mapped but gone from disk
    except Exception:                                  # noqa: BLE001
        return None
    return found or None


def build_info():
    """Which build of the stack this ran on, as far as it is visible from inside.

    `versions` already records torch and HIP, and that pair does not pin a build:
    two images report the same one and ship different hipBLASLt binaries.  The
    container image tag is not observable from inside the container, so the
    closest available proxy is the identity of the libraries that dispatched.
    All best-effort -- a field that cannot be read is None, never a guess.
    """
    info = {"rocm_version": None, "os_release": None, "uname": None,
            "libs": _loaded_libs()}
    try:
        with open("/opt/rocm/.info/version") as fh:
            info["rocm_version"] = fh.read().strip()
    except Exception:                                  # noqa: BLE001
        pass
    try:
        with open("/etc/os-release") as fh:
            for line in fh:
                if line.startswith("PRETTY_NAME="):
                    info["os_release"] = line.split("=", 1)[1].strip().strip('"')
                    break
    except Exception:                                  # noqa: BLE001
        pass
    try:
        u = os.uname()
        info["uname"] = "%s %s" % (u.sysname, u.release)
    except Exception:                                  # noqa: BLE001
        pass
    return info


def contention_info():
    """Everything observable about who else is using this device, right now.

    All of it is best-effort: a field that cannot be determined is None, never a
    guess, because the replay uses these to decide how many co-tenants to start
    and a fabricated 1 would quietly recreate the wrong experiment.
    """
    info = {
        # argv and parent_argv are where a hidden fan-out flag is recoverable
        # from, and they are not interchangeable.  Measured on the 2nd NaN: a
        # capture inside an mp.spawn worker records its own argv as
        #   python3 -c 'from multiprocessing.spawn import spawn_main; ...'
        #     --multiprocessing-fork
        # -- no script name, no flags.  `--workers 6` was on the PARENT's line.
        # Since a spawn worker is the likeliest place a contention-gated fault
        # gets caught, parent_argv is not a nice-to-have: without it the capsule
        # records the concurrency but not what produced it.
        "argv": _cmdline("self"),
        "parent_argv": None,
        "pid": os.getpid(),
        "ppid": None,
        # Two different observations of "who else is on this device", and they
        # compose rather than compete -- see _declared_containers_per_gpu.
        # gpu_procs is what this PID namespace can see; containers_per_gpu is
        # what it structurally cannot.
        "gpu_procs": _gpu_procs(),
        # who they are: [(pid, holds /dev/kfd, cmdline)], so a reader can tell a
        # co-tenant training job from an idle launcher or a metrics agent.
        "gpu_proc_list": _gpu_proc_scan(),
        # launcher ancestors (torchrun) holding a context: listed, not counted in gpu_procs
        "gpu_proc_launchers": None,
        # torch.compile helpers (inductor compile workers) are listed, not counted in
        # gpu_procs; older capsules counted them, and replay discounts them there
        "gpu_proc_helpers_excluded": True,
        "containers_per_gpu": _declared_containers_per_gpu(),
        "mp_child": False,
        "mp_children": None,
        "device_count": None,
        "mem_used_by_others_bytes": None,
        "visible_devices": os.environ.get("HIP_VISIBLE_DEVICES")
                           or os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        info["gpu_proc_launchers"] = _launcher_ancestors(info["gpu_proc_list"])
    except Exception:                                  # noqa: BLE001
        pass
    try:
        info["ppid"] = os.getppid()
        info["parent_argv"] = _cmdline(info["ppid"])
    except Exception:                                  # noqa: BLE001
        pass
    try:
        import multiprocessing as _mp
        # True in a spawn/fork worker, which is exactly the shape that produces
        # N contexts on one device without any launcher flag showing it.
        info["mp_child"] = _mp.parent_process() is not None
        info["mp_children"] = len(_mp.active_children())
    except Exception:                                  # noqa: BLE001
        pass
    try:
        info["device_count"] = torch.cuda.device_count()
        free, total = torch.cuda.mem_get_info()
        ours = torch.cuda.memory_reserved()
        # What the allocator cannot see but the device has lost: a decent proxy
        # for co-tenants even when the /proc scan cannot see into other
        # containers.
        info["mem_used_by_others_bytes"] = max(0, total - free - ours)
    except Exception:                                  # noqa: BLE001
        pass
    return info


# --------------------------------------------------------------------------- #
# packing
# --------------------------------------------------------------------------- #

def _digest(flat_cpu):
    if flat_cpu.numel() == 0:
        return "empty"
    return hashlib.blake2b(flat_cpu.view(torch.uint8).numpy().tobytes(),
                           digest_size=16).hexdigest()


def _window(t):
    """(span, pad) -- the element window t's view covers, plus its leading pad.

    span is the elements from the view's first to its last, inclusive: for a
    padded / transposed view that is exactly the window the kernel reads.

    pad keeps the low bits of the storage offset, so a replayed pointer lands at
    the same alignment relative to a fresh (512B-aligned) allocation.

    Shared by the CPU pack and the device-side snapshot, so the two can never
    disagree about which bytes constitute "this tensor".
    """
    sizes, strides = tuple(t.shape), tuple(t.stride())
    if any(s < 0 for s in strides):
        # Not produced by any aten GEMM today; bail loudly rather than capture
        # something that replays as a different problem.
        raise RuntimeError("negative stride %r -- refusing to capture" % (strides,))
    span = (0 if t.numel() == 0
            else 1 + sum((sz - 1) * st for sz, st in zip(sizes, strides)))
    align_elems = max(_ALIGN_BYTES // t.element_size(), 1)
    return span, t.storage_offset() % align_elems


def _snap_value(v):
    """Layout-preserving device copy of a tensor, or of a list/tuple of them.

    TensorList arguments are not a corner case: `_foreach_add_(Tensor(a!)[] self,
    ...)` is how every fused optimizer writes, so a snapshot that only understood
    bare tensors would quietly skip the largest writes in the step.  None means
    "cannot snapshot this", which the caller treats as no snapshot at all.
    """
    if isinstance(v, torch.Tensor):
        return _clone_keeping_layout(v)
    if isinstance(v, (list, tuple)):
        snapped = []
        for x in v:
            s = _snap_value(x)
            if s is None:
                return None
            snapped.append(s)
        return tuple(snapped) if isinstance(v, tuple) else snapped
    return None


def _snap_bytes(v):
    """Bytes _snap_value(v) would allocate: the byte window of each tensor."""
    if isinstance(v, torch.Tensor):
        if v.layout is not torch.strided:
            return 0
        span, pad = _window(v)
        return (span + pad) * v.element_size()
    if isinstance(v, (list, tuple)):
        return sum(_snap_bytes(x) for x in v)
    return 0


def _snap_breakdown(label, v):
    """One line per written tensor: what the snapshot would copy, and from what."""
    if isinstance(v, torch.Tensor):
        st = v.untyped_storage().nbytes() if v.layout is torch.strided else 0
        return ["%s %s %s stride=%s off=%d copy=%.3f GiB storage=%.3f GiB"
                % (label, tuple(v.shape), str(v.dtype).replace("torch.", ""),
                   tuple(v.stride()) if v.layout is torch.strided else v.layout,
                   v.storage_offset() if v.layout is torch.strided else 0,
                   _snap_bytes(v) / 2**30, st / 2**30)]
    if isinstance(v, (list, tuple)):
        big = max((x for x in v if isinstance(x, torch.Tensor)),
                  key=_snap_bytes, default=None)
        head = ["%s list[%d] copy=%.3f GiB total" % (label, len(v), _snap_bytes(v) / 2**30)]
        return head + (_snap_breakdown("%s largest" % label, big) if big is not None else [])
    return []


def _clone_keeping_layout(t):
    """A device-side copy of `t` with its size, stride and relative alignment intact.

    For snapshotting an argument an out-param kernel is about to overwrite.  A
    plain .clone() returns a contiguous tensor, which at replay time would flip
    transA/lda and hand hipBLASLt a different kernel (see the strides note in the
    module docstring) -- so copy the byte window instead and re-derive the view
    over it.  Stays on the device: the CPU transfer happens only if we capture.
    """
    if t.layout is not torch.strided:
        return None
    span, pad = _window(t)
    if span == 0:
        base = torch.empty(pad, dtype=t.dtype, device=t.device)
    else:
        flat = torch.as_strided(t, (span + pad,), (1,), t.storage_offset() - pad)
        base = flat.detach().clone()
    return torch.as_strided(base, tuple(t.shape), tuple(t.stride()), pad)


def tensor_stats(t):
    """A few numbers describing t's values, for synthesizing look-alike operands.

    Float: counts (numel, nan, posinf, neginf, zero) and, over the finite
    elements, min/max/mean/std.  Integer: min/max.  Bool: the fraction true.
    None for an empty, complex or unreadable tensor.  Sliced like _finite_all,
    with one device read per tensor; never raises (a capsule is worth more than
    its statistics).
    """
    try:
        if t.numel() == 0 or t.is_complex():
            return None
        if t.dtype == torch.bool:
            return {"kind": "bool", "numel": t.numel(),
                    "frac_true": float(t.float().mean())}
        if not t.is_floating_point():
            return {"kind": "int", "numel": t.numel(),
                    "min": int(t.min()), "max": int(t.max())}
        parts = []
        for c in _scan_chunks(t):
            x = _scannable(c).float()
            fin = torch.isfinite(x)
            xf = torch.where(fin, x, torch.zeros((), dtype=x.dtype, device=x.device))
            inf = float("inf")
            # Sum of squares scaled by the chunk's max|x|: fp32 squares overflow
            # past 1.8e19 (the very range an overflow hunt cares about), and a
            # float64 copy of the chunk would double its transient.
            m = xf.abs().amax().clamp_min(torch.finfo(torch.float32).tiny)
            parts.append(torch.stack([
                fin.sum(dtype=torch.float64),
                torch.isnan(x).sum(dtype=torch.float64),
                (x == inf).sum(dtype=torch.float64),
                (x == -inf).sum(dtype=torch.float64),
                (x == 0).sum(dtype=torch.float64),
                xf.sum(dtype=torch.float64),
                ((xf / m) ** 2).sum(dtype=torch.float64) * m.double() ** 2,
                torch.where(fin, x, inf).amin().double(),
                torch.where(fin, x, -inf).amax().double()]))
        p = torch.stack(parts)
        s = p[:, :7].sum(0).tolist() + [p[:, 7].amin().item(), p[:, 8].amax().item()]
        n_fin, n_nan, n_pinf, n_ninf, n_zero, tot, sq, lo, hi = s
        mean = tot / n_fin if n_fin else None
        var = max(sq / n_fin - mean * mean, 0.0) if n_fin else None
        return {"kind": "float", "numel": t.numel(), "nan": int(n_nan),
                "posinf": int(n_pinf), "neginf": int(n_ninf), "zero": int(n_zero),
                "min": lo if n_fin else None, "max": hi if n_fin else None,
                "mean": mean, "std": var ** 0.5 if n_fin else None}
    except Exception:                                    # noqa: BLE001
        return None


def _is_index_dtype(dtype):
    """Integer tensors: cu_seqlens, indices, offsets -- structure, not data."""
    return not (dtype.is_floating_point or dtype.is_complex or dtype == torch.bool)


def _pack_tensor(t, max_bytes, record_data=True):
    """Copy the exact bytes this tensor views, keeping size/stride/alignment.

    record_data=False keeps only the metadata for floating-point tensors: no
    bytes and no digest of them (a digest still says whether two runs saw the
    same inputs).  `base` is None, which unpack() refuses unless synthesizing.
    Integer tensors keep their bytes anyway: they are small, carry structure
    rather than the user's data, and a random draw from [min, max] is not a
    valid cu_seqlens / index tensor (the op rejects it instead of replaying).
    """
    if t.layout is not torch.strided:
        raise RuntimeError("cannot capture non-strided tensor: %r" % (t.layout,))

    sizes, strides = tuple(t.shape), tuple(t.stride())
    esz = t.element_size()
    span, pad = _window(t)

    if not record_data and _is_index_dtype(t.dtype) and (span + pad) * esz <= max_bytes:
        packed = _pack_tensor(t, max_bytes, record_data=True)
        packed["base_numel"] = span + pad
        packed["stats"] = tensor_stats(t)
        return packed

    if not record_data:
        return {
            _TENSOR_TAG: True,
            "base": None,
            "base_numel": span + pad,  # what `base` would have held
            "size": sizes,
            "stride": strides,
            "offset": pad,
            "dtype": t.dtype,
            "requires_grad": t.requires_grad,
            "sha": None,
            "stats": tensor_stats(t),  # what replay --synthesize draws from
        }

    if (span + pad) * esz > max_bytes:
        raise RuntimeError(
            "capture would be %.1f GB (span=%d pad=%d esz=%d); raise max_bytes "
            "if that is really what you want" % ((span + pad) * esz / 1e9, span, pad, esz))

    if span == 0:
        base = torch.empty(pad, dtype=t.dtype, device="cpu")
    else:
        # as_strided views the SAME storage -- no contiguous copy, no layout change.
        flat = torch.as_strided(t, (span + pad,), (1,), t.storage_offset() - pad)
        base = flat.detach().to("cpu", copy=True)

    return {
        _TENSOR_TAG: True,
        "base": base,              # flat 1-D buffer, length span+pad
        "size": sizes,
        "stride": strides,
        "offset": pad,             # offset into `base`
        "dtype": t.dtype,
        "requires_grad": t.requires_grad,
        "sha": _digest(base[pad:] if pad else base),   # digest of the viewed window only
    }


def _pack(obj, max_bytes, record_data=True):
    if isinstance(obj, torch.Tensor):
        return _pack_tensor(obj, max_bytes, record_data)
    if isinstance(obj, tuple):
        return tuple(_pack(o, max_bytes, record_data) for o in obj)
    if isinstance(obj, list):
        return [_pack(o, max_bytes, record_data) for o in obj]
    if isinstance(obj, dict):
        return dict((k, _pack(v, max_bytes, record_data)) for k, v in obj.items())
    return obj                     # int / float / bool / None / str / ScalarType


def _picklable(obj, path, bad):
    """`obj` with every non-tensor leaf that pickle refuses replaced by a marker
    dict ({"unpicklable": type, "repr": ...}); each replaced leaf's path is
    appended to `bad`.  _pack and _refs pass unknown argument types through
    verbatim, and one that holds a module (or a lock, a code object, ...) makes
    torch.save fail and the capsule be lost."""
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, tuple):
        return tuple(_picklable(o, "%s[%d]" % (path, i), bad) for i, o in enumerate(obj))
    if isinstance(obj, list):
        return [_picklable(o, "%s[%d]" % (path, i), bad) for i, o in enumerate(obj)]
    if isinstance(obj, dict):
        return dict((k, _picklable(v, "%s[%r]" % (path, k), bad)) for k, v in obj.items())
    try:
        pickle.dumps(obj)
        return obj
    except Exception:
        bad.append("%s: %s" % (path, type(obj).__name__))
        return {"unpicklable": type(obj).__name__, "repr": repr(obj)[:200]}


SHAPES_ONLY_MSG = ("capsule was recorded without tensor data (shapes only, the "
                   "capture default); replay it with --synthesize (look-alike "
                   "values from the recorded statistics), or re-capture with "
                   "NAN_CAPTURE_DATA=1 (OpNanCapture(record_data=True)).")


def has_data(capsule):
    """False for a shapes-only capsule.  Capsules older than the option hold data."""
    return capsule.get("data", "values") == "values"


# Synthesis: unpack() fills a shapes-only tensor (base None) with values drawn
# from its recorded statistics.  Off unless synthesize_operands() is called
# (replay --synthesize).  A tensor with recorded bytes is never touched.
_SYNTH = {"seed": None, "cache": {}}


def synthesize_operands(seed=0):
    _SYNTH["seed"] = int(seed)
    _SYNTH["cache"].clear()


def synthesizing():
    return _SYNTH["seed"] is not None


def _synth_base(obj):
    """The flat CPU buffer `base` stands for, synthesized; cached per packed dict.

    Seeded from the global seed and the tensor's own description, not from the
    order of unpack() calls, so every process (5b copies, co-tenants, sweep
    trials) and the CPU fp64 reference see identical operands.
    """
    hit = _SYNTH["cache"].get(id(obj))
    if hit is not None and hit[0] is obj:
        return hit[1]
    stats = obj.get("stats")
    n = obj["base_numel"]
    dtype = obj["dtype"]
    key = repr((obj["size"], obj["stride"], obj["offset"], str(dtype),
                sorted((stats or {}).items())))
    seed = int.from_bytes(hashlib.blake2b(("%d|%s" % (_SYNTH["seed"], key)).encode(),
                                          digest_size=8).digest(), "little") >> 1
    g = torch.Generator().manual_seed(seed)
    kind = (stats or {}).get("kind")
    if kind == "bool":
        base = torch.rand(n, generator=g) < stats["frac_true"]
    elif kind == "int":
        base = torch.randint(stats["min"], stats["max"] + 1, (n,), generator=g,
                             dtype=torch.int64).to(dtype)
    elif not dtype.is_floating_point:
        base = torch.zeros(n, dtype=dtype)        # int/bool without statistics
    else:
        base = _synth_float(n, stats, g).to(dtype)
    _SYNTH["cache"][id(obj)] = (obj, base)
    return base


def _synth_float(n, s, g):
    """n float32 values shaped like `s`: normal(mean, std) clipped to [min, max],
    with the recorded fractions of exact zeros, NaN, +inf and -inf."""
    if not s or s.get("mean") is None:
        # No statistics (an older capsule), or not one finite element recorded.
        x = torch.randn(n, generator=g)
        if s and s.get("numel"):
            x.fill_(float("nan") if s["nan"] else float("inf"))
        return x
    x = torch.randn(n, generator=g, dtype=torch.float64) * (s["std"] or 0.0) + s["mean"]
    x = x.clamp(s["min"], s["max"]).float()
    numel = s["numel"]
    u = torch.rand(n, generator=g, dtype=torch.float64)
    edge = 0.0
    for frac, val in ((s["zero"], 0.0), (s["nan"], float("nan")),
                      (s["posinf"], float("inf")), (s["neginf"], float("-inf"))):
        if frac:
            lo, edge = edge, edge + frac / numel
            x[(u >= lo) & (u < edge)] = val
    return x


def upgrade_capsule(obj):
    """Rename a loaded capsule's pre-rename keys (_LEGACY_KEYS) in place; returns it."""
    if isinstance(obj, dict):
        for old, new in _LEGACY_KEYS.items():
            if old in obj:
                obj[new] = obj.pop(old)
        for v in obj.values():
            upgrade_capsule(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            upgrade_capsule(v)
    return obj


def unpack(obj, device):
    """Inverse of _pack.  Used by the replay scripts."""
    if isinstance(obj, dict) and obj.get(_TENSOR_TAG):
        if obj["base"] is None and synthesizing():
            base = _synth_base(obj).to(device)
            return torch.as_strided(base, obj["size"], obj["stride"], obj["offset"])
        if obj["base"] is None:
            raise RuntimeError(SHAPES_ONLY_MSG)
        base = obj["base"].to(device)
        return torch.as_strided(base, obj["size"], obj["stride"], obj["offset"])
    if isinstance(obj, tuple):
        return tuple(unpack(o, device) for o in obj)
    if isinstance(obj, list):
        return [unpack(o, device) for o in obj]
    if isinstance(obj, dict):
        return dict((k, unpack(v, device)) for k, v in obj.items())
    return obj


def tensor_shas(packed_args):
    """['<sha of arg0>', None, '<sha of arg2>', ...] for cross-rank comparison."""
    out = []
    for a in packed_args:
        out.append(a["sha"] if isinstance(a, dict) and a.get(_TENSOR_TAG) else None)
    return out


# --------------------------------------------------------------------------- #
# torch -> hipBLASLt problem key
# --------------------------------------------------------------------------- #

def hipblaslt_geometry(func, args):
    """Map an aten GEMM onto the column-major problem hipBLASLt will log.

    For mm(X[p,q], Y[q,r]) -> out[p,r]:  M=r, N=p, K=q.
    hipBLASLt's A is Y and its B is X (the operands swap, because a row-major
    product is the column-major product of the transposes).  trans/ld then follow
    from which dimension of each operand is unit-stride.

    Returns None for ops this mapping does not cover.
    """
    if func is torch.ops.aten.mm.default:
        X, Y = args[0], args[1]
    elif func is torch.ops.aten.addmm.default:
        X, Y = args[1], args[2]
    else:
        return None
    if X.dim() != 2 or Y.dim() != 2:
        return None

    p, q = X.shape
    r = Y.shape[1]
    M, N, K = r, p, q

    # A <- Y[q,r] = [K,M]
    if Y.stride(1) == 1:
        transA, lda = "N", M       # memory is row-major (K,M) == col-major (M,K)
    elif Y.stride(0) == 1:
        transA, lda = "T", K       # memory is row-major (M,K) == col-major (K,M)
    else:
        return None                # neither dim unit-stride: not a plain GEMM

    # B <- X[p,q] = [N,K]
    if X.stride(1) == 1:
        transB, ldb = "N", K
    elif X.stride(0) == 1:
        transB, ldb = "T", N
    else:
        return None

    return {"M": M, "N": N, "K": K, "lda": lda, "ldb": ldb, "ldc": M, "ldd": M,
            "transA": transA, "transB": transB, "batch_count": 1}


def geometry_key(g):
    if not g:
        return "unmapped"
    return ("M%d_N%d_K%d_%s%s_lda%d_ldb%d_ldd%d"
            % (g["M"], g["N"], g["K"], g["transA"], g["transB"],
               g["lda"], g["ldb"], g["ldd"]))


#TODO(cluk): improve error messages
def _check_outfile(outfile_path_str:str) -> None:
    print(f">>> outfile_path={outfile_path_str}")
    from pathlib import Path
    outfile_path = Path(outfile_path_str)
    parent_dir = outfile_path.parent


    
# --------------------------------------------------------------------------- #
# provenance: which op produced the tensor this op just read?
#
# `inputs_finite=True` is the single most misread field a capsule carries.  It
# says the operands contained no NaN -- not that they were CORRECT.  An op one
# step upstream can return a finite but wrong value (a corrupted accumulator, a
# logsumexp off by an exponent) and the op that finally goes non-finite is then
# an innocent bystander faithfully computing garbage.  Nothing in a single-op
# capsule can tell those two apart, so the walk below records ancestry.
#
# `inputs_finite=None` is the second most misread, and reads as a non-answer
# when it is often the most informative value in the capsule.  For an op with
# positional written arguments it means: every operand that is unambiguously an
# input was CLEAN, and the only dirty ones were buffers the kernel writes --
# whose pre-call contents are uninitialised memory.  That is not "cannot tell
# anything"; combined with an empty `nonfinite_operands["inputs"]` it is good
# evidence the op created the NaN itself.  Read the two together.
#
# Two windows, because on a real model one window cannot be both long enough to
# reach a producer and small enough to hold in memory:
#
#   structure  metadata only -- op name, storage ids, byte counts.  Negligible
#              per op, so it can run thousands of ops deep.  Survives eviction,
#              which is what lets a capsule say "op X produced your bad input"
#              even when X's tensors are long gone.
#   value      strong references to the actual tensors, bounded by bytes and
#              evicted oldest-first.  This is the part a replay can re-execute.
#
# A node in the structure window whose values were evicted is reported as
# `values_evicted`, never silently dropped: on a transformer the producer of a
# backward op's input is thousands of ops back, so this is the ORDINARY outcome,
# not an error, and a chain that stopped early must never read like one that
# reached a root.
# --------------------------------------------------------------------------- #

_DIGESTS = {}       # (shape, stride, offset, dtype) -> tkey view digest

_CHAIN_STOP_REASONS = (
    "depth_reached",     # the walk hit chain_depth; raise it to go further
    "values_evicted",    # producer known, its tensors aged out of the value window
    "outside_window",    # no producer recorded at all -- older than the structure window
    "data_dependent",    # .item()/nonzero: the value left the tensor world
    "collective",        # a c10d op: its input lives on another rank
    "version_moved",     # the tensor was mutated after it was produced
)


def _storage_bytes(t):
    try:
        return int(t.untyped_storage().nbytes())
    except Exception:                                  # noqa: BLE001
        return int(t.numel() * t.element_size())


class _ChainTracker:
    """Records op ancestry so a capture can walk back from the failing op.

    Lives for the duration of one OpNanCapture.  Every method is on the hot
    path of every dispatched op, so the steady-state cost is deliberately a few
    dict operations and no copies -- the only clone it ever takes is for an
    argument an op is about to overwrite, which would otherwise be unrecoverable.
    """

    def __init__(self, depth, struct_ops, value_bytes, store_intermediates,
                 track_only, snapshot_mutated):
        self.depth = depth
        self.struct_ops = struct_ops
        self.value_bytes = value_bytes
        self.store_intermediates = store_intermediates
        self.track_only = track_only
        self.snapshot_mutated = snapshot_mutated

        self.records = collections.deque()      # oldest first
        # Keyed on the tensor KEY, not the storage id.  A view shares its base's
        # storage, so a storage-keyed map lets `m.transpose(0,1)` overwrite the
        # entry for the mm that produced `m` and the walk can then never reach
        # it.  Keys carry the view digest, so base and view are distinct
        # producers and the path runs base -> view -> consumer the way it ran.
        self.produced_by = {}                   # tensor key -> record id
        self._sid = WeakIdKeyDictionary()       # storage object -> storage-id
        self._gen = {}                          # storage-id -> write generation
        self._next_sid = 0
        self._next_rec = 0
        self._held_bytes = 0
        self._vfront = 0                        # records[:_vfront] hold no values

    # -- identity -----------------------------------------------------------
    def sid(self, t):
        """A stable integer id for the storage `t` views, assigned on first sight.

        Ids are monotonic and never recycled: a storage that dies takes its entry
        with it (the map is weak), and a new storage reusing the same address
        gets a fresh id.  A view and its base share a storage and therefore an
        id, which is exactly what the walk wants -- consuming a view of X should
        find the op that produced X.
        """
        try:
            st = t.untyped_storage()
        except Exception:                              # noqa: BLE001
            return None
        s = self._sid.get(st)
        if s is None:
            s = self._next_sid
            self._next_sid += 1
            self._sid[st] = s
        return s

    def tkey(self, t):
        """Graph identity of one tensor VALUE: storage, generation, and view.

        The generation is ours, not torch's `_version`.  An in-place op bumps
        _version in VariableType, ABOVE __torch_dispatch__, so at the moment this
        mode sees `add_` return, the counter still reads its pre-call value and
        only reads the new one by the time the next op consumes the tensor.  A key
        built on _version therefore records the producer under one number and the
        consumer under another, and the edge silently fails to form -- which is
        precisely the in-place case the chain most needs to get right.  The
        tracker bumps its own counter in record(), after reading the inputs and
        before keying the outputs, so both sides always agree.

        The view digest keeps a tensor and a reshape of it distinct even though
        they share a storage: a replay has to re-create the exact size/stride its
        consumer read, and a view's producer is not its base's producer.
        """
        s = self.sid(t)
        if s is None:
            return None
        # The digest depends only on the view's metadata, and a model reuses a
        # few thousand shapes at most: memoise it rather than repr+hash per tensor.
        meta = (tuple(t.shape), tuple(t.stride()), t.storage_offset(), t.dtype)
        d = _DIGESTS.get(meta)
        if d is None:
            d = hashlib.blake2b(repr(meta[:3] + (str(meta[3]),)).encode(),
                                digest_size=5).hexdigest()
            if len(_DIGESTS) > (1 << 16):
                _DIGESTS.clear()
            _DIGESTS[meta] = d
        return "k%d_g%d_%s" % (s, self._gen.get(s, 0), d)

    def _refs(self, obj, keys=None):
        """`obj` with every tensor replaced by {_CHAIN_REF: key}.  `keys` maps
        id(tensor) -> key already computed for this op, to avoid keying twice."""
        if isinstance(obj, torch.Tensor):
            k = keys.get(id(obj)) if keys else None
            if k is None:
                k = self.tkey(obj)
            return {_CHAIN_REF: k} if k else None
        if isinstance(obj, tuple):
            return tuple(self._refs(o, keys) for o in obj)
        if isinstance(obj, list):
            return [self._refs(o, keys) for o in obj]
        if isinstance(obj, dict):
            return dict((k, self._refs(v, keys)) for k, v in obj.items())
        return obj

    # -- recording ----------------------------------------------------------
    def tracks(self, func):
        if self.track_only is None:
            return True
        return _matches(_op_names(func), self.track_only)

    def record(self, func, args, kwargs, out, pre, call_index, rng_pre=None):
        """One op's provenance.  `pre` is _snapshot_writes' pre-call copy, if any.

        `rng_pre` must be the generator state from BEFORE the op ran: this is
        called after func(), by which point a seeded op has already consumed its
        draw, and restoring the post-draw state at replay would reproduce a
        different random sequence than the one that failed.
        """
        rid = self._next_rec
        self._next_rec += 1

        # The operands as the kernel READ them: for a mutating op the live `args`
        # already hold its output, so prefer the pre-call snapshot -- the same
        # substitution _capture does, for the same reason.
        cap_args, cap_kwargs = args, kwargs
        if pre:
            if pre["args"]:
                cap_args = tuple(pre["args"].get(i, a) for i, a in enumerate(args))
            if pre["kwargs"]:
                cap_kwargs = dict(kwargs)
                cap_kwargs.update(pre["kwargs"])

        # Ordering below is load-bearing, and it is the whole in-place story:
        #   1. key the INPUTS at the current generation, off the LIVE args -- for
        #      `m.add_(1)` that is m's pre-call key, the one mm produced, so the
        #      edge to mm forms;
        #   2. bump the generation of everything this op wrote;
        #   3. key the OUTPUTS, which for an in-place op is the same storage one
        #      generation on, so the next consumer matches the writer and not the
        #      op before it.
        # The VALUE stored against an input key is the pre-call snapshot, not the
        # live operand, which by now holds the kernel's output.
        orig_t = _tensors((args, kwargs))
        snap_t = orig_t if not pre else _tensors((cap_args, cap_kwargs))
        in_keys = [self.tkey(t) for t in orig_t]
        keys = dict((id(t), k) for t, k in zip(orig_t, in_keys))
        args_ref = self._refs(args, keys)
        kwargs_ref = self._refs(kwargs, keys)
        held = list(zip(in_keys, snap_t if len(snap_t) == len(orig_t) else orig_t))

        w = op_writes(func)
        if w["any"]:
            written = [args[i] for i in w["idx"] if i < len(args)]
            written += [kwargs[n] for n in w["names"] if n in kwargs]
            for t in _tensors(written):
                s = self.sid(t)
                if s is not None:
                    self._gen[s] = self._gen.get(s, 0) + 1

        outs = _tensors(out)
        out_keys = [self.tkey(t) for t in outs]
        held += list(zip(out_keys, outs))

        rec = {
            "id": rid,
            "op": _op_names(func)[0],
            "func": func,
            "call_index": call_index,
            "is_view": is_view(func),
            "needs_rng": needs_rng(func),
            "breaks": breaks_chain(func),
            "in_keys": in_keys,
            "args_ref": args_ref,
            "kwargs_ref": kwargs_ref,
            "out_keys": out_keys,
            # Strong refs.  Dropped by _evict(), after which "values" is False and
            # the record still answers "who produced this key?".
            "args": cap_args,
            "kwargs": cap_kwargs,
            "outs": outs,
            "values": True,
            # (key, tensor-holding-that-key's-value).  torch's _version is kept
            # alongside purely as a tamper check: if it moves between here and
            # capture, something mutated a tensor we are holding and the recorded
            # value is no longer what this op saw.
            "held": held,
            "versions": [t._version for _, t in held],
            "bytes": sum(_storage_bytes(t) for _, t in held),
            # Pre-call, and only for ops that actually draw (torch tags those for
            # us).  A few bytes each, so recording per-op costs nothing and lets
            # the replay restore the generator immediately before each op that
            # needs it -- which is more faithful than restoring once at the top,
            # since the chain omits the non-chain seeded ops that ran in between.
            "rng": rng_pre,
        }
        self.records.append(rec)
        self._held_bytes += rec["bytes"]
        # An identity view (detach, alias) returns a tensor with the same storage,
        # generation, size, stride and offset as its input, so its output key IS
        # its input key.  Registering it as that key's producer would make the op
        # its own ancestor -- the walk stops at the self-loop and never reaches
        # the real producer, and the key is counted as "produced" so it is never
        # packed as a root either.  An op that handed the same value back did not
        # produce it.
        for k in rec["out_keys"]:
            if k is not None and k not in rec["in_keys"]:
                self.produced_by[k] = rid
        self._evict()

    @staticmethod
    def _rng_state():
        st = {"cpu": torch.get_rng_state()}
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            st["cuda"] = torch.cuda.get_rng_state()
        return st

    def _evict(self):
        """Drop values past the byte budget, then whole records past the op budget.

        Values go oldest-first, so every record before `_vfront` has already
        lost its values; start there instead of rescanning from 0 on every op
        (that rescan was ~7% of capture time in Lux profile 369551).
        """
        i = min(self._vfront, len(self.records))
        while self._held_bytes > self.value_bytes and i < len(self.records):
            rec = self.records[i]
            if rec["values"]:
                rec["args"] = rec["kwargs"] = rec["outs"] = None
                rec["held"] = rec["versions"] = None
                rec["values"] = False
                self._held_bytes -= rec["bytes"]
            i += 1
        self._vfront = i
        while len(self.records) > self.struct_ops:
            old = self.records.popleft()
            self._vfront = max(0, self._vfront - 1)
            if old["values"]:
                self._held_bytes -= old["bytes"]
            for k in old["out_keys"]:
                if self.produced_by.get(k) == old["id"]:
                    del self.produced_by[k]

    # -- the walk -----------------------------------------------------------
    def walk(self, terminal_rid):
        """(ordered records, {sid: stop reason}) for the ancestry of `terminal_rid`.

        Breadth-first over a DAG, not a path: addmm has three tensor parents and
        fan-in is the normal case, so this collects a subgraph and returns it in
        topological (record id) order.  Views are traversed without spending
        depth -- they compute nothing, so a replay has nothing to re-execute for
        them and they must not be what exhausts the budget.
        """
        by_id = dict((r["id"], r) for r in self.records)
        if terminal_rid not in by_id:
            return [], {}

        chosen = {terminal_rid: 0}
        stops = {}
        frontier = [(terminal_rid, 0)]
        while frontier:
            rid, d = frontier.pop(0)
            rec = by_id[rid]
            for key in rec["in_keys"]:
                if key is None:
                    continue
                pid = self.produced_by.get(key)
                if pid is None or pid not in by_id:
                    # Ordinary and expected: a model parameter, an activation
                    # older than the window, anything created before the capture
                    # armed.  It simply becomes a root.
                    stops[key] = "outside_window"
                    continue
                prod = by_id[pid]
                # A view costs nothing, so step through it at the same depth.
                nd = d if prod["is_view"] else d + 1
                if prod["breaks"]:
                    stops[key] = prod["breaks"]
                    continue
                if not prod["values"]:
                    stops[key] = "values_evicted"
                    continue
                if nd > self.depth:
                    stops[key] = "depth_reached"
                    continue
                if pid in chosen and chosen[pid] <= nd:
                    continue
                chosen[pid] = nd
                frontier.append((pid, nd))

        # Record ids are assigned in dispatch order, so sorting them IS the
        # topological order -- a producer always ran before its consumer.
        # A view that survived into the subgraph stays: a replay still has to
        # re-create the exact size/stride its consumer read.
        return [by_id[i] for i in sorted(chosen)], stops

    def versions_ok(self, rec):
        """False if anything this record holds was mutated after we recorded it.

        None when the values were evicted -- "we cannot tell" is a different
        answer from "it is fine", and the capsule reports it as such.
        """
        if not rec["values"]:
            return None
        return not any(self.moved(rec))

    @staticmethod
    def moved(rec):
        """Per held tensor: was it mutated between this record and now?"""
        return [t._version != ver
                for (_, t), ver in zip(rec["held"], rec["versions"])]


# --------------------------------------------------------------------------- #
# the dispatch mode
# --------------------------------------------------------------------------- #

class OpNanCapture(TorchDispatchMode):
    """Dump a replay capsule at the first non-finite op output, per rank.

    out           path; '{rank}' is substituted, or '.rankN' is appended
    record_data   False (default) = the capsule keeps each tensor's shape,
                  stride, offset and dtype plus a few statistics (tensor_stats)
                  but none of its values (nor a digest of them): safe to hand
                  over when the data is not yours to share; replayable only as
                  look-alikes (replay --synthesize) -- a NaN that depends on the
                  data needs the data.  Integer tensors (cu_seqlens, indices)
                  are recorded with their values either way.  True = the exact
                  operand bytes.
                  Detection is identical either way.
    max_bytes     per-tensor capture ceiling (guards against a view into a pool)
    check_inputs  also scan the inputs, so the capsule records whether this op
                  CREATED the NaN or merely propagated one
    raise_on_nan  'auto' (default) = raise immediately only when world_size == 1.
                  Under distributed, raising here deadlocks the other ranks in
                  RCCL -- use sync_stop() instead.  True/False force it.
    snapshot_mutated
                  copy an op's written arguments before the call, so a capsule
                  for an out-param kernel holds the operands it actually ran on.
                  Costs one device-side copy per considered mutating op and
                  nothing at all for aten's (non-mutating) matmul family, so the
                  bill is set by how wide the filter is.  False = pack the
                  post-call operands and flag it with args_are_pre_call=False.
    check_views   also check the outputs of pure view ops (default False: a view
                  aliases data already checked where it was produced; only a
                  tensor that was NaN before arming and is then only viewed
                  goes unseen).
    snapshot_max_bytes
                  skip the snapshot for any op whose written arguments total
                  more than this; that op's capsule records
                  args_are_pre_call=False, every other op keeps its pre-call
                  copy.  None (default) = no cap.  Exists because one optimizer
                  buffer -- fbgemm's rowwise-adagrad backward, 52.45 GiB on the
                  1st NaN -- is a copy no co-tenanted GPU can afford every step,
                  and an unfiltered capture otherwise had to give up snapshots
                  for every op to avoid it.

    The four filter options below each take a comma-separated string (a sequence
    works too), e.g. include_only_ops="aten.mm.default, aten.addmm.default" or
    exclude_op_classes="gemm".  Unset = no restriction.

    include_only_op_classes  only ops in these classes are considered
    include_only_ops         only these ops are considered
    exclude_op_classes       ops in these classes are never considered
    exclude_ops              these ops are never considered

    The two include_only_* sets are OR'd (an op passes if it is in either), then
    the exclude_* sets are subtracted -- exclusion always wins.  A name may be
    the full overload 'aten.addmm.default', the packet 'aten.addmm' (every
    overload of that op), or the bare namespace 'aten' / 'aiter' (every op that
    library registered).  The only op class today is 'gemm' (the matmul family).

    op_call_index in the capsule counts only the ops that passed the filter, so
    capsules are comparable across ranks only when the ranks used the same filter.

    Chain capture (all off by default; chain_depth=0 is the single-op capsule)

    chain_depth   how many COMPUTING ops of ancestry to record for the op that
                  failed.  Views cost nothing, so this counts real arithmetic
                  rather than detach/reshape hops.  Exists because
                  inputs_finite=True means "no NaN on the operands", NOT "the
                  operands were right" -- see the provenance note above
                  _ChainTracker.  2-4 is usually enough to tell those apart.
    chain_struct_ops
                  how many ops of metadata-only history to keep.  Cheap; the
                  default reaches back far enough to find the forward that
                  produced a backward op's input on a mid-sized model.
    chain_value_bytes
                  ceiling on the tensors held live for replay, evicted
                  oldest-first.  This is the knob that keeps chain capture
                  survivable on a real model, and the one that decides whether a
                  node comes back replayable or merely `values_evicted`.
    chain_store_intermediates
                  store every tensor on the chain, not just its roots, so a
                  replay can tell which op's output DIVERGED from the recording
                  rather than only which one went non-finite.  Costs capsule size.
    chain_track_only
                  restrict provenance tracking to these ops/namespaces, in the
                  same syntax as the filters above.  Leave it None unless
                  overhead forces your hand: tracking is a SEPARATE population
                  from NaN detection on purpose, and narrowing it punches holes
                  in the graph exactly where the walk needs an edge.

    Settle mode (off by default here; capture_from_env turns it on)

    settle        hold the first NaN an op creates as a pending candidate and
                  write it only if the NaN outlives its step (see "Settle mode"
                  in the module docstring).  Requires settle_step() once per
                  step -- _ArmedCapture.step() does it -- and finish() after the
                  loop.  Without settle_step() nothing settles: candidates pend
                  until survival or exit.
    settle_steps  extra steps a candidate waits for its NaN to surface before it
                  settles as MASKED.  1 (default) covers a NaN that reaches the
                  parameters through an optimizer outside the dispatcher: the
                  next forward reads it.
    """

    def __init__(self, out="nan_capsule.rank{rank}.pt", max_bytes=2 << 30,
                 check_inputs=True, raise_on_nan="auto", verbose=True,
                 include_only_op_classes=None, include_only_ops=None,
                 exclude_op_classes=None, exclude_ops=None,
                 snapshot_mutated=True, snapshot_max_bytes=None, check_views=False,
                 chain_depth=0, chain_struct_ops=4096,
                 chain_value_bytes=8 << 30, chain_store_intermediates=True,
                 chain_track_only=None, settle=False, settle_steps=1, check_batch=0,
                 late_norms=False, record_data=False, verify_every=0, bounds=False,
                 record_collectives=None):
        self.rank, self.world, self.local_rank, _ = rank_info()
        self.out = _rank_path(out, self.rank)
        _check_outfile(self.out)
        self.record_data = bool(record_data)
        self.max_bytes = max_bytes
        self.check_inputs = check_inputs
        self.verbose = verbose
        self.snapshot_mutated = snapshot_mutated
        self.snapshot_max_bytes = snapshot_max_bytes
        self.check_views = check_views
        self._snapshot_warned = False
        self._snapshot_skipped = set()          # (op, GiB) already reported over the cap

        inc_cls = _expand_op_classes(include_only_op_classes, "include_only_op_classes")
        inc_ops = _parse_name_list(include_only_ops, "include_only_ops")
        exc_cls = _expand_op_classes(exclude_op_classes, "exclude_op_classes")
        exc_ops = _parse_name_list(exclude_ops, "exclude_ops")
        self.include_ops = ((inc_cls or frozenset()) | (inc_ops or frozenset())
                            if (inc_cls or inc_ops) else None)
        self.exclude_ops = ((exc_cls or frozenset()) | (exc_ops or frozenset())
                            if (exc_cls or exc_ops) else None)
        if self.verbose and (self.include_ops or self.exclude_ops):
            print("[op-nan][rank %d] op filter: include=%s exclude=%s"
                  % (self.rank,
                     sorted(self.include_ops) if self.include_ops else "(all)",
                     sorted(self.exclude_ops) if self.exclude_ops else "(none)"),
                  flush=True)
        if self.verbose:
            print("[op-nan][rank %d] capsule data: %s"
                  % (self.rank, "values (replayable)" if self.record_data else
                     "shapes only, no floating-point values (integer tensors "
                     "kept; replay with --synthesize; NAN_CAPTURE_DATA=1 to "
                     "record all values)"), flush=True)
        if raise_on_nan == "auto":
            raise_on_nan = self.world <= 1
        self.raise_on_nan = raise_on_nan
        self.n = 0
        self.captured = False
        self._considers_cache = {}      # func -> considers(func); the filters never change
        self.check_batch = int(check_batch or 0)
        self.late_norms = bool(late_norms)
        self._queue = []            # deferred checks, oldest first (see _enqueue)
        self._qptrs = set()         # storages the queue holds
        self._ctx = None            # the queued op being handled, during _flush
        self.flushes = 0
        self.flush_why = {}         # reason -> count, for tuning check_batch

        self.settle = bool(settle)
        self.settle_steps = int(settle_steps)
        self._step = 0              # settle_step() calls so far
        self._born = set()          # storage addresses allocated this step
        # Uninitialized memory (settle mode): storages an allocator op returned and
        # nothing has written in full since, and storages whose only non-finite
        # values came from such memory.  FSDP2 all-gathers out of a padded buffer
        # from torch.empty that its copy-in writes only partly; the leftover bytes
        # can be NaN patterns, and the escape rule took them for a real NaN handed
        # to a collective (2026-11-09 torchtitan capture, ranks 2-7, step 0).
        self._raw = set()
        self._garbage = set()
        self._just_marked = ()      # garbage marked by the op being dispatched now
        # Cross-rank causality (see "cross-rank causality" above).
        self._coll_seq = {}         # group name -> collectives issued so far
        self._groups = {}           # group name -> member global ranks
        self._coll_out = WeakIdKeyDictionary() if WeakIdKeyDictionary else None
        self._first_arrival = None  # first NaN read from a collective's output
        # R3: collectives to record on every member rank, "group:seq[:step],..."
        self._rec_coll = _parse_record_collectives(record_collectives)
        self._emitted = None        # summary of the capsule written, for the manifest
        self._nan_step = False      # a non-finite output was seen this step
        self._pending = []          # [{site, step, capsule, origin}], oldest first
        self._sites = {}            # site -> census entry (see _census)
        self._masked_files = 0
        self._finished = False
        self._exiting = False
        if self.settle:
            atexit.register(self._at_exit)
            # A fork-started mp child leaves through os._exit after its target
            # returns, so atexit never runs there; its finalizers do (a spawn child
            # exits through sys.exit and runs both).  And a parent that terminate()s
            # its workers once one reports the NaN kills the reporter mid-write --
            # every capture trial of a 6-worker SDPA loop wrote nothing [E46].
            # _at_exit is idempotent.
            if multiprocessing.parent_process() is not None:
                multiprocessing.util.Finalize(None, self._at_exit, exitpriority=10)
                self._catch_terminate()
            if self.verbose:
                print("[op-nan][rank %d] settle mode: a NaN is written only if it "
                      "outlives its step (settles after %d extra step(s)); masked "
                      "NaNs are logged as MASKED" % (self.rank, self.settle_steps),
                      flush=True)

        # Wrong finite values (see "wrong FINITE values" above): the GEMM family only.
        self.verify_every = int(verify_every or 0)
        self.bounds = bool(bounds)
        self._vsites = {}           # (op, dtype, N) -> {calls, checked, errs, max_err}
        if self.verbose and (self.verify_every or self.bounds):
            print("[op-nan][rank %d] wrong-value checks on the GEMM family: bounds=%s "
                  "reference=%s" % (self.rank, self.bounds,
                                    "every %d call(s) per site" % self.verify_every
                                    if self.verify_every else "off"), flush=True)

        self.chain = None
        if chain_depth:
            if WeakIdKeyDictionary is None:
                raise RuntimeError(
                    "chain_depth=%d needs torch.utils.weak.WeakIdKeyDictionary, "
                    "absent in torch %s. Keying provenance on data_ptr() instead "
                    "would mis-attribute freed tensors, so this refuses rather "
                    "than capture a wrong chain." % (chain_depth, torch.__version__))
            self.chain = _ChainTracker(
                depth=chain_depth, struct_ops=chain_struct_ops,
                value_bytes=chain_value_bytes,
                store_intermediates=chain_store_intermediates,
                track_only=_parse_name_list(chain_track_only, "chain_track_only"),
                snapshot_mutated=snapshot_mutated)
            if self.verbose:
                print("[op-nan][rank %d] chain capture: depth=%d struct=%d ops "
                      "values<=%.1f GB intermediates=%s%s"
                      % (self.rank, chain_depth, chain_struct_ops,
                         chain_value_bytes / 1e9, chain_store_intermediates,
                         "" if chain_track_only is None
                         else "  tracking only %s (graph will have holes)"
                              % chain_track_only), flush=True)

    # -- op filter ----------------------------------------------------------
    def considers(self, func):
        """True if `func` is in the population this capture watches."""
        if self.include_ops is None and self.exclude_ops is None:
            return True
        c = self._considers_cache.get(func)
        if c is None:
            names = _op_names(func)        # (overload, packet, namespace)
            c = not ((self.include_ops is not None and not _matches(names, self.include_ops))
                     or (self.exclude_ops is not None and _matches(names, self.exclude_ops)))
            self._considers_cache[func] = c
        return c

    # -- the hot path -------------------------------------------------------
    # Higher-order operators (flex_attention, cond, while_loop, invoke_subgraph, ...)
    # refuse to run under a mode that does not declare support -- the 2026-11-09
    # torchtitan capture died on flex_attention.  Declared, they reach
    # __torch_dispatch__ with this mode already popped and run unchecked: their
    # bodies are invisible to the capture, like a compiled region.
    supports_higher_order_operators = True

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if isinstance(func, _HOP):
            return func(*args, **kwargs)
        sub = _subclass_kind(types)
        if sub == "nodata" or _has_nodata_arg(args, kwargs):
            return func(*args, **kwargs)
        if sub:
            return NotImplemented   # the subclass re-dispatches its local ops to us
        if self._queue:
            # Deferred checks: verify the queue before anything that could change
            # what a queued op saw (an in-place write into a tensor it holds), or
            # that needs the NaN state up to date (a collective).  See _enqueue.
            if _is_collective(func):
                self._flush("collective")
            elif op_writes(func)["any"] and self._touches_queue(func, args, kwargs):
                self._flush("inplace")
        settling = self.settle and not self.captured
        if _is_collective(func):
            # Settle mode: a NaN handed to another rank has escaped -- no local
            # where() can mask it any more.  Checked only once a NaN is about.
            # Only an op that talks to a process group sends anything: wait_tensor and
            # _wrap_tensor_autograd live in the same namespace but move no data.
            if settling and (self._pending or self._nan_step) \
                    and _collective_group(func, args, kwargs)[0] is not None:
                self._guarded(self._check_escape, func, args, kwargs, None, "collective")
            rec = self._record_collective_pre(func, args, kwargs) if self._rec_coll else None
            out = func(*args, **kwargs)
            if rec is not None:
                self._guarded(self._record_collective_post, rec, func, args, kwargs, out)
            if not self.captured and self._coll_out is not None:
                self._note_collective(func, args, kwargs, out)
            if settling and (self._raw or self._garbage):
                if _collective_group(func, args, kwargs)[0] is not None:
                    self._note_raw(func, args, kwargs, out, collective=True)
                else:
                    # wait/wrap: the result aliases what the collective wrote; its
                    # garbage mark (if any) carries over unchanged.
                    for t in _tensors(out):
                        if any(_storage_ptr(t) == _storage_ptr(a) for a in _tensors(args)):
                            continue
                        if any(_storage_ptr(a) in self._garbage for a in _tensors(args)):
                            self._garbage.add(_storage_ptr(t))
            return out
        watching = not self.captured and self.considers(func)
        # Tracking is a DIFFERENT population from detection, deliberately.  The
        # filters aim which ops get checked for a NaN; provenance has to see
        # every op or the graph is missing exactly the edge the walk needs -- an
        # include_only_op_classes="gemm" capture would otherwise record a chain
        # made solely of GEMMs and miss the softmax between them.
        tracking = (self.chain is not None and not self.captured
                    and self.chain.tracks(func))
        # Before the call: for an out-param kernel this is the last moment the
        # operands still hold what the kernel will actually read.  No-op (and no
        # copy) for every non-mutating op, which is all of aten's matmul family.
        pre = (self._snapshot_writes(func, args, kwargs)
               if (watching or tracking) else None)
        # Likewise the last moment the generator still holds what this op will
        # draw from.  Gated on the tag so the cost is paid only by seeded ops.
        rng_pre = (self.chain._rng_state()
                   if (tracking and needs_rng(func)) else None)
        out = func(*args, **kwargs)
        if tracking:
            self.chain.record(func, args, kwargs, out, pre, self.n + 1, rng_pre)
        if settling:
            self._note_born(func, args, kwargs, out)
        if (self._queue or (settling and (self._pending or self._nan_step))) \
                and _is_host_read(func, args, kwargs, out):
            self._flush("host")     # the read synchronised anyway; settle needs it current
            if settling and not self.captured and (self._pending or self._nan_step):
                self._guarded(self._check_escape, func, args, kwargs, out, "host")
        if self.captured:
            return out
        if watching:
            self.n += 1
            # A view (every return aliases an argument, nothing written) computes
            # nothing: its data was checked where it was produced or last
            # written.  Skipping it saves a check kernel per view op, which is
            # what the check costs under co-tenancy.  check_views=True restores
            # it (e.g. for a tensor that was already NaN before arming).
            if self.check_views or not is_view(func):
                cands = self._out_cands(func, args, kwargs, out)
                dev = cands[0][0].device if cands else None
                if (self.check_batch > 1 and dev is not None and dev.type != "cpu"
                        and all(t.device == dev for t, _ in cands)
                        and _norm_propagates_nan(dev)):
                    self._enqueue(func, args, kwargs, pre, cands, tracking)
                else:
                    # CPU (or mixed-device) outputs: checking them stalls no GPU.
                    bad, where = _first_bad(cands)
                    if bad is not None:
                        if self._queue:
                            self._flush()       # earlier ops first, in program order
                        if not self.captured:
                            self._guarded(self._settle_hit if self.settle else self._capture,
                                          func, args, kwargs, bad, pre, where)
        if (watching and (self.verify_every or self.bounds) and not self.captured
                and _op_names(func)[1] in _GEMM_OPS):
            self._guarded(self._verify, func, args, kwargs, out, pre)
        if settling:
            # After the check, which judged against the state BEFORE this op; what
            # the check itself just marked as garbage is this op's verdict, kept.
            self._note_raw(func, args, kwargs, out)
            self._just_marked = ()
        return out

    def _verify(self, func, args, kwargs, out, pre):
        """Bounds and sampled reference check of one GEMM-family call; captures the
        op at once on a wrong value (no settling: a wrong finite value is not
        something a later where() masks)."""
        outs = [t for t in _tensors(out) if _checkable(t)]
        if len(outs) != 1 or outs[0].dtype not in _REF_FLOOR:
            return
        c = outs[0]
        if not math.isfinite(float(_maxabs32(c))):
            return                  # non-finite: the NaN path owns it
        key = (str(func), str(c.dtype).replace("torch.", ""), tuple(c.shape[-1:]))
        s = self._vsites.get(key)
        if s is None:
            s = self._vsites[key] = {"calls": 0, "checked": 0, "errs": [], "max_err": 0.0}
        s["calls"] += 1
        hit = None
        if self.bounds:
            r = _gemm_bound_ratio(func, args, kwargs, c)
            if r is not None and not r <= 1.0 + _BOUND_SLACK[c.dtype]:
                hit = {"kind": "bound", "ratio": r, "slack": _BOUND_SLACK[c.dtype]}
        if hit is None and self.verify_every and (s["calls"] - 1) % self.verify_every == 0:
            err = _ref_rel_err(func, args, kwargs, c)
            if err is not None:
                s["checked"] += 1
                errs = s["errs"]
                med = sorted(errs)[len(errs) // 2] if len(errs) >= 16 else None
                floor = _ref_floor(c.dtype)
                thr = max(floor, 50.0 * med) if med else floor
                if not err <= thr:
                    hit = {"kind": "reference", "rel_err": err, "threshold": thr,
                           "site_median": med}
                else:
                    if len(errs) < 256:
                        errs.append(err)
                    s["max_err"] = max(s["max_err"], err)
        if hit is None:
            return
        if self._queue:
            self._flush()           # an earlier queued NaN comes first, in program order
            if self.captured:
                return
        hit.update(site="%s %s N=%s" % key, site_calls=s["calls"], site_checked=s["checked"],
                   site_max_ok_err=s["max_err"], step=getattr(self, "_step", None))
        if self.verbose:
            print("[op-nan][rank %d] WRONG VALUE at %s (call %d of the site): %s"
                  % (self.rank, hit["site"], s["calls"],
                     "bound ratio %.3g" % hit["ratio"] if hit["kind"] == "bound" else
                     "rel err %.3g > %.3g" % (hit["rel_err"], hit["threshold"])), flush=True)
        origin = {"class": "wrong_value", "nonfinite": 0, "math": None, "overflow": None,
                  "unexplained": None,
                  "note": "finite output failed the %s check" % hit["kind"]}
        self._capture(func, args, kwargs, c, pre, "return",
                      extra={"verify": hit, "nan_origin": origin})

    def _note_raw(self, func, args, kwargs, out, collective=False):
        """Keep _raw/_garbage current (settle mode; see __init__).

        An allocator op's outputs become raw.  A write that spans a whole storage --
        or any write by a collective, whose data is real -- makes it initialized.
        A non-view op returning a tensor in a storage none of its inputs held made a
        fresh allocation at that address: whatever was marked there is gone.
        """
        raw, garbage = self._raw, self._garbage
        if _op_names(func)[0] in _ALLOC_OPS:
            for t in _tensors(out):
                p = _storage_ptr(t)
                if p is not None:
                    raw.add(p)
                    garbage.discard(p)
            return
        if not (raw or garbage) or is_view(func):
            return
        w = op_writes(func)
        written = []
        if w["any"]:
            for i in w["idx"]:
                if i < len(args):
                    written.extend(_tensors(args[i]))
            for nm in w["names"]:
                written.extend(_tensors(kwargs.get(nm)))
        ins = set(_storage_ptr(t) for t in _tensors((args, kwargs)))
        if collective and garbage and ins & garbage:
            # Communication of garbage is garbage: FSDP2's all-gather hands every
            # rank's padding to every rank.  A real NaN in another rank's shard is
            # caught on THAT rank (a persistent read), so nothing is lost here.
            for t in written + list(_tensors(out)):
                p = _storage_ptr(t)
                if p is not None:
                    garbage.add(p)
            return
        for t in written:
            if collective or _covers_storage(t):
                p = _storage_ptr(t)
                raw.discard(p)
                garbage.discard(p)
        keep = self._just_marked
        for t in _tensors(out):
            p = _storage_ptr(t)
            if (collective or p not in ins) and p not in keep:
                raw.discard(p)
                garbage.discard(p)

    def _note_collective(self, func, args, kwargs, out):
        """Count this collective in its group and remember the storages it wrote."""
        name, pg = _collective_group(func, args, kwargs)
        if name is None:
            return                  # wait_tensor & co: the storage was noted at issue
        if name not in self._groups:
            self._groups[name] = _group_ranks(pg) if pg is not None else None
        seq = self._coll_seq[name] = self._coll_seq.get(name, 0) + 1
        info = {"group": name, "seq": seq, "op": str(func), "step": self._step}
        w = op_writes(func)
        written = list(_tensors(out))
        for i in w["idx"]:
            if i < len(args):
                written.extend(_tensors(args[i]))
        for nm in w["names"]:
            written.extend(_tensors(kwargs.get(nm)))
        for t in written:
            try:
                self._coll_out[t.untyped_storage()] = info
            except Exception:                           # noqa: BLE001
                pass

    # -- R3: record one collective on every member rank --------------------------
    def _record_collective_pre(self, func, args, kwargs):
        """If this call is a requested (group, seq[, step]): CPU copies of its inputs."""
        name, pg = _collective_group(func, args, kwargs)
        if name is None:
            return None
        seq = self._coll_seq.get(name, 0) + 1
        want = [r for r in self._rec_coll if r[0] == name and r[1] == seq
                and (r[2] is None or r[2] == self._step)]
        if not want:
            return None
        with _disable_current_modes():
            return {"op": str(func), "group": name, "seq": seq, "step": self._step,
                    "group_ranks": _group_ranks(pg) if pg is not None else None,
                    "rank": self.rank, "args_pre": _cpu_copy(args),
                    "kwargs_pre": _cpu_copy(kwargs)}

    def _record_collective_post(self, rec, func, args, kwargs, out):
        """Wait for the collective, copy its outputs and written arguments, save
        <capsule dir>/collective_g<group>_s<seq>.rank<r>.pt (collective_r3.py reads it)."""
        with _disable_current_modes():
            for x in (out if isinstance(out, (tuple, list)) else [out]):
                if isinstance(x, torch.ScriptObject) and hasattr(x, "wait"):
                    x.wait()                            # c10d Work: the data has to land
            if _op_names(func)[2] != "c10d":
                for t in _tensors(out):
                    torch.ops._c10d_functional.wait_tensor(t)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            rec["out_post"] = _cpu_copy(out)
            rec["args_post"] = _cpu_copy(args)
            d = os.path.dirname(self.out)
            path = os.path.join(d, "collective_g%s_s%d.rank%d.pt" % (rec["group"], rec["seq"], self.rank))
            torch.save(rec, path)
            print("[op-nan][rank %d] recorded collective %s #%d of group %s -> %s"
                  % (self.rank, rec["op"], rec["seq"], rec["group"], path), flush=True)

    def _arrived_via(self, tensors):
        """The collective that last wrote the storage of any of `tensors`, or None."""
        if not self._coll_out:
            return None
        for t in tensors:
            try:
                info = self._coll_out.get(t.untyped_storage())
            except Exception:                           # noqa: BLE001
                info = None
            if info is not None:
                return info
        return None

    def _note_arrival(self, func, dirty):
        """First NaN this rank read from a collective's output: record it once."""
        if self._first_arrival is None and dirty:
            arr = self._arrived_via(dirty)
            if arr is not None:
                self._first_arrival = dict(arr, read_by=str(func), read_at_step=self._step)

    def _uninit(self, t):
        """True if `t` lives in raw or garbage storage -- as of the op being judged
        (a deferred check carries the membership it had when its op ran)."""
        p = _storage_ptr(t)
        if p is None:
            return False
        if self._ctx is not None:
            snap = self._ctx.get("uninit")
            return snap is not None and p in snap
        return p in self._raw or p in self._garbage

    # -- deferred checks (check_batch > 1) ----------------------------------
    def _enqueue(self, func, args, kwargs, pre, cands, tracking):
        """Queue this op's check instead of reading it back now.

        The read-back (.tolist()) is what made the check expensive: it waits for
        the GPU after every watched op, so the CPU can never run ahead and every
        op's Python cost lands on the critical path.  Queued, the max-|value|
        reductions stay on the device and one read covers check_batch ops.
        What the capture needs from the moment the op ran is kept with it: its
        call index, chain record, Python frames and autograd node.  Its operands
        are held by reference, and _touches_queue flushes before any in-place
        op writes into one, so they still hold what the op saw.
        """
        e = {"func": func, "args": args, "kwargs": kwargs, "pre": pre,
             "cands": cands,
             # late_norms: computed at flush, batched (see _flush); else now, per op
             "norms": None if self.late_norms else [_maxabs32(t).reshape(1)
                                                     for t, _ in cands],
             "n": self.n,
             "rid": self.chain._next_rec - 1 if tracking else None,
             "frames": _grab_frames(sys._getframe(1)),
             "node": _node_name() if self.settle else None,
             "uninit": None}
        if self.settle and (self._raw or self._garbage):
            u = self._raw | self._garbage
            e["uninit"] = frozenset(p for p in (_storage_ptr(t) for t in
                                                _tensors((args, kwargs, [c for c, _ in cands])))
                                    if p in u)
        self._queue.append(e)
        qp = self._qptrs
        for t in _tensors((args, kwargs)):
            qp.add(_storage_ptr(t))
        for t, _ in cands:
            qp.add(_storage_ptr(t))
        if len(self._queue) >= self.check_batch:
            self._flush("full")

    def _touches_queue(self, func, args, kwargs):
        """True if `func` writes into a storage a queued op holds."""
        w = op_writes(func)
        qp = self._qptrs
        for i in w["idx"]:
            if i < len(args):
                for t in _tensors(args[i]):
                    if _storage_ptr(t) in qp:
                        return True
        for nm in w["names"]:
            for t in _tensors(kwargs.get(nm)):
                if _storage_ptr(t) in qp:
                    return True
        return False

    def flush(self):
        """Verify every queued check now (deferred mode); no-op otherwise.  Called
        at each step boundary by _ArmedCapture.step() and by sync_stop()."""
        self._flush()

    def _flush(self, why="step"):
        q = self._queue
        if not q:
            return
        self._queue, self._qptrs = [], set()
        self.flushes += 1
        self.flush_why[why] = self.flush_why.get(why, 0) + 1
        with _disable_current_modes():
            if self.late_norms:
                self._late_norms(q)
            # One read per device (a process may drive several GPUs).
            by_dev = {}
            for e in q:
                for n in e["norms"]:
                    by_dev.setdefault(n.device, []).append(n)
            read = {}
            for d, ns in by_dev.items():
                read[d] = iter((ns[0] if len(ns) == 1 else torch.cat(ns)).tolist())
            for e in q:
                vs = [next(read[n.device]) for n in e["norms"]]
                if self.captured:
                    break
                for (t, where), v in zip(e["cands"], vs):
                    if not math.isfinite(v):
                        self._ctx = e
                        try:
                            self._guarded(self._settle_hit if self.settle else self._capture,
                                          e["func"], e["args"], e["kwargs"], t, e["pre"], where)
                        finally:
                            self._ctx = None
                        break

    @staticmethod
    def _late_norms(q):
        """Fill every entry's "norms" now: one _foreach_norm per (device, dtype)
        instead of one vector_norm launch per op output (~23x cheaper for 64
        small tensors on gfx1250).  Valid only because nothing has written into
        a queued tensor since its op ran -- _touches_queue flushes first."""
        groups = {}
        for e in q:
            e["norms"] = [None] * len(e["cands"])
            for j, (t, _) in enumerate(e["cands"]):
                if t.numel() > _SCAN_CHUNK_ELEMS or _needs_promote(t.dtype):
                    e["norms"][j] = _maxabs32(t).reshape(1)      # chunked / fp8 path
                else:
                    groups.setdefault((t.device, t.dtype), []).append((e, j, t))
        for items in groups.values():
            ts = [t for _, _, t in items]
            res = torch._foreach_norm(ts, float("inf"), dtype=torch.float32)
            flat = torch.stack(res)
            for k, (e, j, _) in enumerate(items):
                e["norms"][j] = flat[k:k + 1]

    def __enter__(self):
        # Processes on the device as the capture arms, while every sibling is still
        # alive.  The scan in contention_info() runs when the capsule is written,
        # and by then a sibling that NaN'd or finished may have exited: an mp.spawn
        # trio has recorded gpu_procs=2.  capture_gpu_procs() takes the larger.
        if getattr(self, "_gpu_procs_at_arm", None) is None:
            self._gpu_procs_at_arm = _gpu_procs()
        return super().__enter__()

    def __exit__(self, exc_type, exc, tb):
        r = super().__exit__(exc_type, exc, tb)
        if self._queue and not self.captured:
            if exc_type is None:
                self._flush()
            else:
                try:
                    self._flush()
                except Exception:                       # noqa: BLE001
                    pass                                # do not mask the exception in flight
        return r

    def _guarded(self, fn, *a):
        """Call a capture-side handler.  Multi-rank: never let a capture-side
        failure (e.g. a save error) unwind THIS rank alone -- the others would
        wait in the next collective forever.  When a capsule was being written,
        captured is already set, so the next sync_stop() stops every rank
        together."""
        if self.world <= 1:
            return fn(*a)
        try:
            return fn(*a)
        except Exception as e:                  # noqa: BLE001
            print("[op-nan][rank %d] !! %s failed (%s: %s)%s"
                  % (self.rank, fn.__name__, type(e).__name__, str(e)[:200],
                     "; capsule NOT written, stopping all ranks at the next "
                     "sync_stop()" if self.captured else ""), flush=True)

    # -- finding the output -------------------------------------------------
    def _first_nonfinite(self, func, args, kwargs, out):
        """(tensor, where) for the first non-finite output, or (None, None)."""
        return _first_bad(self._out_cands(func, args, kwargs, out))

    def _out_cands(self, func, args, kwargs, out):
        """[(tensor, where)]: every output to check, in checking order.

        'Output' is not just the return value: an out-param op writes its result
        into an argument and may return None, so the written arguments count too.
        """
        # Same order as before -- returns, then written args by index, then
        # written kwargs by name -- but gathered first and checked in one go
        # (_first_bad).  A tensor seen twice (an in-place op returns self) is
        # checked once, under its first label.
        cands, seen = [], set()

        def add(t, where):
            if _checkable(t) and id(t) not in seen:
                seen.add(id(t))
                cands.append((t, where))

        for t in _tensors(out):
            add(t, "return")
        w = op_writes(func)
        if w["any"]:
            # tree_leaves, not a bare isinstance: a written argument may be a
            # TensorList (every fused optimizer writes that way).
            for i in sorted(w["idx"]):
                if i < len(args):
                    for t in _tensors(args[i]):
                        add(t, "arg%d" % i)
            for nm in sorted(w["names"]):
                for t in _tensors(kwargs.get(nm)):
                    add(t, "kwarg:%s" % nm)
        return cands

    # -- pre-call snapshot of mutated arguments ------------------------------
    def _snapshot_writes(self, func, args, kwargs):
        """Device-side copies of the arguments `func` is about to overwrite.

        None when the op writes nothing -- the aten GEMM path, which stays
        copy-free.  Otherwise {'args': {index: tensor}, 'kwargs': {name: tensor}}.
        """
        if not self.snapshot_mutated:
            return None
        w = op_writes(func)
        if not w["any"]:
            return None
        if self.snapshot_max_bytes is not None:
            # All or nothing per op: a partial snapshot would let the capsule
            # claim args_are_pre_call=True while one operand is post-call.
            need = (sum(_snap_bytes(args[i]) for i in w["idx"] if i < len(args))
                    + sum(_snap_bytes(kwargs[n]) for n in w["names"] if n in kwargs))
            if need > self.snapshot_max_bytes:
                # Keyed on (op, size), not op alone: one op name can be called on
                # very different buffers -- fbgemm's TBE wrapper runs once per
                # table group -- and reporting only the first call hid a 52 GiB
                # one behind a 4.81 GiB one.
                key = (str(func), round(need / 2**30, 2))
                if self.verbose and key not in self._snapshot_skipped:
                    self._snapshot_skipped.add(key)
                    print("[op-nan][rank %d] pre-call snapshot of %s skipped: "
                          "%.2f GiB written > snapshot_max_bytes %.2f GiB; its "
                          "capsules record args_are_pre_call=False"
                          % (self.rank, key[0], need / 2**30,
                             self.snapshot_max_bytes / 2**30), flush=True)
                    for i in w["idx"]:
                        if i < len(args):
                            for line in _snap_breakdown("arg%d" % i, args[i]):
                                print("[op-nan][rank %d]     %s" % (self.rank, line), flush=True)
                    for n in w["names"]:
                        if n in kwargs:
                            for line in _snap_breakdown(n, kwargs[n]):
                                print("[op-nan][rank %d]     %s" % (self.rank, line), flush=True)
                return None
        snap_a, snap_k = {}, {}
        try:
            for i in w["idx"]:
                if i < len(args):
                    c = _snap_value(args[i])
                    if c is not None:
                        snap_a[i] = c
            for nm in w["names"]:
                if nm in kwargs:
                    c = _snap_value(kwargs[nm])
                    if c is not None:
                        snap_k[nm] = c
        except Exception as e:          # noqa: BLE001
            # A failed snapshot (OOM, exotic layout) must not take the run down
            # with it -- degrade to a post-call capsule, which says so.
            if self.verbose and not self._snapshot_warned:
                self._snapshot_warned = True
                print("[op-nan][rank %d] pre-call snapshot of %s failed (%s: %s); "
                      "capsules for mutating ops will record args_are_pre_call=False"
                      % (self.rank, func, type(e).__name__, e), flush=True)
            return None
        return {"args": snap_a, "kwargs": snap_k} if (snap_a or snap_k) else None

    # -- cross-rank replica check (T5) ---------------------------------------
    def watch_replicas(self, params, every=1):
        """Compare, every `every` steps at sync_stop(), the parameters that must be
        identical across ranks: a DTensor along each mesh dimension where it is
        Replicate (TP, HSDP replicas), a plain tensor across the whole job (DDP).

        The check is a parallelism-bug detector, not a NaN one: every op can be
        correct while the combination is wrong (a dropped or doubled gradient
        reduction -- torchtitan #4054), and replicas then drift apart.  Digest per
        tensor: fp64 sum and abs-sum of the local data (bit-identical replicas give
        bit-identical digests).  A mismatch stops every rank and the manifest says
        PARALLELISM, naming the tensor, the group and the step."""
        self._replicas = [(n, p) for n, p in (params.items() if isinstance(params, dict)
                                              else enumerate(params))]
        self._replicas_every = max(1, int(every))
        self._replica_divergence = None

    def _replica_groups(self):
        """[(group key, process group, [(name, local tensor)])] for watched params."""
        import torch.distributed as dist
        groups = {}
        for name, p in self._replicas:
            t = p.data if hasattr(p, "data") else p
            spec = getattr(t, "placements", None)
            if spec is not None:                        # DTensor
                mesh = t.device_mesh
                local = t.to_local()
                for d, pl in enumerate(spec):
                    if pl.is_replicate() and mesh.size(d) > 1:
                        g = mesh.get_group(d)
                        groups.setdefault(("mesh", id(g)), (g, []))[1].append((str(name), local))
            elif self.world > 1:
                groups.setdefault(("world", 0), (dist.group.WORLD, []))[1].append((str(name), t))
        return [(k, g, items) for k, (g, items) in groups.items()]

    def _check_replicas(self):
        """T5 at a step boundary; sets captured + _replica_divergence on a mismatch."""
        import torch.distributed as dist
        if not getattr(self, "_replicas", None) or self.captured \
                or self._step % self._replicas_every:
            return
        with _disable_current_modes(), torch.no_grad():
            for key, g, items in self._replica_groups():
                dig = torch.stack([torch.stack([t.double().sum(), t.double().abs().sum()])
                                   for _, t in items])
                allg = [torch.empty_like(dig) for _ in range(dist.get_world_size(g))]
                dist.all_gather(allg, dig, group=g)
                allg = torch.stack(allg)                # (ranks, tensors, 2)
                bad = (allg != allg[0:1]).any(dim=2).any(dim=0)
                if bool(bad.any()):
                    i = int(bad.nonzero()[0])
                    ranks = dist.get_process_group_ranks(g)
                    self._replica_divergence = {
                        "tensor": items[i][0], "group_ranks": ranks, "step": self._step,
                        "digests": {r: allg[j, i].tolist() for j, r in enumerate(ranks)}}
                    self.captured = True
                    print("[op-nan][rank %d] REPLICAS DIVERGED: %s across ranks %s at step %d"
                          % (self.rank, items[i][0], ranks, self._step), flush=True)
                    return

    # -- coordinated stop ---------------------------------------------------
    def sync_stop(self, group=None):
        """Raise on ALL ranks if ANY rank captured.  Call once per step, on every
        rank, outside any other collective.  No-op in a single-process run."""
        self._flush()               # deferred checks first: a queued NaN must count
        if self.world > 1:
            self._check_replicas()
        if self.world <= 1:
            if self.captured and self.raise_on_nan:
                raise RuntimeError("[op-nan] captured to %s" % self.out)
            return self.captured

        import torch.distributed as dist
        if not (dist.is_available() and dist.is_initialized()):
            return self.captured

        dev = (torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available()
               else torch.device("cpu"))           # gloo / CPU-only runs (the CPU tests)
        flag = torch.tensor([1 if self.captured else 0], dtype=torch.int32, device=dev)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=group)
        if flag.item() == 0:
            return False

        # Every rank learns about it at the same point, so every rank can unwind.
        who = torch.zeros(self.world, dtype=torch.int32, device=dev)
        who[self.rank] = 1 if self.captured else 0
        dist.all_reduce(who, op=dist.ReduceOp.MAX, group=group)
        hits = [i for i, v in enumerate(who.tolist()) if v]
        if self.rank == 0:
            print("[op-nan] ranks that captured: %s of %d" % (hits, self.world), flush=True)
        self._write_manifest(group)
        raise RuntimeError(
            "[op-nan] rank %d stopping; capturing ranks=%s; my capsule=%s"
            % (self.rank, hits, self.out if self.captured else "(none)"))

    def _rank_summary(self):
        """This rank's part of the job manifest: small, JSON-able."""
        return {"rank": self.rank, "local_rank": self.local_rank, "world_size": self.world,
                "host": os.uname().nodename, "step": self._step,
                "captured": bool(self.captured), "capsule": self._emitted,
                # A real NaN: allocator garbage and 0/0 sites (soft origins) are not one.
                "nan_seen": any(s.get("origin") not in _SOFT_ORIGINS for s in self._sites.values()),
                "first_arrival": self._first_arrival,
                "replica_divergence": getattr(self, "_replica_divergence", None),
                "coll_seq": dict(self._coll_seq), "groups": dict(self._groups),
                "job": os.environ.get("SLURM_JOB_ID") or os.environ.get("TORCHELASTIC_RUN_ID")}

    @staticmethod
    def job_verdict(ranks):
        """origin ranks + a sentence, from every rank's summary (see the manifest).

        Origin: a rank whose capsule's NaN did not arrive through a collective
        (inputs finite or its own op made it); else, failing that, a rank that saw a
        NaN and never read one from a collective.  Ranks whose first NaN arrived are
        listed with the (group, seq) it came through: the origin is upstream of the
        EARLIEST such collective, among that group's members."""
        made = [r["rank"] for r in ranks if r.get("captured") and r.get("capsule")
                and not (r["capsule"] or {}).get("arrived_via")
                and (r["capsule"] or {}).get("inputs_finite") is not False]
        if not made:
            # Evidence of a NaN only: a rank stopped by the replica check captured
            # nothing and made nothing.
            made = [r["rank"] for r in ranks if (r.get("capsule") or r.get("nan_seen"))
                    and not r.get("first_arrival")]
        arrived = sorted(((r["first_arrival"]["step"], r["first_arrival"]["group"],
                           r["first_arrival"]["seq"], r["rank"])
                          for r in ranks if r.get("first_arrival")))
        div = next((r["replica_divergence"] for r in ranks if r.get("replica_divergence")), None)
        if div and not made:
            return {"origin_ranks": [], "parallelism": div,
                    "arrivals": [{"step": a[0], "group": a[1], "seq": a[2], "rank": a[3]}
                                 for a in arrived],
                    "note": "PARALLELISM: replicas of %s diverged across ranks %s at step %s "
                            "with no NaN made by any op -- a reduction or placement bug, not "
                            "a kernel (step 5N, multi-rank)" % (div["tensor"], div["group_ranks"],
                                                                 div["step"])}
        if made:
            note = "origin: rank(s) %s made the NaN locally" % made
        elif arrived:
            st, g, seq, _ = arrived[0]
            note = ("no rank made it locally in view of the capture; earliest arrival: "
                    "collective #%d of group %s at step %d -- the origin is a member of "
                    "that group that did not capture (an op the capture cannot see, or the "
                    "collective itself)" % (seq, g, st))
        else:
            note = "no NaN seen by any rank's capture"
        return {"origin_ranks": made,
                "arrivals": [{"step": a[0], "group": a[1], "seq": a[2], "rank": a[3]}
                             for a in arrived],
                "note": note}

    def _write_manifest(self, group=None):
        """Pool every rank's summary at the coordinated stop; rank 0 writes
        <capsule dir>/manifest[.<n>].json.  Never raises."""
        import json
        import torch.distributed as dist
        try:
            ranks = [None] * self.world
            with _disable_current_modes():
                dist.all_gather_object(ranks, self._rank_summary(), group=group)
        except Exception as e:                          # noqa: BLE001
            print("[op-nan][rank %d] !! manifest gather failed (%s: %s)"
                  % (self.rank, type(e).__name__, str(e)[:200]), flush=True)
            return
        if self.rank != 0:
            return
        try:
            d = os.path.dirname(self.out)
            path, n = os.path.join(d, "manifest.json"), 1
            while os.path.exists(path):
                path, n = os.path.join(d, "manifest.%d.json" % n), n + 1
            doc = {"format": 1, "world_size": self.world, "ranks": ranks,
                   "verdict": self.job_verdict(ranks)}
            with open(path + ".tmp", "w") as fh:
                json.dump(doc, fh, indent=2, default=str)
            os.replace(path + ".tmp", path)
            print("[op-nan] job manifest -> %s: %s" % (path, doc["verdict"]["note"]), flush=True)
        except Exception as e:                          # noqa: BLE001
            print("[op-nan] !! manifest write failed (%s: %s)" % (type(e).__name__, e), flush=True)

    # -- chain --------------------------------------------------------------
    def _chain_payload(self, terminal_rid=None):
        """The format-3 keys, or None when there is no chain to report.

        Everything here runs exactly once, after the NaN is already in hand, so
        it is allowed to be the expensive part.  What it must not do is fail: the
        single-op capsule is already worth writing, so every per-tensor pack is
        individually guarded and a tensor too big for max_bytes degrades that one
        node to `too_big` rather than costing the whole capture.
        """
        tr = self.chain
        if tr is None or not tr.records:
            return None
        if terminal_rid is None:
            terminal = tr.records[-1]
        else:       # a deferred check: later ops have been recorded since
            terminal = next((r for r in reversed(tr.records) if r["id"] == terminal_rid), None)
            if terminal is None:
                return None
        ordered, stops = tr.walk(terminal["id"])
        if not ordered:
            return None

        # Same exclusion as produced_by: a key an op merely passed through is not
        # produced by the chain, so it stays a root and gets packed by value.
        produced = set()
        for rec in ordered:
            produced.update(k for k in rec["out_keys"]
                            if k and k not in rec["in_keys"])

        pool, rng, chain = {}, None, []
        for rec in ordered:
            ok = tr.versions_ok(rec)
            moved = tr.moved(rec) if rec["values"] else []
            # A key consumed but produced by nobody in the chain is a root: the
            # replay cannot recompute it, so it must be packed by value.  An
            # intermediate is packed only to let the replay detect divergence.
            for (key, t), was_moved in zip(rec["held"] or [], moved):
                if not key or key in pool:
                    continue
                if was_moved:
                    # This tensor no longer holds what this op saw -- something
                    # mutated it in place afterwards.  Packing it anyway would
                    # make the replay report a DIVERGED that is an artefact of
                    # the recording, not of the kernel.  Skipping it is also how
                    # the right value gets in: the mutating op's own pre-call
                    # snapshot carries the same key, is not itself mutated, and
                    # is reached later in this loop.
                    stops[key] = "version_moved"
                    continue
                want = (key not in produced) or tr.store_intermediates
                if not want:
                    continue
                try:
                    pool[key] = _pack_tensor(t, self.max_bytes, self.record_data)
                    # Only clear the mark THIS loop set.  The walk's own reasons
                    # (depth_reached, values_evicted) explain why we did not go
                    # further UPSTREAM of this key, which stays true even though
                    # the key itself got packed as a root.
                    if stops.get(key) == "version_moved":
                        del stops[key]
                except Exception as e:                 # noqa: BLE001
                    stops[key] = "too_big"
                    if self.verbose:
                        print("[op-nan][rank %d] chain: not packing %s (%s)"
                              % (self.rank, key, e), flush=True)
            if rec["rng"] and rng is None:
                rng = rec["rng"]                       # oldest chain node wins
            chain.append({
                "op": rec["op"],
                "args": rec["args_ref"],
                "kwargs": rec["kwargs_ref"],
                "produces": rec["out_keys"],
                "call_index": rec["call_index"],
                "is_view": rec["is_view"],
                "needs_rng": rec["needs_rng"],
                "rng": rec["rng"],
                "version_ok": ok,
            })

        roots = sorted(k for k in pool if k not in produced)
        return {
            "chain": chain,
            "pool": pool,
            "roots": roots,
            "rng": rng,
            # Never absent.  A chain that stopped early must not read like one
            # that reached a root, so every frontier that did not expand says why.
            "truncation": dict(stops),
            "config": {
                "depth": tr.depth, "struct_ops": tr.struct_ops,
                "value_bytes": tr.value_bytes,
                "store_intermediates": tr.store_intermediates,
                "track_only": sorted(tr.track_only) if tr.track_only else None,
            },
        }

    # -- capture ------------------------------------------------------------
    def _capture(self, func, args, kwargs, bad, pre=None, where="return", extra=None):
        self.captured = True        # before the dump: a failure in here must not re-enter
        capsule, cap_args, cap_kwargs = self._build_capsule(func, args, kwargs, bad, pre, where)
        if self.settle:
            capsule["nan_origin"] = self._origin_for(func, cap_args, cap_kwargs, bad,
                                                     where, capsule["inputs_finite"], pre)
        if extra:
            capsule.update(extra)
        self._emit(capsule)
        if self.raise_on_nan:
            raise RuntimeError("%s non-finite op output captured to %s"
                               % ("[op-nan]", self.out))

    @staticmethod
    def _pre_call_operands(args, kwargs, pre):
        # `args` holds whatever the operands look like NOW, which for a mutating
        # op is after the kernel overwrote them.  Pack the pre-call snapshot
        # instead, or the capsule replays a different problem than the one that
        # failed.  `bad` stays post-call -- it is the output we are reporting.
        cap_args, cap_kwargs = args, kwargs
        if pre:
            if pre["args"]:
                cap_args = tuple(pre["args"].get(i, a) for i, a in enumerate(args))
            if pre["kwargs"]:
                cap_kwargs = dict(kwargs)
                cap_kwargs.update(pre["kwargs"])
        return cap_args, cap_kwargs

    @staticmethod
    def _origin_for(func, cap_args, cap_kwargs, bad, where, inputs_finite, pre):
        """_nan_origin, except for a mutating op without a pre-call snapshot,
        whose operands are already the kernel's output."""
        if op_writes(func)["any"] and not pre:
            return {"class": "no_reference", "nonfinite": None, "math": None,
                    "overflow": None, "unexplained": None,
                    "note": "mutating op without a pre-call snapshot"}
        return _nan_origin(func, cap_args, cap_kwargs, bad, where, inputs_finite)

    def _build_capsule(self, func, args, kwargs, bad, pre=None, where="return"):
        """(capsule, cap_args, cap_kwargs): the capsule for this op, built now --
        the only moment its operands exist -- but not written."""
        w = op_writes(func)
        cap_args, cap_kwargs = self._pre_call_operands(args, kwargs, pre)

        inputs_finite = None
        nonfinite_operands = None
        if self.check_inputs:
            # Three populations, because "written" and "is an input" are NOT the
            # same question and the schema cannot answer the second one:
            #
            #   out-buffer   written, kwarg_only.  Definitively not an input;
            #                pre-call it is allocator garbage.  Excluded.
            #   read-only    no write alias at all.  Definitively an input, and
            #                the ONLY population that can prove `False`.
            #   written      has a write alias but is positional.  AMBIGUOUS: an
            #                in-place `self` is read, a positional out-param like
            #                flash_attn's `dq` is not, and `Tensor(a!)` looks
            #                identical in both.  Most fused-library kernels take
            #                their outputs positionally.
            #
            # Scoring the ambiguous population as `False` was a bug: it reported
            # "the NaN arrived on the inputs" for every positional out-param op,
            # whose buffers come straight from empty_like and are routinely full
            # of NaN before the kernel has done anything.  That reading sends you
            # hunting an upstream producer that does not exist.  `None` -- which
            # already means "cannot honestly tell" here -- is the correct verdict,
            # and `nonfinite_operands` below says exactly which operand forced it.
            skip_idx, skip_names = set(w["outbuf_idx"]), set(w["outbuf_names"])
            unknown = False
            if w["any"] and not pre:
                # No snapshot: the written operands now hold the kernel's OWN
                # output and say nothing about what went in.
                skip_idx |= set(w["idx"])
                skip_names |= set(w["names"])
                unknown = bool(w["idx"] - w["outbuf_idx"])

            read_only, written = [], []
            for i, a in enumerate(cap_args):
                if i in skip_idx:
                    continue
                (written if i in w["idx"] else read_only).append(("arg%d" % i, a))
            for k, v in cap_kwargs.items():
                if k in skip_names:
                    continue
                (written if k in w["names"] else read_only).append((k, v))

            bad_in = _nonfinite_report(read_only)
            bad_wr = _nonfinite_report(written)
            nonfinite_operands = {"inputs": bad_in, "written_ambiguous": bad_wr}

            if bad_in:
                inputs_finite = False          # an unambiguous input was dirty
            elif bad_wr or unknown:
                inputs_finite = None           # cannot tell whether it is read
            else:
                inputs_finite = True
        # Did the NaN on the inputs come from a collective (another rank)?
        arrived = None
        if inputs_finite is False:
            ro = [a for i, a in enumerate(cap_args) if i not in w["idx"]] + \
                 [v for k, v in cap_kwargs.items() if k not in w["names"]]
            arrived = self._arrived_via([t for t in _tensors(ro) if _nonfinite(t)])
            if arrived is not None and self._first_arrival is None:
                self._first_arrival = dict(arrived, read_by=str(func), read_at_step=self._step)

        ctx = self._ctx
        stack = _frames_text(ctx["frames"]) if ctx else "".join(traceback.format_stack())
        call_index = ctx["n"] if ctx else self.n
        packed_args = _pack(cap_args, self.max_bytes, self.record_data)
        geom = hipblaslt_geometry(func, cap_args)

        chain = self._chain_payload(ctx["rid"] if ctx else None)

        capsule = {
            # 3 adds the chain keys below and changes nothing else: every field a
            # format-2 reader looks at still describes the terminal op, so
            # compare_nan_capsules groups these alongside single-op capsules and
            # a plain replay still replays just the op that failed.
            "format": 3 if chain else 2,
            "op": str(func),                       # e.g. "aten.addmm.default"
            # "shapes" = every packed tensor (args, kwargs, chain pool) has
            # base=None and sha=None; "values" = replayable.  Absent = "values".
            "data": "values" if self.record_data else "shapes",
            "args": packed_args,
            "kwargs": _pack(cap_kwargs, self.max_bytes, self.record_data),
            "op_call_index": call_index,
            # What op_call_index counted.  None/None means "every aten op".
            "op_filter": {
                "include_ops": sorted(self.include_ops) if self.include_ops else None,
                "exclude_ops": sorted(self.exclude_ops) if self.exclude_ops else None,
            },
            "inputs_finite": inputs_finite,
            # Which operands were actually dirty, split by whether they are
            # unambiguously inputs.  A non-empty `written_ambiguous` with an
            # empty `inputs` is the ordinary out-param case -- uninitialised
            # buffers, nothing upstream to chase -- and is why `inputs_finite`
            # is None rather than False.
            "nonfinite_operands": nonfinite_operands,
            # Where the non-finite output was found: "return", or the written
            # argument an out-param kernel put its result in.
            "nan_found_in": where,
            # None for a functional op (every aten GEMM).  Present when the op
            # declares written arguments, because then the capsule's `args` are
            # a pre-call snapshot rather than the live operands -- the replay
            # must know which, and must re-create the out buffer's prior
            # contents in case the kernel accumulates into it.
            "mutation": ({
                "writes_args": sorted(w["idx"]),
                "writes_kwargs": sorted(w["names"]),
                "out_buffer_kwargs": sorted(w["outbuf_names"]),
                "args_are_pre_call": bool(pre),
            } if w["any"] else None),
            # _scannable for the same reason as the scan above: isnan/isinf are
            # as unimplemented for fp8 as isfinite, and dying here would lose the
            # capsule we have already decided to write.
            "out_nan": _count_where(torch.isnan, bad),
            "out_inf": _count_where(torch.isinf, bad),
            "out_numel": int(bad.numel()),
            "out_size": tuple(bad.shape),
            "out_stride": tuple(bad.stride()),
            "out_dtype": bad.dtype,
            "geometry": geom,
            "arg_shas": tensor_shas(packed_args),
            "stack": stack,
            # Identity of the *program point*, independent of the data.  Two ranks
            # hit the same op when these agree.
            "identity": {
                "op": str(func),
                "call_index": call_index,
                "geometry_key": geometry_key(geom),
                "stack_digest": hashlib.blake2b(stack.encode(), digest_size=8).hexdigest(),
                "dtypes": [str(a.dtype) for a in tree_leaves(args)
                           if isinstance(a, torch.Tensor)],
            },
            "dist": {
                "rank": self.rank,
                "world_size": self.world,
                "local_rank": self.local_rank,
                # CPU-only runs (the CPU tests) have no device to name.
                "device": torch.cuda.current_device() if torch.cuda.is_available() else None,
                "device_name": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
                "hostname": os.uname().nodename,
            },
            # The values, under the rule in ENV_PREFIXES/ENV_EXACT above.  Kept
            # as a flat dict at this key because that is what every existing
            # reader of a capsule expects; env_info carries the rule and the
            # names of what fell outside it.
            "env": dict((k, v) for k, v in os.environ.items() if _env_recorded(k)),
            "env_info": env_info(),
            # Which build dispatched.  Absent from older capsules, so read it
            # with .get(); see build_info().
            "build": build_info(),
            # How many processes were hammering this device when it failed.  A
            # replay that ignores this is running a different experiment; see
            # the note above contention_info().
            "contention": dict(contention_info(),
                               gpu_procs_at_arm=getattr(self, "_gpu_procs_at_arm", None)),
            "precision": {
                "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "allow_fp16_reduced": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            },
            "versions": {
                "torch": torch.__version__,
                "hip": getattr(torch.version, "hip", None),
            },
            # Callables run with the capture off (suspend_capture_in): blind spots.
            "suspended": sorted(_SUSPENDED),
            # Cross-rank causality: collectives issued per group so far, whether this
            # op's non-finite input came from one, and this rank's first such read.
            "dist_causality": {
                "coll_seq": dict(self._coll_seq),
                "groups": dict(self._groups),
                "arrived_via": arrived,
                "first_arrival": self._first_arrival,
                "step": self._step,
            },
        }
        if chain:
            capsule.update(chain)
        return capsule, cap_args, cap_kwargs

    def _emit(self, capsule, path=None):
        """Write `capsule` (atomically) to `path` (default: self.out) and report it."""
        path = path or self.out
        tmp = path + ".tmp"
        try:
            torch.save(capsule, tmp)
        except (TypeError, AttributeError, pickle.PicklingError) as e:
            # Losing the capsule is worse than losing one argument's value: save
            # it with the offending leaves replaced, and name them.
            bad = []
            capsule = _picklable(capsule, "capsule", bad)
            capsule["unpicklable"] = bad
            print("[op-nan][rank %d] capsule not picklable (%s: %s); replaced %d leaf(s): %s"
                  % (self.rank, type(e).__name__, e, len(bad), "; ".join(bad)), flush=True)
            torch.save(capsule, tmp)
        os.replace(tmp, path)       # atomic: a reader never sees a half-written capsule
        if path == self.out:
            dc = capsule.get("dist_causality") or {}
            self._emitted = {"capsule": path, "op": capsule["op"],
                             "inputs_finite": capsule["inputs_finite"],
                             "origin": (capsule.get("nan_origin") or {}).get("class"),
                             "arrived_via": dc.get("arrived_via"),
                             "settle": (capsule.get("settle") or {}).get("result")}

        where, mut = capsule["nan_found_in"], capsule["mutation"]
        inputs_finite, geom = capsule["inputs_finite"], capsule["geometry"]
        chain = capsule if capsule["format"] == 3 else None
        if self.verbose:
            p = "[op-nan][rank %d]" % self.rank
            print("\n%s captured -> %s" % (p, path), flush=True)
            print("%s   op=%s  op call #%d" % (p, capsule["op"], capsule["op_call_index"]),
                  flush=True)
            print("%s   out=%s nan=%d inf=%d  found in %s" %
                  (p, capsule["out_size"], capsule["out_nan"], capsule["out_inf"],
                   where), flush=True)
            if mut:
                print("%s   mutating op: writes %s  <- %s" %
                      (p, mut["writes_kwargs"],
                       "args captured pre-call" if mut["args_are_pre_call"] else
                       "args captured AFTER the call; the operands in this capsule "
                       "are the kernel's own output, not its input"), flush=True)
            nfo = capsule["nonfinite_operands"] or {}
            if inputs_finite:
                why = "this op CREATED the NaN"
            elif inputs_finite is False:
                why = "NaN arrived on the inputs; walk further up"
            elif not self.check_inputs:
                why = "input check disabled"
            elif nfo.get("written_ambiguous"):
                why = ("every unambiguous input is CLEAN; only written operands "
                       "were dirty pre-call. For a positional out-param kernel "
                       "that is uninitialised memory, so this op most likely "
                       "CREATED the NaN -- see the operand list below")
            else:
                why = ("unknown: this op overwrote an argument that was also an "
                       "input, and snapshot_mutated is off")
            print("%s   inputs_finite=%s  <- %s" % (p, inputs_finite, why), flush=True)
            for kind, label in (("inputs", "dirty INPUT"),
                                ("written_ambiguous", "dirty written operand")):
                for d in nfo.get(kind, []):
                    print("%s     %s %s %s%s  nan=%d inf=%d"
                          % (p, label, d["operand"], d["dtype"], tuple(d["shape"]),
                             d["nan"], d["inf"]), flush=True)
            ci = capsule["contention"]
            if ci.get("gpu_procs") or ci.get("mp_child"):
                print("%s   contention: %s process(es) on this device%s%s"
                      % (p, ci.get("gpu_procs") or "?",
                         ", this one is an mp worker" if ci["mp_child"] else "",
                         ", %.1f GB held by others" % (ci["mem_used_by_others_bytes"] / 1e9)
                         if ci.get("mem_used_by_others_bytes") else ""), flush=True)
                launchers = set(ci.get("gpu_proc_launchers") or [])
                others = [x for x in (ci.get("gpu_proc_list") or []) if x[0] != ci.get("pid")]
                helpers = [x for x in others if _is_compile_helper(x[2])]
                if helpers:
                    print("%s     %d torch.compile helper(s) (inductor compile workers, not counted)"
                          % (p, len(helpers)), flush=True)
                for pid, kfd, cmd in [x for x in others if not _is_compile_helper(x[2])][:6]:
                    if pid in launchers:
                        print("%s     launcher (ancestor, not counted): pid %d %s"
                              % (p, pid, cmd or "?"), flush=True)
                        continue
                    print("%s     other: pid %d %s %s"
                          % (p, pid, "compute (kfd)" if kfd else "monitor (render node only)",
                             cmd or "?"), flush=True)
                if (ci.get("gpu_procs") or 0) > 1 or ci["mp_child"]:
                    print("%s     ^ replay this capsule ALONE and you are running the"
                          " 1-process case, which for a contention-gated fault is the"
                          " case that never fails. Use --contend." % p, flush=True)
            if chain:
                n_int = len(chain["pool"]) - len(chain["roots"])
                print("%s   chain: %d ops, %d roots, %d intermediates stored"
                      % (p, len(chain["chain"]), len(chain["roots"]), n_int),
                      flush=True)
                for rec in chain["chain"]:
                    print("%s     [%d] %s%s%s"
                          % (p, rec["call_index"], rec["op"],
                             "  (view)" if rec["is_view"] else "",
                             "" if rec["version_ok"] is not False
                             else "  MUTATED AFTER PRODUCTION"), flush=True)
                if chain["truncation"]:
                    # Printed, not summarised away: "the walk stopped" and "the
                    # walk finished" look identical in a capsule otherwise, and
                    # on a real model values_evicted is the ordinary outcome.
                    seen = {}
                    for reason in chain["truncation"].values():
                        seen[reason] = seen.get(reason, 0) + 1
                    print("%s   chain stopped at: %s"
                          % (p, ", ".join("%s x%d" % (r, c)
                                          for r, c in sorted(seen.items()))),
                          flush=True)
                if inputs_finite:
                    print("%s   NB inputs_finite=True means no NaN on the operands, "
                          "NOT that they were correct -- replay with --chain"
                          % p, flush=True)
            if geom:
                print("%s   hipBLASLt problem: M=%d N=%d K=%d transA=%s transB=%s "
                      "lda=%d ldb=%d ldd=%d" %
                      (p, geom["M"], geom["N"], geom["K"], geom["transA"], geom["transB"],
                       geom["lda"], geom["ldb"], geom["ldd"]), flush=True)
            if capsule.get("nan_origin"):
                print("%s   nan origin: %s" % (p, _origin_text(capsule["nan_origin"])),
                      flush=True)
            st = capsule.get("settle")
            if st:
                print("%s   settle: %s -- %s" % (p, st["result"], st.get("reason", "")),
                      flush=True)
                for other in st.get("other_pending") or []:
                    print("%s     also pending (not written): %s" % (p, other), flush=True)

    # -- settle mode (see "Settle mode" in the module docstring) -------------
    def _note_born(self, func, args, kwargs, out):
        """Add the storages `func` allocated to the born-this-step set.  A view
        allocates nothing, and a mutating op's returns alias its written
        arguments, which existed before it -- neither is born here."""
        if is_view(func):
            return
        w = op_writes(func)
        old = set()
        if w["any"]:
            for i in w["idx"]:
                if i < len(args):
                    old.update(_storage_ptr(t) for t in _tensors(args[i]))
            for nm in w["names"]:
                old.update(_storage_ptr(t) for t in _tensors(kwargs.get(nm)))
        for t in _tensors(out):
            ptr = _storage_ptr(t)
            if ptr and ptr not in old:
                self._born.add(ptr)

    def _is_born(self, t):
        """True if `t` lives in a storage allocated this step.  Unknown -> True:
        never claim a tensor outlives the step without evidence."""
        ptr = _storage_ptr(t)
        return ptr is None or ptr in self._born

    def _check_escape(self, func, args, kwargs, out, kind):
        """Survival if a non-finite tensor leaves the process's reach: handed to a
        collective (kind='collective', before the call) or read to the host
        (kind='host': .item(), or any op that turns device tensors into CPU ones)."""
        ins = _tensors((args, kwargs))
        if kind == "host":
            if str(func) != "aten._local_scalar_dense.default":
                outs = _tensors(out)
                if not outs or any(t.device.type != "cpu" for t in outs):
                    return
                ins = [t for t in ins if t.device.type != "cpu"]
            how = "was read to the host by %s" % func
        else:
            how = "was handed to collective %s" % func
        # Garbage storage (uninitialized memory, see __init__) is not a NaN anyone
        # computed: handing it to a collective is not an escape.
        garbage = self._garbage
        if any(_nonfinite(t) and _storage_ptr(t) not in garbage for t in ins):
            self._survive("a non-finite tensor %s" % how, func)

    def _settle_hit(self, func, args, kwargs, bad, pre, where):
        """A watched op produced a non-finite output: survival, a new candidate,
        or one more hit on a site already pending or settled."""
        seen_before, self._nan_step = self._nan_step, True
        w = op_writes(func)
        read_only = ([a for i, a in enumerate(args) if i not in w["idx"]]
                     + [v for k, v in kwargs.items() if k not in w["names"]])
        dirty = [t for t in _tensors(read_only) if _nonfinite(t)]
        self._note_arrival(func, dirty)
        # Uninitialized memory: every non-finite operand lives in storage an
        # allocator returned and nothing wrote in full (or that holds only such
        # bytes), and the result stays in fresh or such storage.  Garbage copied
        # into a tensor that outlives the step is NOT this: that falls through to
        # survival below.
        if dirty and all(self._uninit(t) for t in dirty) \
                and (self._is_born(bad) or self._uninit(bad)):
            marked = set(_storage_ptr(t) for t in dirty + [bad])
            marked.discard(None)
            self._garbage |= marked
            self._just_marked = marked
            site = _call_site(func, where, self._ctx)
            s = self._sites.get(site)
            if s is not None and (s["status"] == "pending" or s["masked"] >= 1):
                s["hits"] += 1
                s["last_step"] = self._step
                return
            return self._new_candidate(func, args, kwargs, bad, pre, where, site,
                                       origin="uninitialized")
        # Data movement of what another rank sent: a copy cannot create a NaN, so if
        # its source came from a collective and nothing of this rank's own is pending,
        # the NaN (or another rank's allocator garbage -- FSDP2 copies every rank's
        # padding into the padded unsharded parameter, 2026-11-09 e4b rank 1) is that
        # rank's to judge.  Record the arrival, let the destination inherit it, go on;
        # if it then enters arithmetic here, that op is captured, marked arrived.
        if self._coll_out is not None and _is_data_movement(func) \
                and not any(q["capsule"] is not None and q["capsule"]["nan_origin"]["class"]
                            not in _SOFT_ORIGINS for q in self._pending):
            arr = self._arrived_via(list(_tensors(read_only)))
            if arr is not None:
                if self._first_arrival is None:
                    self._first_arrival = dict(arr, read_by=str(func), read_at_step=self._step)
                for t in [bad] + [x for x in _tensors((args, kwargs)) if x is bad]:
                    try:
                        self._coll_out[t.untyped_storage()] = arr
                    except Exception:                   # noqa: BLE001
                        pass
                return
        # A NaN computed into this storage: it no longer holds only garbage.
        p = _storage_ptr(bad)
        self._raw.discard(p)
        self._garbage.discard(p)
        if not self._is_born(bad):
            return self._survive("wrote a non-finite value into %s, a tensor that "
                                 "outlives the step" % where,
                                 func, args, kwargs, bad, pre, where)
        if any(not self._is_born(t) for t in dirty):
            return self._survive("read a non-finite tensor that outlives the step",
                                 func, args, kwargs, bad, pre, where)
        if dirty and (self._pending or seen_before):
            return              # downstream of a NaN already pending or counted this step
        site = _call_site(func, where, self._ctx)
        s = self._sites.get(site)
        if s is not None and s["status"] == "pending":
            s["hits"] += 1
            s["last_step"] = self._step
            return
        if s is not None and s["status"] == "masked":
            if s["origin"] in _BENIGN_ORIGINS:
                # Re-verify: a real fault at a site that usually computes 0/0 must
                # not hide behind its history.
                ca, ck = self._pre_call_operands(args, kwargs, pre)
                o = self._origin_for(func, ca, ck, bad, where, not dirty or None, pre)
                if o["class"] in _BENIGN_ORIGINS:
                    s["hits"] += 1
                    s["last_step"] = self._step
                    return
            elif s["masked"] >= 3:
                s["hits"] += 1          # saved three times already; stop paying for it
                s["last_step"] = self._step
                return
        self._new_candidate(func, args, kwargs, bad, pre, where, site)

    def _new_candidate(self, func, args, kwargs, bad, pre, where, site, origin=None):
        try:
            capsule, ca, ck = self._build_capsule(func, args, kwargs, bad, pre, where)
            if origin == "uninitialized":
                capsule["nan_origin"] = {
                    "class": origin, "nonfinite": capsule.get("out_nan", 0) + capsule.get("out_inf", 0),
                    "math": None, "overflow": None, "unexplained": None,
                    "note": "every non-finite operand is uninitialized memory (allocated "
                            "by an empty* op, never written in full)"}
            else:
                capsule["nan_origin"] = self._origin_for(func, ca, ck, bad, where,
                                                         capsule["inputs_finite"], pre)
        except Exception as e:                          # noqa: BLE001
            # A masked NaN must not take the run down through the instrumentation.
            # Survival with no capsule falls back to capturing where it was seen.
            capsule = None
            print("[op-nan][rank %d] !! could not build the candidate capsule for %s "
                  "(%s: %s)" % (self.rank, _site_text(site), type(e).__name__,
                                str(e)[:200]), flush=True)
        s = self._sites.get(site)
        if s is None:
            s = self._sites[site] = {"hits": 0, "masked": 0, "first_step": self._step,
                                     "file": None}
        s.update(status="pending", last_step=self._step,
                 origin=capsule["nan_origin"]["class"] if capsule else "no_reference")
        s["hits"] += 1
        self._pending.append({"site": site, "step": self._step, "capsule": capsule})
        if self.verbose:
            print("[op-nan][rank %d] pending: %s at step %d, out=%s nan=%s inf=%s; "
                  "origin: %s -- written only if the NaN outlives the step"
                  % (self.rank, _site_text(site), self._step, tuple(bad.shape),
                     capsule["out_nan"] if capsule else "?",
                     capsule["out_inf"] if capsule else "?",
                     _origin_text(capsule["nan_origin"]) if capsule else "?"), flush=True)

    def _survive(self, reason, func, args=None, kwargs=None, bad=None, pre=None,
                 where=None):
        """The NaN outlived its step: write the pending candidate it most likely
        came from (the earliest not of a benign origin), else capture `func`."""
        cands = [p for p in self._pending if p["capsule"] is not None]
        pick = next((p for p in cands
                     if p["capsule"]["nan_origin"]["class"] not in _SOFT_ORIGINS), None)
        if pick is None and cands and bad is None:
            if self.world > 1 and all(p["capsule"]["nan_origin"]["class"] == "uninitialized"
                                      for p in cands):
                # An escape in a multi-rank job with only allocator-garbage candidates here:
                # the NaN is not this rank's (it arrived, or came out of an op the capture
                # cannot see) and the garbage is not evidence.  Stop with the job, write
                # nothing; the manifest attributes it (2026-11-09: 3 torchtitan ranks wrote
                # their FSDP-padding candidate when rank 4's NaN passed an all_reduce).  A
                # `math` candidate (sqrt(-x)) IS this rank's NaN: it falls through, written.
                self.captured = True
                self._pending = []
                print("[op-nan][rank %d] a non-finite tensor escaped through %s with only "
                      "allocator-garbage candidates here (%s): no capsule; stopping with the job"
                      % (self.rank, func, ", ".join(_site_text(p["site"]) for p in cands)),
                      flush=True)
                return
            pick = cands[0]         # an escape with only soft candidates: the best we have
        # Only soft candidates (allocator garbage, 0/0) and a real survival at an op in
        # hand: that op is the evidence, not the garbage -- capture it (below).  On
        # 2026-11-09 five torchtitan ranks wrote their FSDP-padding candidate instead
        # of the op where a NaN from another rank surfaced.
        settle = {"result": "survived", "reason": "%s (seen at %s, step %d)"
                  % (reason, func, self._step),
                  "seen_at": str(func), "seen_at_step": self._step}
        if pick is not None:
            self.captured = True
            settle["candidate_step"] = pick["step"]
            settle["other_pending"] = [_site_text(p["site"]) for p in self._pending
                                       if p is not pick]
            self._sites[pick["site"]]["status"] = "survived"
            self._pending = []
            c = pick["capsule"]
            c["settle"] = settle
            c["masked_sites"] = self._census()
            self._emit(c)
            self._print_census()
            if self.raise_on_nan:
                raise RuntimeError("%s non-finite op output captured to %s"
                                   % ("[op-nan]", self.out))
        elif bad is not None:
            settle["candidate_step"] = None
            settle["reason"] += "; no candidate was pending, so this op is the first place it was seen"
            self._pending = []
            self._capture(func, args, kwargs, bad, pre, where,
                          extra={"settle": settle, "masked_sites": self._census()})
        else:
            # Escaped with nothing pending: it came from a site counted as masked.
            # Forget those verdicts so the next occurrence becomes a candidate.
            print("[op-nan][rank %d] !! %s, but no candidate is pending -- it came "
                  "from a site settled as masked earlier; re-watching every masked "
                  "site" % (self.rank, settle["reason"]), flush=True)
            for s in self._sites.values():
                if s["status"] == "masked":
                    s["status"] = None

    def settle_step(self):
        """Step boundary: settle every candidate that has waited settle_steps full
        steps as MASKED, and start a new born-this-step set.  Call once per step,
        at the top of the iteration (_ArmedCapture.step() does)."""
        self._flush()               # the step's queued checks belong to the step
        if not self.settle or self.captured:
            return
        with _disable_current_modes():
            self._step += 1
            keep = []
            for p in self._pending:
                if self._step - p["step"] > self.settle_steps:
                    self._settle_masked(p, at_close=False)
                else:
                    keep.append(p)
            self._pending = keep
            self._born = set()
            self._nan_step = False

    def _settle_masked(self, p, at_close):
        s = self._sites[p["site"]]
        s["status"] = "masked"
        s["masked"] += 1
        c = p["capsule"]
        origin = c["nan_origin"] if c else None
        path = None
        if c is not None and origin["class"] not in _BENIGN_ORIGINS and s["masked"] == 1:
            # Masked is not benign: a model's own nan_to_num masks a real kernel
            # NaN too.  Keep the first one per site, under a name no glob for
            # capsules (*.pt) will pick up.
            self._masked_files += 1
            path = "%s.masked-%d" % (self.out, self._masked_files)
            c["settle"] = {"result": "masked", "candidate_step": p["step"],
                           "settled_at_step": self._step, "at_close": at_close,
                           "reason": "the NaN never outlived its step"}
            c["masked_sites"] = self._census()
            tmp = path + ".tmp"
            torch.save(c, tmp)
            os.replace(tmp, path)
            s["file"] = path
        if self.verbose:
            print("[op-nan][rank %d] MASKED(%s): %s -- created at step %d, never "
                  "outlived it%s; origin: %s%s"
                  % (self.rank, origin["class"] if origin else "?", _site_text(p["site"]),
                     p["step"], " (settled at close)" if at_close else "",
                     _origin_text(origin),
                     "; saved for inspection -> %s" % path if path else
                     "; not a capsule" if origin and origin["class"] in _BENIGN_ORIGINS
                     else ""), flush=True)

    def finish(self):
        """After the loop: settle what is still pending (as MASKED -- nothing it
        did surfaced before the run ended) and print the census.  Idempotent."""
        self._flush()
        if self._finished:
            return
        if not self.settle:
            self._finished = True
            self._print_census()        # the wrong-value census, if those checks ran
            return
        self._finished = True
        with _disable_current_modes():
            if not self.captured:
                for p in self._pending:
                    self._settle_masked(p, at_close=True)
                self._pending = []
            self._print_census()

    def _catch_terminate(self):
        """mp workers only: a parent that stops its workers with terminate() once one
        reports the NaN (run_workload.py does) SIGTERMs the very worker whose exit
        hook is still writing the capsule, and SIGTERM's default is instant death.
        Write first, then die the way SIGTERM would have; if the exit hook is
        already writing, let it finish.  Never replaces a handler the model set."""
        import signal
        import threading
        if threading.current_thread() is not threading.main_thread():
            return
        if signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL:
            return

        def on_term(signum, frame):
            if self._exiting:
                return                      # the exit hook is mid-write: let it finish
            try:
                self._at_exit()
            finally:
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
                os.kill(os.getpid(), signal.SIGTERM)
        signal.signal(signal.SIGTERM, on_term)

    def _at_exit(self):
        """The process is exiting without finish(): usually the run died, often from
        the NaN itself (an assert on a NaN-derived length).  Write the earliest
        pending candidate of a non-benign origin rather than lose it."""
        self._exiting = True
        if self._finished or self.captured:
            return
        try:
            self._flush()           # may itself write a capsule (survival)
        except Exception:                               # noqa: BLE001
            pass                    # raise_on_nan: the capsule is already on disk
        if self.captured:
            return
        self._finished = True
        with _disable_current_modes():
            cands = [p for p in self._pending if p["capsule"] is not None
                     and p["capsule"]["nan_origin"]["class"] not in _SOFT_ORIGINS]
            if cands:
                p, self.captured = cands[0], True
                c = p["capsule"]
                c["settle"] = {"result": "unsettled_at_exit", "candidate_step": p["step"],
                               "reason": "the process exited at step %d with this "
                                         "candidate pending -- did the NaN end the run?"
                                         % self._step}
                c["masked_sites"] = self._census()
                self._emit(c)
            self._print_census()

    def _census(self):
        """Every site a NaN was created at, with its verdict -- the record of what
        settle mode let through and why."""
        return [{"site": _site_text(k), "op": k[0], "place": k[1], "where": k[2],
                 "status": s["status"], "origin": s["origin"], "hits": s["hits"],
                 "masked": s["masked"], "first_step": s["first_step"],
                 "last_step": s["last_step"], "file": s["file"]}
                for k, s in self._sites.items()]

    def _print_census(self):
        if self.verbose and self._vsites:
            # Evidence the wrong-value checks ran, and how close clean calls came.
            p = "[op-nan][rank %d]" % self.rank
            print("%s wrong-value check census (site: calls, reference-checked, max clean "
                  "rel err):" % p, flush=True)
            for k, s in sorted(self._vsites.items(), key=lambda kv: -kv[1]["calls"]):
                print("%s   %s %s N=%s: %d calls, %d checked, max %.3g"
                      % ((p,) + k + (s["calls"], s["checked"], s["max_err"])), flush=True)
        if not (self.verbose and self._sites):
            return
        p = "[op-nan][rank %d]" % self.rank
        print("%s census of NaN-creating sites (status, origin, hits, steps):" % p, flush=True)
        for e in self._census():
            print("%s   %-8s %-12s %5d  %s..%s  %s%s"
                  % (p, e["status"], e["origin"], e["hits"], e["first_step"], e["last_step"],
                     e["site"], "  -> %s" % e["file"] if e["file"] else ""), flush=True)


# Backwards-compatible alias for the check-only mode in docs/hipblaslt-nan-check.md.
OpNanCheck = OpNanCapture


# ---------------------------------------------------------------------------
# capture_from_env: the debug-nan skill's defaults in one call, for ANY workload.
#
#   sys.path.insert(0, os.environ["NAN_DOCTOR_ROOT"] + "/scripts/user")
#   from op_nan_capture import capture_from_env
#   cap = capture_from_env()        # after model/optimizer construction
#   for ...:                        # the training loop, a tight repro loop, a worker loop
#       cap.step()                  # top of every iteration
#       ...
#   cap.close()
#
# Where to put these three lines is the only model-specific decision.  Everything
# else is configuration, from the environment, so the launcher -- not an edit --
# sets it, and the same source runs as a control arm when NAN_CAPTURE_DIR is unset:
#
#   NAN_CAPTURE_DIR               capsule directory (unset = capture off: a no-op)
#   NAN_CAPTURE_DATA              0 (default) = shapes + statistics, no float
#                                 values or digests; integer tensors keep their
#                                 values (replay --synthesize only);
#                                 1 = record all values
#   NAN_CAPTURE_ARM_AFTER_STEPS   arm at the step() call with this index (default 0;
#                                 1 for lazily-built pipelines -- see the skill)
#   NAN_CAPTURE_CHAIN_DEPTH       default 4
#   NAN_CAPTURE_SNAPSHOT          1 (default) | 0 = snapshot_mutated off
#   NAN_CAPTURE_SNAPSHOT_MAX_BYTES  default 4 GiB
#   NAN_CAPTURE_OP_CLASSES        include_only_op_classes, e.g. "gemm" (default: none)
#   NAN_CAPTURE_EXCLUDE_OPS       extra exclude_ops, comma-separated; the allocator
#                                 family below is always excluded
#   NAN_CAPTURE_SETTLE            1 (default) = settle mode: write a NaN only if it
#                                 outlives its step, log masked ones as MASKED with
#                                 their nan_origin (module docstring); 0 = capture
#                                 the first non-finite output, as before
#   NAN_CAPTURE_SETTLE_STEPS      extra steps before a candidate settles (default 1)
#   NAN_CAPTURE_CHECK_BATCH       read finiteness back once per N watched ops instead
#                                 of after every op (default 64; 0/1 = every op).
#                                 See OpNanCapture._enqueue for what is kept per op.
#   NAN_CAPTURE_LATE_NORMS        1 = compute the queued checks' norms at flush, batched
#                                 (default 0: no gain measured on a real model once
#                                 checks are deferred, and it relies on no write
#                                 bypassing the dispatcher -- see _late_norms)
#   NAN_CAPTURE_VERIFY_EVERY      N > 0 = recompute every N-th call of each GEMM-family
#                                 site in higher precision and capture a WRONG finite
#                                 value (origin wrong_value); 0 (default) = off
#   NAN_CAPTURE_BOUNDS            1 = check every GEMM-family output against the
#                                 Cauchy-Schwarz bound (mm/addmm/bmm/baddbmm); default 0
#   NAN_CAPTURE_RECORD_COLLECTIVE "group:seq[:step],..." (R3): every member rank saves that
#                                 collective's inputs and outputs to <capsule dir>/
#                                 collective_g<group>_s<seq>.rank<r>.pt; collective_r3.py
#                                 checks and replays them.  Take group/seq from the
#                                 manifest's earliest arrival.
#   NAN_CAPTURE_SUSPEND           comma-separated "pkg.module:Qual.name" callables to run
#                                 with the capture off (suspend_capture_in): compiled
#                                 entry points torch.compile will not trace under a
#                                 dispatch mode, e.g. torchtitan's
#                                 torchtitan.models.common.attention:FlexAttention.compiled_flex_attn
#   NAN_PROGRESS_EVERY            print a progress line every N step() calls (0 = off)
#
# step() also calls sync_stop() when torch.distributed runs with world size > 1:
# once per step, on every rank, at the top of the iteration -- outside any other
# collective, as the capture requires.
ALLOC_EXCLUDE_OPS = ("aten.empty.memory_format,aten.empty_strided.default,"
                     "aten.empty_like.default,aten.new_empty.default,"
                     "aten.new_empty_strided.default,aten.empty_permuted.default")


class _ArmedCapture:
    def __init__(self, capture, arm_after, progress_every):
        self.capture, self.arm_after, self.progress_every = capture, arm_after, progress_every
        self.calls, self.armed = 0, False

    def step(self):
        i = self.calls
        self.calls += 1
        if self.capture is None:
            return
        if self.armed:
            self.capture.flush()            # deferred checks of the step just run
            self.capture.settle_step()      # local; before the collective below
        if self.armed and self.capture.world > 1:
            self.capture.sync_stop()
        if not self.armed and i >= self.arm_after:
            self.capture.__enter__()
            self.armed = True
            print("[op-nan] nan-capture armed at step() %d: out=%s gpu_procs=%s "
                  "containers_per_gpu=%s" % (i, self.capture.out, _gpu_procs(),
                                             _declared_containers_per_gpu()), flush=True)
        if self.progress_every and i % self.progress_every == 0:
            print("[op-nan] Progress step %d watched_ops=%d armed=%s"
                  % (i, getattr(self.capture, "n", 0), self.armed), flush=True)

    def close(self):
        if self.capture is not None and self.armed:
            self.capture.finish()
            if self.capture.check_batch > 1:
                print("[op-nan] deferred checks: %d read-backs for %d watched ops over "
                      "%d step(s); by reason %s" % (self.capture.flushes, self.capture.n,
                                                    self.calls - self.arm_after,
                                                    self.capture.flush_why), flush=True)
            self.capture.__exit__(None, None, None)
            self.armed = False
            print("[op-nan] nan-capture disarmed", flush=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


_SUSPENDED = {}     # "module:Qual.name" -> original attribute (patched once per process)


def suspend_capture_in(paths):
    """Run each callable in `paths` ("pkg.module:Class.attr" or "pkg.module:func",
    comma-separated or a list) with every user dispatch mode switched off.

    torch.compile refuses to trace while a non-infra dispatch mode is on the stack
    ("non-infra torch dispatch mode present, this is not supported today in
    torch.compile"), so a model that calls a compiled function from eager code --
    torchtitan's FlexAttention.compiled_flex_attn, any torch.compile'd block --
    dies under the capture.  Suspending the capture around the compiled entry point
    lets it run; its body is then invisible to the capture, exactly like any other
    compiled region.  Patch the attribute the model looks up at call time (a class
    attribute or module global read at each call), after the model is imported.
    Returns the list of paths patched."""
    import importlib
    import inspect
    if isinstance(paths, str):
        paths = [p.strip() for p in paths.split(",") if p.strip()]
    done = []
    for path in paths or []:
        if path in _SUSPENDED:
            done.append(path)
            continue
        mod_name, _, qual = path.partition(":")
        owner = importlib.import_module(mod_name)
        parts = qual.split(".")
        for p in parts[:-1]:
            owner = getattr(owner, p)
        raw = inspect.getattr_static(owner, parts[-1])
        fn = raw.__func__ if isinstance(raw, (staticmethod, classmethod)) else raw

        def wrapper(*a, _fn=fn, **k):
            with _disable_current_modes():
                return _fn(*a, **k)
        wrapper.__wrapped__ = fn
        wrapper.__name__ = getattr(fn, "__name__", parts[-1])
        if isinstance(raw, staticmethod):
            new = staticmethod(wrapper)
        elif isinstance(raw, classmethod):
            new = classmethod(wrapper)
        else:
            new = wrapper
        setattr(owner, parts[-1], new)
        _SUSPENDED[path] = raw
        done.append(path)
    return done


def capture_from_env(**overrides):
    """A OpNanCapture configured from the environment (see above), wrapped so the
    model needs only cap.step() per iteration and cap.close() after the loop.
    Keyword overrides go to OpNanCapture.  NAN_CAPTURE_DIR unset -> a no-op."""
    env = os.environ.get
    arm_after = int(env("NAN_CAPTURE_ARM_AFTER_STEPS", "0"))
    progress = int(env("NAN_PROGRESS_EVERY", "0"))
    out = env("NAN_CAPTURE_DIR")
    if not out:
        return _ArmedCapture(None, arm_after, progress)
    extra = [x for x in env("NAN_CAPTURE_EXCLUDE_OPS", "").split(",") if x]
    kw = dict(out=out,
              record_data=env("NAN_CAPTURE_DATA", "0") == "1",
              chain_depth=int(env("NAN_CAPTURE_CHAIN_DEPTH", "4")),
              snapshot_mutated=env("NAN_CAPTURE_SNAPSHOT", "1") == "1",
              snapshot_max_bytes=int(env("NAN_CAPTURE_SNAPSHOT_MAX_BYTES", str(4 << 30))),
              exclude_ops=",".join([ALLOC_EXCLUDE_OPS] + extra),
              settle=env("NAN_CAPTURE_SETTLE", "1") == "1",
              settle_steps=int(env("NAN_CAPTURE_SETTLE_STEPS", "1")),
              check_batch=int(env("NAN_CAPTURE_CHECK_BATCH", "64")),
              late_norms=env("NAN_CAPTURE_LATE_NORMS", "0") == "1",
              verify_every=int(env("NAN_CAPTURE_VERIFY_EVERY", "0")),
              bounds=env("NAN_CAPTURE_BOUNDS", "0") == "1",
              record_collectives=env("NAN_CAPTURE_RECORD_COLLECTIVE"))
    if env("NAN_CAPTURE_OP_CLASSES"):
        kw["include_only_op_classes"] = env("NAN_CAPTURE_OP_CLASSES")
    if env("NAN_CAPTURE_SUSPEND"):
        done = suspend_capture_in(env("NAN_CAPTURE_SUSPEND"))
        print("[op-nan] capture suspended inside (compiled regions, unchecked): %s"
              % ", ".join(done), flush=True)
    kw.update(overrides)
    return _ArmedCapture(OpNanCapture(**kw), arm_after, progress)


# Framework-owned loops: the same capture as a callback, for loops the user code
# does not contain.  Built lazily, so importing this module never needs either
# framework.  Arming/progress/sync_stop come from the environment as above.
#
#   L.Trainer(..., callbacks=[lightning_capture_callback()])        # Lightning
#   Trainer(..., callbacks=[hf_capture_callback()])                  # HF transformers
def lightning_capture_callback(**overrides):
    try:
        import lightning.pytorch as pl
    except ImportError:                      # the older package name
        import pytorch_lightning as pl

    class NanCaptureCallback(pl.Callback):
        def on_train_start(self, trainer, pl_module):
            self.cap = capture_from_env(**overrides)

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            self.cap.step()                  # top of each batch: outside any collective

        def on_train_end(self, trainer, pl_module):
            self.cap.close()

        def on_exception(self, trainer, pl_module, exception):
            if hasattr(self, "cap"):
                self.cap.close()

    return NanCaptureCallback()


def hf_capture_callback(**overrides):
    from transformers import TrainerCallback

    class NanCaptureCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kw):
            self.cap = capture_from_env(**overrides)

        def on_step_begin(self, args, state, control, **kw):
            self.cap.step()                  # per optimizer step (all its micro-batches)

        def on_train_end(self, args, state, control, **kw):
            self.cap.close()

    return NanCaptureCallback()
