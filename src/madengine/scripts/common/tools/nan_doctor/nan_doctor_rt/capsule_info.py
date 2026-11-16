#!/usr/bin/env python3
"""What a capsule says, in ~20 lines, without a replay (mad-debug-nan steps 4.5 / 5a).

    capsule_info.py <capsule.pt> [...] [--frames N]

Per capsule: op, call index, output (shape, dtype, nan/inf, where), inputs_finite
and which operands were dirty, the chain's ops and why it stopped, nan_origin
and settle (settle-mode capsules), and the last N MODEL frames of the stack
(torch, stdlib and nan-doctor frames dropped -- in backward the stack is the
autograd engine's, so the op's autograd node is the location instead).

For a pointwise op it also re-runs the op on CPU from the capsule's operands and
in fp64: "math" there (fp64 non-finite at the same positions: 0/0, sqrt(-1))
means the arithmetic is undefined, not a kernel fault -- the usual false
positive (e.g. torch's own sinc_backward at x == 0).  Any other op: replay it on
the GPU with replay_nan_op.py.  Runs on CPU; needs torch -- a login node without it:
`madengine nan exec --image <the model's image> -- python3 -m nan_doctor_rt.capsule_info ...` (CPU only).

`args` lists the op's operands (dtype, shape, stride) and the values of small integer
operands (group offsets, cu_seqlens: a grouped GEMM's per-group sizes are there).
"""
if not __package__:  # run by path: make the package importable so relative imports work
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    __package__ = "nan_doctor_rt"
import argparse
import os
import sys
import torch  # noqa: E402
from . import op_nan_capture as g  # noqa: E402


def resolve(name):
    ns, op, overload = name.split(".")
    return getattr(getattr(getattr(torch.ops, ns), op), overload)


def model_frames(stack, n):
    keep = []
    for line in stack.splitlines():
        line = line.strip()
        if not line.startswith('File "'):
            continue
        path = line.split('"')[1]
        if any(s in path for s in ("/torch/", "/python3.", "nan-doctor/scripts", "<frozen")):
            continue
        keep.append(line.replace('File "', "").replace('", line ', ":").replace(", in ", "  "))
    return keep[-n:]


FP32_MAX = 3.4028234663852886e38
REDUCTIONS = ("_foreach_norm", "norm", "linalg_vector_norm", "frobenius_norm", "sum", "mean",
              "nansum", "_local_scalar_dense", "var", "std", "dot", "vdot")


def _float_stats(x, out):
    """Every recorded float tensor's stats under x (args may nest: a foreach op's list)."""
    if isinstance(x, dict):
        st = x.get("stats")
        if isinstance(st, dict) and st.get("kind") == "float":
            out.append(st)
        else:
            for v in x.values():
                _float_stats(v, out)
    elif isinstance(x, (list, tuple)):
        for v in x:
            _float_stats(v, out)
    return out


def overflow_line(c):
    """A reduction that returned inf (no NaN) from finite inputs large enough to overflow fp32
    is a correct op: the cause is the exploding values upstream, not this kernel."""
    name = c["op"].split(".")[1] if c["op"].count(".") >= 2 else c["op"]
    if name not in REDUCTIONS or not c.get("inputs_finite") or c.get("out_nan") or not c.get("out_inf"):
        return None
    stats = _float_stats(c.get("args"), [])
    if not stats:
        return None
    mx = max(max(abs(s.get("min") or 0.0), abs(s.get("max") or 0.0)) for s in stats)
    numel = max(s.get("numel") or 1 for s in stats)
    norm = "norm" in name or name in ("var", "std", "dot", "vdot")
    certain = mx * mx > FP32_MAX if norm else mx > FP32_MAX
    possible = (mx * mx * numel if norm else mx * numel) > FP32_MAX
    if not possible:
        return None
    return ("overflow %s: finite inputs up to %.3g (%d tensor(s), up to %d elements) overflow fp32 "
            "in %s -- a correct op; the cause is upstream (exploding values). Log per-step "
            "gradient norms: nan_doctor_rt.grad_trajectory"
            % ("expected" if certain else "likely", mx, len(stats), numel, name))


INT_DTYPES = (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64, torch.bool)


def describe_arg(x, pool, nvals=64):
    """One operand as text: 'dtype (shape) stride=(..)', plus the values of an integer
    tensor of <= nvals elements; chain references are looked up in the pool."""
    if isinstance(x, dict) and g._CHAIN_REF in x:
        x = (pool or {}).get(x[g._CHAIN_REF], x)
        if g._CHAIN_REF in x:
            return "chain tensor %s" % x[g._CHAIN_REF]
    if isinstance(x, dict) and x.get(g._TENSOR_TAG):
        s = "%s %s stride=%s" % (str(x["dtype"]).replace("torch.", ""), tuple(x["size"]), tuple(x["stride"]))
        n = 1
        for d in x["size"]:
            n *= d
        if x["dtype"] in INT_DTYPES and x.get("base") is not None and n <= nvals:
            s += " = %s" % g.unpack(x, "cpu").flatten().tolist()
        return s
    if isinstance(x, (list, tuple)):
        return "[%s]" % ", ".join(describe_arg(v, pool, nvals) for v in x)
    return repr(x)[:40]


def info(path, nframes):
    c = g.upgrade_capsule(torch.load(path, map_location="cpu", weights_only=False))
    p = print
    p("== %s" % path)
    p("op       %s  call #%s" % (c["op"], c.get("op_call_index")))
    p("data     %s" % c.get("data", "values"))
    p("out      %s %s  nan=%s inf=%s of %s  found in %s"
      % (tuple(c["out_size"]), str(c["out_dtype"]).replace("torch.", ""), c["out_nan"],
         c["out_inf"], c["out_numel"], c.get("nan_found_in")))
    nfo = c.get("nonfinite_operands") or {}
    dirty = ["%s %s nan=%d inf=%d" % (d["operand"], tuple(d["shape"]), d["nan"], d["inf"])
             for k in ("inputs", "written_ambiguous") for d in nfo.get(k, [])]
    p("inputs   inputs_finite=%s%s" % (c.get("inputs_finite"),
                                       ("  dirty: " + "; ".join(dirty)) if dirty else ""))
    try:
        ops = ["[%d] %s" % (i, describe_arg(x, c.get("pool"))) for i, x in enumerate(c.get("args") or [])]
        ops += ["%s=%s" % (k, describe_arg(v, c.get("pool"))) for k, v in (c.get("kwargs") or {}).items()]
        p("args     %s" % "\n         ".join(ops or ["-"]))
    except Exception as e:                              # noqa: BLE001 -- informational only
        p("args     (not readable: %s)" % e)
    if c.get("chain"):
        ops = [r["op"].replace("aten.", "") for r in c["chain"] if not r.get("is_view")]
        stops = {}
        for r in (c.get("truncation") or {}).values():
            stops[r] = stops.get(r, 0) + 1
        p("chain    %s  (stopped: %s)" % (" -> ".join(ops[-8:]),
                                          ", ".join("%s x%d" % kv for kv in stops.items()) or "-"))
    if c.get("nan_origin"):
        p("origin   %s" % g._origin_text(c["nan_origin"]))
    ovf = overflow_line(c)
    if ovf:
        p(ovf)
    if c.get("settle"):
        s = c["settle"]
        p("settle   %s: %s" % (s.get("result"), s.get("reason", "")))
    if c.get("masked_sites"):
        p("sites    %s" % "; ".join("%s %s/%s x%d" % (m["site"], m["status"], m["origin"], m["hits"])
                                    for m in c["masked_sites"]))
    fr = model_frames(c.get("stack", ""), nframes)
    p("frames   %s" % ("\n         ".join(fr) if fr else
                       "(none outside torch: a backward op -- see the chain / autograd node)"))
    try:
        func = resolve(c["op"])
    except Exception:                                   # noqa: BLE001
        func = None
        p("cpu      %s is not registered here (its library is not imported): replay on "
          "the GPU (replay_nan_op.py --chain)" % c["op"])
    if not g.has_data(c):
        p("cpu      shapes-only capsule (no float values; integer tensors kept): no re-run; replay with --synthesize")
    elif func is not None and "pointwise" in g._tag_names(func):
        try:
            args, kwargs = g.unpack(c["args"], "cpu"), g.unpack(c["kwargs"], "cpu")
            out = func(*args, **kwargs)
            ws = c.get("nan_found_in") or "return"
            bad = out if ws == "return" else (args[int(ws[3:])] if ws.startswith("arg") else None)
            if isinstance(bad, (tuple, list)):
                bad = bad[0]
            o = g._nan_origin(func, args, kwargs, bad, "return" if ws == "return" else ws, True)
            p("cpu      re-run on CPU: %d non-finite (capsule: %d); %s"
              % (o["nonfinite"], c["out_nan"] + c["out_inf"], g._origin_text(o)))
        except Exception as e:                          # noqa: BLE001
            p("cpu      re-run failed: %s: %s" % (type(e).__name__, str(e)[:120]))
    elif func is not None:
        p("cpu      not pointwise: replay on the GPU (replay_nan_op.py --chain)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("capsules", nargs="+")
    ap.add_argument("--frames", type=int, default=4)
    a = ap.parse_args()
    for path in a.capsules:
        info(path, a.frames)


if __name__ == "__main__":
    main()
