"""Replay one NaN capsule under load on every GPU of this node (mad-debug-numeric step 5b).

    madengine nan replay <capsule.pt> --image <image> --out-dir <dir> [options]

Every copy is one container pinned to one GPU running
    python3 -m nan_doctor_rt.replay_nan_op CAPSULE --chain --chain-fallback --chain-iters N --restore-env
i.e. the [C1] per-op walk (with its fp64 reference column), then the whole chain re-run N
times; a capsule without a chain falls back to the single op x N (`--mode op` forces it).
The copies on one GPU are each other's co-tenants and wait for each other at a barrier
before their first run, so they overlap from the start.

Deterministic probe: before the fan-out, ONE copy on GPU 0 runs the capsule
--deterministic-probe times (100; 0 off; skipped with --cotenant).  Every run non-finite =
a deterministic fault: the ladder ends there, REPRODUCED at concurrency 1, instead of
loading the capsule into NGPUS x concurrency copies.

Ladder: start at the capture's processes per GPU + 1 (read from the capsule), so the replay
is at least as loaded as the model was; an out-of-memory attempt is retried with one copy
fewer, down to 1.  The first attempt that runs to completion ends the ladder: REPRODUCED,
CLEAN, or NOT_SET_UP (0 hits in `futility x the capsule's op call index` node-wide runs: the
replay is not recreating the model's trigger, which more runs cannot fix).  Runs in the
first --futility-grace seconds of an attempt do not count toward futility (default 180
with --cotenant, else 0): a co-tenant loads the device in seconds but computes only after
its own start-up, and solo-speed runs in between would end a reproducing attempt
NOT_SET_UP.  Every copy of an attempt is stopped together once the node has --stop-hits
non-finite runs.

The last line of output is LADDER_RESULT=...; each attempt directory holds summary.txt.
Paths are the same on the host and in the containers (the working directory and the
directories of CAPSULE and OUT_DIR are bind-mounted at their own path).

Environment defaults (so a reproducer's run.sh can override without editing it):
CAPSULE_ITER_COUNT, CAPSULE_CONCURRENCY, CAPSULE_MODE, CAPSULE_COTENANT, CAPSULE_EXTRA_ARGS,
CAPSULE_STOP_HITS, CAPSULE_FUTILITY, CAPSULE_FUTILITY_GRACE, CAPSULE_DETERMINISTIC_PROBE,
CAPSULE_SECONDS, NGPUS.
"""

# flake8: noqa: E741,E731,E402  (terse variable names, kept from the original scripts)

import argparse
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import runner, summary

TRIAL_RC = "TRIAL_RC="


def _env(name, default):
    v = os.environ.get(name)
    return default if v in (None, "") else v


def count_gpus():
    """GPUs on this host: rocm-smi, else amd-smi, else 8."""
    for cmd, rx in (
        (["rocm-smi", "--showid"], r"^GPU\[(\d+)\]"),
        (["amd-smi", "list"], r"^GPU: (\d+)"),
    ):
        if shutil.which(cmd[0]):
            try:
                out = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=60
                ).stdout
            except (OSError, subprocess.SubprocessError):
                continue
            ids = {m.group(1) for l in out.splitlines() for m in [re.match(rx, l)] if m}
            if ids:
                return len(ids)
    return 8


def read_progress(attempt_dir):
    """(runs, hits) summed over every copy's --progress-file ("<runs> <hits>")."""
    runs = hits = 0
    for p in os.listdir(attempt_dir):
        if not p.endswith(".progress"):
            continue
        try:
            parts = Path(os.path.join(attempt_dir, p)).read_text().split()
            runs += int(parts[0])
            hits += int(parts[1])
        except (OSError, ValueError, IndexError):
            continue
    return runs, hits


class Ladder:
    def __init__(
        self,
        capsule,
        out_dir,
        image=None,
        ngpus=None,
        mode="chain",
        iters=1000000,
        concurrency=None,
        cotenant="",
        extra_args="",
        stop_hits=10,
        futility=100,
        futility_grace=None,
        det_runs=100,
        seconds=0,
        mounts=(),
        env=None,
        runtime_dir=None,
        poll=3.0,
        popen=None,
        probe=None,
        out=print,
    ):
        self.capsule = os.path.realpath(capsule)
        self.out_dir = os.path.abspath(out_dir)
        self.image, self.mode, self.iters = image, mode, int(iters)
        self.ngpus = int(ngpus) if ngpus else count_gpus()
        self.concurrency = concurrency
        self.cotenant, self.extra_args = cotenant or "", extra_args or ""
        self.stop_hits, self.futility, self.seconds = (
            int(stop_hits),
            int(futility),
            seconds,
        )
        self.grace = (
            (180 if self.cotenant else 0)
            if futility_grace is None
            else float(futility_grace)
        )
        self.det_runs = int(det_runs)
        self.mounts, self.env, self.runtime_dir, self.poll = (
            list(mounts),
            dict(env or {}),
            runtime_dir,
            poll,
        )
        self.out = out
        self._popen = popen or self._docker_popen
        self._probe = probe or self._docker_probe
        self.tag = os.getpid()
        self.host = socket.gethostname().split(".")[0]
        self._names, self._term = [], False
        if mode not in ("chain", "op"):
            raise ValueError("mode must be chain or op")
        if "'" in self.cotenant:
            raise ValueError("--cotenant must not contain single quotes")

    # ---- what runs in each container -------------------------------------------------
    def payload(self, iters=None):
        iters = self.iters if iters is None else iters
        base = "python3 -m nan_doctor_rt.replay_nan_op %s" % shlex.quote(self.capsule)
        if self.mode == "op":
            p = base + " --iters %d --restore-env" % iters
        else:
            p = (
                base
                + " --chain --chain-fallback --chain-iters %d --restore-env" % iters
            )
        if self.cotenant:
            p += " --contend-with %s" % shlex.quote(self.cotenant)
        if self.seconds and str(self.seconds) != "0":
            p += " --seconds %s" % self.seconds
        return (p + " " + self.extra_args).strip()

    def _mounts(self):
        cwd = os.path.realpath(os.getcwd())
        extra = []
        for path in (self.capsule, self.out_dir):
            d = path if os.path.isdir(path) else os.path.dirname(path)
            if not (d == cwd or d.startswith(cwd + "/")):
                extra.append((d, d, "rw"))
        return extra + self.mounts

    def _kw(self, gpu):
        return dict(
            gpus=str(gpu),
            mounts=self._mounts(),
            env=self.env,
            runtime_dir=self.runtime_dir,
        )

    def _docker_popen(self, gpu, cmd, log, name):
        argv = runner.build_command(self.image, cmd, name=name, **self._kw(gpu))
        logf = open(log, "w")
        p = subprocess.Popen(
            argv, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True
        )
        p._nan_logf = logf
        return p

    def _docker_probe(self):
        cmd = "python3 -m nan_doctor_rt.replay_nan_op %s --capture-procs" % shlex.quote(
            self.capsule
        )
        argv = runner.build_command(self.image, cmd, **self._kw(0))
        return subprocess.run(argv, capture_output=True, text=True).stdout

    # ---- the ladder -------------------------------------------------------------------
    def _terminate(self):
        self._term = True
        for n in self._names:
            subprocess.run(["docker", "rm", "-f", n], capture_output=True)

    def read_capture(self):
        """(capture processes per GPU, detail, op call index) from the capsule."""
        probe = self._probe()

        def grab(key):
            v = [
                m.group(1)
                for l in probe.splitlines()
                for m in [re.match(key + r"=(.*)", l)]
                if m
            ]
            return v[-1].strip() if v else None

        return (
            grab("CAPTURE_PROCS"),
            grab("CAPTURE_PROCS_DETAIL") or "",
            grab("CAPTURE_OP_INDEX"),
            probe,
        )

    def attempt(
        self, c, expect, op_index, ngpus=None, iters=None, suffix="", stop_hits=None
    ):
        """One attempt: (ATTEMPT_RESULT, dir, hits, runs) -- hits and runs from the totals
        line of summary.txt (the copies' final counts; .progress lags behind)."""
        ngpus = ngpus or self.ngpus
        d = os.path.join(self.out_dir, "c%d%s.%s.%s" % (c, suffix, self.tag, self.host))
        os.makedirs(d, exist_ok=True)
        os.chmod(d, 0o777)
        self.out(
            "[%s] attempt: concurrency=%d -> %d copies, logs in %s"
            % (time.strftime("%c"), c, ngpus * c, d)
        )
        procs = []
        for gpu in range(ngpus):
            for k in range(c):
                loop = "--progress-file %s --stop-file %s" % (
                    os.path.join(d, "g%d_k%d.progress" % (gpu, k)),
                    os.path.join(d, "STOP"),
                )
                if c > 1:
                    loop += " --start-barrier %s --barrier-n %d" % (
                        os.path.join(d, "barrier.g%d" % gpu),
                        c,
                    )
                cmd = "PYTHONUNBUFFERED=1 NAN_REPLAY_CONCURRENCY=%d %s %s" % (
                    c,
                    self.payload(iters),
                    loop,
                )
                log = os.path.join(d, "g%d_k%d.%s.log" % (gpu, k, self.host))
                name = "nan_replay_%d_c%d%s_g%d_k%d" % (self.tag, c, suffix, gpu, k)
                self._names.append(name)
                procs.append((self._popen(gpu, cmd, log, name), log))
        self._wait(d, procs, expect, self.stop_hits if stop_hits is None else stop_hits)
        for p, log in procs:
            with open(log, "a") as f:
                f.write("%s%s\n" % (TRIAL_RC, p.returncode))
            lf = getattr(p, "_nan_logf", None)
            if lf:
                lf.close()
        text, res = summary.summarize(d)
        self.out(text.rstrip("\n"))
        with open(os.path.join(d, "summary.txt"), "w") as f:
            f.write(text)
        tot = [
            m
            for l in text.splitlines()
            if l.startswith("=== totals:")
            for m in [re.search(r"non-finite (\d+)/(\d+)", l)]
            if m
        ]
        hits, runs = (int(tot[-1].group(1)), int(tot[-1].group(2))) if tot else (0, 0)
        return res, d, hits, runs

    def _wait(self, d, procs, expect, stop_hits):
        """Wait for the copies; stop ALL of them together once the node has --stop-hits non-finite
        runs, or once it has run `expect` times clean (futility; runs inside the grace do
        not count, hits always do).  Stopping one copy early would de-stress its
        co-tenants, so the stop file is node-wide."""
        stop = os.path.join(d, "STOP")
        t0, base = time.time(), 0
        while any(p.poll() is None for p, _ in procs):
            if self._term:
                break
            if not os.path.exists(stop):
                runs, hits = read_progress(d)
                if time.time() - t0 < self.grace:
                    base = runs
                runs -= base
                if stop_hits > 0 and hits >= stop_hits:
                    self.out(
                        "[%s] node-wide non-finite runs: %d >= --stop-hits=%d -- stopping every copy"
                        % (time.strftime("%c"), hits, stop_hits)
                    )
                    Path(stop).write_text(
                        "hits=%d target=%d at %s\n"
                        % (hits, stop_hits, time.strftime("%F %T"))
                    )
                elif expect > 0 and hits == 0 and runs >= expect:
                    self.out(
                        "[%s] futility: 0 non-finite in %d node-wide runs >= %d -- stopping every copy (NOT_SET_UP)"
                        % (time.strftime("%c"), runs, expect)
                    )
                    Path(stop).write_text(
                        "futility runs=%d expect=%d at %s\n"
                        % (runs, expect, time.strftime("%F %T"))
                    )
                    Path(os.path.join(d, "FUTILE")).write_text("%d\n" % runs)
            time.sleep(self.poll)
        for p, _ in procs:
            p.wait()

    def run(self):
        os.makedirs(self.out_dir, exist_ok=True)
        os.chmod(self.out_dir, 0o777)
        cap_procs, detail, op_index, probe = self.read_capture()
        if not cap_procs:
            self.out(
                "--capture-procs printed no CAPTURE_PROCS line:\n"
                + "\n".join(probe.splitlines()[-20:])
            )
            self.out(
                "LADDER_RESULT=ERROR reading the capture's contention from the capsule"
            )
            return 1
        if self.cotenant:
            start, why = 1, "one replay per GPU beside --cotenant"
        elif self.concurrency:
            start, why = int(self.concurrency), "--concurrency"
        elif cap_procs.isdigit():
            start, why = int(cap_procs) + 1, "capture: %s; +1" % detail
        else:
            start, why = (
                2,
                "capture contention not recorded (%s); assumed 1, +1" % detail,
            )
        expect = (
            self.futility * int(op_index)
            if self.futility > 0 and op_index and op_index.isdigit()
            else 0
        )
        self.out(
            "[%s] capsule=%s mode=%s iters=%d start=c%d (%s), one less per OOM out=%s"
            % (
                time.strftime("%c"),
                self.capsule,
                self.mode,
                self.iters,
                start,
                why,
                self.out_dir,
            )
        )
        self.out(
            "[%s] futility: %s"
            % (
                time.strftime("%c"),
                (
                    "NOT_SET_UP after %d clean node-wide runs (--futility=%d x capture op "
                    "index %s), counted after a %gs grace"
                    % (expect, self.futility, op_index, self.grace)
                    if expect
                    else "off"
                ),
            )
        )
        self.out("[%s] payload: %s" % (time.strftime("%c"), self.payload()))
        tried, c, result = [], start, None
        # A deterministic fault needs no fan-out: one solo copy that NaNs on every run
        # settles it (a contention-gated fault does not hit every run alone).
        if self.det_runs > 0 and not self.cotenant:
            r, d, hits, runs = self.attempt(
                1,
                expect,
                op_index,
                ngpus=1,
                iters=self.det_runs,
                suffix="det",
                stop_hits=0,
            )
            tried.append("c1det:%s:%d/%d" % (r, hits, runs))
            if r == "REPRODUCED" and runs > 0 and hits == runs:
                result = (
                    "REPRODUCED at concurrency 1 -- deterministic: %d/%d solo runs on GPU 0 "
                    "non-finite, no fan-out" % (hits, runs)
                )
            elif r == "ERROR":
                result = (
                    "ERROR at concurrency 1 (deterministic probe) -- read the copy logs"
                )
            self.out(
                "[%s] deterministic probe: %s %d/%d -> %s"
                % (
                    time.strftime("%c"),
                    r,
                    hits,
                    runs,
                    result or "not deterministic, ladder from c%d" % start,
                )
            )
        while result is None:
            if self._term:
                result = "ERROR interrupted"
                break
            r, d, _, _ = self.attempt(c, expect, op_index)
            tried.append("c%d:%s" % (c, r))
            if r == "REPRODUCED":
                result = "REPRODUCED at concurrency %d" % c
            elif r == "CLEAN":
                result = "NOT_REPRODUCED at concurrency %d" % c
            elif r == "NOT_SET_UP":
                runs = Path(os.path.join(d, "FUTILE")).read_text().strip()
                result = (
                    "NOT_SET_UP at concurrency %d -- 0 in %s runs, the model hit it within %s op calls: "
                    "escalate (step5-pressure.md)" % (c, runs, op_index)
                )
            elif r == "OOM":
                if c > 1:
                    c -= 1
                    continue
                result = "NOT_REPRODUCED -- ladder ended on OOM at concurrency 1"
            else:
                result = "ERROR at concurrency %d -- read the copy logs" % c
            break
        self.out(
            "[%s] attempts: %s  (iters=%d per copy)"
            % (time.strftime("%c"), " ".join(tried), self.iters)
        )
        self.out("LADDER_RESULT=%s" % result)
        return 0 if not result.startswith("ERROR") else 1


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="madengine nan replay",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("capsule")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image")
    src.add_argument(
        "--manifest", help="madengine build_manifest.json naming the image"
    )
    ap.add_argument(
        "--model", help="with --manifest: the model, when the manifest has several"
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="logs and summaries; one subdirectory per attempt",
    )
    ap.add_argument(
        "--mode", default=_env("CAPSULE_MODE", "chain"), choices=["chain", "op"]
    )
    ap.add_argument(
        "--iters",
        type=int,
        default=int(_env("CAPSULE_ITER_COUNT", 1000000)),
        help="runs per copy; an upper bound, the stops end an attempt sooner (default 1000000)",
    )
    ap.add_argument(
        "--concurrency",
        type=int,
        default=int(_env("CAPSULE_CONCURRENCY", 0)) or None,
        help="starting copies per GPU (default: the capture's processes per GPU + 1)",
    )
    ap.add_argument(
        "--cotenant",
        default=_env("CAPSULE_COTENANT", ""),
        help="a command run as co-tenant beside ONE replay per GPU; forces concurrency 1",
    )
    ap.add_argument(
        "--extra-args",
        default=_env("CAPSULE_EXTRA_ARGS", ""),
        help="extra replay flags for every copy",
    )
    ap.add_argument(
        "--stop-hits",
        type=int,
        default=int(_env("CAPSULE_STOP_HITS", 10)),
        help="stop every copy once the node has this many non-finite runs; 0 = never (default 10)",
    )
    ap.add_argument(
        "--futility",
        type=int,
        default=int(_env("CAPSULE_FUTILITY", 100)),
        help="NOT_SET_UP after this many times the capsule's op call index of clean "
        "node-wide runs; 0 = off (default 100)",
    )
    ap.add_argument(
        "--futility-grace",
        type=float,
        default=_env("CAPSULE_FUTILITY_GRACE", None),
        help="seconds after an attempt starts whose runs futility does not count "
        "(default 180 with --cotenant, else 0)",
    )
    ap.add_argument(
        "--deterministic-probe",
        type=int,
        default=int(_env("CAPSULE_DETERMINISTIC_PROBE", 100)),
        help="runs of ONE solo copy on GPU 0 before the fan-out; every run non-finite "
        "ends the ladder REPRODUCED at concurrency 1; 0 = off, skipped with "
        "--cotenant (default 100)",
    )
    ap.add_argument(
        "--seconds",
        default=_env("CAPSULE_SECONDS", "0"),
        help="also stop every copy this long after its loop starts",
    )
    ap.add_argument(
        "--ngpus",
        type=int,
        default=int(_env("NGPUS", 0)) or None,
        help="GPUs to use (default: counted)",
    )
    ap.add_argument(
        "--mount", action="append", default=[], metavar="HOST[:CONT[:ro|rw]]"
    )
    ap.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="environment variable for every copy (e.g. a loader path the capture held back)",
    )
    ap.add_argument("--runtime-dir")
    ap.add_argument("--pull", action="store_true")
    a = ap.parse_args(argv)
    if not os.path.isfile(a.capsule):
        print("capsule not readable: %s" % a.capsule, file=sys.stderr)
        return 2
    try:
        image = a.image or runner.image_from_manifest(a.manifest, a.model)
        runner.ensure_image(image, a.pull)
        lad = Ladder(
            a.capsule,
            a.out_dir,
            image=image,
            ngpus=a.ngpus,
            mode=a.mode,
            iters=a.iters,
            concurrency=a.concurrency,
            cotenant=a.cotenant,
            extra_args=a.extra_args,
            stop_hits=a.stop_hits,
            futility=a.futility,
            futility_grace=a.futility_grace,
            det_runs=a.deterministic_probe,
            seconds=a.seconds,
            mounts=[runner.parse_mount(m) for m in a.mount],
            env=dict(kv.partition("=")[::2] for kv in a.env),
            runtime_dir=a.runtime_dir,
        )
    except (ValueError, RuntimeError, OSError) as e:
        print("nan replay: %s" % e, file=sys.stderr)
        return 2
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, lambda n, f: lad._terminate())
        signal.signal(signal.SIGINT, lambda n, f: lad._terminate())
    return lad.run()


if __name__ == "__main__":
    sys.exit(main())
