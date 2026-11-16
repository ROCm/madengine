"""The 5b concurrency ladder, run against fake copies (no docker, no GPU).

Each fake copy is a small Python process that behaves like `replay_nan_op --progress-file
--stop-file`: it counts runs, writes "<runs> <hits>" to its progress file, honours the node-wide
STOP file, and ends with the "non-finite on H/R chain runs" line the summary reads.
"""

import subprocess
import sys

import pytest

from madengine.nan_doctor import replay

FAKE = r"""
import os, re, sys, time
cmd = sys.argv[1]
scenario = sys.argv[2]
c = int(re.search(r"NAN_REPLAY_CONCURRENCY=(\d+)", cmd).group(1))
progress = re.search(r"--progress-file (\S+)", cmd).group(1)
stop = re.search(r"--stop-file (\S+)", cmd).group(1)
iters = int(re.search(r"--chain-iters (\d+)", cmd).group(1))
if scenario == "oom_above_1" and c > 1:
    print("torch.OutOfMemoryError: HIP out of memory"); sys.exit(1)
runs = hits = 0
while runs < iters and not os.path.exists(stop):
    runs += 10
    if scenario in ("repro", "oom_above_1"):
        hits += 1
    open(progress, "w").write("%d %d\n" % (runs, hits))
    time.sleep(0.005)
print("[C1] chain walk")
print("[verdict] walk done")
print("chain loop: non-finite on %d/%d chain runs" % (hits, runs))
sys.exit(1 if hits == 0 else 0)
"""

PROBE = (
    "CAPTURE_PROCS=2\nCAPTURE_PROCS_DETAIL=2 processes at capture\nCAPTURE_OP_INDEX=5\n"
)


def make(tmp_path, scenario, **kw):
    lines = []
    cap = tmp_path / "capsule.pt"
    cap.write_bytes(b"x")
    log_dir = tmp_path / "out"
    launched = []

    def popen(gpu, cmd, log, name):
        launched.append((gpu, cmd, name))
        f = open(log, "w")
        p = subprocess.Popen(
            [sys.executable, "-c", FAKE, cmd, scenario],
            stdout=f,
            stderr=subprocess.STDOUT,
        )
        p._nan_logf = f
        return p

    opts = dict(
        image="img",
        ngpus=2,
        iters=200,
        stop_hits=3,
        futility=2,
        poll=0.02,
        popen=popen,
        probe=lambda: PROBE,
        out=lines.append,
    )
    opts.update(kw)
    lad = replay.Ladder(str(cap), str(log_dir), **opts)
    return lad, lines, launched


def result(lines):
    return [x for x in lines if x.startswith("LADDER_RESULT=")][-1]


def test_a_reproducing_capsule_stops_every_copy_at_the_hit_target(tmp_path):
    lad, lines, launched = make(tmp_path, "repro")
    assert lad.run() == 0
    assert (
        result(lines) == "LADDER_RESULT=REPRODUCED at concurrency 3"
    )  # capture 2 procs + 1
    assert len(launched) == 2 * 3  # 2 GPUs x 3 copies
    assert sorted({g for g, _, _ in launched}) == [0, 1]
    d = next(p for p in (tmp_path / "out").iterdir() if p.is_dir())
    assert "hits=" in (d / "STOP").read_text()
    assert (
        (d / "summary.txt").read_text().rstrip().endswith("ATTEMPT_RESULT=REPRODUCED")
    )


def test_copies_overlap_at_a_barrier_and_share_the_stop_file(tmp_path):
    lad, _, launched = make(tmp_path, "repro")
    lad.run()
    _, cmd, _ = launched[0]
    assert "--start-barrier" in cmd and "--barrier-n 3" in cmd and "--stop-file" in cmd
    assert "--chain --chain-fallback --chain-iters 200 --restore-env" in cmd
    assert "python3 -m nan_doctor_rt.replay_nan_op" in cmd


def test_a_clean_attempt_is_not_reproduced_and_does_not_climb(tmp_path):
    lad, lines, launched = make(tmp_path, "clean", futility=0)
    lad.run()
    assert result(lines) == "LADDER_RESULT=NOT_REPRODUCED at concurrency 3"
    assert len(launched) == 6


def test_futility_ends_a_hitless_attempt_as_not_set_up(tmp_path):
    lad, lines, _ = make(
        tmp_path, "clean", iters=100000, futility=2
    )  # 2 x op index 5 = 10 runs
    lad.run()
    assert "NOT_SET_UP at concurrency 3" in result(lines)
    assert "escalate" in result(lines)


def test_out_of_memory_steps_down_one_copy_at_a_time(tmp_path):
    lad, lines, launched = make(tmp_path, "oom_above_1")
    lad.run()
    assert result(lines) == "LADDER_RESULT=REPRODUCED at concurrency 1"
    assert "c3:OOM c2:OOM c1:REPRODUCED" in "\n".join(lines)


def test_an_explicit_concurrency_and_a_cotenant_set_the_start(tmp_path):
    lad, lines, launched = make(tmp_path, "repro", concurrency=2)
    lad.run()
    assert result(lines).endswith("at concurrency 2") and len(launched) == 4
    lad, lines, launched = make(
        tmp_path / "b" if (tmp_path / "b").mkdir() is None else tmp_path,
        "repro",
        cotenant="python3 -m nan_doctor_rt.transformer_hog --mode fwdbwd",
    )
    lad.run()
    assert result(lines).endswith("at concurrency 1")
    assert (
        "--contend-with 'python3 -m nan_doctor_rt.transformer_hog --mode fwdbwd'"
        in launched[0][1]
    )


def test_an_unreadable_capsule_contention_is_an_error_not_a_guess(tmp_path):
    lad, lines, launched = make(tmp_path, "repro", probe=lambda: "no such line\n")
    assert lad.run() == 1
    assert result(lines).startswith("LADDER_RESULT=ERROR") and not launched


def test_the_payload_forms(tmp_path):
    lad, _, _ = make(
        tmp_path, "repro", mode="op", seconds=30, extra_args="--synthesize"
    )
    p = lad.payload()
    assert "--iters 200 --restore-env" in p and "--chain" not in p
    assert "--seconds 30" in p and p.endswith("--synthesize")


def test_bad_inputs_are_refused(tmp_path):
    with pytest.raises(ValueError):
        make(tmp_path, "repro", mode="other")
    with pytest.raises(ValueError):
        make(tmp_path, "repro", cotenant="echo 'x'")


def test_read_progress_sums_every_copy(tmp_path):
    (tmp_path / "g0_k0.progress").write_text("100 2\n")
    (tmp_path / "g1_k0.progress").write_text("50 1\n")
    (tmp_path / "junk.progress").write_text("x")
    (tmp_path / "STOP").write_text("")
    assert replay.read_progress(str(tmp_path)) == (150, 3)
