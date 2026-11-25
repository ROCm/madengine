"""handoff.py: native/handoff.md for step 5N from the capsule path's artifacts."""

import sys

from madengine.nan_doctor import handoff


def write_handoff(A):
    old = sys.argv
    sys.argv = ["handoff", str(A)]
    try:
        handoff.main()
    finally:
        sys.argv = old
    return (A / "native" / "handoff.md").read_text()


def test_workload():
    assert handoff.workload(["RUN_SCRIPT=scripts/vllm/run.sh"]) == "inference"
    assert handoff.workload(["RUN_SCRIPT=train.sh"]) == "training"
    assert handoff.workload(["something else"]) == ""


def test_collective_replay_is_an_attempt_and_a_null(tmp_path):
    (tmp_path / "ledger.md").write_text(
        "2026-01-01 10:00:00  step1  RUN_SCRIPT=scripts/vllm/run.sh "
        "NAN_SIGNAL=wrong=yes\n"
    )
    d = tmp_path / "replay" / "G1"
    d.mkdir(parents=True)
    (d / "collective.42.host1.out").write_text(
        "collective vllm.all_reduce\nR3 vllm: path quick_reduce\n"
        "VERDICT CLEAN ranks [0, 1]\nLADDER_RESULT=NOT_REPRODUCED 0/2000\n"
    )
    md = write_handoff(tmp_path)
    assert "(no 5b attempt on disk)" not in md
    assert (
        "COLLECTIVE groups" in md and "| G1 | `replay/G1/collective.42.host1.out`" in md
    )
    assert "collective G1 replay (`vllm.all_reduce`" in md and "0/2000" in md
    assert "Collective G1 replayed clean on its recorded inputs" in md
    assert "A finite signal (wrong values, no NaN)" in md
