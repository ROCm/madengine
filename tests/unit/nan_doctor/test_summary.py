"""summary.py: one attempt's copy logs -> per-copy status, totals, ATTEMPT_RESULT."""

import os

from madengine.nan_doctor import summary


def attempt(tmp_path, logs, stop=None, futile=False):
    d = tmp_path / "c3.1.host"
    d.mkdir()
    for name, text in logs.items():
        (d / name).write_text(text)
    if stop is not None:
        (d / "STOP").write_text(stop)
    if futile:
        (d / "FUTILE").write_text("999\n")
    return str(d)


NAN = "[C1] chain walk\n[verdict] ok\nchain loop: non-finite on 2/1000 chain runs\n"
CLEAN = "chain loop: non-finite on 0/1000 chain runs\n"


def test_any_nan_copy_means_reproduced(tmp_path):
    text, res = summary.summarize(
        attempt(tmp_path, {"g0_k0.h.log": NAN, "g0_k1.h.log": CLEAN})
    )
    assert res == "REPRODUCED"
    assert "NAN" in text and "CLEAN" in text and "non-finite 2/2000" in text
    assert text.rstrip().endswith("ATTEMPT_RESULT=REPRODUCED")


def test_all_clean_is_clean(tmp_path):
    assert summary.summarize(attempt(tmp_path, {"g0_k0.h.log": CLEAN}))[1] == "CLEAN"


def test_oom_beats_clean_but_not_a_hit(tmp_path):
    oom = "torch.OutOfMemoryError: out of memory\n"
    assert (
        summary.summarize(
            attempt(tmp_path, {"g0_k0.h.log": oom, "g0_k1.h.log": CLEAN})
        )[1]
        == "OOM"
    )


def test_a_copy_without_a_loop_line_is_an_error(tmp_path):
    assert (
        summary.summarize(attempt(tmp_path, {"g0_k0.h.log": "Traceback: boom\n"}))[1]
        == "ERROR"
    )


def test_no_logs_is_an_error(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    assert summary.summarize(str(d))[1] == "ERROR"


def test_stopped_copies_with_no_hits_are_not_clean(tmp_path):
    d = attempt(tmp_path, {"g0_k0.h.log": CLEAN}, stop="hits=0")
    text, res = summary.summarize(d)
    assert res == "ERROR" and "STOPPED" in text  # nothing clean, nothing hit


def test_futility_is_not_set_up(tmp_path):
    d = attempt(
        tmp_path, {"g0_k0.h.log": CLEAN}, stop="futility runs=9 expect=9", futile=True
    )
    text, res = summary.summarize(d)
    assert res == "NOT_SET_UP" and "futility stop" in text and "not a negative" in text


def test_runtime_noise_in_the_walk_is_dropped_and_counted(tmp_path):
    log = (
        "[C1] chain walk\n[aiter] WARNING x\nUserWarning: y\nreal row\n[verdict] done\n"
        + NAN.split("\n", 2)[2]
    )
    text, _ = summary.summarize(attempt(tmp_path, {"g0_k0.h.log": log}))
    assert (
        "real row" in text
        and "WARNING x" not in text
        and "2 runtime-noise line(s) dropped" in text
    )


def test_main_writes_summary_txt(tmp_path, capsys):
    d = attempt(tmp_path, {"g0_k0.h.log": NAN})
    assert summary.main([d]) == 0
    assert os.path.exists(os.path.join(d, "summary.txt"))
    assert "ATTEMPT_RESULT=REPRODUCED" in capsys.readouterr().out
