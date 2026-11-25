"""`madengine nan ledger`: status.json, the investigation's state beside its ledger."""

import json

from madengine.nan_doctor import ledger

SHA = "77ab3035" + "0" * 56


def write(path, rows):
    path.write_text("".join("2026-11-24 23:%02d:00  %s  %s\n" % r for r in rows))


def test_status_reads_the_milestones(tmp_path):
    art = tmp_path / "2026-11-24-qwen3"
    art.mkdir()
    L = art / "ledger.md"
    write(
        L,
        [
            (1, "step1", "start"),
            (2, "step1", "RUN_SCRIPT=scripts/x/run.sh"),
            (3, "step1", "end  total=0m42s"),
            (4, "step2", "skipped  folded"),
            (5, "step6[G1]", "start"),
            (6, "step6[G1]", "LADDER_RESULT=REPRODUCED at concurrency 1 host=n1"),
            (
                7,
                "step6[G1]",
                "TARBALL %s repro-G1.tar.gz size=90411477 proposed=KERNEL_BUG" % SHA,
            ),
            (8, "step6[G1]", "HANDOVER announced"),
            (9, "step6[G1]", "end  total=4m00s"),
            (10, "step7[G1]", "start"),
            (11, "step7[G1]", "job=mad-nan-rc1  recipe sweep"),
        ],
    )
    st = ledger.status(str(L))
    assert st["schema"] == ledger.STATUS_SCHEMA and st["run"] == "2026-11-24-qwen3"
    assert st["run_script"] == "scripts/x/run.sh" and st["state"] == "running"
    assert st["current"] == {
        "step": "step7[G1]",
        "since": "2026-11-24 23:10:00",
        "job": "mad-nan-rc1 recipe sweep",
    }
    assert {"step": "step1", "total": "0m42s"} in st["steps"]
    assert {"step": "step2", "skipped": "folded"} in st["steps"]
    rp = st["reproducer"]
    assert rp["sha256"] == SHA and rp["group"] == "G1" and rp["verdict"] == "KERNEL_BUG"
    assert rp["host"] == "n1" and rp["handover_at"] == "2026-11-24 23:08:00"
    assert rp["tarball"] == str(tmp_path / "repro-G1.tar.gz")
    assert st["line"].startswith("2026-11-24-qwen3: repro repro-G1.tar.gz")


def test_root_cause_fix_and_done(tmp_path):
    L = tmp_path / "ledger.md"
    write(
        L,
        [
            (1, "step7[G1]", "ROOT_CAUSE=kernel: lgkmcnt race in the epilogue"),
            (2, "step8[G1]", "FIX=WORKAROUND TENSILE_FIXED_WGMXCC=1; validated 0/100"),
            (3, "step9", "start"),
            (4, "step9", "end  total=1m00s"),
        ],
    )
    st = ledger.status(str(L))
    assert st["root_cause"]["status"] == "FOUND"
    assert st["root_cause"]["text"] == "lgkmcnt race in the epilogue"
    assert st["fix"]["kind"] == "WORKAROUND" and st["state"] == "done"
    assert st["line"].endswith("| done")


def test_every_ledger_line_rewrites_status_json(tmp_path, monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
    L = tmp_path / "ledger.md"
    ledger.append(str(L), "step1", "start")
    first = json.loads((tmp_path / "status.json").read_text())
    assert first["current"]["step"] == "step1"
    ledger.append(str(L), "step1", "end  total=0m01s")
    assert json.loads((tmp_path / "status.json").read_text())["current"] is None


def test_the_session_record_points_at_the_run(tmp_path, monkeypatch):
    sessions = tmp_path / "sessions"
    monkeypatch.setattr(ledger, "SESSIONS_TMP", str(sessions))
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "abcd1234-ef")
    art = tmp_path / "artifacts" / "run1"
    art.mkdir(parents=True)
    ledger.append(str(art / "ledger.md"), "step1", "start")
    assert (sessions / "abcd1234-ef").read_text().strip() == str(art)
    assert (tmp_path / "artifacts" / ".sessions" / "abcd1234-ef").exists()
