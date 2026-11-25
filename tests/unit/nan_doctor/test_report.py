"""report.py: the step-9 draft from the artifact dir (stdlib only, no torch)."""

import json
import sys

from madengine.nan_doctor import report


def draft(A, capsys):
    old = sys.argv
    sys.argv = ["report", str(A)]
    try:
        report.main()
    finally:
        sys.argv = old
    return capsys.readouterr().out


def test_group_of_op_and_chain_packages():
    assert report.group_of("verify-G1-op/x/out/c1.1.h") == "G1"
    assert report.group_of("verify-G2-chain/x/out/c1.1.h") == "G2"
    assert report.group_of("replay/G3/c2.1.h") == "G3"


def test_redraft_keeps_the_old_report_and_drafts_a_result_card(tmp_path, capsys):
    (tmp_path / "ledger.md").write_text(
        "2026-01-01 10:00:00  step5n  skipped -- G1 reproduced\n"
    )
    (tmp_path / "groups_priority.txt").write_text(
        "G1 3 caps/a.pt aten.mm.default\nG2 1 caps/b.pt aten.mm.default\n"
        "G3 1 caps/c.pt aten.bmm.default\n"
    )
    att = tmp_path / "replay" / "G1" / "c1.1.host"
    att.mkdir(parents=True)
    (att / "summary.txt").write_text(
        "=== totals: copies NAN 1/1; non-finite 3/100 ===\nATTEMPT_RESULT=REPRODUCED\n"
    )
    out = draft(tmp_path, capsys)
    assert "DRAFT" in out and (tmp_path / "result.json").exists()
    json.loads((tmp_path / "result.json").read_text())
    rep = (tmp_path / "report.md").read_text()
    assert "same op as G1 (replayed) -- folded into it" in rep
    assert "G3 1 caps/c.pt aten.bmm.default  -- **not replayed**: TODO(agent)" in rep
    assert "Step 5N did not run: -- G1 reproduced" in rep
    assert "## Result card" in rep
    out = draft(tmp_path, capsys)
    assert "the earlier report.md is now report.prev1.md" in out
    assert (tmp_path / "report.prev1.md").exists()
