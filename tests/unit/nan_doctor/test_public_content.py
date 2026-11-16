"""madengine is a public repository: nothing that identifies a customer, a compute site, a person,
an unreleased product or a credential may ship with the nan-doctor code.

Built-in patterns (credentials, home directories of people, evidence citations, e-mail
addresses) always run. A private list of extra terms -- kept OUT of the repository, so the list
does not publish what it protects -- is read from the file named by $PUBLIC_CONTENT_DENYLIST;
CI sets it from a secret. Without it only the built-in patterns run.
"""

import os
from pathlib import Path

import pytest

from .public_content_scan import load_patterns, scan

ROOT = Path(__file__).resolve().parents[3]
# Files that hold the scanner's own pattern table and its test data.
SELF = ("public_content_scan.py", "test_public_content.py")
PATHS = [
    ROOT / "src" / "madengine" / "nan_doctor",
    ROOT / "src" / "madengine" / "scripts" / "common" / "tools" / "nan_doctor",
    ROOT
    / "src"
    / "madengine"
    / "scripts"
    / "common"
    / "pre_scripts"
    / "nan_doctor_start.sh",
    ROOT
    / "src"
    / "madengine"
    / "scripts"
    / "common"
    / "post_scripts"
    / "nan_doctor_collect.sh",
    ROOT / "src" / "madengine" / "cli" / "commands" / "nan.py",
    ROOT / "tests" / "unit" / "nan_doctor",
    ROOT / "tests" / "fixtures" / "nan_doctor",
    ROOT / "tests" / "integration" / "test_nan_doctor_tool.py",
    ROOT / "docs" / "nan-doctor.md",
]


def _format(hits):
    return "\n".join(f"{p}:{n}: [{why}] {m!r}" for p, n, why, m, _ in hits)


def test_no_public_content_violations_with_builtin_patterns():
    # The scanner's own pattern table is excluded: it names the categories it looks for.
    hits = [
        h
        for h in scan([str(p) for p in PATHS if p.exists()])
        if not h[0].endswith(SELF)
    ]
    assert not hits, "non-public content:\n" + _format(hits)


def test_no_public_content_violations_with_the_private_list():
    listing = os.environ.get("PUBLIC_CONTENT_DENYLIST")
    if not listing:
        pytest.skip("PUBLIC_CONTENT_DENYLIST is not set (CI sets it from a secret)")
    hits = [
        h
        for h in scan(
            [str(p) for p in PATHS if p.exists()], extra=load_patterns(listing)
        )
        if not h[0].endswith(SELF)
    ]
    assert not hits, "non-public content:\n" + _format(hits)


def test_the_scanner_catches_what_it_claims_to(tmp_path):
    bad = tmp_path / "x.py"
    bad.write_text(
        "tok = 'ghp_" + "a" * 36 + "'\n"
        "# see /home/someone/work\n"
        "# measured [E41]\n"
        "# mail a.b@corp.com\n"
    )
    why = {h[2] for h in scan([str(bad)])}
    assert why == {
        "GitHub token",
        "home directory of a person",
        "evidence citation",
        "e-mail address",
    }
