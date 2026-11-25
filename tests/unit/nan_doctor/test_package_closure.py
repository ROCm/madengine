"""The reproducer's runtime closure covers every module replay_nan_op imports."""

import ast
from pathlib import Path

from madengine.nan_doctor import package

RT = (
    Path(package.__file__).resolve().parents[1]
    / "scripts"
    / "common"
    / "tools"
    / "nan_doctor"
    / "nan_doctor_rt"
)


def _relative_imports(path):
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.level == 1:
            if node.module:
                names.add(node.module.split(".")[0])
            else:
                names.update(a.name for a in node.names)
    return names


def test_closure_has_every_module_the_replay_imports():
    shipped = {Path(f).stem for f in package.CLOSURE if f.endswith(".py")}
    todo, seen = ["replay_nan_op", "transformer_hog", "pressure_hog"], set()
    while todo:
        mod = todo.pop()
        if mod in seen or not (RT / (mod + ".py")).exists():
            continue
        seen.add(mod)
        todo.extend(_relative_imports(RT / (mod + ".py")))
    assert seen - shipped == set(), "not shipped in the reproducer: %s" % sorted(
        seen - shipped
    )


def test_runtime_output_diff_is_the_host_copy():
    """diff_tap's check mode imports nan_doctor_rt.output_diff; `madengine nan diff` uses the host copy."""
    host = Path(package.__file__).resolve().parent / "output_diff.py"
    assert (RT / "output_diff.py").read_text() == host.read_text()
