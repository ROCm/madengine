"""--additional-context-file and --additional-context combine key by key.

A cluster profile kept in a file (partition, GPUs per node) plus a per-run string
(node count) must give both. A shallow update let the string's `slurm` object replace
the file's whole, so the run silently lost its partition.
"""

import json

from madengine.cli.validators import merge_additional_context_from_sources


def _file(tmp_path, data):
    p = tmp_path / "cluster.json"
    p.write_text(json.dumps(data))
    return str(p)


def test_cli_keys_merge_into_file_keys(tmp_path):
    f = _file(tmp_path, {"slurm": {"partition": "amd-rccl", "gpus_per_node": 8},
                         "env_vars": {"A": "1"}})
    ctx, _ = merge_additional_context_from_sources(
        '{"slurm": {"nodes": 4}, "env_vars": {"B": "2"}}', f)
    assert ctx["slurm"] == {"partition": "amd-rccl", "gpus_per_node": 8, "nodes": 4}
    assert ctx["env_vars"] == {"A": "1", "B": "2"}


def test_cli_wins_on_the_same_key(tmp_path):
    f = _file(tmp_path, {"slurm": {"partition": "amd-rccl", "exclusive": True}})
    ctx, _ = merge_additional_context_from_sources('{"slurm": {"exclusive": false}}', f)
    assert ctx["slurm"] == {"partition": "amd-rccl", "exclusive": False}


def test_either_source_alone(tmp_path):
    f = _file(tmp_path, {"slurm": {"partition": "p"}})
    assert merge_additional_context_from_sources("{}", f)[0] == {"slurm": {"partition": "p"}}
    assert merge_additional_context_from_sources('{"x": 1}', None)[0] == {"x": 1}
