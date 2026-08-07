"""Finding the results CSV a model wrote, when the card did not say where it is.

The header fixtures here are copied verbatim out of MAD's own scripts rather than
invented, so a change in what the scripts emit shows up as a failure here:

- ``model,performance,metric``                       scripts/dummy/run_multi.sh:1
- ``model,performance,metric``                       scripts/mochi/run_mochi.sh:70
- ``model, performance, metric``                     scripts/pyt_chai1_inference/run.sh:39
- ``model,performance,metric,mode,precision,...``    scripts/primus_megatron-lm/
                                                     primus_megatron-lm_benchmark_report.sh:283
- ``hf_pipeline_tag,model,...,performance,metric,unit``  scripts/atom/run_atom.py:43

and the near-misses that must stay rejected:

- ``Model,xP/yD,ISL,...``             scripts/sglang_disagg/benchmark_parser.py:188
- ``model_name,model_unique_name,...``  scripts/kvcache_transfer_bench/kv_cache_estimator.py:1319
"""

import os
import time
from pathlib import Path

import pytest

from madengine.reporting import result_csv


# Real headers, as the scripts write them.
DUMMY_HEADER = "model,performance,metric"
CHAI_HEADER = "model, performance, metric"
PRIMUS_HEADER = (
    "model,performance,metric,mode,precision,batch_size,global_batch_size,"
    "seq_len,device,num_gpus"
)
# The three columns are neither first nor adjacent here.
ATOM_HEADER = (
    "hf_pipeline_tag,model,benchmark,tp,inp,out,kv_cache_dtype,num_prompts,"
    "max_concurrency,bs,cmd,performance,metric,unit"
)


def write_csv(path: Path, header: str, *rows: str) -> Path:
    """Write a CSV with *header* and *rows* verbatim, and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


def make_result_csv(path: Path, header: str = DUMMY_HEADER, count: int = 1) -> Path:
    """A results CSV with *count* measured rows."""
    rows = [f"m{i},{100 + i},tokens_per_second" for i in range(count)]
    if header == PRIMUS_HEADER:
        rows = [
            f"m{i},{100 + i},tok/s/GPU,train,BF16,1,8,4096,gfx950,16"
            for i in range(count)
        ]
    return write_csv(path, header, *rows)


class TestHeaderPredicate:
    """A results CSV is recognised by its columns, however they are spelled."""

    @pytest.mark.parametrize("header", [DUMMY_HEADER, CHAI_HEADER, PRIMUS_HEADER])
    def test_real_headers_are_recognised(self, tmp_path, header):
        assert result_csv.has_result_shape(make_result_csv(tmp_path / "r.csv", header))

    def test_the_three_columns_need_not_be_first_or_adjacent(self, tmp_path):
        path = write_csv(tmp_path / "r.csv", ATOM_HEADER, "text,m," + ",".join([""] * 9) + ",42,tok/s,x")
        assert result_csv.has_result_shape(path)

    @pytest.mark.parametrize(
        "header",
        [
            # scripts/sglang_disagg/benchmark_parser.py:188 -- a Model column, no metric
            "Model,xP/yD,ISL,OSL,Concurrency,Request Throughput (req/s)",
            # kv_cache_estimator.py:1319 -- model_name is a different column
            "model_name,model_unique_name,concurrency,seq_length,kv_cache_mb",
            # sglang_benchmark_report.py:99 -- throughput instead of performance/metric
            "model,total_throughput (tok/sec),output_throughput (tok/sec),tp",
        ],
    )
    def test_near_miss_headers_are_rejected(self, tmp_path, header):
        path = write_csv(tmp_path / "r.csv", header, "a,b,c,d")
        assert not result_csv.has_result_shape(path)

    def test_a_raw_training_log_named_csv_is_rejected(self, tmp_path):
        """benchmark_report.sh:106 tees stdout into a file with a .csv extension."""
        path = tmp_path / "primus-megatron-Megatron-LM-pretrain.csv"
        path.write_text("[INFO] iteration 1/100 | throughput 17967.4\n[INFO] done\n")
        assert not result_csv.has_result_shape(path)

    def test_column_order_does_not_matter(self, tmp_path):
        path = write_csv(tmp_path / "r.csv", "metric,model,performance", "tok/s,m,42")
        assert result_csv.has_result_shape(path)

    def test_quoted_and_uppercase_columns_are_recognised(self, tmp_path):
        path = write_csv(tmp_path / "r.csv", '"Model","Performance","Metric"', "m,42,tok/s")
        assert result_csv.has_result_shape(path)

    def test_byte_order_mark_does_not_hide_the_first_column(self, tmp_path):
        path = tmp_path / "r.csv"
        path.write_text("\ufeffmodel,performance,metric\nm,42,tok/s\n", encoding="utf-8")
        assert result_csv.has_result_shape(path)

    def test_a_csv_without_the_three_columns_is_not_a_result(self, tmp_path):
        path = write_csv(tmp_path / "gpu_info.csv", "gpu,power,temperature", "0,300,45")
        assert not result_csv.has_result_shape(path)
        assert "header lacks" in result_csv.rejection_reason(path)

    def test_a_file_that_is_not_csv_is_reported_as_such(self, tmp_path):
        path = tmp_path / "notes.csv"
        path.write_bytes(b"\x00\x01\x02")
        assert result_csv.rejection_reason(path) is not None

    def test_missing_performance_column_is_named_for_a_declared_file(self, tmp_path):
        path = write_csv(tmp_path / "r.csv", "model,metric", "m,tok/s")
        reason = result_csv.metric_rejection_reason(path)
        assert "no 'performance' column" in reason
        assert "model, metric" in reason

    def test_a_declared_file_needs_only_a_performance_column(self, tmp_path):
        """A card that named the file is trusted; only the metric has to be there."""
        path = write_csv(tmp_path / "r.csv", "run,performance", "a,42")
        assert result_csv.metric_rejection_reason(path) is None
        assert not result_csv.has_result_shape(path)

    def test_all_rows_empty_is_not_a_measurement(self, tmp_path):
        path = write_csv(tmp_path / "r.csv", DUMMY_HEADER, "m,,tok/s", "m2, ,tok/s")
        assert result_csv.metric_rejection_reason(path) == (
            "every row has an empty 'performance' value"
        )
        assert result_csv.rejection_reason(path) is not None


class TestOwnOutputsAreNeverInput:
    """madengine's own files match the predicate by construction."""

    @pytest.mark.parametrize(
        "name", ["perf.csv", "perf_super.csv", "perf_super_1.csv", "perf_entry.csv"]
    )
    def test_own_outputs_are_excluded(self, tmp_path, name):
        make_result_csv(tmp_path / name)
        assert result_csv.is_own_output(tmp_path / name)
        assert result_csv.discover([tmp_path]).winner is None

    def test_own_output_does_not_hide_a_real_result(self, tmp_path):
        make_result_csv(tmp_path / "perf.csv", count=9)
        make_result_csv(tmp_path / "perf_dummy.csv", count=1)
        assert result_csv.discover([tmp_path]).winner == tmp_path / "perf_dummy.csv"


class TestRanking:
    """Which candidate wins, and the same one every time."""

    def test_the_file_with_more_measured_rows_wins(self, tmp_path):
        thin = make_result_csv(tmp_path / "node_0" / "r.csv", count=1)
        rich = make_result_csv(tmp_path / "node_1" / "r.csv", count=8)
        assert result_csv.select_best([thin, rich]) == rich
        assert result_csv.select_best([rich, thin]) == rich

    def test_an_empty_file_loses_to_a_measured_one(self, tmp_path):
        empty = write_csv(tmp_path / "node_0" / "r.csv", DUMMY_HEADER, "m,,tok/s")
        measured = make_result_csv(tmp_path / "node_1" / "r.csv", count=1)
        assert result_csv.select_best([empty, measured]) == measured

    def test_ties_break_on_the_newer_file(self, tmp_path):
        older = make_result_csv(tmp_path / "a" / "r.csv", count=2)
        newer = make_result_csv(tmp_path / "b" / "r.csv", count=2)
        os.utime(older, (1_000_000, 1_000_000))
        os.utime(newer, (2_000_000, 2_000_000))
        assert result_csv.select_best([older, newer]) == newer
        assert result_csv.select_best([newer, older]) == newer

    def test_a_full_tie_keeps_the_order_it_was_given(self, tmp_path):
        first = make_result_csv(tmp_path / "a" / "r.csv", count=2)
        second = make_result_csv(tmp_path / "b" / "r.csv", count=2)
        os.utime(first, (1_000_000, 1_000_000))
        os.utime(second, (1_000_000, 1_000_000))
        assert result_csv.select_best([first, second]) == first
        assert result_csv.select_best([second, first]) == second

    def test_no_candidates_is_not_an_error(self):
        assert result_csv.select_best([]) is None


class TestDiscovery:
    """What a depth-1 search finds, and what it refuses to."""

    def test_a_results_csv_in_the_run_directory_is_found(self, tmp_path):
        run_dir = tmp_path / "run_directory"
        found = make_result_csv(run_dir / "perf_primus-megatron-Megatron-LM.csv", PRIMUS_HEADER)
        assert result_csv.discover([run_dir, tmp_path]).winner == found

    def test_the_parent_of_the_run_directory_is_searched_too(self, tmp_path):
        """Primus writes to $(pwd)/../ -- benchmark_report.sh:109."""
        run_dir = tmp_path / "run_directory"
        run_dir.mkdir()
        found = make_result_csv(tmp_path / "perf_primus-megatron-Megatron-LM.csv", PRIMUS_HEADER)
        assert result_csv.discover([run_dir, tmp_path]).winner == found

    def test_the_search_does_not_walk_the_tree(self, tmp_path):
        make_result_csv(tmp_path / "deep" / "nested" / "r.csv")
        assert result_csv.discover([tmp_path]).winner is None

    def test_a_decoy_csv_is_rejected_and_explained(self, tmp_path):
        write_csv(tmp_path / "gpu_info_power.csv", "gpu,power", "0,300")
        discovery = result_csv.discover([tmp_path])
        assert discovery.winner is None
        assert discovery.seen == 1
        rejected_paths = [str(path) for path, _ in discovery.rejected]
        assert str(tmp_path / "gpu_info_power.csv") in rejected_paths

    def test_the_same_file_reached_twice_is_counted_once(self, tmp_path):
        make_result_csv(tmp_path / "r.csv")
        discovery = result_csv.discover([tmp_path, tmp_path, tmp_path / "missing"])
        assert discovery.seen == 1
        assert len(discovery.searched) == 1

    def test_a_file_from_an_earlier_model_is_not_this_run(self, tmp_path):
        """The workspace root is shared, so a stale CSV must not be adopted."""
        stale = make_result_csv(tmp_path / "r.csv", count=4)
        os.utime(stale, (1_000_000, 1_000_000))
        run_started = time.time()
        discovery = result_csv.discover([tmp_path], min_mtime=run_started)
        assert discovery.winner is None
        assert discovery.rejected[0][1] == "written before this run started"

    def test_a_file_written_during_the_run_is_kept(self, tmp_path):
        run_started = time.time()
        fresh = make_result_csv(tmp_path / "r.csv", count=4)
        assert result_csv.discover([tmp_path], min_mtime=run_started).winner == fresh

    def test_an_excluded_path_is_skipped(self, tmp_path):
        skip_me = make_result_csv(tmp_path / "r.csv")
        assert result_csv.discover([tmp_path], excluded=[skip_me]).winner is None


class TestLogReportingModelsStaySilent:
    """Some models report only through stdout, e.g. scripts/huggingface_gpt2/run.sh:86.

    Of the 39 cards that declare no multiple_results, four write no results CSV at all;
    most of the rest write one to /run_logs on shared storage, which no depth-1 search
    reaches. Either way the search must come back empty without adding noise.
    """

    def test_a_directory_with_no_csv_yields_nothing_and_no_noise(self, tmp_path):
        (tmp_path / "run.log").write_text("performance: 14164 samples_per_second\n")
        discovery = result_csv.discover([tmp_path])
        assert discovery.winner is None
        assert discovery.seen == 0
        assert discovery.rejected == []

    def test_a_missing_directory_is_not_an_error(self, tmp_path):
        discovery = result_csv.discover([tmp_path / "never_created"])
        assert discovery.winner is None
        assert discovery.searched == []


class TestDescription:
    """The diagnostic says where it looked and why each file was refused."""

    def test_it_names_the_directories_and_the_reasons(self, tmp_path):
        write_csv(tmp_path / "gpu_info.csv", "gpu,power", "0,300")
        lines = result_csv.describe(result_csv.discover([tmp_path]))
        assert any(str(tmp_path) in line for line in lines)
        assert any("CSV files seen: 1" in line for line in lines)
        assert any("header lacks" in line for line in lines)

    def test_it_stays_short_when_there_is_a_lot_to_say(self, tmp_path):
        for index in range(12):
            write_csv(tmp_path / f"decoy_{index}.csv", "gpu,power", "0,300")
        lines = result_csv.describe(result_csv.discover([tmp_path]), limit=3)
        assert len(lines) == 6
        assert "and 9 more" in lines[-1]


class TestSettleResultsCsv:
    """Which file the Docker path reports from, and what it says about the choice."""

    def settle(self, tmp_path, monkeypatch, declared=None, min_mtime=None):
        from madengine.execution.container_runner import _settle_results_csv

        monkeypatch.chdir(tmp_path)
        said = []
        model_info = {"name": "dummy"}
        if declared is not None:
            model_info["multiple_results"] = declared
        path, discovery = _settle_results_csv(
            model_info, "run_directory", said.append, min_mtime=min_mtime
        )
        return path, discovery, "\n".join(said)

    def test_a_declared_file_that_exists_wins(self, tmp_path, monkeypatch):
        """Even against a candidate with more rows: the card said where to look."""
        (tmp_path / "run_directory").mkdir()
        make_result_csv(tmp_path / "run_directory" / "declared.csv", count=1)
        make_result_csv(tmp_path / "run_directory" / "richer.csv", count=9)
        path, discovery, said = self.settle(tmp_path, monkeypatch, declared="declared.csv")
        assert Path(path).name == "declared.csv"
        assert discovery is None
        assert said == ""

    def test_a_typo_recovers_and_says_both_things(self, tmp_path, monkeypatch):
        (tmp_path / "run_directory").mkdir()
        make_result_csv(tmp_path / "run_directory" / "perf_dummy.csv", count=2)
        path, _, said = self.settle(tmp_path, monkeypatch, declared="perf_dumy.csv")
        assert Path(path).name == "perf_dummy.csv"
        assert "declares multiple_results='perf_dumy.csv' but no such file" in said
        assert "found by its header" in said
        assert "the declared file was not there" in said

    def test_an_undeclared_file_is_found_and_named(self, tmp_path, monkeypatch):
        (tmp_path / "run_directory").mkdir()
        make_result_csv(tmp_path / "run_directory" / "perf_dummy.csv", count=2)
        path, _, said = self.settle(tmp_path, monkeypatch)
        assert Path(path).name == "perf_dummy.csv"
        assert "declares no multiple_results" in said
        assert "Warning: model" not in said

    def test_the_workspace_root_is_searched_after_the_run_directory(self, tmp_path, monkeypatch):
        (tmp_path / "run_directory").mkdir()
        make_result_csv(tmp_path / "perf_primus-megatron-Megatron-LM.csv", PRIMUS_HEADER, count=4)
        path, _, said = self.settle(tmp_path, monkeypatch)
        assert Path(path).name == "perf_primus-megatron-Megatron-LM.csv"

    def test_nothing_found_reports_nothing_and_does_not_raise(self, tmp_path, monkeypatch):
        (tmp_path / "run_directory").mkdir()
        path, discovery, said = self.settle(tmp_path, monkeypatch)
        assert path is None
        assert discovery.winner is None
        assert said == ""

    def test_a_log_only_model_stays_silent(self, tmp_path, monkeypatch):
        """A card with no CSV writer must not gain a warning it never had."""
        (tmp_path / "run_directory").mkdir()
        (tmp_path / "run.log").write_text("performance: 14164 samples_per_second\n")
        path, _, said = self.settle(tmp_path, monkeypatch)
        assert path is None
        assert said == ""

    def test_a_previous_model_result_is_not_adopted(self, tmp_path, monkeypatch):
        (tmp_path / "run_directory").mkdir()
        stale = make_result_csv(tmp_path / "perf_other_model.csv", count=3)
        os.utime(stale, (1_000_000, 1_000_000))
        path, _, _ = self.settle(tmp_path, monkeypatch, min_mtime=time.time())
        assert path is None

    def test_madengine_own_perf_csv_is_never_adopted(self, tmp_path, monkeypatch):
        (tmp_path / "run_directory").mkdir()
        make_result_csv(tmp_path / "perf.csv", count=5)
        path, _, _ = self.settle(tmp_path, monkeypatch)
        assert path is None
