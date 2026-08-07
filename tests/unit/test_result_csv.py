"""Recognising a results CSV by its header, and saying so when a run read no metric.

The header fixtures here are copied verbatim out of MAD's own scripts rather than
invented, so a change in what the scripts emit shows up as a failure here:

- ``model,performance,metric``                       scripts/dummy/run_multi.sh:1
- ``model, performance, metric``                     scripts/pyt_chai1_inference/run.sh:39
- ``model,performance,metric,mode,precision,...``    scripts/primus_megatron-lm/
                                                     primus_megatron-lm_benchmark_report.sh:283
- ``hf_pipeline_tag,model,...,performance,metric,unit``  scripts/atom/run_atom.py:43

and the near-misses that must stay rejected:

- ``Model,xP/yD,ISL,...``               scripts/sglang_disagg/benchmark_parser.py:188
- ``model_name,model_unique_name,...``  scripts/kvcache_transfer_bench/
                                        kv_cache_estimator.py:1319

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

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
DISAGG_HEADER = "Model,xP/yD,ISL,OSL,concurrency"
ESTIMATOR_HEADER = "model_name,model_unique_name,num_layers"


def write_csv(path: Path, header: str, *rows: str) -> Path:
    """Write a CSV with *header* and *rows* verbatim, and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


def make_result_csv(path: Path, header: str = DUMMY_HEADER, count: int = 1) -> Path:
    """A results CSV with *count* measured rows."""
    if header == PRIMUS_HEADER:
        rows = [
            f"m{i},{100 + i},tok/s/GPU,train,BF16,1,8,4096,gfx950,16"
            for i in range(count)
        ]
    else:
        rows = [f"m{i},{100 + i},tokens_per_second" for i in range(count)]
    return write_csv(path, header, *rows)


class TestHeaderPredicate:
    """A results CSV is recognised by its columns, however they are spelled."""

    @pytest.mark.parametrize("header", [DUMMY_HEADER, CHAI_HEADER, PRIMUS_HEADER])
    def test_real_headers_are_recognised(self, tmp_path, header):
        assert result_csv.has_result_shape(make_result_csv(tmp_path / "r.csv", header))

    def test_the_three_columns_need_not_be_first_or_adjacent(self, tmp_path):
        path = write_csv(
            tmp_path / "r.csv", ATOM_HEADER, "text,m," + ",".join([""] * 9) + ",42,tok/s,x"
        )
        assert result_csv.has_result_shape(path)

    @pytest.mark.parametrize("header", [DISAGG_HEADER, ESTIMATOR_HEADER])
    def test_near_misses_are_rejected(self, tmp_path, header):
        """A capitalised ``Model`` is fine, but a header without all three is not."""
        assert not result_csv.has_result_shape(write_csv(tmp_path / "r.csv", header, "a,b,c"))

    def test_case_and_quoting_do_not_matter(self, tmp_path):
        path = write_csv(tmp_path / "r.csv", '"Model", PERFORMANCE ,"metric"', "m,1,t/s")
        assert result_csv.has_result_shape(path)

    def test_a_missing_file_is_not_a_results_csv(self, tmp_path):
        assert result_csv.missing_columns(tmp_path / "absent.csv") is None
        assert not result_csv.has_result_shape(tmp_path / "absent.csv")

    def test_an_empty_file_has_no_header(self, tmp_path):
        empty = tmp_path / "r.csv"
        empty.write_text("", encoding="utf-8")
        assert result_csv.read_columns(empty) is None


class TestOwnOutputs:
    """madengine's own files satisfy the predicate, so they are never candidates."""

    @pytest.mark.parametrize(
        "name", ["perf.csv", "PERF.CSV", "perf_super.csv", "perf_entry_1.csv"]
    )
    def test_own_outputs_are_recognised(self, name):
        assert result_csv.is_own_output(name)

    @pytest.mark.parametrize("name", ["results.csv", "perf_of_model.csv", "my_perf.csv"])
    def test_other_files_are_not(self, name):
        assert not result_csv.is_own_output(name)


class TestMetricRejectionReason:
    """What a declared file is asked: can a metric be read out of it?"""

    def test_a_measured_file_is_accepted(self, tmp_path):
        assert result_csv.metric_rejection_reason(make_result_csv(tmp_path / "r.csv")) is None

    def test_a_two_column_csv_still_reports(self, tmp_path):
        """A declared file is trusted, so the three-column shape is not demanded of it."""
        path = write_csv(tmp_path / "r.csv", "model,performance", "m,42")
        assert result_csv.metric_rejection_reason(path) is None

    def test_a_missing_performance_column_names_what_was_found(self, tmp_path):
        path = write_csv(tmp_path / "r.csv", "model,throughput", "m,42")
        reason = result_csv.metric_rejection_reason(path)
        assert "no 'performance' column" in reason
        assert "throughput" in reason

    def test_rows_without_a_number_are_no_metric(self, tmp_path):
        path = write_csv(tmp_path / "r.csv", DUMMY_HEADER, "m,,tokens_per_second", "m2, ,x")
        assert result_csv.metric_rejection_reason(path) == (
            "every row has an empty 'performance' value"
        )

    def test_an_unreadable_file_says_so(self, tmp_path):
        assert result_csv.metric_rejection_reason(tmp_path / "absent.csv") == (
            "not readable as CSV"
        )

    def test_counting_measured_rows(self, tmp_path):
        path = write_csv(tmp_path / "r.csv", DUMMY_HEADER, "m,1,t/s", "m2,,t/s", "m3,3,t/s")
        assert result_csv.count_rows(path) == (2, 3)


class TestSuggestions:
    """What a run says about the files beside it when it read no metric."""

    def test_shape_matching_files_are_suggested(self, tmp_path):
        make_result_csv(tmp_path / "results.csv")
        write_csv(tmp_path / "params.csv", "name,value", "lr,3e-4")
        assert result_csv.suggest_candidates([tmp_path]) == [tmp_path / "results.csv"]

    def test_own_outputs_are_not_suggested(self, tmp_path):
        make_result_csv(tmp_path / "perf.csv")
        make_result_csv(tmp_path / "perf_super.csv")
        assert result_csv.suggest_candidates([tmp_path]) == []

    def test_the_search_is_depth_one(self, tmp_path):
        make_result_csv(tmp_path / "nested" / "results.csv")
        assert result_csv.suggest_candidates([tmp_path]) == []

    def test_directories_that_do_not_exist_are_skipped(self, tmp_path):
        make_result_csv(tmp_path / "results.csv")
        found = result_csv.suggest_candidates([tmp_path / "absent", None, tmp_path])
        assert found == [tmp_path / "results.csv"]

    def test_the_same_file_is_offered_once(self, tmp_path):
        make_result_csv(tmp_path / "results.csv")
        assert result_csv.suggest_candidates([tmp_path, tmp_path]) == [
            tmp_path / "results.csv"
        ]

    def test_lines_name_the_candidates_and_the_field_to_declare(self, tmp_path):
        make_result_csv(tmp_path / "results.csv")
        lines = result_csv.suggestion_lines([tmp_path])
        assert "multiple_results" in lines[0]
        assert str(tmp_path / "results.csv") in lines[1]

    def test_lines_say_where_it_looked_when_nothing_matches(self, tmp_path):
        write_csv(tmp_path / "params.csv", "name,value", "lr,3e-4")
        lines = result_csv.suggestion_lines([tmp_path])
        assert len(lines) == 1
        assert str(tmp_path) in lines[0]

    def test_a_long_list_is_cut_off(self, tmp_path):
        for index in range(5):
            make_result_csv(tmp_path / f"r{index}.csv")
        lines = result_csv.suggestion_lines([tmp_path], limit=2)
        assert lines[-1] == "  ... and 3 more"
