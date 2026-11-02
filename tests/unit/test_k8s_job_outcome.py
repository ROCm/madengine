"""When a Kubernetes Job counts as finished, and how.

A Job's pod counters say less than they seem: `succeeded` is non-zero once ONE pod
of a multi-pod Job finishes, and `failed` is non-zero while retries are still to
come. The Job's own conditions say when it is done.
"""

from types import SimpleNamespace

from madengine.deployment.kubernetes import job_finished_failed, job_finished_successfully


def _job(conditions=(), succeeded=0, failed=0, completions=2):
    conds = [SimpleNamespace(type=t, status="True") for t in conditions]
    return SimpleNamespace(status=SimpleNamespace(conditions=conds, succeeded=succeeded, failed=failed),
                           spec=SimpleNamespace(completions=completions))


class TestJobOutcome:
    def test_one_finished_pod_of_two_is_not_success(self):
        assert not job_finished_successfully(_job(succeeded=1))
        assert job_finished_successfully(_job(succeeded=2))

    def test_conditions_decide(self):
        assert job_finished_successfully(_job(["Complete"], succeeded=1))
        assert job_finished_failed(_job(["Failed"], failed=1))

    def test_a_failed_pod_is_not_a_failed_job(self):
        # Retries under backoffLimit are still to come until the Failed condition.
        assert not job_finished_failed(_job(failed=1))


def test_collector_pod_names_are_valid_and_distinct():
    import re

    from madengine.deployment.k8s_results import collector_pod_name

    a = collector_pod_name("madengine-vllm-multinode-pyt-vllm-kimi-k3-mi300x-pp2xtp8")
    b = collector_pod_name("madengine-vllm-disagg-pyt-vllm-disagg-mori-deepseek-v3")
    for n in (a, b, collector_pod_name("madengine-x.y")):
        assert re.match(r"^[a-z0-9]([-a-z0-9.]*[a-z0-9])?$", n) and len(n) <= 63, n
    assert a != b
    assert collector_pod_name("madengine-m") == collector_pod_name("madengine-m")
