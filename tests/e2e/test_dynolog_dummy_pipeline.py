"""Whole-pipeline `torch_profiler_dynolog` tests that run wherever CI runs.

On-demand PyTorch tracing needs three things madengine cannot provide in CI: the
dynolog daemon (published only as a GitHub release asset), a GPU, and a PyTorch
workload for the daemon to attach to. These tests substitute a dummy ``dynolog``
and ``dyno`` on ``PATH`` and then run the tool's real scripts against them, which
covers what madengine owns: the daemon lifecycle, the retry loop that waits for
the workload to register, the flags the trace request carries, and the
diagnostics printed when nothing was captured.

The stand-in for ``dyno`` also writes the trace file, which in a real run is
written by the workload's own Kineto instance.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""

# built-in modules
import contextlib
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

# third-party modules
import pytest

# project modules
from madengine.utils.path_utils import get_madengine_root

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="drives the Linux in-container profiling scripts"
)

COMMON_SCRIPTS = get_madengine_root() / "scripts" / "common"
START_SCRIPT = COMMON_SCRIPTS / "pre_scripts" / "dynolog_start.sh"
TRIGGER_SCRIPT = COMMON_SCRIPTS / "tools" / "dynolog_trigger.sh"
STOP_SCRIPT = COMMON_SCRIPTS / "post_scripts" / "dynolog_stop.sh"

# The scripts hand off to each other through fixed paths, because the pre-script,
# the trigger, and the post-script are three separate processes in a container.
DYNOLOG_PID_FILE = Path("/tmp/madengine_dynolog.pid")
TRIGGER_PID_FILE = Path("/tmp/madengine_dynolog_trigger.pid")
STARTED_FILE = Path("/tmp/madengine_dynolog.started")
RESULT_FILE = Path("/tmp/madengine_dynolog_trigger.result")
HANDOFF_FILES = (
    DYNOLOG_PID_FILE,
    TRIGGER_PID_FILE,
    STARTED_FILE,
    RESULT_FILE,
    Path("/tmp/madengine_dynolog.log"),
    Path("/tmp/madengine_dynolog_trigger.log"),
)

# A daemon that stays up until it is signalled, so the stop script has something
# real to terminate.
DYNOLOG_STUB = """#!/bin/sh
echo "dummy dynolog listening: $*"
while : ; do sleep 0.2; done
"""

# Stands in for `dyno gputrace` from the pinned dynolog release, and holds to the
# same contract, because both halves of that contract are easy to get wrong:
#
# * it validates its options the way clap does, rejecting anything it does not
#   know with exit 2 and a usage message
# * it exits 0 whether or not it matched a process, reporting the matched pids in
#   its response instead
#
# $DUMMY_DYNO_REJECTS requests report no match, as they do before the workload has
# registered. $DUMMY_DYNO_UNSUPPORTED drops one option from the supported set, the
# way an older or newer dynolog than madengine expects would.
DYNO_STUB = """#!/bin/sh
printf '%s\\n' "$*" >> "$DUMMY_DYNO_LOG"

supported=" --duration-ms --iterations --job-id --log-file --pids --process-limit\
 --profile-memory --profile-start-iteration-roundup --profile-start-time\
 --record-shapes --with-flops --with-modules --with-stacks "
if [ -n "${DUMMY_DYNO_UNSUPPORTED:-}" ]; then
    supported=$(printf '%s' "$supported" | sed "s| ${DUMMY_DYNO_UNSUPPORTED} | |")
fi

for arg in "$@"; do
    case "$arg" in
        --port|gputrace) continue;;
        --*)
            case "$supported" in
                *" $arg "*) ;;
                *)
                    echo "error: Found argument '$arg' which wasn't expected, or isn't valid in this context"
                    echo ""
                    echo "USAGE:"
                    echo "    dyno gputrace --log-file <LOG_FILE> --job-id <JOB_ID> --process-limit <PROCESS_LIMIT>"
                    echo ""
                    echo "For more information try --help"
                    exit 2;;
            esac;;
    esac
done

log_file=""
previous=""
for arg in "$@"; do
    if [ "$previous" = "--log-file" ]; then log_file=$arg; fi
    previous=$arg
done

echo "Kineto config = "
echo "ACTIVITIES_LOG_FILE=$log_file"
requests=$(wc -l < "$DUMMY_DYNO_LOG")
if [ "$requests" -le "${DUMMY_DYNO_REJECTS:-0}" ]; then
    echo 'response = {"activityProfilersBusy":0,"activityProfilersTriggered":[],"eventProfilersBusy":0,"eventProfilersTriggered":[],"processesMatched":[]}'
    echo "No processes were matched, please check --job-id or --pids flags"
    exit 0
fi
echo 'response = {"activityProfilersBusy":0,"activityProfilersTriggered":[4242],"eventProfilersBusy":0,"eventProfilersTriggered":[],"processesMatched":[4242]}'
echo "Matched 1 processes"
echo "Trace output files will be written to:"
echo "    ${log_file%.json}_4242.json"
if [ "${DUMMY_DYNO_WRITE_TRACE:-0}" = "1" ] && [ -n "$log_file" ]; then
    # Kineto appends the process id to the requested filename.
    printf '{"traceEvents": [], "schemaVersion": 1}' > "${log_file%.json}_4242.json"
fi
exit 0
"""


class DummyDynolog:
    """Dummy ``dynolog`` and ``dyno`` binaries, installed the way the .deb is.

    ``pre_scripts/trace.sh dynolog`` ends with both binaries on ``PATH``, which
    is all the rest of the tool depends on.
    """

    def __init__(self, bin_dir: Path) -> None:
        bin_dir.mkdir(parents=True)
        self.bin_dir = bin_dir
        self.log = bin_dir / "dyno_requests.log"
        self.log.touch()
        for name, body in (("dynolog", DYNOLOG_STUB), ("dyno", DYNO_STUB)):
            script = bin_dir / name
            script.write_text(body, encoding="utf-8")
            script.chmod(0o755)

    def environ(self, on_path: bool = True, **overrides: str) -> dict:
        env = dict(os.environ, DUMMY_DYNO_LOG=str(self.log))
        if on_path:
            env["PATH"] = f"{self.bin_dir}{os.pathsep}{env.get('PATH', '')}"
        env.update(overrides)
        return env

    def requests(self) -> list:
        """Return the ``dyno gputrace`` requests made so far, one per line."""
        return [
            line for line in self.log.read_text(encoding="utf-8").splitlines() if line
        ]

    def cleanup(self) -> None:
        """Kill anything left running and clear the handoff files."""
        for pid_file in (TRIGGER_PID_FILE, DYNOLOG_PID_FILE):
            if pid_file.is_file():
                with contextlib.suppress(ValueError, OSError):
                    os.kill(int(pid_file.read_text().strip()), signal.SIGKILL)
        for path in HANDOFF_FILES:
            with contextlib.suppress(OSError):
                path.unlink()


@pytest.fixture
def dummy_dynolog(tmp_path):
    dummy = DummyDynolog(tmp_path / "dynolog-bin")
    dummy.cleanup()
    yield dummy
    dummy.cleanup()


@pytest.fixture
def workdir(tmp_path):
    """A working directory with scripts/common staged, as a run has."""
    work = tmp_path / "workdir"
    (work / "scripts").mkdir(parents=True)
    shutil.copytree(COMMON_SCRIPTS, work / "scripts" / "common")
    return work


def run_script(script: Path, work: Path, env: dict, timeout: int = 120):
    return subprocess.run(
        ["bash", str(script)],
        cwd=work,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def alive(pid_file: Path) -> bool:
    if not pid_file.is_file():
        return False
    try:
        os.kill(int(pid_file.read_text().strip()), 0)
    except (ValueError, OSError):
        return False
    return True


def wait_until(predicate, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.2)
    return predicate()


class TestDaemonLifecycle:
    """Starting and stopping the daemon around a model run."""

    def test_start_arms_the_daemon_and_the_trigger(self, workdir, dummy_dynolog):
        """The pre-script leaves a daemon running and a trigger waiting."""
        result = run_script(
            START_SCRIPT,
            workdir,
            dummy_dynolog.environ(TORCH_PROFILE_WARMUP_S="30"),
        )

        assert result.returncode == 0, result.stdout
        assert "dynolog daemon started" in result.stdout
        assert "trace trigger armed" in result.stdout
        assert alive(DYNOLOG_PID_FILE), result.stdout
        assert alive(TRIGGER_PID_FILE), result.stdout
        # The post-script uses this marker to tell "not started" from "failed".
        assert STARTED_FILE.is_file()
        # Kineto writes into this directory itself, so it must exist up front.
        assert (workdir / "torch_profiler_output").is_dir()

    def test_start_without_the_daemon_installed_fails(self, workdir, dummy_dynolog):
        """Without the pre-script's install step there is nothing to start."""
        result = run_script(START_SCRIPT, workdir, dummy_dynolog.environ(on_path=False))

        assert result.returncode != 0
        assert "pre-script must run first" in result.stdout
        assert not STARTED_FILE.is_file()

    def test_stop_leaves_nothing_running(self, workdir, dummy_dynolog):
        """The post-script has to reap both processes; they outlive the workload."""
        run_script(
            START_SCRIPT, workdir, dummy_dynolog.environ(TORCH_PROFILE_WARMUP_S="30")
        )

        result = run_script(STOP_SCRIPT, workdir, dummy_dynolog.environ())

        assert result.returncode == 0, result.stdout
        assert "dynolog cleanup complete" in result.stdout
        assert wait_until(lambda: not alive(DYNOLOG_PID_FILE)), result.stdout
        assert wait_until(lambda: not alive(TRIGGER_PID_FILE)), result.stdout
        assert not STARTED_FILE.is_file()
        # Both processes answer SIGTERM. Reaching the force-kill path instead
        # would cost the full grace period at the end of every profiled run.
        assert "did not stop gracefully" not in result.stdout

    def test_stop_without_start_is_a_no_op(self, workdir, dummy_dynolog):
        """Stacking the tool on a failed run must not turn into a second failure."""
        result = run_script(STOP_SCRIPT, workdir, dummy_dynolog.environ())

        assert result.returncode == 0, result.stdout
        assert "dynolog was not started" in result.stdout


class TestTraceRequest:
    """The trigger, which is what actually asks for a trace."""

    @staticmethod
    def run_trigger(work: Path, dummy: DummyDynolog, **env: str):
        settings = dict(
            TORCH_PROFILE_WARMUP_S="0",
            TORCH_PROFILE_RETRY_INTERVAL_S="0",
            TORCH_PROFILE_MAX_ATTEMPTS="5",
        )
        settings.update(env)
        return run_script(TRIGGER_SCRIPT, work, dummy.environ(**settings))

    def test_trigger_retries_until_the_workload_registers(self, workdir, dummy_dynolog):
        """A trace cannot be requested until PyTorch has registered, so we poll.

        dyno reports that in its response and still exits 0, so a request that
        matched nothing has to be told apart from one that succeeded by output.
        """
        result = self.run_trigger(workdir, dummy_dynolog, DUMMY_DYNO_REJECTS="2")

        assert result.returncode == 0, result.stdout
        assert len(dummy_dynolog.requests()) == 3
        assert "accepted on attempt 3" in result.stdout
        assert RESULT_FILE.read_text().strip() == "accepted"

    def test_an_option_dyno_rejects_fails_fast_and_says_so(
        self, workdir, dummy_dynolog
    ):
        """A request dyno refuses to parse can never succeed, so stop retrying it.

        Retrying it instead spends every attempt on a permanent error and then
        blames the workload for never registering.
        """
        result = self.run_trigger(
            workdir,
            dummy_dynolog,
            DUMMY_DYNO_UNSUPPORTED="--with-modules",
            TORCH_PROFILE_MAX_ATTEMPTS="5",
        )

        assert result.returncode != 0
        assert len(dummy_dynolog.requests()) == 1, dummy_dynolog.requests()
        assert "not retrying" in result.stdout
        assert "wasn't expected" in result.stdout, result.stdout
        assert "no PyTorch process matched" not in result.stdout
        assert RESULT_FILE.read_text().strip() == "request_rejected"

    def test_request_carries_the_data_tracelens_needs(self, workdir, dummy_dynolog):
        """Shapes, stacks, and modules are what make the TraceLens reports useful.

        Every option here also has to exist in the dynolog release the pre-script
        installs; the dyno stand-in rejects anything else.
        """
        result = self.run_trigger(workdir, dummy_dynolog)

        assert result.returncode == 0, result.stdout
        request = dummy_dynolog.requests()[0]
        for flag in (
            "--record-shapes",
            "--with-stacks",
            "--with-modules",
            "--iterations 5",
            "--process-limit 64",
        ):
            assert flag in request, request
        # An absolute path, because the workload's working directory is its own.
        assert (
            f"--log-file {workdir}/torch_profiler_output/libkineto_trace.json"
            in request
        )

    def test_disabling_iterations_switches_to_a_timed_capture(
        self, workdir, dummy_dynolog
    ):
        """Iteration counting needs an optimizer step, which not every model has."""
        self.run_trigger(
            workdir,
            dummy_dynolog,
            TORCH_PROFILE_ITERATIONS="0",
            TORCH_PROFILE_DURATION_MS="750",
        )

        request = dummy_dynolog.requests()[0]
        assert "--duration-ms 750" in request
        assert "--iterations" not in request

    def test_optional_capture_flags_follow_their_env_vars(self, workdir, dummy_dynolog):
        """The expensive captures are opt-in, and the default ones opt-out."""
        self.run_trigger(
            workdir,
            dummy_dynolog,
            TORCH_PROFILE_RECORD_SHAPES="0",
            TORCH_PROFILE_WITH_STACKS="0",
            TORCH_PROFILE_WITH_MODULES="0",
            TORCH_PROFILE_WITH_FLOPS="1",
            TORCH_PROFILE_PROFILE_MEMORY="1",
        )

        request = dummy_dynolog.requests()[0]
        assert "--record-shapes" not in request
        assert "--with-stacks" not in request
        assert "--with-modules" not in request
        assert "--with-flops" in request
        assert "--profile-memory" in request

    def test_giving_up_says_what_to_check(self, workdir, dummy_dynolog):
        """Nothing registering is the common failure, so it must be self-explaining."""
        result = self.run_trigger(
            workdir,
            dummy_dynolog,
            DUMMY_DYNO_REJECTS="99",
            TORCH_PROFILE_MAX_ATTEMPTS="2",
        )

        assert result.returncode != 0
        assert len(dummy_dynolog.requests()) == 2
        assert "gave up after 2 attempts" in result.stdout
        assert "KINETO_USE_DAEMON=1" in result.stdout
        assert "TORCH_PROFILE_WARMUP_S" in result.stdout
        assert RESULT_FILE.read_text().strip() == "no_process"


class TestCaptureReporting:
    """What the post-script tells the user about the traces it found."""

    def test_captured_traces_are_reported(self, workdir, dummy_dynolog):
        """A successful capture is confirmed with a count, per rank."""
        run_script(
            START_SCRIPT, workdir, dummy_dynolog.environ(TORCH_PROFILE_WARMUP_S="30")
        )
        TestTraceRequest.run_trigger(workdir, dummy_dynolog, DUMMY_DYNO_WRITE_TRACE="1")

        result = run_script(STOP_SCRIPT, workdir, dummy_dynolog.environ())

        assert result.returncode == 0, result.stdout
        assert "Captured 1 torch.profiler trace(s)" in result.stdout
        traces = list((workdir / "torch_profiler_output").glob("*.json"))
        assert [p.name for p in traces] == ["libkineto_trace_4242.json"]

    def test_missing_traces_are_explained(self, workdir, dummy_dynolog):
        """An empty output directory needs a reason, not just a warning."""
        run_script(
            START_SCRIPT, workdir, dummy_dynolog.environ(TORCH_PROFILE_WARMUP_S="30")
        )
        TestTraceRequest.run_trigger(
            workdir,
            dummy_dynolog,
            DUMMY_DYNO_REJECTS="99",
            TORCH_PROFILE_MAX_ATTEMPTS="1",
        )

        result = run_script(STOP_SCRIPT, workdir, dummy_dynolog.environ())

        assert result.returncode == 0, result.stdout
        assert "No torch.profiler traces were captured" in result.stdout
        assert "never matched a PyTorch process" in result.stdout

    def test_a_rejected_request_is_reported_as_such(self, workdir, dummy_dynolog):
        """A request dyno never accepted is a different problem from a quiet workload."""
        run_script(
            START_SCRIPT, workdir, dummy_dynolog.environ(TORCH_PROFILE_WARMUP_S="30")
        )
        TestTraceRequest.run_trigger(
            workdir,
            dummy_dynolog,
            DUMMY_DYNO_UNSUPPORTED="--record-shapes",
            TORCH_PROFILE_MAX_ATTEMPTS="1",
        )

        result = run_script(STOP_SCRIPT, workdir, dummy_dynolog.environ())

        assert result.returncode == 0, result.stdout
        assert "dyno rejected the trace request" in result.stdout
        assert "never matched a PyTorch process" not in result.stdout
