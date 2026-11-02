"""The host package probe is informational and must not abort a run."""

from unittest.mock import MagicMock, patch

import pytest

from madengine.orchestration.run_orchestrator import RunOrchestrator


@pytest.mark.unit
class TestShowNodeInfo:
    def test_sles_missing_rocm_libs_does_not_fail_the_run(self):
        """zypper exit 104 (package not found) is printed and the run continues.

        TheRock installs ROCm from a tarball, so SLES has no rocm-libs RPM.
        Treating that exit as fatal blocked every DLM run (ROCM-8759).
        """
        mock_args = MagicMock()
        mock_args.additional_context = None
        mock_args.live_output = False
        mock_args.require_pinned_image = False

        orchestrator = RunOrchestrator(mock_args)
        orchestrator.context = MagicMock()
        orchestrator.context.ctx = {"host_os": "HOST_SLES"}

        commands = []

        class _Pipe:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        class _Proc:
            def __init__(self, returncode, output):
                self.returncode = returncode
                self.stdin = _Pipe()
                self.stdout = _Pipe()
                self._output = output

            def communicate(self, timeout=None):
                return self._output, None

        def fake_popen(command, **kwargs):
            commands.append(command)
            if "zypper" in command:
                return _Proc(104, "Package 'rocm-libs' not found.\n")
            return _Proc(0, "MAD Run Models\n")

        with patch("madengine.core.console.subprocess.Popen", side_effect=fake_popen):
            orchestrator._show_node_info()

        assert commands[-1] == "timeout 10 zypper info rocm-libs"
