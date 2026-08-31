#!/usr/bin/env python3
"""Module to keep containers from outliving the madengine process.

When a CI job is cancelled the runner signals only the top-level process of
the step and, ten seconds later, kills the whole process tree.  madengine's
benchmark container is started detached, so it is not part of that tree: it
survives, keeps ``/dev/kfd`` and the render nodes open, and the next job on
that shared GPU runner fails.  If the GPU has wedged, the host needs a reboot.

Three mechanisms defend against that, in decreasing order of reliability:

1. A dead man's switch inside the container itself (see
   :class:`madengine.core.docker.Docker`).  Its PID 1 watches a heartbeat file
   that this process refreshes every few seconds through the bind-mounted
   workspace; when the heartbeat goes stale the container stops itself.  This
   is what covers ``SIGKILL``, where none of our cleanup code gets to run at
   all, and it relies on nothing but a shell loop -- no signal, no daemon
   behaviour, no cooperation from the CI runner.
2. The signal handlers installed by :func:`install_signal_handlers`, which
   tear containers down promptly when a signal does reach us.
3. :func:`reap`, a bounded escalation shared by the handlers, by normal
   teardown and by ``madengine cleanup``.

Every docker command issued while tearing down has a hard deadline.  A wedged
GPU blocks the docker daemon too, and a cleanup path that can hang is worse
than none: we would be killed part-way through and orphan the container
anyway, having spent the grace period achieving nothing.

Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
"""
# built-in modules
import glob
import json
import os
import signal
import subprocess
import threading
import time
import typing
import uuid


# Labels stamped on every container madengine starts, so a container can be
# traced back to the run that owns it long after that run's process is gone.
LABEL_SESSION = "madengine.session"
LABEL_OWNER_PID = "madengine.owner_pid"

# Hard deadlines (seconds) for the docker CLI calls used during teardown.
# Container states in which processes may still be holding the GPU.
_LIVE_STATES = ("running", "restarting", "paused", "removing")

STOP_TIMEOUT = 10
KILL_TIMEOUT = 10
RM_TIMEOUT = 15
INSPECT_TIMEOUT = 10

# Total budget for teardown from a signal handler.  GitHub Actions allows
# 7.5s between SIGINT and SIGTERM and 2.5s more before killing the process
# tree; staying inside the first window means cleanup completes rather than
# being cut in half.
SIGNAL_CLEANUP_BUDGET = 6.0

# Where the container registry and the wedged-GPU marker are written.
DEFAULT_STATE_DIR = "~/.madengine"

# The dead man's switch. The file lives in the bind-mounted workspace, so the
# host writes it and the container reads it. Refreshing it more often than a
# third of the staleness threshold leaves room for a stalled host to miss a
# couple of beats without the container giving up on a run that is still fine.
HEARTBEAT_FILENAME = ".madengine_heartbeat"
HEARTBEAT_INTERVAL = int(os.environ.get("MADENGINE_HEARTBEAT_INTERVAL", "15"))
HEARTBEAT_STALE_AFTER = int(os.environ.get("MADENGINE_HEARTBEAT_STALE_AFTER", "120"))

# Locations of a container's cgroup process list, tried in order.  Reading
# these needs no help from the docker daemon, which is the point: when the
# daemon is the thing that is stuck, this is how we still find the processes.
_CGROUP_PROCS_GLOBS = (
    "/sys/fs/cgroup/system.slice/docker-{cid}.scope/cgroup.procs",
    "/sys/fs/cgroup/docker/{cid}/cgroup.procs",
    "/sys/fs/cgroup/*/docker/{cid}/cgroup.procs",
    "/sys/fs/cgroup/*/system.slice/docker-{cid}.scope/cgroup.procs",
)

_SESSION_ID = os.environ.get("MADENGINE_SESSION_ID") or uuid.uuid4().hex[:12]

_handlers_installed = False
_cleanup_in_progress = False


def session_id() -> str:
    """Return the id shared by every container started by this process.

    Returns:
        str: The session id.
    """
    return _SESSION_ID


def state_dir() -> str:
    """Return the directory holding the container registry.

    Returns:
        str: Absolute path to the state directory. Created if missing.
    """
    path = os.path.expanduser(
        os.environ.get("MADENGINE_STATE_DIR", DEFAULT_STATE_DIR)
    )
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        # A read-only or otherwise unusable HOME must not fail a benchmark;
        # the registry is a backstop, not a prerequisite.
        pass
    return path


def _registry_path(pid: typing.Optional[int] = None) -> str:
    """Return the registry file for a process.

    Args:
        pid (Optional[int]): The owning process id. Defaults to this process.

    Returns:
        str: Path to the per-process registry file.
    """
    return os.path.join(state_dir(), f"containers-{pid or os.getpid()}.json")


def _read_registry(path: str) -> typing.List[typing.Dict]:
    """Read a registry file, tolerating absence and corruption.

    Args:
        path (str): Path to the registry file.

    Returns:
        list: The recorded container entries, empty if unreadable.
    """
    try:
        with open(path, "r") as handle:
            entries = json.load(handle)
        return entries if isinstance(entries, list) else []
    except (OSError, ValueError):
        return []


def _write_registry(path: str, entries: typing.List[typing.Dict]) -> None:
    """Write a registry file, ignoring failures.

    Args:
        path (str): Path to the registry file.
        entries (list): The container entries to record.
    """
    try:
        if entries:
            with open(path, "w") as handle:
                json.dump(entries, handle)
        elif os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def register(container_id: str, container_name: str) -> None:
    """Record a container as owned by this process.

    Args:
        container_id (str): The full container id.
        container_name (str): The container name.
    """
    path = _registry_path()
    entries = [e for e in _read_registry(path) if e.get("id") != container_id]
    entries.append(
        {
            "id": container_id,
            "name": container_name,
            "session": _SESSION_ID,
            "owner_pid": os.getpid(),
            "started": time.time(),
        }
    )
    _write_registry(path, entries)


def unregister(container_id: str) -> None:
    """Drop a container from this process's registry.

    Args:
        container_id (str): The full container id.
    """
    path = _registry_path()
    _write_registry(path, [e for e in _read_registry(path) if e.get("id") != container_id])


def registered() -> typing.List[typing.Dict]:
    """Return the containers currently owned by this process.

    Returns:
        list: The recorded container entries.
    """
    return _read_registry(_registry_path())


def docker(
    args: typing.List[str], timeout: int
) -> typing.Tuple[typing.Optional[int], str]:
    """Run a docker CLI command under a hard deadline.

    The command is started in its own session so that a signal aimed at
    madengine's process group -- the very situation this module exists to
    survive -- cannot kill the cleanup that is still in flight.

    Args:
        args (list): Arguments to pass to the docker binary.
        timeout (int): Deadline in seconds.

    Returns:
        tuple: ``(returncode, output)``. The return code is ``None`` if the
            command timed out or could not be started, in which case the
            output describes why.
    """
    try:
        proc = subprocess.run(
            ["docker"] + args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            timeout=timeout,
            start_new_session=True,
        )
        return proc.returncode, (proc.stdout or "").strip()
    except subprocess.TimeoutExpired:
        return None, f"docker {' '.join(args[:2])} timed out after {timeout}s"
    except (OSError, ValueError) as exc:
        return None, str(exc)


def container_state(ref: str) -> typing.Optional[str]:
    """Return a container's state.

    Args:
        ref (str): Container id or name.

    Returns:
        Optional[str]: ``"running"``, ``"exited"`` and so on, ``""`` if the
            container is gone, or ``None`` if docker did not answer in time.
    """
    code, out = docker(
        ["inspect", "-f", "{{.State.Status}}", ref], INSPECT_TIMEOUT
    )
    if code is None:
        return None
    if code != 0:
        return ""
    return out.strip()


def container_host_pids(container_id: str) -> typing.List[int]:
    """Return the host pids of a container's processes.

    Read straight from the cgroup filesystem so that this still works when
    the docker daemon itself is blocked behind a wedged GPU.

    Args:
        container_id (str): The full container id.

    Returns:
        list: Host process ids, empty if none were found.
    """
    pids: typing.List[int] = []
    for pattern in _CGROUP_PROCS_GLOBS:
        for path in glob.glob(pattern.format(cid=container_id)):
            try:
                with open(path, "r") as handle:
                    pids.extend(
                        int(line) for line in handle.read().split() if line.isdigit()
                    )
            except OSError:
                continue
        if pids:
            break
    return sorted(set(pids))


def uninterruptible_pids(pids: typing.Iterable[int]) -> typing.List[int]:
    """Return the pids stuck in uninterruptible sleep.

    A process in ``D`` state is blocked inside a kernel driver and ignores
    every signal, ``SIGKILL`` included. For a benchmark container that means
    the GPU has wedged and only a host reboot will release it.

    Args:
        pids (Iterable[int]): Host process ids to inspect.

    Returns:
        list: The subset of pids in ``D`` state.
    """
    stuck = []
    for pid in pids:
        try:
            with open(f"/proc/{pid}/stat", "r") as handle:
                stat = handle.read()
        except OSError:
            continue
        if _state_char(stat) == "D":
            stuck.append(pid)
    return stuck


def _state_char(stat: str) -> str:
    """Return the state character from a ``/proc/<pid>/stat`` line.

    Args:
        stat (str): The raw contents of the stat file.

    Returns:
        str: The one-character state, or ``""`` if it cannot be found. The
            comm field is parenthesised and may itself contain spaces and
            parentheses, so the state is the first field after the *last*
            closing parenthesis rather than the second whitespace token.
    """
    head, paren, tail = stat.rpartition(")")
    if not paren:
        return ""
    fields = tail.split()
    return fields[0] if fields else ""


def _kill_host_pids(pids: typing.Iterable[int]) -> None:
    """Send SIGKILL directly to host pids.

    Args:
        pids (Iterable[int]): Host process ids to kill.
    """
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def reap(
    ref: str,
    container_id: typing.Optional[str] = None,
    verbose: bool = True,
) -> typing.Dict:
    """Stop and remove a container, escalating until it is gone.

    Each step is bounded, and the next one is only attempted if the container
    is still there, so a healthy container costs one ``docker stop`` while a
    wedged one cannot stall the caller indefinitely.

    Args:
        ref (str): Container id or name to reap.
        container_id (Optional[str]): The full container id, when known. Used
            to find host pids without asking the docker daemon.
        verbose (bool): Whether to narrate the escalation.

    Returns:
        dict: ``ref``, ``removed``, ``wedged``, ``stuck_pids``,
            ``docker_responsive`` and the list of ``steps`` attempted.
    """
    result: typing.Dict = {
        "ref": ref,
        "removed": False,
        "wedged": False,
        "stuck_pids": [],
        "docker_responsive": True,
        "steps": [],
    }

    def note(step: str) -> None:
        result["steps"].append(step)
        if verbose:
            print(f"   {step}", flush=True)

    state = container_state(ref)
    if state == "":
        result["removed"] = True
        return result
    if state is None:
        result["docker_responsive"] = False

    if state and state not in _LIVE_STATES:
        # Already dead, just not reaped -- the usual case for `madengine
        # cleanup`. Stopping and killing it would only add noise.
        docker(["rm", "-f", ref], RM_TIMEOUT)
        note("docker rm -f")
        if container_state(ref) == "":
            result["removed"] = True
            return result

    docker(["stop", "-t", "1", ref], STOP_TIMEOUT)
    note("docker stop")
    if container_state(ref) == "":
        result["removed"] = True
        return result

    docker(["kill", "-s", "KILL", ref], KILL_TIMEOUT)
    note("docker kill -s KILL")

    docker(["rm", "-f", ref], RM_TIMEOUT)
    note("docker rm -f")
    if container_state(ref) == "":
        result["removed"] = True
        return result

    # The daemon could not remove it. Go around the daemon and signal the
    # container's processes on the host directly.
    cid = container_id or ref
    pids = container_host_pids(cid)
    if pids:
        note(f"SIGKILL host pids {pids}")
        _kill_host_pids(pids)
        time.sleep(1)
        docker(["rm", "-f", ref], RM_TIMEOUT)
        if container_state(ref) == "":
            result["removed"] = True
            return result
        result["stuck_pids"] = uninterruptible_pids(pids)

    # Only claim a wedge we can actually evidence. Processes stuck in D state
    # are proof; so is a container the daemon still reports after being told
    # twice to remove it. A daemon that never answered and left no host pids
    # behind proves nothing -- saying "reboot this host" on that basis would
    # train people to ignore the marker.
    final_state = container_state(ref)
    if final_state == "":
        result["removed"] = True
        return result
    result["docker_responsive"] = result["docker_responsive"] and final_state is not None
    result["wedged"] = bool(result["stuck_pids"]) or bool(final_state)
    if result["wedged"]:
        _report_wedged(ref, result["stuck_pids"], verbose=verbose)
    elif verbose:
        note(
            "could not confirm removal: docker did not respond and no host "
            "pids were found"
        )
    return result


def _report_wedged(
    ref: str, stuck_pids: typing.List[int], verbose: bool = True
) -> None:
    """Record and announce a container that refuses to die.

    Args:
        ref (str): Container id or name.
        stuck_pids (list): Host pids in uninterruptible sleep, if any.
        verbose (bool): Whether to print the diagnostic.
    """
    marker = os.path.join(state_dir(), "gpu_wedged.json")
    payload = {
        "container": ref,
        "stuck_pids": stuck_pids,
        "session": _SESSION_ID,
        "hostname": os.uname().nodename,
        "timestamp": time.time(),
    }
    try:
        with open(marker, "w") as handle:
            json.dump(payload, handle, indent=2)
    except OSError:
        marker = "(could not be written)"

    if not verbose:
        return
    print("=" * 80, flush=True)
    print(f"GPU WEDGED: container '{ref}' could not be removed.", flush=True)
    if stuck_pids:
        print(
            f"  Host pids {stuck_pids} are in uninterruptible sleep (D state):",
            flush=True,
        )
        print(
            "  they are blocked inside the GPU driver and will not respond to",
            flush=True,
        )
        print("  any signal, SIGKILL included.", flush=True)
    print("  This host needs a reboot before it can run another benchmark.", flush=True)
    print(f"  Marker written to: {marker}", flush=True)
    print("=" * 80, flush=True)


def reap_registered(verbose: bool = True, budget: float = 0.0) -> typing.List[typing.Dict]:
    """Reap every container owned by this process.

    Args:
        verbose (bool): Whether to narrate the teardown.
        budget (float): Total seconds to spend, or 0 for no limit. When the
            budget runs out the remaining containers are left to their dead
            man's switch, which does not need us alive.
        
    Returns:
        list: One :func:`reap` result per container attempted.
    """
    deadline = (time.time() + budget) if budget else None
    results = []
    for entry in registered():
        if deadline is not None and time.time() >= deadline:
            if verbose:
                print(
                    "   cleanup budget exhausted; remaining containers will stop "
                    "on their own when this process exits",
                    flush=True,
                )
            break
        results.append(
            reap(entry.get("name") or entry["id"], entry.get("id"), verbose=verbose)
        )
        unregister(entry["id"])
    return results


_heartbeat_paths: typing.Set[str] = set()
_heartbeat_lock = threading.Lock()
_heartbeat_thread: typing.Optional[threading.Thread] = None


def _write_heartbeat(path: str) -> None:
    """Stamp the current time into a heartbeat file.

    Args:
        path (str): Path to the heartbeat file.
    """
    try:
        with open(path, "w") as handle:
            handle.write(str(int(time.time())))
    except OSError:
        pass


def _beat_once() -> None:
    """Refresh every heartbeat file that is still live."""
    with _heartbeat_lock:
        paths = list(_heartbeat_paths)
    for path in paths:
        _write_heartbeat(path)


def _heartbeat_loop() -> None:
    """Refresh the live heartbeat files for as long as the process runs."""
    while True:
        time.sleep(HEARTBEAT_INTERVAL)
        _beat_once()


def start_heartbeat(path: str) -> None:
    """Begin refreshing a heartbeat file for a container.

    The first beat is written synchronously: the container's watchdog exits
    immediately if the file is missing, so it has to exist before the
    container starts.

    Args:
        path (str): Path to the heartbeat file, on the host side of the mount.
    """
    global _heartbeat_thread
    _write_heartbeat(path)
    with _heartbeat_lock:
        _heartbeat_paths.add(path)
        if _heartbeat_thread is None or not _heartbeat_thread.is_alive():
            # A daemon thread keeps beating while the main thread is blocked
            # in a long benchmark, and does not hold up interpreter exit.
            _heartbeat_thread = threading.Thread(
                target=_heartbeat_loop, name="madengine-heartbeat", daemon=True
            )
            _heartbeat_thread.start()


def stop_heartbeat(path: str) -> None:
    """Stop refreshing a heartbeat file and remove it.

    Removing the file is the quickest way to tell the container to stop: its
    watchdog checks for the file's existence on every pass and does not have
    to wait for the staleness threshold.

    Args:
        path (str): Path to the heartbeat file.
    """
    with _heartbeat_lock:
        _heartbeat_paths.discard(path)
    try:
        os.remove(path)
    except OSError:
        pass


def watchdog_script() -> str:
    """Return the shell script a container runs as PID 1.

    The container stops itself once the heartbeat file disappears or goes
    stale, which is what makes a container outliving madengine impossible
    rather than merely unlikely. Written for POSIX ``sh`` so it runs on any
    image that already supported the ``cat`` this replaces, and quoted with
    double quotes only so it can be embedded in a single-quoted ``sh -c``.

    Returns:
        str: The script body, without the ``sh -c`` wrapper.
    """
    return (
        f"while [ -f /myworkspace/{HEARTBEAT_FILENAME} ]; do "
        f"hb=$(cat /myworkspace/{HEARTBEAT_FILENAME} 2>/dev/null); "
        'case "$hb" in ""|*[!0-9]*) hb=0 ;; esac; '
        f"if [ $(($(date +%s) - $hb)) -ge {HEARTBEAT_STALE_AFTER} ]; then "
        'echo "madengine heartbeat stale; stopping container"; break; fi; '
        "sleep 5; "
        "done"
    )


def container_watchdog_command() -> str:
    """Return the watchdog wrapped for a ``docker run`` command line.

    Returns:
        str: The command, ready to append to a ``docker run`` line.
    """
    return "sh -c '" + watchdog_script() + "' "


def install_signal_handlers() -> None:
    """Tear containers down when this process is signalled.

    A second signal exits immediately: if the first teardown is itself stuck,
    the caller should not have to wait for it twice.
    """
    global _handlers_installed
    if _handlers_installed:
        return

    def handle(signum, frame):  # noqa: ANN001 - signal handler signature
        global _cleanup_in_progress
        name = signal.Signals(signum).name
        if _cleanup_in_progress:
            print(f"\nReceived {name} again; exiting now.", flush=True)
            os._exit(130)
        _cleanup_in_progress = True
        print(f"\nReceived {name}; stopping containers...", flush=True)
        # Drop the heartbeats before anything else. Even if every docker call
        # below times out, the containers now stop by themselves.
        for path in list(_heartbeat_paths):
            stop_heartbeat(path)
        try:
            reap_registered(verbose=True, budget=SIGNAL_CLEANUP_BUDGET)
        finally:
            # os._exit skips interpreter shutdown deliberately: atexit hooks
            # and destructors here would re-enter the teardown we just did,
            # and any one of them could block past the grace period.
            os._exit(130)

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(signum, handle)
        except (ValueError, OSError):
            # Not the main thread, or the platform disallows it. The dead
            # man's switch still covers us.
            pass
    _handlers_installed = True
