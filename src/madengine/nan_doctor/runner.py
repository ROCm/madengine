"""Run one command in a model's image on the host's AMD GPUs (the `<GPU_EXEC>` of nan-doctor).

Standard library only, so this package can be copied next to a reproducer.

The container gets:
  * the GPUs: /dev/kfd, /dev/dri, HIP_VISIBLE_DEVICES for a subset;
  * the same flags a replay needs to see the timing the model saw (--ipc=host, a large
    --shm-size, memlock and stack limits, seccomp off) -- they shape the very races
    that are being replayed, so they are not tunable here;
  * the working directory, bind-mounted at its own absolute path (paths printed inside
    are valid on the host), plus any --mount;
  * the nan_doctor_rt runtime, read-only at /opt/nan_doctor, with PYTHONPATH set, so
    `python3 -m nan_doctor_rt.<module>` works in any image that has torch; and this host
    package at /opt/nan_doctor_host, so `python3 -m nan_doctor <command>` does too.

Nothing here takes the host GPU lock: callers that need one (MAD's mad-run.sh) hold it
around the whole command.
"""

import argparse
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time

RUNTIME_MOUNT = "/opt/nan_doctor"
# This host package too, read-only, so the standard-library tools also run in the image
# (`python3 -m nan_doctor gpu-check --torch` asks torch about device 0).
HOST_MOUNT = "/opt/nan_doctor_host"

# Flags that shape the timing of what is replayed.  Keep in step with the flags the
# model's own run used (madengine's container runner).
DOCKER_FLAGS = [
    "--network=host",
    "--ipc=host",
    "--shm-size=64g",
    "--device=/dev/kfd",
    "--device=/dev/dri",
    "--group-add",
    "video",
    "--cap-add=SYS_PTRACE",
    "--cap-add=IPC_LOCK",
    "--ulimit",
    "memlock=-1:-1",
    "--ulimit",
    "stack=67108864:67108864",
    "--security-opt",
    "seccomp=unconfined",
]

_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def runtime_parent():
    """Directory holding the nan_doctor_rt package: the madengine copy, or ./runtime next to a
    reproducer's host/ directory."""
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (
        os.path.join(os.path.dirname(here), "scripts", "common", "tools", "nan_doctor"),
        os.path.join(os.path.dirname(os.path.dirname(here)), "runtime"),
    ):
        if os.path.isdir(os.path.join(cand, "nan_doctor_rt")):
            return cand
    raise RuntimeError(
        "nan_doctor_rt not found next to this package; pass --runtime-dir"
    )


def image_from_manifest(path, model=None):
    """The docker image a madengine build_manifest.json names (exactly one model, or --model)."""
    with open(path) as f:
        manifest = json.load(f)
    built = manifest.get("built_images") or {}
    models = manifest.get("built_models") or {}
    pairs = []
    for key, info in built.items():
        name = (models.get(key) or {}).get("name", key)
        pairs.append((name, info.get("docker_image")))
    if model:
        pairs = [p for p in pairs if p[0] == model or p[0].endswith("/" + model)]
    pairs = [p for p in pairs if p[1]]
    if len(pairs) != 1:
        raise ValueError(
            "%s names %d image(s)%s; pass --model <name> (%s)"
            % (
                path,
                len(pairs),
                " for model %r" % model if model else "",
                ", ".join(p[0] for p in pairs) or "none",
            )
        )
    return pairs[0][1]


def image_present(image):
    return (
        subprocess.run(
            ["docker", "image", "inspect", image], capture_output=True
        ).returncode
        == 0
    )


def ensure_image(image, pull=False):
    if image_present(image):
        return
    if pull:
        r = subprocess.run(
            ["docker", "pull", "-q", image], capture_output=True, text=True
        )
        if r.returncode == 0:
            return
        raise RuntimeError(
            "docker pull %s failed: %s" % (image, r.stderr.strip()[-300:])
        )
    raise RuntimeError(
        "image %s is not on this host: build it with `madengine build`, "
        "or pass --pull to fetch it" % image
    )


def parse_mount(spec):
    """HOST[:CONTAINER[:ro|rw]] -> (host, container, mode)."""
    parts = spec.split(":")
    host = os.path.abspath(parts[0])
    cont = parts[1] if len(parts) > 1 and parts[1] else host
    mode = parts[2] if len(parts) > 2 else "rw"
    if mode not in ("ro", "rw"):
        raise ValueError("mount mode must be ro or rw: %r" % spec)
    return host, cont, mode


def _under(path, roots):
    path = os.path.realpath(path)
    return any(path == r or path.startswith(r.rstrip("/") + "/") for r in roots)


def build_command(
    image,
    payload,
    gpus="0",
    mounts=(),
    env=None,
    workdir=None,
    runtime_dir=None,
    as_user=False,
    name=None,
    cwd=None,
):
    """The `docker run` argv that runs `payload` (a shell string) in `image`."""
    cwd = os.path.realpath(cwd or os.getcwd())
    workdir = workdir or cwd
    rt = runtime_dir or runtime_parent()
    argv = ["docker", "run", "--rm"]
    if name:
        argv += ["--name", name]
    argv += DOCKER_FLAGS
    envs = {
        "PYTHONPATH": RUNTIME_MOUNT + ":" + HOST_MOUNT,
        "NAN_DOCTOR_RT": RUNTIME_MOUNT + "/nan_doctor_rt",
    }
    if gpus != "all":
        if not re.match(r"^\d+(,\d+)*$", gpus):
            raise ValueError("--gpus must be 'all' or a list like 0 or 0,1: %r" % gpus)
        envs["HIP_VISIBLE_DEVICES"] = gpus
        # One port per first-listed GPU, so concurrent single-GPU runs on a node do not
        # all try to claim 29500.
        envs["MASTER_PORT"] = str(29500 + int(gpus.split(",")[0]))
    if as_user:
        uid, gid = os.getuid(), os.getgid()
        argv += ["--user", "%d:%d" % (uid, gid)]
        try:
            argv += ["--group-add", str(os.stat("/dev/kfd").st_gid)]
        except OSError:
            pass
        user = os.environ.get("USER") or str(uid)
        envs.update(
            {
                "USER": user,
                "LOGNAME": user,
                "HOME": "/tmp",
                "TORCHINDUCTOR_CACHE_DIR": "/tmp/inductor-%d" % os.getpid(),
            }
        )
    envs.update(env or {})
    for k, v in envs.items():
        if not _ENV_KEY.match(k):
            raise ValueError("invalid environment variable name: %r" % k)
        argv += ["-e", "%s=%s" % (k, v)]
    mount_list = [(cwd, cwd, "rw")] + [m for m in mounts]
    seen = set()
    for host, cont, mode in mount_list:
        if (host, cont) in seen:
            continue
        seen.add((host, cont))
        argv += ["-v", "%s:%s:%s" % (host, cont, mode)]
    argv += [
        "-v",
        "%s:%s:ro" % (rt, RUNTIME_MOUNT),
        "-v",
        "%s:%s/nan_doctor:ro"
        % (os.path.dirname(os.path.abspath(__file__)), HOST_MOUNT),
        "-w",
        workdir,
        image,
        "bash",
        "-lc",
        payload,
    ]
    return argv


def run(image, payload, log=None, quiet=False, **kw):
    """Run `payload`; stream output to stdout (and `log`); return the exit status.

    The container is removed on exit, SIGTERM and SIGINT -- a container is a child of the
    docker daemon, not of this process, so a killed job would otherwise leave it holding GPU
    memory."""
    name = kw.pop("name", None) or "nan_exec_%d_%d" % (
        os.getpid(),
        int(time.time() * 1000) % 10**9,
    )
    argv = build_command(image, payload, name=name, **kw)
    logf = None
    if log and log != os.devnull:
        os.makedirs(os.path.dirname(os.path.abspath(log)), exist_ok=True)
        logf = open(log, "a")

    def cleanup(*_):
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)

    old = {
        s: signal.signal(s, lambda n, f: (cleanup(), sys.exit(128 + n)))
        for s in (signal.SIGTERM, signal.SIGINT)
        if hasattr(signal, "signal") and _main_thread()
    }
    try:
        p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for chunk in iter(lambda: p.stdout.readline(), b""):
            text = chunk.decode("utf-8", "replace")
            if not quiet:
                sys.stdout.write(text)
                sys.stdout.flush()
            if logf:
                logf.write(text)
        return p.wait()
    finally:
        cleanup()
        for s, h in old.items():
            signal.signal(s, h)
        if logf:
            logf.close()


def _main_thread():
    import threading

    return threading.current_thread() is threading.main_thread()


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="madengine nan exec",
        description="Run a command in a model's image on the host's AMD GPUs. Everything after -- is the command.",
        epilog="Example:  madengine nan exec --manifest build_manifest.json --gpus 0 -- "
        "python3 -m nan_doctor_rt.replay_nan_op capsule.pt --chain --chain-iters 1000",
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="docker image to run in")
    src.add_argument(
        "--manifest", help="madengine build_manifest.json naming the image"
    )
    ap.add_argument(
        "--model", help="with --manifest: the model, when the manifest has several"
    )
    ap.add_argument(
        "--gpus",
        default="0",
        help="'all', or a list such as 0 or 0,1 (default 0: a replay "
        "is reproducible only if it lands on the same device)",
    )
    ap.add_argument(
        "--mount",
        action="append",
        default=[],
        metavar="HOST[:CONT[:ro|rw]]",
        help="extra bind mount; the current directory is always mounted at its own path",
    )
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument(
        "--workdir",
        help="working directory in the container (default: the current directory)",
    )
    ap.add_argument(
        "--runtime-dir",
        help="directory holding nan_doctor_rt (default: the one shipped with madengine)",
    )
    ap.add_argument(
        "--as-user",
        action="store_true",
        help="run as the host uid:gid with HOME=/tmp "
        "(files written to a shared filesystem stay yours)",
    )
    ap.add_argument("--log", help="also append the output to this file")
    ap.add_argument(
        "--pull",
        action="store_true",
        help="docker pull the image if it is not on this host",
    )
    ap.add_argument(
        "--print-command", action="store_true", help="print the docker command and exit"
    )
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- <command ...>")
    a = ap.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        ap.error("no command given after --")
    try:
        image = a.image or image_from_manifest(a.manifest, a.model)
        env = {}
        for kv in a.env:
            k, _, v = kv.partition("=")
            env[k] = v
        kw = dict(
            gpus=a.gpus,
            mounts=[parse_mount(m) for m in a.mount],
            env=env,
            workdir=a.workdir,
            runtime_dir=a.runtime_dir,
            as_user=a.as_user,
        )
        payload = " ".join(shlex.quote(c) for c in cmd) if len(cmd) > 1 else cmd[0]
        if a.print_command:
            print(" ".join(shlex.quote(x) for x in build_command(image, payload, **kw)))
            return 0
        if not shutil.which("docker"):
            raise RuntimeError("docker is not on PATH")
        ensure_image(image, a.pull)
    except (ValueError, RuntimeError, OSError) as e:
        print("nan exec: %s" % e, file=sys.stderr)
        return 2
    return run(image, payload, log=a.log, **kw)


if __name__ == "__main__":
    sys.exit(main())
