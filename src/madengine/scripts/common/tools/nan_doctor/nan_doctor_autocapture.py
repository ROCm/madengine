"""Arm op_nan_capture in an UNMODIFIED PyTorch workload -- no source wiring.

install_autocapture.py drops this module, op_nan_capture.py, and a one-line
nan_doctor_autocapture.pth into site-packages.  Python then imports this module
at start-up in every interpreter of the container: the launcher, each torchrun
rank, DataLoader workers, pip.  It is inert unless BOTH are set:

    NAN_CAPTURE_DIR=<capsule dir>       (the same switch capture_from_env uses)
    NAN_CAPTURE_AUTO=optimizer

and even then it imports nothing until torch itself imports torch.optim: a
meta-path finder waits for `torch.optim.optimizer`, and once that module has
loaded registers one GLOBAL optimizer-step post-hook.  So bash helpers, pip and
any process that never imports torch pay one environment lookup.

The hook stands in for the three wired lines of reference/step4-capture.md:

    cap = capture_from_env()     # first optimizer.step() returns: model and
                                 # optimizer exist, dist is initialised
    cap.step()                   # after every optimizer.step() == top of the
                                 # next iteration, outside any collective
    cap.close()                  # atexit

Every other knob (NAN_CAPTURE_DATA, _ARM_AFTER_STEPS, _OP_CLASSES, ...) is read
by capture_from_env as usual.  Consequences of counting optimizer steps:

  * the first iteration is never watched (the capture is built after it);
  * only the FIRST optimizer object that steps is counted, so a model with
    several optimizers still advances one step per iteration;
  * a step that GradScaler skips on inf/NaN gradients does not count, so settle
    mode waits for the next real step (finish() settles what is left at exit);
  * a loop with no optimizer (inference, serving) is never armed -- use source
    wiring (wire_capture.py) for those.

Status lines go to stderr with the "[op-nan]" prefix so they land in the same
log the madengine tool collects.
"""
import os
import sys

MODE_VAR = "NAN_CAPTURE_AUTO"
_TARGET = "torch.optim.optimizer"

_cap = None          # the _ArmedCapture, built at the first counted step
_owner = None        # id() of the optimizer whose steps are counted
_failed = False      # the capture could not be built: never try again


def enabled(environ=None):
    env = os.environ if environ is None else environ
    return bool(env.get("NAN_CAPTURE_DIR")) and env.get(MODE_VAR, "").strip().lower() == "optimizer"


def _log(msg):
    print("[op-nan] autocapture pid %d: %s" % (os.getpid(), msg), file=sys.stderr, flush=True)


def _close():
    if _cap is not None:
        try:
            _cap.close()
        except Exception as e:  # noqa: BLE001 -- never turn exit into a crash
            _log("close failed: %r" % (e,))


def _on_step(optimizer, args, kwargs):
    global _cap, _owner, _failed
    if _failed:
        return
    if _owner is None:
        _owner = id(optimizer)
    elif id(optimizer) != _owner:
        return
    if _cap is None:
        try:
            os.makedirs(os.environ["NAN_CAPTURE_DIR"], exist_ok=True)   # every rank, racing: fine
            from op_nan_capture import capture_from_env
            _cap = capture_from_env()
        except Exception as e:  # noqa: BLE001 -- a broken capture must not kill the run
            _failed = True
            _log("could not build the capture, running unwatched: %r" % (e,))
            return
        import atexit
        atexit.register(_close)
        _log("counting steps of %s" % type(optimizer).__name__)
    _cap.step()     # may raise on every rank together (sync_stop): that ends the run


def _install(optim_module):
    optim_module.register_optimizer_step_post_hook(_on_step)
    _log("optimizer-step hook registered")


class _Finder:
    """Wraps the loader of torch.optim.optimizer so _install runs right after it."""

    def find_spec(self, name, path=None, target=None):
        if name != _TARGET:
            return None
        import importlib.machinery
        sys.meta_path.remove(self)      # one shot; also keeps the lookup below from recursing
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return spec
        loader = spec.loader
        exec_module = loader.exec_module

        def exec_and_install(module):
            exec_module(module)
            try:
                _install(module)
            except Exception as e:  # noqa: BLE001
                _log("could not register the optimizer-step hook: %r" % (e,))
        loader.exec_module = exec_and_install
        return spec


def activate():
    """Arm the lazy hook in this interpreter (what the .pth line calls)."""
    if not enabled():
        return False
    mod = sys.modules.get(_TARGET)
    if mod is not None:                 # torch already imported (e.g. called by hand)
        _install(mod)
    elif not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
    return True


activate()
