"""Parallel helpers for the gridded lane: one worker process per CPU for the CPU-bound encode / decode-back, bounded
ordered pipelines for downloads and for that work, and peak-memory reporting.

Nothing here changes what is computed: every job is a pure function of its inputs, results are consumed in the
same order as the serial loop, and the encoders are deterministic (one zstd frame per block at a fixed level, no
multi-threaded zstd), so stored bytes are the same whichever process built them.

FEEDER_PROCS (environment): worker processes; blank = os.cpu_count(); 0 or 1 = no worker processes (every job runs
inline in the main process, the serial path).
"""

import atexit
import collections
import concurrent.futures as cf
import multiprocessing
import os
import threading

from . import log

_lock = threading.Lock()
_pool = None
_pool_n = 0


def procs():
    raw = (os.environ.get("FEEDER_PROCS") or "").strip()
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    return os.cpu_count() or 1


def _context():
    """forkserver (spawn where there is none); never fork: the main process runs download threads, and a forked
    child could inherit a held lock. As with any non-fork start, a script that starts the pool must keep its work
    under ``if __name__ == "__main__":`` (feeder.run and feeder.daily do): each worker imports the main module."""
    if "forkserver" in multiprocessing.get_all_start_methods():
        ctx = multiprocessing.get_context("forkserver")
        ctx.set_forkserver_preload(["__main__", "feeder.noaa"])  # workers start with numpy and the encoder loaded
        return ctx
    return multiprocessing.get_context("spawn")


def pool():
    """The shared worker pool, or None when FEEDER_PROCS asks for no worker processes."""
    global _pool, _pool_n
    n = procs()
    if n <= 1:
        return None
    with _lock:
        broken = _pool is not None and getattr(_pool, "_broken", False)
        if broken:
            log("worker-pool", "broken; starting a new one")  # e.g. a worker was killed: the retry gets new workers
        if _pool is None or _pool_n != n or broken:
            if _pool is not None:
                _pool.shutdown(wait=not broken, cancel_futures=True)
            _pool = cf.ProcessPoolExecutor(max_workers=n, mp_context=_context())
            _pool_n = n
        return _pool


def shutdown():
    global _pool, _pool_n
    with _lock:
        if _pool is not None:
            _pool.shutdown(wait=True, cancel_futures=True)
        _pool, _pool_n = None, 0


atexit.register(shutdown)


def width():
    """Jobs one CPU pipeline keeps in flight: the workers plus two queued, so no worker waits for the next input."""
    p = pool()
    return 1 if p is None else _pool_n + 2


def submit(fn, *args):
    """fn(*args) in a worker process, or inline (an already finished future) without workers."""
    p = pool()
    if p is not None:
        return p.submit(fn, *args)
    return inline(fn, *args)


def inline(fn, *args):
    """fn(*args) now, in this process, as an already finished future."""
    f = cf.Future()
    try:
        f.set_result(fn(*args))
    except BaseException as err:  # re-raised by .result(), at the same point in the loop as the serial code
        f.set_exception(err)
    return f


def picklable(obj):
    import pickle

    try:
        pickle.dumps(obj)
        return True
    except Exception:
        return False


def ordered(start, items, ahead):
    """Yields (item, result) for each item, in item order, keeping at most ``ahead`` started and not yet yielded.

    ``start(item)`` returns a Future (work started now) or any other value (passed through as the result).
    An exception from an item is raised when that item's turn comes, as in a serial loop. On early exit the work
    not yet started is cancelled."""
    it = iter(items)
    queue = collections.deque()
    ahead = max(1, int(ahead))

    def top_up(n):
        while len(queue) < n:
            try:
                item = next(it)
            except StopIteration:
                return
            queue.append((item, start(item)))

    try:
        top_up(ahead)
        while queue:
            item, head = queue.popleft()
            top_up(ahead - 1)  # keep the others busy while waiting for the head (never starts inline work early)
            yield item, (head.result() if isinstance(head, cf.Future) else head)
            top_up(ahead)
    finally:
        for _, f in queue:
            if isinstance(f, cf.Future):
                f.cancel()


def prefetch(executor, fn, items, ahead):
    """fn(item) on a thread pool, results (only) in item order, at most ``ahead`` fetched and not yet consumed."""
    return (r for _, r in ordered(lambda x: executor.submit(fn, x), items, ahead))


# ---------------------------------------------------------------- memory

def _hwm_kb(pid="self"):
    try:
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return None


def peak_memory():
    """{main_mb, workers_mb (sum of each live worker's peak; an upper bound on their peak together), workers}."""
    main = _hwm_kb()
    if main is None:
        import resource

        main = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    wk = []
    p = _pool
    if p is not None:
        for pid in list(getattr(p, "_processes", {}) or {}):
            v = _hwm_kb(pid)
            if v is not None:
                wk.append(v)
    return {"main_mb": int(main // 1024), "workers_mb": int(sum(wk) // 1024), "workers": len(wk)}


def log_memory(*what):
    m = peak_memory()
    log("peak-mem", *what, f"main_mb={m['main_mb']}", f"workers_mb={m['workers_mb']}", f"workers={m['workers']}")
    return m
