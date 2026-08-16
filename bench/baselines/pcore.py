#!/usr/bin/env python3
"""P-core scheduling bias and measured thread occupancy, shared by both drivers.

Two facts drive this module, both learned the hard way on this project:

1. macOS exposes no thread-to-core pinning API -- THREAD_AFFINITY_POLICY is not
   supported on Apple Silicon.  The only lever is Quality of Service, and it is
   asymmetric: USER_INTERACTIVE *biases* a thread toward the P cluster, while
   BACKGROUND *confines* it to the E cluster.  So "run on the performance
   cores" is a request that has to be verified, never a guarantee.

2. A thread-count setting is not evidence that the setting took effect.  The
   single-threaded baseline comparison was wrong for exactly this reason:
   torch.set_num_threads(1) does not reach ExecuTorch's own pthreadpool, which
   was quietly running 10 workers.  What exposed it was CPU-seconds per
   wall-second, so that ratio is measured here for every timed row.
"""

from __future__ import annotations

import ctypes
import resource
import sys
import time

# From <sys/qos.h>.  Only the two ends of the range matter here.
QOS_CLASS_USER_INTERACTIVE = 0x21
QOS_CLASS_BACKGROUND = 0x09


def request_p_core(qos: int = QOS_CLASS_USER_INTERACTIVE) -> bool:
    """Bias this thread toward a P-core.  Returns whether the call succeeded.

    Must run BEFORE any framework thread pool is created: pool workers inherit
    the QoS of the thread that creates them, and neither PyTorch nor ExecuTorch
    re-requests it per worker.  Calling this after the first parallel op leaves
    the workers on whatever they were born with.
    """
    if sys.platform != "darwin":
        return False
    libc = ctypes.CDLL(None, use_errno=True)
    fn = getattr(libc, "pthread_set_qos_class_self_np", None)
    if fn is None:
        return False
    fn.argtypes = [ctypes.c_int, ctypes.c_int]
    fn.restype = ctypes.c_int
    return fn(ctypes.c_int(qos), ctypes.c_int(0)) == 0


def cpu_seconds() -> float:
    """Process-wide CPU time, summed across every thread."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_utime + ru.ru_stime


class Occupancy:
    """Measured concurrency over a block: ~1.0 serial, ~N for N busy threads.

    This is the requested-versus-achieved check.  A row claiming four threads
    whose occupancy reads 1.0 did not get them.
    """

    def __init__(self) -> None:
        self.ratio = 0.0

    def __enter__(self) -> "Occupancy":
        self._cpu0 = cpu_seconds()
        self._wall0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        wall = time.perf_counter() - self._wall0
        self.ratio = (cpu_seconds() - self._cpu0) / wall if wall > 0 else 0.0
