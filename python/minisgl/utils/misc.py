from __future__ import annotations

import time
from typing import Any


def call_if_main(name: str = "__main__", discard: bool | None = None):
    """Decorator to ensure a function will call when the script is run as main."""
    if name != "__main__":
        discard = False if discard is None else discard
        if discard:
            return lambda _: None
        else:
            return lambda f: f
    else:
        discard = True if discard is None else discard
        if discard:
            return lambda f: (f() or True) and None
        else:
            return lambda f: (f() and None) or f


def div_even(a: int, b: int, allow_replicate: bool = False) -> int:
    """Divides two integers. If allow_replicate=True, allows b > a when b % a == 0, returning 1."""
    if allow_replicate and b > a:
        assert b % a == 0, f"{b = } must be divisible by {a = } for KV head replication"
        return 1
    assert a % b == 0, f"{a = } must be divisible by {b = }"
    return a // b


def div_ceil(a: int, b: int) -> int:
    """Divides two integers, rounding up"""
    return (a + b - 1) // b


def align_ceil(a: int, b: int) -> int:
    """Aligns a to the next multiple of b"""
    return div_ceil(a, b) * b


def align_down(a: int, b: int) -> int:
    """Aligns a to the previous multiple of b"""
    return (a // b) * b


class Unset:
    pass


UNSET = Unset()


def profile_enabled() -> bool:
    from minisgl.env import ENV

    return ENV.ENABLE_PROFILING_LOGS.value


def elapsed_ms(start_time: float) -> float:
    return (time.perf_counter() - start_time) * 1000


def maybe_log_perf(logger: Any, label: str, start_time: float, *, rank0: bool = False) -> float:
    elapsed = elapsed_ms(start_time)
    if not profile_enabled():
        return elapsed

    from minisgl.env import ENV

    if elapsed < ENV.PROFILE_LOG_MIN_MS.value:
        return elapsed

    log_fn = getattr(logger, "info_rank0", None) if rank0 else None
    if log_fn is None:
        log_fn = logger.info
    log_fn("[profile] %s took %.2f ms", label, elapsed)
    return elapsed
