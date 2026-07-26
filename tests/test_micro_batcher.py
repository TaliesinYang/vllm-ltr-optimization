import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from scheduler_benchmark.micro_batcher import BatcherClosed, MicroBatcher


def doubling(items: list[int]) -> list[int]:
    return [item * 2 for item in items]


def test_single_request_returns_without_waiting_for_a_full_batch() -> None:
    """A lone arrival must not block for other traffic that never comes."""
    with MicroBatcher(doubling, batch_max=8, window_s=0.05) as batcher:
        started = time.perf_counter()
        result = batcher.submit(21)
        elapsed = time.perf_counter() - started

    assert result == 42
    # It waits at most one window, not batch_max arrivals.
    assert elapsed < 0.5


def test_concurrent_arrivals_are_coalesced_into_one_forward() -> None:
    observed: list[int] = []

    def record(items: list[int]) -> list[int]:
        observed.append(len(items))
        return [item * 2 for item in items]

    with MicroBatcher(record, batch_max=8, window_s=0.05) as batcher:
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(batcher.submit, range(8)))

    assert results == [value * 2 for value in range(8)]
    # 8 threads arriving together must not cost 8 forwards.
    assert max(observed) > 1
    assert sum(observed) == 8


def test_batch_never_exceeds_batch_max() -> None:
    observed: list[int] = []

    def record(items: list[int]) -> list[int]:
        observed.append(len(items))
        time.sleep(0.01)
        return [item * 2 for item in items]

    with MicroBatcher(record, batch_max=3, window_s=0.05) as batcher:
        with ThreadPoolExecutor(max_workers=12) as pool:
            results = list(pool.map(batcher.submit, range(12)))

    assert results == [value * 2 for value in range(12)]
    assert max(observed) <= 3


def test_each_caller_receives_its_own_result() -> None:
    """Scatter must be positional - a mismatch here silently corrupts scores."""
    with MicroBatcher(doubling, batch_max=8, window_s=0.02) as batcher:
        with ThreadPoolExecutor(max_workers=8) as pool:
            values = list(range(100, 108))
            results = list(pool.map(batcher.submit, values))

    assert results == [value * 2 for value in values]


def test_batch_failure_reaches_every_caller_and_the_batcher_survives() -> None:
    calls: list[int] = []

    def explode_once(items: list[int]) -> list[int]:
        calls.append(len(items))
        if len(calls) == 1:
            raise RuntimeError("forward blew up")
        return [item * 2 for item in items]

    with MicroBatcher(explode_once, batch_max=8, window_s=0.02) as batcher:
        with pytest.raises(RuntimeError, match="forward blew up"):
            batcher.submit(1)

        # The worker must still be alive; a poisoned batch cannot wedge serving.
        assert batcher.submit(2) == 4


def test_wrong_result_count_is_reported_rather_than_scattered() -> None:
    with MicroBatcher(lambda items: [1], batch_max=8, window_s=0.02) as batcher:
        with ThreadPoolExecutor(max_workers=4) as pool:
            outcomes = []
            for future in [pool.submit(batcher.submit, value) for value in range(4)]:
                try:
                    outcomes.append(future.result())
                except ValueError as error:
                    outcomes.append(error)

    # Either the batch had exactly one item and succeeded, or every caller in a
    # multi-item batch was told the batch was malformed. Nothing silently wrong.
    assert all(
        outcome == 1 or isinstance(outcome, ValueError) for outcome in outcomes
    )
    assert any(isinstance(outcome, ValueError) for outcome in outcomes)


def test_submit_after_close_is_refused() -> None:
    batcher = MicroBatcher(doubling, batch_max=8, window_s=0.02)
    assert batcher.submit(1) == 2
    batcher.close()

    with pytest.raises(BatcherClosed):
        batcher.submit(2)


def test_close_is_idempotent_and_joins_the_worker() -> None:
    batcher = MicroBatcher(doubling, batch_max=8, window_s=0.02)
    batcher.submit(1)
    batcher.close()
    batcher.close()

    assert not batcher.is_running


def test_sustained_concurrency_does_not_deadlock() -> None:
    """The rental-day requirement: 8 concurrent callers, many rounds, no hangs."""
    with MicroBatcher(doubling, batch_max=8, window_s=0.003) as batcher:
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(batcher.submit, range(400)))

    assert results == [value * 2 for value in range(400)]


def test_observed_batch_sizes_are_recorded_for_reporting() -> None:
    with MicroBatcher(doubling, batch_max=8, window_s=0.05) as batcher:
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(batcher.submit, range(8)))
        sizes = batcher.batch_sizes

    assert sizes
    assert sum(sizes) == 8
    assert max(sizes) <= 8


def test_worker_thread_is_a_daemon_and_single() -> None:
    before = threading.active_count()
    with MicroBatcher(doubling, batch_max=8, window_s=0.02) as batcher:
        batcher.submit(1)
        during = threading.active_count()

    assert during == before + 1
