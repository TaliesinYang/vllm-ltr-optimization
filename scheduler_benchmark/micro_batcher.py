"""Server-side request coalescing for a single-owner model.

Concurrent decision requests each arrive on their own HTTP handler thread. Run
one model forward per thread and they contend for the GIL and for the device;
the 201-box prototype measured that naive path at p99 687 ms against 53 ms for
the same model with coalescing. This module is that coalescing, with the model
factored out so the threading can be tested on CPU without a GPU or transformers.

Design is deliberately the boring one: ONE worker thread owns the batch
callable, callers hand over an item and block on their own Event. A single
owner means no tokenizer thread-safety hazard and no lock ordering to get
wrong. It runs for a rental day; it must not deadlock, and clever is not a
requirement.

Failure handling: if the batch callable raises, every caller in that batch
receives the exception and the worker keeps serving. One poisoned batch must
never wedge the queue.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Callable, Generic, Sequence, TypeVar

Item = TypeVar("Item")
Result = TypeVar("Result")

_SHUTDOWN = object()


class BatcherClosed(RuntimeError):
    """Raised when a request arrives after the batcher has been closed."""


class _Job(Generic[Item, Result]):
    __slots__ = ("item", "result", "error", "done")

    def __init__(self, item: Item) -> None:
        self.item = item
        self.result: Result | None = None
        self.error: BaseException | None = None
        self.done = threading.Event()


class MicroBatcher(Generic[Item, Result]):
    """Coalesce concurrent submissions into one call of ``run_batch``.

    ``run_batch`` receives a list of items and must return one result per item,
    positionally aligned. It is only ever called from the worker thread, so it
    may own non-thread-safe resources (a tokenizer, a CUDA stream).
    """

    def __init__(
        self,
        run_batch: Callable[[list[Item]], Sequence[Result]],
        *,
        batch_max: int,
        window_s: float,
    ) -> None:
        if batch_max < 1:
            raise ValueError("batch_max must be at least 1")
        if window_s < 0.0:
            raise ValueError("window_s must be non-negative")
        self._run_batch = run_batch
        self._batch_max = batch_max
        self._window_s = window_s
        self._queue: queue.Queue = queue.Queue()
        self._closed = threading.Event()
        self._sizes_lock = threading.Lock()
        self._batch_sizes: list[int] = []
        self._worker = threading.Thread(
            target=self._loop, name="micro-batcher", daemon=True
        )
        self._worker.start()

    @property
    def is_running(self) -> bool:
        return self._worker.is_alive()

    @property
    def batch_sizes(self) -> list[int]:
        """Observed batch sizes, for reporting how much coalescing happened."""
        with self._sizes_lock:
            return list(self._batch_sizes)

    def submit(self, item: Item) -> Result:
        if self._closed.is_set():
            raise BatcherClosed("micro-batcher is closed")
        job: _Job[Item, Result] = _Job(item)
        self._queue.put(job)
        job.done.wait()
        if job.error is not None:
            raise job.error
        return job.result  # type: ignore[return-value]

    def close(self, *, timeout: float = 5.0) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._queue.put(_SHUTDOWN)
        self._worker.join(timeout=timeout)

    def __enter__(self) -> "MicroBatcher[Item, Result]":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def _collect(self) -> list[_Job[Item, Result]] | None:
        """Block for one job, then drain until batch_max or the window expires."""
        first = self._queue.get()
        if first is _SHUTDOWN:
            return None
        batch = [first]
        deadline = time.perf_counter() + self._window_s
        while len(batch) < self._batch_max:
            remaining = deadline - time.perf_counter()
            if remaining <= 0.0:
                break
            try:
                nxt = self._queue.get(timeout=remaining)
            except queue.Empty:
                break
            if nxt is _SHUTDOWN:
                # Finish the work in hand, then stop after this batch.
                self._queue.put(_SHUTDOWN)
                break
            batch.append(nxt)
        return batch

    def _loop(self) -> None:
        while True:
            batch = self._collect()
            if batch is None:
                return
            with self._sizes_lock:
                self._batch_sizes.append(len(batch))
            self._dispatch(batch)

    def _dispatch(self, batch: list[_Job[Item, Result]]) -> None:
        try:
            results = self._run_batch([job.item for job in batch])
            if len(results) != len(batch):
                raise ValueError(
                    f"batch callable returned {len(results)} results "
                    f"for {len(batch)} items"
                )
        except BaseException as error:  # noqa: BLE001 - relayed to every caller
            for job in batch:
                job.error = error
                job.done.set()
            return
        for job, result in zip(batch, results):
            job.result = result
            job.done.set()
