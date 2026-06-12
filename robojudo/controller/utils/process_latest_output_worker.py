from __future__ import annotations

import atexit
import logging
import multiprocessing as mp
import queue
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


def _split_output(output: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    output_copy = dict(output)
    commands = list(output_copy.pop("_commands", []))
    return output_copy, commands


def _put_latest(output_queue, output: dict[str, Any]) -> None:
    while True:
        try:
            output_queue.get_nowait()
        except queue.Empty:
            break

    try:
        output_queue.put_nowait(output)
    except queue.Full:
        try:
            output_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            output_queue.put_nowait(output)
        except queue.Full:
            pass


class _ProfileAccumulator:
    def __init__(self, *, name: str, enabled: bool, interval: int):
        self._name = name
        self._enabled = bool(enabled)
        self._interval = max(1, int(interval))
        self._count = 0
        self._period_start = time.perf_counter()
        self._sums: dict[str, float] = {}

    def reset(self) -> None:
        self._count = 0
        self._period_start = time.perf_counter()
        self._sums = {}

    def record(self, timings_ms: dict[str, float]) -> str | None:
        if not self._enabled:
            return None
        for key, value in timings_ms.items():
            self._sums[key] = self._sums.get(key, 0.0) + float(value)
        self._count += 1
        if self._count < self._interval:
            return None

        now = time.perf_counter()
        period_s = max(now - self._period_start, 1e-9)
        update_hz = self._count / period_s
        parts = [f"{key}={value / self._count:.2f}ms" for key, value in self._sums.items()]
        message = (
            f"{self._name} latest-cache profile over {self._count} updates: "
            f"cache_update_hz={update_hz:.1f}, {', '.join(parts)}"
        )
        self.reset()
        return message


def _drain_control_queue(
    producer: Callable[[], dict[str, Any]],
    control_queue,
    profiler: _ProfileAccumulator,
    generation: int,
) -> int:
    while True:
        try:
            message = control_queue.get_nowait()
        except queue.Empty:
            return generation
        if not isinstance(message, dict):
            continue
        if message.get("type") == "reset":
            generation = int(message.get("generation", generation + 1))
            if hasattr(producer, "reset"):
                producer.reset()
            profiler.reset()
    return generation


def _process_worker_entry(
    *,
    name: str,
    producer_factory: Callable[..., Callable[[], dict[str, Any]]],
    producer_args: tuple[Any, ...],
    producer_kwargs: dict[str, Any],
    output_queue,
    command_queue,
    profile_queue,
    control_queue,
    stop_event,
    sleep_s: float,
    error_sleep_s: float,
    profile_enabled: bool,
    profile_interval: int,
) -> None:
    producer = producer_factory(*producer_args, **producer_kwargs)
    profiler = _ProfileAccumulator(name=name, enabled=profile_enabled, interval=profile_interval)
    error_logged = False
    generation = 0
    sleep_s = float(max(sleep_s, 0.0))
    error_sleep_s = float(max(error_sleep_s, 0.0))

    while not stop_event.is_set():
        generation = _drain_control_queue(producer, control_queue, profiler, generation)
        cycle_start = time.perf_counter()
        try:
            output = producer()
        except Exception as exc:
            if not error_logged:
                profile_queue.put(f"{name} producer failed: {exc!r}")
                error_logged = True
            if error_sleep_s > 0.0:
                time.sleep(error_sleep_s)
            continue

        cycle_ms = (time.perf_counter() - cycle_start) * 1000.0
        output = dict(output)
        step_timings = dict(output.pop("_profile_timings", {}))
        step_timings["worker_cycle"] = cycle_ms
        latest_output, commands = _split_output(output)
        latest_output["_worker_generation"] = generation
        if commands:
            command_queue.put({"generation": generation, "commands": commands})
        _put_latest(output_queue, latest_output)
        error_logged = False

        profile_message = profiler.record(step_timings)
        if profile_message is not None:
            profile_queue.put(profile_message)
        if sleep_s > 0.0:
            time.sleep(sleep_s)


class ProcessLatestOutputWorker:
    def __init__(
        self,
        *,
        name: str,
        producer_factory: Callable[..., Callable[[], dict[str, Any]]],
        initial_output: dict[str, Any],
        producer_args: tuple[Any, ...] = (),
        producer_kwargs: dict[str, Any] | None = None,
        sleep_s: float = 0.0,
        error_sleep_s: float = 0.05,
        profile_enabled: bool = False,
        profile_interval: int = 50,
        queue_size: int = 0,
        start_method: str = "spawn",
    ):
        self._name = name
        self._ctx = mp.get_context(start_method)
        self._producer_factory = producer_factory
        self._producer_args = tuple(producer_args)
        self._producer_kwargs = dict(producer_kwargs or {})
        self._sleep_s = float(max(sleep_s, 0.0))
        self._error_sleep_s = float(max(error_sleep_s, 0.0))
        self._profile_enabled = bool(profile_enabled)
        self._profile_interval = max(1, int(profile_interval))
        queue_size = int(queue_size)
        self._output_queue = self._ctx.Queue(maxsize=max(0, queue_size))
        self._command_queue = self._ctx.Queue()
        self._profile_queue = self._ctx.Queue()
        self._control_queue = self._ctx.Queue()
        self._stop_event = self._ctx.Event()
        self._process = None
        self._generation = 0
        self._latest_output, self._pending_commands = _split_output(initial_output)
        atexit.register(self.stop)

    def start(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        self._stop_event.clear()
        self._process = self._ctx.Process(
            target=_process_worker_entry,
            kwargs={
                "name": self._name,
                "producer_factory": self._producer_factory,
                "producer_args": self._producer_args,
                "producer_kwargs": self._producer_kwargs,
                "output_queue": self._output_queue,
                "command_queue": self._command_queue,
                "profile_queue": self._profile_queue,
                "control_queue": self._control_queue,
                "stop_event": self._stop_event,
                "sleep_s": self._sleep_s,
                "error_sleep_s": self._error_sleep_s,
                "profile_enabled": self._profile_enabled,
                "profile_interval": self._profile_interval,
            },
            name=self._name,
            daemon=True,
        )
        self._process.start()

    def stop(self, timeout: float | None = 1.0) -> None:
        process = self._process
        if process is None:
            return
        self._stop_event.set()
        if process.is_alive():
            process.join(timeout=timeout)
        if process.is_alive():
            process.terminate()
            process.join(timeout=timeout)
        self._process = None

    def reset(self, initial_output: dict[str, Any]) -> None:
        self._drain_output_queue()
        self._drain_command_queue()
        self._drain_profile_queue()
        self._generation += 1
        latest_output, _commands = _split_output(initial_output)
        self._latest_output = latest_output
        self._pending_commands = []
        try:
            self._control_queue.put_nowait({"type": "reset", "generation": self._generation})
        except queue.Full:
            pass

    def close(self) -> None:
        self.stop()

    def get_data(self) -> dict[str, Any]:
        self._drain_profile_queue()
        self._drain_output_queue()
        self._drain_command_queue()
        output = dict(self._latest_output)
        output["_commands"] = list(self._pending_commands)
        self._pending_commands.clear()
        return output

    def _drain_output_queue(self) -> None:
        while True:
            try:
                output = self._output_queue.get_nowait()
            except queue.Empty:
                return
            generation = int(output.pop("_worker_generation", 0))
            if generation != self._generation:
                continue
            latest_output, commands = _split_output(output)
            self._latest_output = latest_output
            self._pending_commands.extend(commands)

    def _drain_command_queue(self) -> None:
        while True:
            try:
                message = self._command_queue.get_nowait()
            except queue.Empty:
                return
            if not isinstance(message, dict):
                continue
            if int(message.get("generation", -1)) != self._generation:
                continue
            self._pending_commands.extend(message.get("commands", []))

    def _drain_profile_queue(self) -> None:
        while True:
            try:
                message = self._profile_queue.get_nowait()
            except queue.Empty:
                return
            logger.warning("%s", message)

    def __del__(self):
        self.stop()
