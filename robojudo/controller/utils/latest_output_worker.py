from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class LatestOutputWorker:
    def __init__(
        self,
        *,
        name: str,
        producer: Callable[[], dict[str, Any]],
        initial_output: dict[str, Any],
        sleep_s: float = 0.0,
        error_sleep_s: float = 0.05,
        profile_enabled: bool = False,
        profile_interval: int = 50,
    ):
        self._name = name
        self._producer = producer
        self._sleep_s = float(max(sleep_s, 0.0))
        self._error_sleep_s = float(max(error_sleep_s, 0.0))
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._latest_output, self._pending_commands = self._split_output(initial_output)
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._error_logged = False
        self._profile_enabled = bool(profile_enabled)
        self._profile_interval = max(1, int(profile_interval))
        self._profile_count = 0
        self._profile_period_start = time.perf_counter()
        self._profile_sums: dict[str, float] = {}

    @staticmethod
    def _split_output(output: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        output_copy = dict(output)
        commands = list(output_copy.pop("_commands", []))
        return output_copy, commands

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float | None = 0.5) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def reset(self, initial_output: dict[str, Any]) -> None:
        latest_output, _commands = self._split_output(initial_output)
        with self._lock:
            self._latest_output = latest_output
            self._pending_commands = []
            self._error_logged = False
            self._profile_count = 0
            self._profile_period_start = time.perf_counter()
            self._profile_sums = {}

    def get_data(self) -> dict[str, Any]:
        with self._lock:
            output = dict(self._latest_output)
            commands = list(self._pending_commands)
            self._pending_commands.clear()
        output["_commands"] = commands
        return output

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._run_once()
            if self._sleep_s > 0.0:
                time.sleep(self._sleep_s)

    def _run_once(self) -> None:
        cycle_start = time.perf_counter()
        try:
            output = self._producer()
        except Exception:
            if not self._error_logged:
                logger.exception("%s producer failed", self._name)
                self._error_logged = True
            if self._error_sleep_s > 0.0:
                time.sleep(self._error_sleep_s)
            return
        cycle_ms = (time.perf_counter() - cycle_start) * 1000.0

        step_timings = dict(output.pop("_profile_timings", {}))
        step_timings["worker_cycle"] = cycle_ms
        latest_output, commands = self._split_output(output)
        with self._lock:
            self._latest_output = latest_output
            self._pending_commands.extend(commands)
            self._error_logged = False
        self._record_profile(step_timings)

    def _record_profile(self, timings_ms: dict[str, float]) -> None:
        if not self._profile_enabled:
            return
        for key, value in timings_ms.items():
            self._profile_sums[key] = self._profile_sums.get(key, 0.0) + float(value)
        self._profile_count += 1
        if self._profile_count < self._profile_interval:
            return

        now = time.perf_counter()
        period_s = max(now - self._profile_period_start, 1e-9)
        update_hz = self._profile_count / period_s
        parts = [f"{key}={value / self._profile_count:.2f}ms" for key, value in self._profile_sums.items()]
        logger.warning(
            "%s latest-cache profile over %d updates: cache_update_hz=%.1f, %s",
            self._name,
            self._profile_count,
            update_hz,
            ", ".join(parts),
        )
        self._profile_count = 0
        self._profile_period_start = now
        self._profile_sums = {}
