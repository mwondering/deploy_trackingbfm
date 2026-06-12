from __future__ import annotations

import multiprocessing as mp
import time

from robojudo.controller.utils.process_latest_output_worker import ProcessLatestOutputWorker


class _QueueProducer:
    def __init__(self, input_queue):
        self._input_queue = input_queue
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1

    def __call__(self):
        return self._input_queue.get()


def _queue_producer_factory(input_queue):
    return _QueueProducer(input_queue)


def test_process_worker_get_data_returns_cached_output_while_producer_waits():
    ctx = mp.get_context("spawn")
    input_queue = ctx.Queue()
    worker = ProcessLatestOutputWorker(
        name="PicoProcessCacheTestWorker",
        producer_factory=_queue_producer_factory,
        producer_args=(input_queue,),
        initial_output={"state": "idle", "_commands": []},
        start_method="spawn",
    )
    worker.start()
    try:
        start = time.perf_counter()
        output = worker.get_data()
        elapsed = time.perf_counter() - start
    finally:
        worker.stop()

    assert elapsed < 0.05
    assert output == {"state": "idle", "_commands": []}


def test_process_worker_drains_commands_once_from_child_output():
    ctx = mp.get_context("spawn")
    input_queue = ctx.Queue()
    worker = ProcessLatestOutputWorker(
        name="PicoProcessCommandTestWorker",
        producer_factory=_queue_producer_factory,
        producer_args=(input_queue,),
        initial_output={"state": "idle", "_commands": []},
        start_method="spawn",
    )
    worker.start()
    try:
        input_queue.put({"state": "active", "_commands": ["[MOTION_RESET]"]})
        deadline = time.time() + 3.0
        commands = []
        while time.time() < deadline:
            output = worker.get_data()
            commands = output["_commands"]
            if commands:
                break
            time.sleep(0.01)

        assert commands == ["[MOTION_RESET]"]
        assert worker.get_data()["_commands"] == []
    finally:
        worker.stop()


def test_process_worker_preserves_transient_commands_when_latest_output_is_overwritten():
    ctx = mp.get_context("spawn")
    input_queue = ctx.Queue()
    worker = ProcessLatestOutputWorker(
        name="PicoProcessTransientCommandTestWorker",
        producer_factory=_queue_producer_factory,
        producer_args=(input_queue,),
        initial_output={"state": "idle", "_commands": []},
        start_method="spawn",
    )
    worker.start()
    try:
        input_queue.put({"state": "active", "value": 1, "_commands": ["[MOTION_RESET]"]})
        input_queue.put({"state": "active", "value": 2, "_commands": []})

        deadline = time.time() + 3.0
        output = {}
        while time.time() < deadline:
            output = worker.get_data()
            if output.get("value") == 2:
                break
            time.sleep(0.01)

        assert output["state"] == "active"
        assert output["value"] == 2
        assert output["_commands"] == ["[MOTION_RESET]"]
        assert worker.get_data()["_commands"] == []
    finally:
        worker.stop()


def test_process_worker_reset_discards_stale_queued_outputs():
    ctx = mp.get_context("spawn")
    input_queue = ctx.Queue()
    worker = ProcessLatestOutputWorker(
        name="PicoProcessResetTestWorker",
        producer_factory=_queue_producer_factory,
        producer_args=(input_queue,),
        initial_output={"state": "idle", "_commands": []},
        start_method="spawn",
    )

    worker._output_queue.put_nowait({"state": "active", "_commands": ["[OLD]"]})
    worker.reset({"state": "idle", "_commands": []})

    assert worker.get_data() == {"state": "idle", "_commands": []}


def test_process_worker_profile_logs_cache_update_rate_and_step_timings(capsys):
    ctx = mp.get_context("spawn")
    input_queue = ctx.Queue()
    worker = ProcessLatestOutputWorker(
        name="PicoProcessProfileTestWorker",
        producer_factory=_queue_producer_factory,
        producer_args=(input_queue,),
        initial_output={"state": "idle", "_commands": []},
        profile_enabled=True,
        profile_interval=2,
        start_method="spawn",
    )
    worker.start()
    try:
        input_queue.put({"state": "idle", "_commands": [], "_profile_timings": {"pico_read": 1.0}})
        input_queue.put(
            {
                "state": "idle",
                "_commands": [],
                "_profile_timings": {"pico_read": 3.0, "retarget": 5.0},
            }
        )
        deadline = time.time() + 3.0
        output = ""
        while time.time() < deadline and "PicoProcessProfileTestWorker latest-cache profile" not in output:
            worker.get_data()
            output = capsys.readouterr().out
            time.sleep(0.01)
    finally:
        worker.stop()

    output += capsys.readouterr().out
    assert "PicoProcessProfileTestWorker latest-cache profile over 2 updates" in output
    assert "cache_update_hz=" in output
    assert "worker_cycle=" in output
    assert "pico_read=2.00ms" in output
    assert "retarget=2.50ms" in output
