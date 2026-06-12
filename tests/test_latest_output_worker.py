from __future__ import annotations

from robojudo.controller.utils.latest_output_worker import LatestOutputWorker


def test_worker_profile_logs_cache_update_rate_and_step_timings(capsys):
    outputs = iter(
        [
            {"state": "idle", "_commands": [], "_profile_timings": {"pico_read": 1.0}},
            {"state": "idle", "_commands": [], "_profile_timings": {"pico_read": 3.0, "retarget": 5.0}},
        ]
    )
    worker = LatestOutputWorker(
        name="PicoProfileTestWorker",
        producer=lambda: next(outputs),
        initial_output={"state": "idle", "_commands": []},
        profile_enabled=True,
        profile_interval=2,
    )

    worker._run_once()
    worker._run_once()

    output = capsys.readouterr().out
    assert "PicoProfileTestWorker latest-cache profile over 2 updates" in output
    assert "cache_update_hz=" in output
    assert "worker_cycle=" in output
    assert "pico_read=2.00ms" in output
    assert "retarget=2.50ms" in output
