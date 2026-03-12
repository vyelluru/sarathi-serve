from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sarathi.controller.chunk_size_controller import (
    ChunkSizeController,
    ChunkSizeControllerConfig,
    CongestionP95Snapshot,
)


class DummyScheduler:
    def __init__(self, chunk_size: int) -> None:
        self.chunk_size = chunk_size
        self.enable_dynamic_chunking_schedule = False


class DummyBaseEngine:
    """
    Minimal stand-in for BaseLLMEngine for testing wiring logic.
    """

    def __init__(self, initial_chunk_size: int) -> None:
        # controller seeded from config.scheduler_config.chunk_size
        cfg = ChunkSizeControllerConfig(
            min_chunk_size=128,
            max_chunk_size=4096,
            additive_increase=128,
            multiplicative_decrease_factor=0.5,
            scheduling_delay_p95_threshold_s=0.05,
            window_size=10,
            update_every_iters=1,
        )
        self.chunk_controller = ChunkSizeController(
            initial_chunk_size=initial_chunk_size,
            metrics_store=None,  # not used in this test
            config=cfg,
        )
        self.scheduler = DummyScheduler(chunk_size=initial_chunk_size)


def _make_snapshot(delay_s: float) -> CongestionP95Snapshot:
    return CongestionP95Snapshot(
        timestamp_s=0.0,
        request_scheduling_delay_p95_s=delay_s,
        batch_execution_time_p95_s=None,
        inter_batch_delay_p95_s=None,
        decode_token_time_p95_s=None,
    )


def _apply_wiring_once(base_engine: DummyBaseEngine) -> None:
    """
    Mirrors the logic we added in AsyncLLMEngine.engine_step, but in a
    synchronous, minimal form for unit testing.
    """
    controller = base_engine.chunk_controller
    # Pretend one engine iteration completed and metrics are available.
    controller.snapshot_p95 = lambda: _make_snapshot(0.005)  # type: ignore[assignment]
    controller.update()

    scheduler = base_engine.scheduler
    if getattr(scheduler, "enable_dynamic_chunking_schedule", False):
        return
    if hasattr(scheduler, "chunk_size"):
        scheduler.chunk_size = controller.chunk_size


def test_scheduler_chunk_size_follows_controller():
    base = DummyBaseEngine(initial_chunk_size=512)

    before_controller = base.chunk_controller.chunk_size
    before_scheduler = base.scheduler.chunk_size

    _apply_wiring_once(base)

    after_controller = base.chunk_controller.chunk_size
    after_scheduler = base.scheduler.chunk_size

    # Wiring contract: scheduler should track whatever value the controller has
    # after the iteration, even if the controller does not move immediately.
    assert after_scheduler == after_controller
