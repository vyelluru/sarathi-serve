from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sarathi.controller.chunk_size_controller import (
    AIMDChunkSizeController,
    MetricsSnapshot,
    # backward-compat aliases
    ChunkSizeController,
    ChunkSizeControllerConfig,
)


class FakeController(AIMDChunkSizeController):
    """
    Controller variant that bypasses MetricsStore and lets tests inject
    synthetic snapshots directly.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._snap: MetricsSnapshot | None = None

    def set_snapshot(self, snap: MetricsSnapshot) -> None:
        self._snap = snap

    def _collect_snapshot(self) -> MetricsSnapshot:  # type: ignore[override]
        assert self._snap is not None
        return self._snap


def _make_snapshot(delay_s: float) -> MetricsSnapshot:
    return MetricsSnapshot(
        timestamp_s=0.0,
        scheduling_delay_p95_s=delay_s,
        decode_token_time_p95_s=None,
        batch_exec_time_p95_s=None,
        inter_batch_delay_p95_s=None,
        decode_token_time_ewma_s=None,
    )


def test_chunk_size_increases_when_healthy():
    cfg = ChunkSizeControllerConfig(
        min_chunk_size=128,
        max_chunk_size=4096,
        additive_increase=128,
        multiplicative_decrease_factor=0.5,
        scheduling_delay_p95_threshold_s=0.05,
        window_size=10,
        update_every_iters=1,
    )

    ctrl = FakeController(initial_chunk_size=512, metrics_store=None, config=cfg)
    ctrl.set_snapshot(_make_snapshot(0.005))  # well below threshold

    initial = ctrl.chunk_size
    for _ in range(10):
        ctrl.update()

    assert ctrl.chunk_size > initial


def test_chunk_size_decreases_on_congestion():
    cfg = ChunkSizeControllerConfig(
        min_chunk_size=128,
        max_chunk_size=4096,
        additive_increase=128,
        multiplicative_decrease_factor=0.5,
        scheduling_delay_p95_threshold_s=0.05,
        window_size=10,
        update_every_iters=1,
    )

    ctrl = FakeController(initial_chunk_size=2048, metrics_store=None, config=cfg)

    # First, establish a healthy history at low delay.
    ctrl.set_snapshot(_make_snapshot(0.005))
    for _ in range(5):
        ctrl.update()

    before = ctrl.chunk_size

    # Now inject congestion (high scheduling delay) and ensure chunk size drops.
    ctrl.set_snapshot(_make_snapshot(0.2))
    for _ in range(5):
        ctrl.update()

    assert ctrl.chunk_size < before

