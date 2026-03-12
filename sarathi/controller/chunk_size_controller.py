from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

from sarathi.metrics.constants import (
    BatchMetricsTimeDistribution,
    SequenceMetricsTimeDistributions,
    TokenMetricsTimeDistribution,
)


@dataclass(frozen=True)
class CongestionP95Snapshot:
    timestamp_s: float
    request_scheduling_delay_p95_s: Optional[float]
    batch_execution_time_p95_s: Optional[float]
    inter_batch_delay_p95_s: Optional[float]
    decode_token_time_p95_s: Optional[float]


@dataclass(frozen=True)
class ChunkSizeControllerConfig:
    # Controller cadence/state
    window_size: int = 20
    update_every_iters: int = 5

    # AIMD parameters
    min_chunk_size: int = 128
    max_chunk_size: int = 4096
    additive_increase: int = 128
    multiplicative_decrease_factor: float = 0.5  # multiply, then round to int

    # Congestion thresholds (seconds)
    scheduling_delay_p95_threshold_s: float = 0.05
    batch_exec_time_p95_threshold_s: Optional[float] = None
    decode_token_time_p95_threshold_s: Optional[float] = None

    # Early warning: if p95 grows by this fraction across the window, treat as congestion
    scheduling_delay_trend_frac: float = 0.25


def _p95_from_dataseries(ds) -> Optional[float]:
    if ds is None or len(ds) == 0:
        return None
    df = ds.to_df()
    if df.empty:
        return None
    return float(df[ds.y_name].quantile(0.95))


def _p95_from_cdf_sketch(sketch) -> Optional[float]:
    if sketch is None or len(sketch) == 0:
        return None
    return float(sketch.sketch.get_quantile_value(0.95))


class ChunkSizeController:
    """
    Minimal "controller component" scaffold.

    Right now it only snapshots p95 metrics that are useful for congestion
    detection. You can extend this to apply AIMD rules and publish a mutable
    chunk_size that the scheduler reads.
    """

    def __init__(
        self,
        *,
        initial_chunk_size: int,
        metrics_store=None,
        config: Optional[ChunkSizeControllerConfig] = None,
    ) -> None:
        # `metrics_store` is kept generic on purpose. In production, the engine
        # passes a real MetricsStore instance; in tests we often pass None and
        # override `snapshot_p95` to avoid touching metrics at all.
        self.metrics_store = metrics_store
        self.config = config or ChunkSizeControllerConfig()

        self._iter = 0
        self._history: Deque[CongestionP95Snapshot] = deque(maxlen=self.config.window_size)

        initial = int(initial_chunk_size)
        self._chunk_size = max(self.config.min_chunk_size, min(self.config.max_chunk_size, initial))

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    def snapshot_p95(self) -> CongestionP95Snapshot:
        ms = self.metrics_store

        scheduling_delay_ds = ms.seq_metrics_time_distributions.get(
            SequenceMetricsTimeDistributions.REQUEST_SCHEDULING_DELAY
        )
        batch_exec_ds = ms.batch_metrics_time_distribution.get(
            BatchMetricsTimeDistribution.BATCH_EXECUTION_TIME
        )
        inter_batch_ds = ms.batch_metrics_time_distribution.get(
            BatchMetricsTimeDistribution.INTER_BATCH_DELAY
        )
        decode_token_sketch = ms.token_metrics_time_distribution.get(
            TokenMetricsTimeDistribution.DECODE_TOKEN_EXECUTION_PLUS_PREEMPTION_TIME
        )

        return CongestionP95Snapshot(
            timestamp_s=time.time(),
            request_scheduling_delay_p95_s=_p95_from_dataseries(scheduling_delay_ds),
            batch_execution_time_p95_s=_p95_from_dataseries(batch_exec_ds),
            inter_batch_delay_p95_s=_p95_from_dataseries(inter_batch_ds),
            decode_token_time_p95_s=_p95_from_cdf_sketch(decode_token_sketch),
        )

    def update(self) -> CongestionP95Snapshot:
        """
        Called once per engine iteration (batch). Updates history and applies
        AIMD to self._chunk_size every `update_every_iters` iterations.
        """
        self._iter += 1
        snap = self.snapshot_p95()
        self._history.append(snap)

        if self._iter % self.config.update_every_iters != 0:
            return snap

        # Not enough signal yet; keep current chunk size.
        if len(self._history) < max(5, self.config.window_size // 4):
            return snap

        should_decrease = self._is_congested(snap)
        if should_decrease:
            new_size = int(self._chunk_size * self.config.multiplicative_decrease_factor)
        else:
            new_size = self._chunk_size + self.config.additive_increase

        self._chunk_size = max(self.config.min_chunk_size, min(self.config.max_chunk_size, new_size))
        return snap

    def _is_congested(self, snap: CongestionP95Snapshot) -> bool:
        c = self.config

        sd = snap.request_scheduling_delay_p95_s
        if sd is not None and sd >= c.scheduling_delay_p95_threshold_s:
            return True

        if c.batch_exec_time_p95_threshold_s is not None:
            be = snap.batch_execution_time_p95_s
            if be is not None and be >= c.batch_exec_time_p95_threshold_s:
                return True

        if c.decode_token_time_p95_threshold_s is not None:
            dt = snap.decode_token_time_p95_s
            if dt is not None and dt >= c.decode_token_time_p95_threshold_s:
                return True

        # Trend-based early warning on scheduling delay p95.
        if sd is None:
            return False
        oldest = next((h.request_scheduling_delay_p95_s for h in self._history if h.request_scheduling_delay_p95_s is not None), None)
        if oldest is None or oldest <= 0:
            return False
        return (sd - oldest) / oldest >= c.scheduling_delay_trend_frac

