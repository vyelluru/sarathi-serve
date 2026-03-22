from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

from sarathi.metrics.constants import (
    BatchMetricsTimeDistribution,
    SequenceMetricsTimeDistributions,
    TokenMetricsTimeDistribution,
)


# ---------------------------------------------------------------------------
# Shared metrics snapshot
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MetricsSnapshot:
    """All metrics collected each controller update cycle."""
    timestamp_s: float
    # p95 latency signals — used by AIMD for threshold comparisons
    scheduling_delay_p95_s: Optional[float]
    decode_token_time_p95_s: Optional[float]
    batch_exec_time_p95_s: Optional[float]
    inter_batch_delay_p95_s: Optional[float]
    # EWMA of decode token time — used by PID as the continuous process variable
    decode_token_time_ewma_s: Optional[float]


# ---------------------------------------------------------------------------
# Config hierarchy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BaseControllerConfig:
    """Parameters shared by all controller strategies."""
    min_chunk_size: int = 128
    max_chunk_size: int = 4096
    window_size: int = 20
    update_every_iters: int = 5
    ewma_alpha: float = 0.2  # smoothing factor for decode token time EWMA


@dataclass(frozen=True)
class AIMDConfig(BaseControllerConfig):
    """AIMD-specific parameters."""
    additive_increase: int = 16
    multiplicative_decrease_factor: float = 0.75
    # Congestion thresholds (seconds)
    scheduling_delay_p95_threshold_s: float = 0.05
    batch_exec_time_p95_threshold_s: Optional[float] = None
    decode_token_time_p95_threshold_s: Optional[float] = None
    # Early warning: flag congestion if p95 grows by this fraction across window
    scheduling_delay_trend_frac: float = 0.25


@dataclass(frozen=True)
class PIDConfig(BaseControllerConfig):
    """PID-specific parameters."""
    target_decode_token_time_s: float = 0.02   # target TPOT setpoint (seconds)
    kp: float = 512.0    # proportional gain: maps seconds of error → tokens
    ki: float = 64.0     # integral gain: corrects sustained drift
    kd: float = 128.0    # derivative gain: dampens overshoot
    integral_clamp: float = 2048.0  # anti-windup: max absolute integral value


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Base controller
# ---------------------------------------------------------------------------

class BaseChunkSizeController(ABC):
    """
    Shared scaffolding for all chunk size control strategies.

    Handles metric collection, EWMA tracking, history management, and the
    update cadence. Subclasses implement `_compute_new_chunk_size(snap)`
    with their own control logic.
    """

    def __init__(
        self,
        *,
        initial_chunk_size: int,
        metrics_store=None,
        config: BaseControllerConfig,
    ) -> None:
        # `metrics_store` is kept generic on purpose. In production the engine
        # passes a real MetricsStore; in tests override `_collect_snapshot`.
        self.metrics_store = metrics_store
        self.config = config

        self._iter: int = 0
        self._history: Deque[MetricsSnapshot] = deque(maxlen=config.window_size)
        self._decode_token_time_ewma: Optional[float] = None

        initial = int(initial_chunk_size)
        self._chunk_size = max(config.min_chunk_size, min(config.max_chunk_size, initial))

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    def _collect_snapshot(self) -> MetricsSnapshot:
        """Read all metrics from the store and return a unified snapshot."""
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

        decode_p95 = _p95_from_cdf_sketch(decode_token_sketch)

        # Update running EWMA of decode token time (used by PID)
        if decode_p95 is not None:
            if self._decode_token_time_ewma is None:
                self._decode_token_time_ewma = decode_p95
            else:
                a = self.config.ewma_alpha
                self._decode_token_time_ewma = (
                    a * decode_p95 + (1 - a) * self._decode_token_time_ewma
                )

        return MetricsSnapshot(
            timestamp_s=time.time(),
            scheduling_delay_p95_s=_p95_from_dataseries(scheduling_delay_ds),
            decode_token_time_p95_s=decode_p95,
            batch_exec_time_p95_s=_p95_from_dataseries(batch_exec_ds),
            inter_batch_delay_p95_s=_p95_from_dataseries(inter_batch_ds),
            decode_token_time_ewma_s=self._decode_token_time_ewma,
        )

    def update(self) -> MetricsSnapshot:
        """
        Called once per engine iteration. Collects metrics, updates history,
        and adjusts chunk_size every `update_every_iters` iterations once
        there is enough history.
        """
        self._iter += 1
        snap = self._collect_snapshot()
        self._history.append(snap)

        if self._iter % self.config.update_every_iters != 0:
            return snap

        # Wait until there is enough history for a meaningful signal.
        if len(self._history) < max(5, self.config.window_size // 4):
            return snap

        new_size = self._compute_new_chunk_size(snap)
        self._chunk_size = max(
            self.config.min_chunk_size,
            min(self.config.max_chunk_size, new_size),
        )
        return snap

    @abstractmethod
    def _compute_new_chunk_size(self, snap: MetricsSnapshot) -> int:
        """Return the desired new chunk size given the latest snapshot."""
        ...


# ---------------------------------------------------------------------------
# AIMD strategy
# ---------------------------------------------------------------------------

class AIMDChunkSizeController(BaseChunkSizeController):
    """
    Additive Increase / Multiplicative Decrease controller.

    Uses p95 scheduling delay (and optionally batch exec time / decode token
    time) as congestion signals. Binary decision: congested → halve chunk
    size, healthy → add fixed increment.
    """

    def __init__(
        self,
        *,
        initial_chunk_size: int,
        metrics_store=None,
        config: Optional[AIMDConfig] = None,
    ) -> None:
        super().__init__(
            initial_chunk_size=initial_chunk_size,
            metrics_store=metrics_store,
            config=config or AIMDConfig(),
        )

    def _is_congested(self, snap: MetricsSnapshot) -> bool:
        c: AIMDConfig = self.config  # type: ignore[assignment]

        # Threshold-based signals
        sd = snap.scheduling_delay_p95_s
        if sd is not None and sd >= c.scheduling_delay_p95_threshold_s:
            return True

        if c.batch_exec_time_p95_threshold_s is not None:
            be = snap.batch_exec_time_p95_s
            if be is not None and be >= c.batch_exec_time_p95_threshold_s:
                return True

        if c.decode_token_time_p95_threshold_s is not None:
            dt = snap.decode_token_time_p95_s
            if dt is not None and dt >= c.decode_token_time_p95_threshold_s:
                return True

        # Trend-based early warning: growing scheduling delay → treat as congested
        if sd is None:
            return False
        oldest = next(
            (h.scheduling_delay_p95_s for h in self._history if h.scheduling_delay_p95_s is not None),
            None,
        )
        if oldest is None or oldest <= 0:
            return False
        return (sd - oldest) / oldest >= c.scheduling_delay_trend_frac

    def _compute_new_chunk_size(self, snap: MetricsSnapshot) -> int:
        c: AIMDConfig = self.config  # type: ignore[assignment]
        if self._is_congested(snap):
            return int(self._chunk_size * c.multiplicative_decrease_factor)
        return self._chunk_size + c.additive_increase


# ---------------------------------------------------------------------------
# PID strategy
# ---------------------------------------------------------------------------

class PIDChunkSizeController(BaseChunkSizeController):
    """
    Proportional-Integral-Derivative controller.

    Uses the EWMA of decode token time as the process variable against a
    target TPOT setpoint. The continuous error signal produces smoother
    chunk size adjustments than AIMD's binary increase/decrease.
    """

    def __init__(
        self,
        *,
        initial_chunk_size: int,
        metrics_store=None,
        config: Optional[PIDConfig] = None,
    ) -> None:
        super().__init__(
            initial_chunk_size=initial_chunk_size,
            metrics_store=metrics_store,
            config=config or PIDConfig(),
        )
        self._integral: float = 0.0
        self._prev_error: Optional[float] = None

    def _compute_new_chunk_size(self, snap: MetricsSnapshot) -> int:
        c: PIDConfig = self.config  # type: ignore[assignment]

        pv = snap.decode_token_time_ewma_s
        if pv is None:
            return self._chunk_size  # no signal yet, hold steady

        error = pv - c.target_decode_token_time_s  # positive → too slow → decrease

        # Integral with anti-windup clamp
        self._integral = max(
            -c.integral_clamp,
            min(c.integral_clamp, self._integral + error),
        )

        # Derivative (error change between update cycles)
        derivative = 0.0
        if self._prev_error is not None:
            derivative = error - self._prev_error
        self._prev_error = error

        # Negative feedback: positive error → negative adjustment → smaller chunk
        adjustment = -(c.kp * error + c.ki * self._integral + c.kd * derivative)
        return int(self._chunk_size + adjustment)


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------

# Engine wiring (async_llm_engine.py) and existing tests use these names.
ChunkSizeController = AIMDChunkSizeController
ChunkSizeControllerConfig = AIMDConfig

