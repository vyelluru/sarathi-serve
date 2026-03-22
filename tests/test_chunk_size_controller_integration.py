from __future__ import annotations

import asyncio
import importlib.metadata as importlib_metadata
import os
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

_orig_version = importlib_metadata.version


def _patched_version(name: str) -> str:
    if name == "torch":
        return "2.11.0"
    return _orig_version(name)


importlib_metadata.version = _patched_version

from sarathi.config import CacheConfig, MetricsConfig, ParallelConfig, ReplicaConfig, SarathiSchedulerConfig
from sarathi.controller.chunk_size_controller import ChunkSizeController, ChunkSizeControllerConfig
from sarathi.core.datatypes.sampling_params import SamplingParams
from sarathi.core.datatypes.sequence import Sequence
from sarathi.core.scheduler.sarathi_scheduler import SarathiScheduler
from sarathi.engine.async_llm_engine import AsyncLLMEngine, RequestTracker
from sarathi.metrics.constants import (
    BatchMetricsTimeDistribution,
    SequenceMetricsTimeDistributions,
    TokenMetricsTimeDistribution,
)
from sarathi.metrics.metrics_store import MetricsStore


class DummyModelConfig:
    max_model_len = 4096

    def get_total_num_layers(self) -> int:
        return 1


class FakeBaseEngine:
    def __init__(self, tmp_path: Path) -> None:
        self.metrics_store = MetricsStore.get_or_create_instance(
            ReplicaConfig(replica_id=0, output_dir=str(tmp_path)),
            DummyModelConfig(),
            MetricsConfig(
                write_metrics=True,
                enable_chrome_trace=False,
                keep_individual_batch_metrics=True,
            ),
        )
        self.metrics_store.mark_initial_memory_profiling_done()

        self.scheduler = SarathiScheduler(
            DummyModelConfig(),
            SarathiSchedulerConfig(
                chunk_size=512,
                enable_dynamic_chunking_schedule=False,
                max_num_seqs=8,
            ),
            CacheConfig(block_size=16, num_gpu_blocks=256),
            ParallelConfig(pipeline_parallel_size=1, tensor_parallel_size=1),
        )
        self.chunk_controller = ChunkSizeController(
            initial_chunk_size=self.scheduler.chunk_size,
            metrics_store=self.metrics_store,
            config=ChunkSizeControllerConfig(
                window_size=5,
                update_every_iters=1,
                min_chunk_size=128,
                max_chunk_size=4096,
                additive_increase=128,
                multiplicative_decrease_factor=0.5,
                scheduling_delay_p95_threshold_s=0.05,
            ),
        )
        self._iteration = 0

    def record_iteration_metrics(self) -> None:
        idx = self._iteration
        self.metrics_store.seq_metrics_time_distributions[
            SequenceMetricsTimeDistributions.REQUEST_SCHEDULING_DELAY
        ].put(f"req-{idx}", 0.005)
        self.metrics_store.batch_metrics_time_distribution[
            BatchMetricsTimeDistribution.BATCH_EXECUTION_TIME
        ].put_pair(idx, 0.01)
        self.metrics_store.batch_metrics_time_distribution[
            BatchMetricsTimeDistribution.INTER_BATCH_DELAY
        ].put_pair(idx, 0.001)
        self.metrics_store.token_metrics_time_distribution[
            TokenMetricsTimeDistribution.DECODE_TOKEN_EXECUTION_PLUS_PREEMPTION_TIME
        ].put(0.002)
        self._iteration += 1


class FakeAsyncEngineAdapter:
    def __init__(self, base_engine: FakeBaseEngine) -> None:
        self.engine = base_engine

    def add_request(self, **_: object) -> None:
        return None

    async def step_async(self):
        self.engine.record_iteration_metrics()
        return []


def _make_sequence() -> Sequence:
    return Sequence(
        seq_id="seq-1",
        prompt="",
        prompt_token_ids=[1] * 2048,
        block_size=16,
        eos_token_id=2,
        arrival_time=0.0,
        sampling_params=SamplingParams(
            temperature=0.0,
            top_p=1.0,
            ignore_eos=True,
            max_tokens=4,
        ),
    )


def test_async_engine_step_updates_scheduler_chunk_size_from_metrics(tmp_path: Path):
    base_engine = FakeBaseEngine(tmp_path)
    async_engine = AsyncLLMEngine(FakeAsyncEngineAdapter(base_engine))
    async_engine._request_tracker = RequestTracker()

    seq = _make_sequence()
    before = base_engine.scheduler._get_seq_next_num_prefill_tokens(seq, 0)

    for _ in range(5):
        asyncio.run(async_engine.engine_step())

    after_controller = base_engine.chunk_controller.chunk_size
    after_scheduler = base_engine.scheduler.chunk_size
    after_limit = base_engine.scheduler._get_seq_next_num_prefill_tokens(seq, 0)

    assert before == 512
    assert after_controller == 640
    assert after_scheduler == after_controller
    assert after_limit == after_controller
