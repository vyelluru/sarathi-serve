"""Verify all 6 experiment configs parse correctly and generate requests — no GPU needed."""
from __future__ import annotations

import importlib.metadata as importlib_metadata
import os
import sys

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Patch torch version so sarathi config imports succeed without a real GPU build.
_orig_version = importlib_metadata.version


def _patched_version(name: str) -> str:
    if name == "torch":
        return "2.11.0"
    return _orig_version(name)


importlib_metadata.version = _patched_version

from sarathi.benchmark.config import BenchmarkConfig
from sarathi.benchmark.request_generator import RequestGeneratorRegistry
from sarathi.types import ControllerType

CONFIGS = [
    ("experiments/configs/static_steady.yaml", ControllerType.NONE, "traces/steady_trace.csv"),
    ("experiments/configs/static_bursty.yaml", ControllerType.NONE, "traces/bursty_trace.csv"),
    ("experiments/configs/aimd_steady.yaml", ControllerType.AIMD, "traces/steady_trace.csv"),
    ("experiments/configs/aimd_bursty.yaml", ControllerType.AIMD, "traces/bursty_trace.csv"),
    ("experiments/configs/pid_steady.yaml", ControllerType.PID, "traces/steady_trace.csv"),
    ("experiments/configs/pid_bursty.yaml", ControllerType.PID, "traces/bursty_trace.csv"),
]


@pytest.mark.parametrize(
    "config_path, expected_controller, expected_trace",
    CONFIGS,
    ids=[os.path.basename(c[0]).removesuffix(".yaml") for c in CONFIGS],
)
def test_config_parses_and_generates_requests(
    config_path: str,
    expected_controller: ControllerType,
    expected_trace: str,
) -> None:
    full_path = os.path.join(ROOT, config_path)

    # 1. YAML loads without error
    with open(full_path) as f:
        raw = yaml.safe_load(f)
    assert isinstance(raw, dict), f"Expected top-level mapping, got {type(raw)}"

    # 2. Config object is created successfully
    cfg = BenchmarkConfig.create_from_dict(raw)

    # 3. controller_type matches expectations
    assert cfg.scheduler_config.controller_type == expected_controller

    # 4. Trace file referenced in the config exists on disk
    trace_file = cfg.request_generator_config.length_generator_config.trace_file
    assert os.path.isfile(os.path.join(ROOT, trace_file)), (
        f"Trace file not found: {trace_file}"
    )
    assert trace_file == expected_trace

    # 5. Request generator runs and produces the expected number of requests
    gen = RequestGeneratorRegistry.get(
        cfg.request_generator_config.get_type(),
        cfg.request_generator_config,
    )
    requests = gen.generate()
    assert len(requests) == cfg.request_generator_config.num_requests
    assert len(requests) > 0

    # 6. Each request has valid token counts
    for req in requests:
        assert req.num_prefill_tokens > 0
        assert req.num_decode_tokens > 0
