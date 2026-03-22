# Sarathi-Serve: Runtime Chunk Size Controller Extension

Sarathi-Serve's chunk size is a static hyperparameter set at launch time. It controls how many prefill tokens are processed per batch iteration, effectively setting the interleaving ratio between prefill and decode work. A fixed value cannot adapt as workload characteristics change during runtime.

This extension adds a runtime controller that dynamically adjusts chunk size based on live system metrics, inspired by the original paper's own suggestion:

> "The system performance can be further enhanced by dynamically varying the token budget based on workload characteristics. We leave this exploration for future work."

---

## How Chunk Size Works

Each engine iteration, the scheduler slices incoming prompts into chunks of up to `chunk_size` tokens and mixes them with decode steps from in-flight requests. Think of chunk size as a yield point — how much prefill work you do before yielding back to decodes.

- **Too large:** prefill dominates, decode requests stall, TPOT degrades
- **Too small:** new requests take many iterations to complete prefill, TTFT suffers
- **Optimal:** dynamically balances both based on current load

Chunk size has no effect on KV cache memory (blocks are allocated per sequence at admission). It only affects per-step activation memory and the prefill/decode interleaving ratio.

---

## Controller Strategies

Three modes, selected via `scheduler_config.controller_type` in config:

### `NONE` — Static baseline (original sarathi-serve behavior)
Chunk size stays fixed at the value set in config. No controller runs.

### `AIMD` — Additive Increase / Multiplicative Decrease
Inspired by TCP ECN. Reads p95 metrics each iteration and applies binary rules:
- Healthy → `chunk_size += 16` (additive increase)
- Congested → `chunk_size *= 0.75` (multiplicative decrease)

Congestion signals (any one triggers decrease):
- `scheduling_delay_p95` exceeds threshold (primary signal)
- `batch_exec_time_p95` exceeds threshold (optional)
- `decode_token_time_p95` exceeds threshold (optional)
- Trend-based early warning: scheduling delay p95 growing by >25% across window

Limitation: binary signal causes oscillation around the congestion threshold.

### `PID` — Proportional-Integral-Derivative
Uses EWMA of decode token time as a continuous process variable against a target TPOT setpoint. Produces smoother adjustments than AIMD.

- **P:** responds proportionally to current TPOT deviation
- **I:** corrects sustained drift (with anti-windup clamp)
- **D:** dampens overshoot by reacting to rate of change

More aligned with how modern datacenter congestion control works (DCTCP, DCQCN, Swift) — all use continuous signals rather than binary increase/decrease.

---

## Key Files

```
sarathi/
  controller/
    chunk_size_controller.py     # BaseChunkSizeController, AIMDChunkSizeController, PIDChunkSizeController
  engine/
    base_llm_engine.py           # _build_controller() selects strategy from config
    async_llm_engine.py          # engine_step() calls controller.update() after each iteration
  config/
    config.py                    # SarathiSchedulerConfig.controller_type field
  types.py                       # ControllerType enum (NONE, AIMD, PID)

experiments/
  configs/                       # 6 benchmark YAML configs (3 controllers x 2 traces)
  run_experiments.sh             # Runs all 6 conditions sequentially

traces/
  steady_trace.csv               # 500 requests, stable Poisson at 2 req/s
  bursty_trace.csv               # 500 requests, alternating 1/6 req/s in 30s windows

scripts/
  generate_trace.py              # Regenerate traces (ShareGPT-like length distribution)

tests/
  test_chunk_size_controller_logic.py        # AIMD increase/decrease unit tests
  test_chunk_size_controller_wiring.py       # Scheduler chunk_size follows controller
  test_chunk_size_controller_integration.py  # Full metrics store -> controller -> scheduler path
```

---

## Metrics to Compare

All three conditions should be evaluated on:

- **TPOT P50 / P90 / P99** — primary metric, what sarathi-serve optimizes for
- **TTFT P50 / P90 / P99** — ensure prefill is not being starved
- **Request throughput / token throughput** — confirm no throughput regression
- **Chunk size trajectory over time** — visualizes controller behavior
- **KV cache utilization over time** — diagnostic signal

Test under both traces. The static baseline will handle steady load reasonably but show its weakness under bursty load — that is where the controller's advantage should be most visible.

---

## Running Experiments (on GPU machine)

### 1. Clone and set up on the GPU machine

```bash
git clone <repo-url> sarathi-serve
cd sarathi-serve
pip install -e .
```

### 2. Sync traces (if not committed)

From your local machine:
```bash
rsync -avz ./traces/ user@gpu-machine:~/sarathi-serve/traces/
```

### 3. Run all 6 experiments

```bash
bash experiments/run_experiments.sh
```

This runs all conditions sequentially and writes results to `experiments/results/<condition>/`.

### 4. Copy results back to local

```bash
rsync -avz user@gpu-machine:~/sarathi-serve/experiments/results/ ./experiments/results/
```

### 5. Run unit tests

```bash
python -m pytest tests/test_chunk_size_controller_logic.py \
                 tests/test_chunk_size_controller_wiring.py \
                 tests/test_chunk_size_controller_integration.py -v
```

---

## Switching Controllers

Set `controller_type` in any experiment YAML:

```yaml
scheduler_config:
  type: "SARATHI"
  chunk_size: 512          # initial value / static value for NONE
  controller_type: "NONE"  # or "AIMD" or "PID"
```

Or pass directly via CLI:
```bash
python -m sarathi.benchmark.main --scheduler_config.controller_type AIMD
```

---

## Tuning Parameters

**AIMD** (`AIMDConfig` in `chunk_size_controller.py`):
- `additive_increase = 16` — tokens added per update when healthy
- `multiplicative_decrease_factor = 0.75` — fraction kept on congestion
- `scheduling_delay_p95_threshold_s = 0.05` — congestion trigger (seconds)
- `update_every_iters = 5` — how often controller fires
- `window_size = 20` — history window for trend detection

**PID** (`PIDConfig` in `chunk_size_controller.py`):
- `target_decode_token_time_s = 0.02` — target TPOT setpoint (seconds)
- `kp = 512.0`, `ki = 64.0`, `kd = 128.0` — gain terms
- `integral_clamp = 2048.0` — anti-windup bound
- `ewma_alpha = 0.2` — smoothing factor for process variable
