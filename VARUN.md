# Extending Sarathi-Serve for runtime modification of chunk size

Sarathi-Serve's chunk size is set as a fixed configuration parameter at launch time. It's a hyperparameter you tune before serving, not something the system adjusts at runtime. So it might not be the optimal chunk size as your workload evolves and progresses during runtime.

This tool is a runtime controller over the chunk size based on metric indicators of congestion or availability.

Basis from Sarathi-Serve paper:
"The system performance can be further enhanced by dynamically varying the token budget based on workload characteristics. We leave this exploration for future work."

---

## Phase 1: AIMD Controller

Inspired by TCP ECN (Explicit Congestion Notification) and AIMD — routers mark packets when buffers are filling up, before they actually drop anything.

Indicators:
- TPOT trending upward over last N iterations, not yet violated
- Decode batch size growing faster than it's draining
- KV cache utilization crossing some threshold

Rules:
- TPOT healthy → additively increase chunk size (e.g. +128 tokens per N iterations)
- TPOT approaching threshold → multiplicatively decrease chunk size (e.g. halve it)

Limitation: binary congestion signal causes oscillation — chunk size overshoots in both directions rather than settling smoothly.

## Phase 2: PID Controller (refinement)

Replaces the binary increase/decrease logic with a continuous signal. Uses TPOT deviation from a target setpoint as the error term.

- P term: responds proportionally to current TPOT deviation
- I term: corrects for sustained drift (e.g. workload shift that AIMD would oscillate around)
- D term: dampens overshoot by reacting to rate of change in TPOT

This is more aligned with how modern datacenter congestion control works in practice (DCTCP, DCQCN, Swift) — all of which use continuous signals and proportional responses rather than pure AIMD.

---

## Metrics to compare (vs. static baseline)

- TPOT P50 / P90 / P99
- TTFT P50 / P90 / P99
- Request throughput and token throughput
- KV cache utilization over time
- Decode batch size over time
- Chunk size trajectory (controller behavior visualization)

Test under: bursty/variable arrival rates and mixed request lengths — these are where static chunk sizing is most mismatched.

---

## Implementation

1. Find where current static chunk size is configured and consumed. Add hooks to get key metrics.
2. Create a small controller component that periodically reads those metrics and maintains a mutable chunk size. Implement AIMD rules first, then swap in PID.
3. Connect current static chunk size calls with the controller's API.
    You don't have to create a separate thread; you can:
    Add them inside AsyncLLMEngine.engine_step / run_engine_loop (async, but you're just updating Python state each loop).

chunk_size_controller.py is the main file w our logic.
