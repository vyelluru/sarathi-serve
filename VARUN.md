# Extending Sarathi-Serve for runtime modification of chunk size

Sarathi-Serve's chunk size is set as a fixed configuration parameter at launch time. It's a hyperparameter you tune before serving, not something the system adjusts at runtime. So it might not be the optimal chunk size as your workload evolves and progresses during runtime.

This tool is a runtime controller over the chunk size based on metric indicators of congestion or availability. 


Inspired by TCP ECN (Explicit Congestion Notification) and AIMD — routers mark packets when buffers are filling up, before they actually drop anything.
indicators
TPOT trending upward over last N iterations, not yet violated
Decode batch size growing faster than it's draining
KV cache utilization crossing some threshold

Cases:
TPOT healthy → additively increase chunk size (e.g. +128 tokens per N iterations)
TPOT approaching threshold → multiplicatively decrease chunk size (e.g. halve it)


Basis from Sarathi-Serve paper:
"The system performance can be further enhanced by dynamically varying the token budget based on workload characteristics. We leave this exploration for future work."

1. Find where current static chunk size is configured and consumed. Add hooks to get key metrics.
2. Create a small controller component that periodically reads those metrics and maintains a mutable chunk size. This should encode the AIMD style rules
3. Connect current static chunk size calls with the controller's API
    You don’t have to create a separate thread; you can:
    Add them inside AsyncLLMEngine.engine_step / run_engine_loop (async, but you’re just updating Python state each loop).

chunk_size_controller.py is the main file w our logic.