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

