"""
Generate synthetic but realistic request traces for benchmarking.

Produces two CSVs:
  - traces/steady_trace.csv  : stable Poisson arrivals at moderate rate
  - traces/bursty_trace.csv  : alternating low/high rate bursts

Format matches TraceRequestLengthGenerator:
  columns: num_prefill_tokens, num_decode_tokens

Length distribution approximates ShareGPT / LMSYS real-world data:
  - Prompt lengths: log-normal, median ~256 tokens, heavy tail to ~2048
  - Decode lengths: log-normal, median ~128 tokens, heavy tail to ~512
"""

import os
import numpy as np
import pandas as pd

SEED = 42
MAX_TOKENS = 2048
N_REQUESTS = 500
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "traces")


def sample_lengths(n: int, rng: np.random.Generator) -> pd.DataFrame:
    prefill = rng.lognormal(mean=5.5, sigma=1.0, size=n).astype(int).clip(16, MAX_TOKENS)
    decode  = rng.lognormal(mean=4.8, sigma=0.9, size=n).astype(int).clip(16, MAX_TOKENS)

    # enforce total <= MAX_TOKENS
    total = prefill + decode
    over = total > MAX_TOKENS
    ratio = prefill[over] / total[over]
    prefill[over] = (MAX_TOKENS * ratio).astype(int).clip(1)
    decode[over]  = (MAX_TOKENS - prefill[over]).clip(1)

    return pd.DataFrame({"num_prefill_tokens": prefill, "num_decode_tokens": decode})


def make_steady(rng: np.random.Generator) -> pd.DataFrame:
    """Stable Poisson arrivals at 2 req/s for the full trace."""
    df = sample_lengths(N_REQUESTS, rng)
    inter_arrival = rng.exponential(scale=1.0 / 2.0, size=N_REQUESTS)
    df["arrival_time"] = np.cumsum(inter_arrival)
    return df


def make_bursty(rng: np.random.Generator) -> pd.DataFrame:
    """
    Alternates between low load (1 req/s) and high load (6 req/s)
    in 30-second windows. This is where a static chunk size will
    struggle and the controller should show its advantage.
    """
    df = sample_lengths(N_REQUESTS, rng)
    arrival_times = []
    t = 0.0
    window = 30.0

    for _ in range(N_REQUESTS):
        phase = int(t / window) % 2
        rate = 6.0 if phase == 1 else 1.0
        t += rng.exponential(scale=1.0 / rate)
        arrival_times.append(t)

    df["arrival_time"] = arrival_times
    return df


def print_stats(name: str, df: pd.DataFrame) -> None:
    print(f"\n--- {name} ---")
    print(f"  prefill  median={df['num_prefill_tokens'].median():.0f}"
          f"  p95={df['num_prefill_tokens'].quantile(0.95):.0f}"
          f"  max={df['num_prefill_tokens'].max()}")
    print(f"  decode   median={df['num_decode_tokens'].median():.0f}"
          f"  p95={df['num_decode_tokens'].quantile(0.95):.0f}"
          f"  max={df['num_decode_tokens'].max()}")
    duration = df["arrival_time"].max()
    print(f"  duration={duration:.1f}s  avg_rate={len(df)/duration:.2f} req/s")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    steady = make_steady(rng)
    bursty = make_bursty(rng)

    steady_path = os.path.join(OUT_DIR, "steady_trace.csv")
    bursty_path = os.path.join(OUT_DIR, "bursty_trace.csv")

    steady.to_csv(steady_path, index=False)
    bursty.to_csv(bursty_path, index=False)

    print(f"Wrote {len(steady)} requests -> {steady_path}")
    print(f"Wrote {len(bursty)} requests -> {bursty_path}")

    print_stats("steady", steady)
    print_stats("bursty", bursty)


if __name__ == "__main__":
    main()
