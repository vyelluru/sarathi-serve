"""
Offline controller simulation — no GPU required.

Simulates Static, AIMD, and PID chunk-size controllers under steady and
bursty synthetic latency patterns, producing CSV data and comparison plots.

Usage:
    python scripts/simulate_controllers.py

Output:
    scripts/simulation_results/simulation_data.csv
    scripts/simulation_results/chunk_size_trajectory.png
    scripts/simulation_results/reaction_time.png
    scripts/simulation_results/stability_boxplot.png
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sarathi.controller.chunk_size_controller import (
    AIMDChunkSizeController,
    AIMDConfig,
    BaseChunkSizeController,
    MetricsSnapshot,
    PIDChunkSizeController,
    PIDConfig,
)

OUT_DIR = os.path.join(ROOT, "scripts", "simulation_results")
N_ITERS = 2000
INITIAL_CHUNK_SIZE = 512

# ---------------------------------------------------------------------------
# Synthetic latency generators
# ---------------------------------------------------------------------------

def steady_latency(n: int, rng: np.random.Generator) -> np.ndarray:
    """Stable low latency with mild noise — models 2 req/s steady load."""
    base = 0.015  # 15ms base scheduling delay
    return base + rng.normal(0, 0.003, size=n).clip(-0.01, 0.01)


def bursty_latency(n: int, rng: np.random.Generator) -> np.ndarray:
    """Alternating low/high latency in 50-iteration windows — models bursty load."""
    latencies = np.zeros(n)
    for i in range(n):
        phase = (i // 50) % 2
        if phase == 0:
            latencies[i] = 0.015 + rng.normal(0, 0.003)  # low load
        else:
            latencies[i] = 0.08 + rng.normal(0, 0.015)   # high load (above threshold)
    return latencies.clip(0.001, 0.5)


def decode_token_time_from_scheduling_delay(
    scheduling_delays: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Derive decode token times correlated with scheduling delays."""
    base = 0.012
    factor = 0.15  # higher scheduling delay → higher decode time
    noise = rng.normal(0, 0.002, size=len(scheduling_delays))
    return (base + factor * scheduling_delays + noise).clip(0.005, 0.2)


# ---------------------------------------------------------------------------
# Controller wrappers that accept injected snapshots
# ---------------------------------------------------------------------------

class InjectableAIMD(AIMDChunkSizeController):
    def __init__(self, **kw):
        super().__init__(**kw)
        self._snap = None

    def inject(self, snap: MetricsSnapshot):
        self._snap = snap

    def _collect_snapshot(self) -> MetricsSnapshot:
        return self._snap


class InjectablePID(PIDChunkSizeController):
    def __init__(self, **kw):
        super().__init__(**kw)
        self._snap = None

    def inject(self, snap: MetricsSnapshot):
        self._snap = snap

    def _collect_snapshot(self) -> MetricsSnapshot:
        return self._snap


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_simulation(
    controller: BaseChunkSizeController | None,
    scheduling_delays: np.ndarray,
    decode_times: np.ndarray,
    controller_name: str,
) -> list[dict]:
    rows = []
    ewma = None
    alpha = 0.2
    for i in range(len(scheduling_delays)):
        sd = float(scheduling_delays[i])
        dt = float(decode_times[i])

        # Track EWMA for PID
        if ewma is None:
            ewma = dt
        else:
            ewma = alpha * dt + (1 - alpha) * ewma

        if controller is not None:
            snap = MetricsSnapshot(
                timestamp_s=time.time(),
                scheduling_delay_p95_s=sd,
                decode_token_time_p95_s=dt,
                batch_exec_time_p95_s=None,
                inter_batch_delay_p95_s=None,
                decode_token_time_ewma_s=ewma,
            )
            controller.inject(snap)
            controller.update()
            chunk_size = controller.chunk_size
        else:
            chunk_size = INITIAL_CHUNK_SIZE  # static

        rows.append({
            "iteration": i,
            "controller": controller_name,
            "chunk_size": chunk_size,
            "scheduling_delay_s": sd,
            "decode_token_time_s": dt,
        })
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(42)

    # Generate latency patterns
    steady_sd = steady_latency(N_ITERS, rng)
    bursty_sd = bursty_latency(N_ITERS, rng)
    steady_dt = decode_token_time_from_scheduling_delay(steady_sd, rng)
    bursty_dt = decode_token_time_from_scheduling_delay(bursty_sd, rng)

    all_rows = []

    for workload_name, sd, dt in [("steady", steady_sd, steady_dt), ("bursty", bursty_sd, bursty_dt)]:
        # Static (no controller)
        all_rows.extend([
            {**r, "workload": workload_name}
            for r in run_simulation(None, sd, dt, "Static")
        ])

        # AIMD
        aimd = InjectableAIMD(
            initial_chunk_size=INITIAL_CHUNK_SIZE,
            metrics_store=None,
            config=AIMDConfig(update_every_iters=1),
        )
        all_rows.extend([
            {**r, "workload": workload_name}
            for r in run_simulation(aimd, sd, dt, "AIMD")
        ])

        # PID
        pid = InjectablePID(
            initial_chunk_size=INITIAL_CHUNK_SIZE,
            metrics_store=None,
            config=PIDConfig(update_every_iters=1),
        )
        all_rows.extend([
            {**r, "workload": workload_name}
            for r in run_simulation(pid, sd, dt, "PID")
        ])

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(OUT_DIR, "simulation_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Wrote {len(df)} rows to {csv_path}")

    # -----------------------------------------------------------------------
    # Plot 1: Chunk size trajectory (2 panels: steady + bursty)
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    colors = {"Static": "#888888", "AIMD": "#e74c3c", "PID": "#2980b9"}

    for ax, workload in zip(axes, ["steady", "bursty"]):
        wdf = df[df["workload"] == workload]

        # Shade bursty high-load windows
        if workload == "bursty":
            for start in range(0, N_ITERS, 100):
                ax.axvspan(start + 50, min(start + 100, N_ITERS),
                           alpha=0.08, color="red", label="_nolegend_")

        for ctrl_name in ["Static", "AIMD", "PID"]:
            cdf = wdf[wdf["controller"] == ctrl_name]
            ax.plot(cdf["iteration"], cdf["chunk_size"],
                    label=ctrl_name, color=colors[ctrl_name],
                    linewidth=1.5, alpha=0.9)

        ax.set_ylabel("Chunk Size (tokens)")
        ax.set_title(f"{workload.capitalize()} Workload")
        ax.legend(loc="upper right")
        ax.set_ylim(0, 4200)
        ax.axhline(y=128, color="gray", linestyle="--", alpha=0.3, label="_nolegend_")
        ax.axhline(y=4096, color="gray", linestyle="--", alpha=0.3, label="_nolegend_")

    axes[1].set_xlabel("Iteration")
    fig.suptitle("Chunk Size Trajectory: Static vs AIMD vs PID", fontsize=14)
    plt.tight_layout()
    path1 = os.path.join(OUT_DIR, "chunk_size_trajectory.png")
    fig.savefig(path1, dpi=150)
    print(f"Saved {path1}")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Plot 2: Reaction time — iterations to first decrease after burst onset
    # -----------------------------------------------------------------------
    burst_starts = list(range(50, N_ITERS, 100))  # iterations where bursts begin
    reaction_data = []
    bursty_df = df[df["workload"] == "bursty"]

    for ctrl_name in ["AIMD", "PID"]:
        cdf = bursty_df[bursty_df["controller"] == ctrl_name].reset_index(drop=True)
        for bs in burst_starts:
            if bs >= len(cdf):
                break
            pre_size = cdf.loc[cdf["iteration"] == bs, "chunk_size"].values[0]
            for offset in range(1, min(50, N_ITERS - bs)):
                curr = cdf.loc[cdf["iteration"] == bs + offset, "chunk_size"].values
                if len(curr) > 0 and curr[0] < pre_size:
                    reaction_data.append({
                        "controller": ctrl_name,
                        "burst_start": bs,
                        "reaction_iters": offset,
                    })
                    break

    if reaction_data:
        rdf = pd.DataFrame(reaction_data)
        fig, ax = plt.subplots(figsize=(8, 5))
        for ctrl_name in ["AIMD", "PID"]:
            subset = rdf[rdf["controller"] == ctrl_name]
            if not subset.empty:
                ax.bar(
                    [f"Burst@{b}" for b in subset["burst_start"]],
                    subset["reaction_iters"],
                    label=ctrl_name, alpha=0.7,
                    color=colors[ctrl_name],
                    width=0.35,
                    align="edge" if ctrl_name == "AIMD" else "center",
                )
        ax.set_ylabel("Iterations to First Decrease")
        ax.set_xlabel("Burst Onset")
        ax.set_title("Reaction Time: AIMD vs PID")
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        path2 = os.path.join(OUT_DIR, "reaction_time.png")
        fig.savefig(path2, dpi=150)
        print(f"Saved {path2}")
        plt.close(fig)

    # -----------------------------------------------------------------------
    # Plot 3: Stability — chunk size variance during steady-state windows
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    stability_data = []

    for workload in ["steady", "bursty"]:
        wdf = df[df["workload"] == workload]
        for ctrl_name in ["Static", "AIMD", "PID"]:
            cdf = wdf[wdf["controller"] == ctrl_name]
            # Measure variance over rolling windows of 50 iterations
            chunk_sizes = cdf["chunk_size"].values
            for start in range(0, len(chunk_sizes) - 50, 50):
                window = chunk_sizes[start:start + 50]
                stability_data.append({
                    "controller": ctrl_name,
                    "workload": workload,
                    "std": float(np.std(window)),
                })

    sdf = pd.DataFrame(stability_data)
    positions = {"Static": 0, "AIMD": 1, "PID": 2}
    for workload in ["steady", "bursty"]:
        wsdf = sdf[sdf["workload"] == workload]
        offset = -0.15 if workload == "steady" else 0.15
        for ctrl_name in ["Static", "AIMD", "PID"]:
            subset = wsdf[wsdf["controller"] == ctrl_name]["std"]
            bp = ax.boxplot(
                [subset.values],
                positions=[positions[ctrl_name] + offset],
                widths=0.25,
                patch_artist=True,
            )
            color = colors[ctrl_name]
            alpha = 0.6 if workload == "steady" else 0.9
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(alpha)

    ax.set_xticks(list(positions.values()))
    ax.set_xticklabels(list(positions.keys()))
    ax.set_ylabel("Chunk Size Std Dev (per 50-iter window)")
    ax.set_title("Stability: Chunk Size Variance by Controller")
    # Custom legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="gray", alpha=0.5, label="Steady"),
        Patch(facecolor="gray", alpha=0.9, label="Bursty"),
    ])
    plt.tight_layout()
    path3 = os.path.join(OUT_DIR, "stability_boxplot.png")
    fig.savefig(path3, dpi=150)
    print(f"Saved {path3}")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------------
    print("\n=== Summary Statistics ===")
    for workload in ["steady", "bursty"]:
        print(f"\n--- {workload.upper()} ---")
        wdf = df[df["workload"] == workload]
        for ctrl_name in ["Static", "AIMD", "PID"]:
            cdf = wdf[wdf["controller"] == ctrl_name]
            cs = cdf["chunk_size"]
            print(f"  {ctrl_name:8s}: mean={cs.mean():7.1f}  std={cs.std():7.1f}  "
                  f"min={cs.min():5d}  max={cs.max():5d}")

    print(f"\nAll outputs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
