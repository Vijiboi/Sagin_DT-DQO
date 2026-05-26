"""
Generate 4 publication-quality figures from SAGIN-DT-DQO simulation output.

Usage:
    python generate_figures.py results/run_YYYYMMDD_HHMMSS
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── IEEE-style formatting ─────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.3,
    "lines.markersize": 4,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})

FIG_SIZE = (7.16, 3.0)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── Figure 1: Average Delay Cost per Slot (Line) ─────────────────────────────
def figure1_delay(slot_metrics: list[dict], out: Path) -> None:
    slots = [int(r["slot"]) for r in slot_metrics]
    delay = [float(r["average_delay"]) for r in slot_metrics]

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.plot(slots, delay, "-o", color="#2166ac", markersize=3)
    ax.set_xlabel("Time Slot $t$")
    ax.set_ylabel("Average Delay Cost")
    ax.set_xlim(slots[0] - 0.5, slots[-1] + 0.5)

    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 2: Multi-Tier Task Distribution per Slot (Stacked Bar) ────────────
def figure2_tier_distribution(assignments: list[dict], out: Path) -> None:
    tier_counts: dict[int, dict[str, int]] = defaultdict(lambda: {"BS": 0, "HAP": 0, "LEO": 0})
    for row in assignments:
        slot = int(row["slot"])
        tier = row["destination_id"].split("_")[0]
        tier_counts[slot][tier] += 1

    slots = sorted(tier_counts.keys())
    bs = [tier_counts[s]["BS"] for s in slots]
    hap = [tier_counts[s]["HAP"] for s in slots]
    leo = [tier_counts[s]["LEO"] for s in slots]
    x = np.arange(len(slots))

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.bar(x, bs, 0.7, label="BS", color="#2166ac")
    ax.bar(x, hap, 0.7, bottom=bs, label="HAP", color="#b2182b")
    bottom_leo = [bs[i] + hap[i] for i in range(len(slots))]
    ax.bar(x, leo, 0.7, bottom=bottom_leo, label="LEO", color="#1b7837")

    ax.set_xlabel("Time Slot $t$")
    ax.set_ylabel("Number of Tasks")
    ax.set_xticks(x[::5])
    ax.set_xticklabels([str(s) for s in slots[::5]])
    ax.set_xlim(-0.6, len(slots) - 0.4)
    ax.legend(loc="upper right", ncol=3, framealpha=0.9)

    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 3: Average Twin Fidelity per Slot (Line) ──────────────────────────
def figure3_fidelity(twin_metrics: list[dict], out: Path) -> None:
    fidelity_by_slot: dict[int, list[float]] = defaultdict(list)
    for row in twin_metrics:
        fidelity_by_slot[int(row["slot"])].append(float(row["fidelity"]))

    slots = sorted(fidelity_by_slot.keys())
    avg_fid = [np.mean(fidelity_by_slot[s]) for s in slots]

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.plot(slots, avg_fid, "-s", color="#b2182b", markersize=3)
    ax.set_xlabel("Time Slot $t$")
    ax.set_ylabel("Average Twin Fidelity $F_m(t)$")
    ax.set_xlim(slots[0] - 0.5, slots[-1] + 0.5)
    ax.set_ylim(0, 1.05)

    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 4: Average Delay Cost by Tier per Slot (Bar) ──────────────────────
def figure4_tier_delay(assignments: list[dict], out: Path) -> None:
    delay_by_tier: dict[str, list[float]] = {"BS": [], "HAP": [], "LEO": []}
    for row in assignments:
        tier = row["destination_id"].split("_")[0]
        if tier in delay_by_tier:
            delay_by_tier[tier].append(float(row["delay_cost"]))

    tiers = ["BS", "HAP", "LEO"]
    means = [np.mean(delay_by_tier[t]) if delay_by_tier[t] else 0 for t in tiers]
    colors = ["#2166ac", "#b2182b", "#1b7837"]

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.bar(tiers, means, color=colors, width=0.5)
    ax.set_xlabel("Infrastructure Tier")
    ax.set_ylabel("Average Delay Cost")

    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python generate_figures.py <run_directory>")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    if not run_dir.is_dir():
        print(f"Error: {run_dir} is not a directory")
        sys.exit(1)

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    print(f"Loading data from {run_dir} ...")
    slot_metrics = _load_csv(run_dir / "slot_metrics.csv")
    assignments = _load_csv(run_dir / "assignments.csv")
    twin_metrics = _load_csv(run_dir / "twin_metrics.csv")

    print("Generating figures ...")
    figure1_delay(slot_metrics, fig_dir / "fig1_avg_delay.png")
    figure2_tier_distribution(assignments, fig_dir / "fig2_tier_distribution.png")
    figure3_fidelity(twin_metrics, fig_dir / "fig3_avg_fidelity.png")
    figure4_tier_delay(assignments, fig_dir / "fig4_tier_delay.png")

    print(f"\nAll figures saved in: {fig_dir}")


if __name__ == "__main__":
    main()
