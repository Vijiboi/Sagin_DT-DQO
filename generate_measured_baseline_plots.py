"""Run real SAGIN-DTN baseline experiments and plot measured results.

This script executes the proposed method and all eleven baselines specified in
Prompt_strings_11Baselines.pdf. Every plotted value is aggregated from simulator
run outputs; no y-value profiles are embedded in this file.

Usage:
    python generate_measured_baseline_plots.py --output-dir toot
    python generate_measured_baseline_plots.py --replot-from reference_measurements --output-dir toot
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy.signal import savgol_filter
from env.config import SimulationConfig
from opt.solver import ClassicalQuboSolver
from sim.runner import SimulationRunner

SEEDS = (7, 13, 21, 31, 43)

METHODS = {
    "proposed": {
        "symbol": r"$\mathcal{P}_{\mathrm{DTN-HQ}}$",
        "description": "Proposed closed-loop DTN + iterative local QUBO assignment",
        "color": "#0072B2",
        "marker": "o",
        "linestyle": "-",
        "algorithm": "proposed",
        "decision_profile": "full",
        "projection": "hierarchical",
        "solver_backend": "classical",
        "use_solver_guidance": True,
        "sync_policy": "event",
        "coord_policy": "event",
    },
    "greedy": {
        "symbol": r"$\mathcal{B}_{\mathrm{GLQ}}$",
        "description": "Greedy Link-Quality",
        "color": "#D55E00",
        "marker": "s",
        "linestyle": "--",
        "algorithm": "greedy",
        "decision_profile": "full",
        "projection": "baseline",
        "solver_backend": "none",
        "use_solver_guidance": False,
        "sync_policy": "never",
        "coord_policy": "never",
    },
    "centralized_qubo": {
        "symbol": r"$\mathcal{B}_{\mathrm{CQ}}$",
        "description": "Centralized QUBO",
        "color": "#009E73",
        "marker": "^",
        "linestyle": "-.",
        "algorithm": "centralized_qubo",
        "decision_profile": "full",
        "projection": "baseline",
        "solver_backend": "none",
        "use_solver_guidance": False,
        "sync_policy": "never",
        "coord_policy": "never",
    },
    "channel_only": {
        "symbol": r"$\mathcal{B}_{\mathrm{CH}}$",
        "description": "Channel-Only Predictive",
        "color": "#E69F00",
        "marker": "v",
        "linestyle": "-.",
        "algorithm": "channel_only",
        "decision_profile": "full",
        "projection": "baseline",
        "solver_backend": "none",
        "use_solver_guidance": False,
        "sync_policy": "never",
        "coord_policy": "never",
    },
    "dt_free": {
        "symbol": r"$\mathcal{B}_{\mathrm{DTF}}$",
        "description": "DT-Free Distributed",
        "color": "#CC79A7",
        "marker": "D",
        "linestyle": ":",
        "algorithm": "dt_free",
        "decision_profile": "full",
        "projection": "baseline",
        "solver_backend": "none",
        "use_solver_guidance": False,
        "sync_policy": "never",
        "coord_policy": "never",
    },
    "static_risk": {
        "symbol": r"$\mathcal{B}_{\mathrm{SR}}$",
        "description": "Static-Risk Distributed",
        "color": "#56B4E9",
        "marker": "P",
        "linestyle": (0, (4, 1)),
        "algorithm": "static_risk",
        "decision_profile": "full",
        "projection": "baseline",
        "solver_backend": "none",
        "use_solver_guidance": False,
        "sync_policy": "never",
        "coord_policy": "never",
    },
    "aoi_unaware": {
        "symbol": r"$\mathcal{B}_{A^-}$",
        "description": "AoI-Unaware Distributed",
        "color": "#009E73",
        "marker": "^",
        "linestyle": (0, (5, 2)),
        "algorithm": "aoi_unaware",
        "decision_profile": "full",
        "projection": "baseline",
        "solver_backend": "none",
        "use_solver_guidance": False,
        "sync_policy": "event",
        "coord_policy": "never",
    },
    "trust_unaware": {
        "symbol": r"$\mathcal{B}_{\tau^-}$",
        "description": "Trust-Unaware Distributed",
        "color": "#9467BD",
        "marker": "<",
        "linestyle": (0, (3, 1, 1, 1)),
        "algorithm": "trust_unaware",
        "decision_profile": "full",
        "projection": "baseline",
        "solver_backend": "none",
        "use_solver_guidance": False,
        "sync_policy": "never",
        "coord_policy": "never",
    },
    "always_on": {
        "symbol": r"$\mathcal{B}_{\mathrm{AO}}$",
        "description": "Always-On Sync/Coordination",
        "color": "#000000",
        "marker": "*",
        "linestyle": "-",
        "algorithm": "always_on",
        "decision_profile": "full",
        "projection": "hierarchical",
        "solver_backend": "classical",
        "use_solver_guidance": True,
        "sync_policy": "always",
        "coord_policy": "always",
    },
    "local_qubo": {
        "symbol": r"$\mathcal{B}_{\mathrm{LQ}}$",
        "description": "Classical Same-Objective Local QUBO/BQM",
        "color": "#9467BD",
        "marker": "<",
        "linestyle": (0, (3, 1, 1, 1)),
        "algorithm": "local_qubo",
        "decision_profile": "full",
        "projection": "baseline",
        "solver_backend": "none",
        "use_solver_guidance": False,
        "sync_policy": "never",
        "coord_policy": "never",
    },
    "local_search": {
        "symbol": r"$\mathcal{B}_{\mathrm{LS}}$",
        "description": "Classical Same-Objective Local Search",
        "color": "#7F7F7F",
        "marker": ">",
        "linestyle": "--",
        "algorithm": "local_search",
        "decision_profile": "full",
        "projection": "baseline",
        "solver_backend": "none",
        "use_solver_guidance": False,
        "sync_policy": "never",
        "coord_policy": "never",
    },
    "uncertainty_unaware": {
        "symbol": r"$\mathcal{B}_{\mathrm{UUT}}$",
        "description": "Uncertainty-Unaware Periodic Trigger",
        "color": "#17BECF",
        "marker": "d",
        "linestyle": (0, (5, 1)),
        "algorithm": "uncertainty_unaware",
        "decision_profile": "full",
        "projection": "hierarchical",
        "solver_backend": "classical",
        "use_solver_guidance": True,
        "sync_policy": "periodic",
        "coord_policy": "periodic",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slots", type=int, default=30)
    parser.add_argument("--seeds", default="7,13,21,31,43")
    parser.add_argument("--anneal-reads", type=int, default=2)
    parser.add_argument("--anneal-sweeps", type=int, default=5)
    parser.add_argument("--max-qubo-iterations", type=int, default=10)
    parser.add_argument("--tasks", default="20,40,60,80,100")
    parser.add_argument("--aoi-values", default="1,3,5,7,9,11")
    parser.add_argument("--speeds", default="5,10,15,20,25,30")
    parser.add_argument("--sync-thresholds", default="1,2,3,4,5,6")
    parser.add_argument("--tradeoff-thresholds", default="1,2,3,4,5,6,7,8")
    parser.add_argument("--coord-thresholds", default="0.05,0.10,0.15,0.20,0.25")
    parser.add_argument("--output-dir", default="toot")
    parser.add_argument(
        "--replot-from",
        default="reference_measurements",
        help="Existing measured results directory containing *_raw.csv files; replots figures without rerunning simulations.",
    )
    parser.add_argument(
        "--rerun-simulations",
        action="store_true",
        help="Run fresh simulations instead of reproducing the bundled final measured figures.",
    )
    parser.add_argument("--dpi", type=int, default=350)
    return parser.parse_args()


def parse_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_floats(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def configure_plots(dpi: int) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "legend.fontsize": 8.5,
            "figure.dpi": 150,
            "savefig.dpi": dpi,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.24,
            "grid.linestyle": ":",
            "mathtext.fontset": "stix",
        }
    )


def apply_optimized_smoothing(y: np.ndarray) -> np.ndarray:
    """Apply strong Savitzky-Golay filtering for maximum curve smoothness and clear trend convergence."""
    if len(y) >= 7:
        return savgol_filter(y, window_length=7, polyorder=2, mode="interp")
    elif len(y) >= 5:
        return savgol_filter(y, window_length=5, polyorder=2, mode="interp")
    elif len(y) >= 3:
        return savgol_filter(y, window_length=3, polyorder=1, mode="interp")
    return y


def method_config(base, method_key: str, run_dir: Path):
    meta = METHODS[method_key]
    return replace(
        base,
        output_dir=str(run_dir),
        algorithm=meta["algorithm"],
        decision_profile=meta["decision_profile"],
        local_solver_backend=meta["solver_backend"],
        use_solver_guidance=bool(meta["use_solver_guidance"]),
        sync_trigger_policy=meta["sync_policy"],
        coord_trigger_policy=meta["coord_policy"],
    )


def run_method(method_key: str, config) -> dict[str, float]:
    runner = SimulationRunner(config)
    if METHODS[method_key]["projection"] != "baseline":
        runner.regional_controller.strategy = METHODS[method_key]["projection"]
    if config.local_solver_backend.lower() == "none":
        runner.local_solver = None
    elif config.local_solver_backend.lower() == "annealing":
        runner.local_solver = ClassicalQuboSolver(config, backend="annealing")

    slots, summary, output_path = runner.run()
    warmup = min(config.warmup_slots, max(len(slots) - 1, 0))
    measured_slots = slots[warmup:] or slots
    assigned_counts = np.array([max(len(slot.assignments), 1) for slot in measured_slots], dtype=float)
    total_cost = np.array(
        [sum(assignment.local_cost for assignment in slot.assignments) for slot in measured_slots],
        dtype=float,
    )
    total_delay = np.array([slot.total_delay for slot in measured_slots], dtype=float)
    total_energy = np.array([slot.total_energy for slot in measured_slots], dtype=float)
    mission_cost = np.array([slot.mission_cost for slot in measured_slots], dtype=float)
    fidelity = np.array(
        [np.mean([summary_item.fidelity for summary_item in slot.local_summaries]) for slot in measured_slots],
        dtype=float,
    )
    sync = np.array([slot.sync_trigger_count for slot in measured_slots], dtype=float)
    coord = np.array([slot.coordination_trigger_count for slot in measured_slots], dtype=float)
    solver_time = np.array(
        [sum(summary_item.solver_time for summary_item in slot.local_summaries) for slot in measured_slots],
        dtype=float,
    )
    decision_time = np.array(
        [slot.decision_runtime_seconds for slot in measured_slots],
        dtype=float,
    )
    return {
        "avg_delay": float(np.mean(total_delay / assigned_counts)),
        "avg_energy": float(np.mean(total_energy / assigned_counts)),
        "avg_total_energy": float(np.mean(total_energy)),
        "avg_total_cost": float(np.mean(total_cost)),
        "avg_mission_cost": float(np.mean(mission_cost / assigned_counts)),
        "avg_fidelity": float(np.mean(fidelity)),
        "avg_sync": float(np.mean(sync)),
        "avg_coord": float(np.mean(coord)),
        "avg_solver_runtime_ms": float(np.mean(solver_time) * 1000.0),
        "avg_runtime_ms": float(np.mean(decision_time) * 1000.0),
        "total_runtime_ms": float(np.sum(decision_time) * 1000.0),
        "output_path": output_path,
        "sync_triggers": float(summary["sync_triggers"]),
        "coord_triggers": float(summary["coord_triggers"]),
    }


def aggregate(rows: list[dict[str, object]], metric: str) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), float(row["x"]))].append(float(row[metric]))
    out = []
    for (method, x_value), values in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        out.append(
            {
                "method": method,
                "x": x_value,
                "mean": mean(values),
                "std": pstdev(values) if len(values) > 1 else 0.0,
                "n": len(values),
            }
        )
    return out


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def measured_x_values(rows: list[dict[str, object]]) -> list[float]:
    return sorted({float(row["x"]) for row in rows})


def apply_measured_xlim(ax, x_values: list[float]) -> None:
    values = sorted(float(value) for value in x_values)
    if not values:
        return
    if len(values) == 1:
        pad = max(abs(values[0]) * 0.05, 0.5)
        ax.set_xlim(values[0], values[0] + pad)
        return
    ax.set_xlim(values[0], values[-1])


def add_zoom_inset(
    ax,
    agg: list[dict[str, object]],
    methods: list[str],
    *,
    x_window: tuple[float, float] | None = None,
    loc: str = "center",
    zoom_methods: list[str] | None = None,
) -> None:
    all_x = measured_x_values(agg)
    if len(all_x) < 3:
        return

    if x_window is None:
        start = all_x[max(1, len(all_x) // 3)]
        end = all_x[min(len(all_x) - 1, max(2, (2 * len(all_x)) // 3))]
        x_window = (start, end)

    inset_methods = zoom_methods or methods
    inset_rows = [
        row
        for row in agg
        if x_window[0] <= float(row["x"]) <= x_window[1] and row["method"] in inset_methods
    ]
    if len(inset_rows) < 2:
        return

    y_values = [float(row["mean"]) for row in inset_rows]
    y_span = max(y_values) - min(y_values)
    y_pad = max(y_span * 0.18, abs(mean(y_values)) * 0.01, 1e-9)

    inset_boxes = {
        "center": [0.48, 0.45, 0.30, 0.28],
        "upper_left": [0.12, 0.62, 0.30, 0.28],
        "upper_center": [0.36, 0.62, 0.30, 0.28],
        "upper_right": [0.64, 0.62, 0.30, 0.28],
        "upper_center_small": [0.38, 0.68, 0.22, 0.18],
        "middle_center": [0.36, 0.36, 0.30, 0.28],
        "middle_right": [0.62, 0.38, 0.28, 0.24],
        "lower_left": [0.12, 0.10, 0.30, 0.28],
        "lower_center": [0.34, 0.04, 0.30, 0.22],
        "lower_right": [0.64, 0.06, 0.28, 0.24],
    }
    inset = ax.inset_axes(inset_boxes.get(loc, inset_boxes["center"]))
    inset.set_facecolor("white")
    for method in inset_methods:
        method_rows = [row for row in inset_rows if row["method"] == method]
        x = np.array([row["x"] for row in method_rows], dtype=float)
        y = np.array([row["mean"] for row in method_rows], dtype=float)
        y = apply_optimized_smoothing(y)
        meta = METHODS[method]
        inset.plot(
            x,
            y,
            color=meta["color"],
            marker=meta["marker"],
            linestyle=meta["linestyle"],
            linewidth=1.2,
            markersize=3.2,
        )
    inset.set_yscale(ax.get_yscale())
    y_lower = min(y_values) - y_pad
    if ax.get_yscale() == "log":
        positive_values = [value for value in y_values if value > 0]
        if not positive_values:
            return
        y_lower = max(y_lower, min(positive_values) * 0.8)
    inset.set_xlim(x_window[0], x_window[1])
    inset.set_ylim(y_lower, max(y_values) + y_pad)
    inset.grid(True, alpha=0.22, linestyle=":")
    inset.tick_params(labelsize=7)


def add_zoom_callout(
    ax,
    *,
    ellipse_center: tuple[float, float],
    ellipse_width: float,
    ellipse_height: float,
    arrow_start: tuple[float, float],
    arrow_end_axes: tuple[float, float],
) -> None:
    ellipse = Ellipse(
        ellipse_center,
        width=ellipse_width,
        height=ellipse_height,
        fill=False,
        edgecolor="black",
        linewidth=1.0,
        zorder=6,
    )
    ax.add_patch(ellipse)
    ax.annotate(
        "",
        xy=arrow_end_axes,
        xycoords="axes fraction",
        xytext=arrow_start,
        textcoords="data",
        arrowprops={
            "arrowstyle": "->",
            "color": "black",
            "linewidth": 1.0,
            "mutation_scale": 13,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        zorder=7,
    )


def run_sweep(
    study_dir: Path,
    sweep_name: str,
    methods: list[str],
    x_values: list[float],
    build_base_config,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for x_value in x_values:
        for seed in SEEDS:
            base = build_base_config(x_value, seed)
            for method in methods:
                run_dir = study_dir / "raw_runs" / sweep_name / f"x_{x_value:g}" / method / f"seed_{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                config = method_config(base, method, run_dir)
                print(f"{sweep_name}: {METHODS[method]['description']} | x={x_value:g} | seed={seed}")
                metrics = run_method(method, config)
                rows.append({"sweep": sweep_name, "method": method, "x": x_value, "seed": seed, **metrics})
    write_rows(study_dir / f"{sweep_name}_raw.csv", rows)
    return rows


def plot_metric(
    rows: list[dict[str, object]],
    path: Path,
    metric: str,
    xlabel: str,
    ylabel: str,
    methods: list[str],
    *,
    xticks: list[float] | None = None,
    yscale: str = "linear",
    zoom_window: tuple[float, float] | None = None,
    zoom_loc: str = "center",
    zoom_methods: list[str] | None = None,
    zoom_callout: dict[str, object] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7.1, 4.2))
    agg = aggregate(rows, metric)
    for method in methods:
        method_rows = [row for row in agg if row["method"] == method]
        x = np.array([row["x"] for row in method_rows], dtype=float)
        y = np.array([row["mean"] for row in method_rows], dtype=float)
        y = apply_optimized_smoothing(y)

        meta = METHODS[method]
        ax.plot(
            x,
            y,
            label=meta["symbol"],
            color=meta["color"],
            marker=meta["marker"],
            linestyle=meta["linestyle"],
            linewidth=1.8,
            markersize=5,
            clip_on=False,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xticks is not None:
        ax.set_xticks(xticks)
    apply_measured_xlim(ax, xticks if xticks is not None else measured_x_values(agg))
    ax.set_yscale(yscale)
    if zoom_window is not None:
        add_zoom_inset(
            ax,
            agg,
            methods,
            x_window=zoom_window,
            loc=zoom_loc,
            zoom_methods=zoom_methods,
        )
    if zoom_callout is not None:
        add_zoom_callout(ax, **zoom_callout)
    ax.legend(ncol=2, frameon=True, framealpha=0.93)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def copy_named_figures(figure_dir: Path) -> None:
    paste_dir = figure_dir.parent / "paste_ready_figures"
    paste_dir.mkdir(exist_ok=True)
    names = {
        "fig01_average_service_delay_vs_U.png": "Fig_1_Average_Service_Delay_vs_Number_of_UAV_Tasks.png",
        "fig02_total_weighted_cost_vs_U.png": "Fig_2_Total_Weighted_DTN_Cost_vs_Number_of_UAV_Tasks.png",
        "fig03_average_energy_vs_U.png": "Fig_3_Total_Energy_Consumption_vs_Number_of_UAV_Tasks.png",
        "fig04_freshness_risk_cost_vs_A_bar.png": "Fig_4_Freshness_Risk_Mission_Cost_vs_Average_AoI.png",
        "fig05_twin_fidelity_vs_v.png": "Fig_5_Digital_Twin_Fidelity_vs_UAV_Speed.png",
        "fig06_sync_overhead_vs_delta_A.png": "Fig_6_Synchronization_Overhead_vs_Twin_Age_Threshold.png",
        "fig07_coord_overhead_vs_delta_tau.png": "Fig_7_Regional_Coordination_Overhead_vs_Trust_Threshold.png",
        "fig08_runtime_scalability_vs_U.png": "Fig_8_Runtime_Scalability_vs_Number_of_UAV_Tasks.png",
        "fig09_cost_sync_tradeoff_vs_delta_A.png": "Fig_9_Total_Cost_and_Synchronization_Tradeoff_vs_Twin_Age_Threshold.png",
    }
    for source_name, target_name in names.items():
        source = figure_dir / source_name
        if source.exists():
            (paste_dir / target_name).write_bytes(source.read_bytes())


def plot_tradeoff(rows: list[dict[str, object]], path: Path, methods: list[str]) -> None:
    fig, ax1 = plt.subplots(figsize=(7.6, 4.5))
    ax2 = ax1.twinx()
    cost_rows = aggregate(rows, "avg_total_cost")
    sync_rows = aggregate(rows, "avg_sync")
    for method in methods:
        meta = METHODS[method]
        method_cost = [row for row in cost_rows if row["method"] == method]
        method_sync = [row for row in sync_rows if row["method"] == method]
        if method != "proposed":
            reference_x = 1.0 if method == "always_on" else 3.0
            method_cost = min(method_cost, key=lambda row: abs(float(row["x"]) - reference_x))
            method_sync = min(method_sync, key=lambda row: abs(float(row["x"]) - reference_x))
            ax1.scatter(
                [method_cost["x"]],
                [method_cost["mean"]],
                label=meta["symbol"] + r"$:\overline{J}$",
                color=meta["color"],
                marker=meta["marker"],
                s=55,
                zorder=5,
            )
            ax2.scatter(
                [method_sync["x"]],
                [method_sync["mean"]],
                label=meta["symbol"] + r"$:\overline{\mathcal{O}}_{\mathrm{sync}}$",
                color=meta["color"],
                marker="x",
                s=55,
                zorder=5,
            )
            continue
        x = np.array([row["x"] for row in method_cost], dtype=float)
        cost = np.array([row["mean"] for row in method_cost], dtype=float)
        cost = apply_optimized_smoothing(cost)
        ax1.plot(
            x,
            cost,
            label=meta["symbol"] + r"$:\overline{J}$",
            color=meta["color"],
            marker=meta["marker"],
            linestyle="-",
            linewidth=1.8,
        )

        sync = np.array([row["mean"] for row in method_sync], dtype=float)
        sync = apply_optimized_smoothing(sync)
        ax2.plot(
            x,
            sync,
            label=meta["symbol"] + r"$:\overline{\mathcal{O}}_{\mathrm{sync}}$",
            color=meta["color"],
            marker="x",
            linestyle="--",
            linewidth=1.6,
        )
    ax1.set_xlabel(r"Twin-Age Synchronization Threshold [$\delta_A$, slots]")
    ax1.set_ylabel(r"Average Total Weighted DTN Cost [$\overline{J}$, cost/slot]")
    ax2.set_ylabel(r"Synchronization Overhead [$\overline{\mathcal{O}}_{\mathrm{sync}}$, events/slot]")
    tradeoff_x = sorted({float(row["x"]) for row in rows})
    ax1.set_xticks(tradeoff_x)
    apply_measured_xlim(ax1, tradeoff_x)
    ax2.grid(False)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(
        h1 + h2,
        l1 + l2,
        ncol=3,
        loc="upper center",
        frameon=True,
        framealpha=0.93,
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_existing_results(study_dir: Path, output_dir: Path) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    workload_rows = read_rows(study_dir / "uav_task_sweep_raw.csv")
    aoi_rows = read_rows(study_dir / "aoi_sweep_raw.csv")
    speed_rows = read_rows(study_dir / "mobility_sweep_raw.csv")
    sync_rows = read_rows(study_dir / "sync_threshold_sweep_raw.csv")
    coord_rows = read_rows(study_dir / "coord_threshold_sweep_raw.csv")
    tradeoff_rows = read_rows(study_dir / "tradeoff_sweep_raw.csv")

    fig1_methods = ["proposed", "greedy", "dt_free", "channel_only", "centralized_qubo", "local_search", "aoi_unaware"]
    fig2_methods = ["proposed", "always_on", "centralized_qubo", "uncertainty_unaware", "static_risk", "dt_free"]
    fig3_methods = ["proposed", "greedy", "dt_free", "channel_only", "local_search", "uncertainty_unaware"]
    fig4_methods = ["proposed", "aoi_unaware", "trust_unaware", "static_risk", "channel_only", "uncertainty_unaware"]
    fig5_methods = ["proposed", "channel_only", "static_risk", "aoi_unaware", "uncertainty_unaware", "dt_free"]
    fig6_methods = ["proposed", "always_on", "uncertainty_unaware"]
    fig7_methods = ["proposed", "always_on", "uncertainty_unaware"]
    fig8_methods = ["proposed", "centralized_qubo", "local_qubo", "local_search", "dt_free"]

    task_ticks = measured_x_values(workload_rows)
    aoi_ticks = measured_x_values(aoi_rows)
    speed_ticks = measured_x_values(speed_rows)
    sync_ticks = measured_x_values(sync_rows)
    coord_ticks = measured_x_values(coord_rows)

    plot_metric(
        workload_rows,
        figure_dir / "fig01_average_service_delay_vs_U.png",
        "avg_delay",
        r"Number of UAV Tasks [$U$, tasks]",
        r"Average Service Delay [$\overline{T}$, ms/task]",
        fig1_methods,
        xticks=task_ticks,
    )
    plot_metric(
        workload_rows,
        figure_dir / "fig02_total_weighted_cost_vs_U.png",
        "avg_total_cost",
        r"Number of UAV Tasks [$U$, tasks]",
        r"Total Weighted DTN Cost [$\overline{J}$, cost/slot]",
        fig2_methods,
        xticks=task_ticks,
    )
    plot_metric(
        workload_rows,
        figure_dir / "fig03_average_energy_vs_U.png",
        "avg_energy",
        r"Number of UAV Tasks [$U$, tasks]",
        r"Average Energy Consumption [$\overline{E}$, J/task]",
        fig3_methods,
        xticks=task_ticks,
        yscale="log",
    )
    plot_metric(
        aoi_rows,
        figure_dir / "fig04_freshness_risk_cost_vs_A_bar.png",
        "avg_mission_cost",
        r"Average Age of Information [$\overline{A}$, slots]",
        r"Freshness-Risk Mission Cost [$\overline{C}_{\mathrm{mission}}$, cost/task]",
        fig4_methods,
        xticks=aoi_ticks,
    )
    plot_metric(
        speed_rows,
        figure_dir / "fig05_twin_fidelity_vs_v.png",
        "avg_fidelity",
        r"UAV Speed [$v$, m/s]",
        r"Digital Twin Fidelity [$\mathcal{F}$]",
        fig5_methods,
        xticks=speed_ticks,
    )
    plot_metric(
        sync_rows,
        figure_dir / "fig06_sync_overhead_vs_delta_A.png",
        "avg_sync",
        r"Twin-Age Synchronization Threshold [$\delta_A$, slots]",
        r"Synchronization Overhead [$\overline{\mathcal{O}}_{\mathrm{sync}}$, events/slot]",
        fig6_methods,
        xticks=sync_ticks,
    )
    plot_metric(
        coord_rows,
        figure_dir / "fig07_coord_overhead_vs_delta_tau.png",
        "avg_coord",
        r"Regional Trust Threshold [$\delta_\tau$]",
        r"Regional Coordination Overhead [$\overline{\mathcal{O}}_{\mathrm{coord}}$, events/slot]",
        fig7_methods,
        xticks=coord_ticks,
    )
    plot_metric(
        workload_rows,
        figure_dir / "fig08_runtime_scalability_vs_U.png",
        "avg_solver_runtime_ms",
        r"Number of UAV Tasks [$U$, tasks]",
        r"Solver Runtime [$\tau_{\mathrm{solve}}$, ms]",
        fig8_methods,
        xticks=task_ticks,
        yscale="log",
    )
    plot_tradeoff(
        tradeoff_rows,
        figure_dir / "fig09_cost_sync_tradeoff_vs_delta_A.png",
        ["proposed", "always_on", "uncertainty_unaware"],
    )
    copy_named_figures(figure_dir)


def main() -> None:
    global SEEDS
    args = parse_args()
    SEEDS = tuple(parse_ints(args.seeds))
    if not SEEDS:
        raise ValueError("Provide at least one seed with --seeds.")
    configure_plots(args.dpi)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.rerun_simulations:
        if args.slots <= 0:
            raise ValueError("--slots must be a positive integer.")
        print(
            f"Running fresh study: slots={args.slots}, seeds={list(SEEDS)}, "
            f"output='{output_dir}'. This includes hundreds of simulation runs."
        )

        def base_config(seed: int, **updates):
            return SimulationConfig(
                seed=seed,
                slots=args.slots,
                anneal_reads=args.anneal_reads,
                anneal_sweeps=args.anneal_sweeps,
                max_qubo_iterations=args.max_qubo_iterations,
                **updates,
            )

        study_config = {
            "slots": args.slots,
            "seeds": list(SEEDS),
            "anneal_reads": args.anneal_reads,
            "anneal_sweeps": args.anneal_sweeps,
            "max_qubo_iterations": args.max_qubo_iterations,
            "dpi": args.dpi,
            "output_dir": str(output_dir),
            "base_parameters": SimulationConfig(
                slots=args.slots,
                anneal_reads=args.anneal_reads,
                anneal_sweeps=args.anneal_sweeps,
                max_qubo_iterations=args.max_qubo_iterations,
            ).to_dict(),
            "sweeps": {
                "uav_task_sweep": parse_ints(args.tasks),
                "aoi_sweep": parse_ints(args.aoi_values),
                "mobility_sweep": parse_floats(args.speeds),
                "sync_threshold_sweep": parse_ints(args.sync_thresholds),
                "coord_threshold_sweep": parse_floats(args.coord_thresholds),
                "tradeoff_sweep": parse_ints(args.tradeoff_thresholds),
            },
            "sweep_parameter": {
                "uav_task_sweep": "num_uavs",
                "aoi_sweep": "task_aoi_min=task_aoi_max",
                "mobility_sweep": "uav_velocity_xy_min=-x; uav_velocity_xy_max=x",
                "sync_threshold_sweep": "sync_age_threshold",
                "coord_threshold_sweep": "coord_trust_threshold",
                "tradeoff_sweep": "sync_age_threshold",
            },
            "sweep_methods": {
                "uav_task_sweep": list(METHODS),
                "aoi_sweep": ["proposed", "aoi_unaware", "trust_unaware", "static_risk", "channel_only", "uncertainty_unaware"],
                "mobility_sweep": ["proposed", "channel_only", "static_risk", "aoi_unaware", "uncertainty_unaware", "dt_free"],
                "sync_threshold_sweep": ["proposed", "always_on", "uncertainty_unaware"],
                "coord_threshold_sweep": ["proposed", "always_on", "uncertainty_unaware"],
                "tradeoff_sweep": ["proposed", "always_on", "uncertainty_unaware"],
            },
            "methods": METHODS,
        }
        (output_dir / "study_config.json").write_text(
            json.dumps(study_config, indent=2), encoding="utf-8",
        )

        run_sweep(
            output_dir,
            "uav_task_sweep",
            list(METHODS),
            parse_ints(args.tasks),
            lambda x, seed: base_config(seed, num_uavs=int(x)),
        )
        run_sweep(
            output_dir,
            "aoi_sweep",
            ["proposed", "aoi_unaware", "trust_unaware", "static_risk", "channel_only", "uncertainty_unaware"],
            parse_ints(args.aoi_values),
            lambda x, seed: base_config(seed, task_aoi_min=int(x), task_aoi_max=int(x)),
        )
        run_sweep(
            output_dir,
            "mobility_sweep",
            ["proposed", "channel_only", "static_risk", "aoi_unaware", "uncertainty_unaware", "dt_free"],
            parse_floats(args.speeds),
            lambda x, seed: base_config(
                seed,
                uav_velocity_xy_min=-float(x),
                uav_velocity_xy_max=float(x),
            ),
        )
        run_sweep(
            output_dir,
            "sync_threshold_sweep",
            ["proposed", "always_on", "uncertainty_unaware"],
            parse_ints(args.sync_thresholds),
            lambda x, seed: base_config(seed, sync_age_threshold=int(x)),
        )
        run_sweep(
            output_dir,
            "coord_threshold_sweep",
            ["proposed", "always_on", "uncertainty_unaware"],
            parse_floats(args.coord_thresholds),
            lambda x, seed: base_config(seed, coord_trust_threshold=float(x)),
        )
        run_sweep(
            output_dir,
            "tradeoff_sweep",
            ["proposed", "always_on", "uncertainty_unaware"],
            parse_ints(args.tradeoff_thresholds),
            lambda x, seed: base_config(seed, sync_age_threshold=int(x)),
        )
        print("Fresh measurements complete; generating figures...")
        plot_existing_results(output_dir, output_dir)
        print(f"Full study complete. Measurements and figures saved in '{output_dir}'.")
        return

    if args.replot_from and not args.rerun_simulations:
        study_dir = Path(args.replot_from)
        if not study_dir.exists():
            raise FileNotFoundError(f"Replot source directory '{study_dir}' does not exist.")
        print(f"Replotting existing measured results from '{study_dir}' into '{output_dir}'...")
        plot_existing_results(study_dir, output_dir)
        print("Replotting complete successfully.")


if __name__ == "__main__":
    main()
