from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from env.config import SimulationConfig
from opt.solver import ClassicalQuboSolver
from sim.runner import SimulationRunner


METHODS = {
    "proposed": {
        "label": "Proposed (Solver-Guided Hierarchical)",
        "backend": "auto",
        "use_solver_guidance": True,
        "projection": "hierarchical",
        "color": "#1b9e77",
        "linestyle": "-",
        "marker": "o",
    },
    "baseline": {
        "label": "Baseline (Greedy Multi-Candidate Offloading)",
        "backend": None,
        "use_solver_guidance": False,
        "projection": "greedy",
        "color": "#d95f02",
        "linestyle": "--",
        "marker": "s",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run comparison studies and generate performance plots.")
    parser.add_argument("--slots", type=int, default=30, help="Number of simulation slots per run.")
    parser.add_argument("--densities", type=str, default="10,20,30,40,50", help="Comma-separated UAV counts.")
    parser.add_argument("--seeds", type=str, default="7,13", help="Comma-separated random seeds.")
    parser.add_argument("--focus-uavs", type=int, default=20, help="UAV count used for convergence-over-time plots.")
    parser.add_argument("--output-dir", type=str, default="comparison_results", help="Root directory for outputs.")
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Centered moving-average window for convergence figures only. Raw CSV metrics are unchanged.",
    )
    parser.add_argument(
        "--show-raw-convergence",
        action="store_true",
        help="Overlay faint unsmoothed convergence traces behind the smoothed mean curves.",
    )
    return parser.parse_args()


def parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def build_config(slots: int, seed: int, num_uavs: int, output_dir: str) -> SimulationConfig:
    return SimulationConfig(
        slots=slots,
        seed=seed,
        num_uavs=num_uavs,
        output_dir=output_dir,
    )


def run_case(method_key: str, config: SimulationConfig) -> dict[str, object]:
    runner = SimulationRunner(config)
    method = METHODS[method_key]
    runner.use_solver_guidance = method["use_solver_guidance"]
    runner.regional_controller.strategy = method["projection"]
    if method["backend"] is None:
        runner.local_solver = None
    else:
        runner.local_solver = ClassicalQuboSolver(config, backend=method["backend"])
    slot_results, summary, _ = runner.run()

    slot_delay = np.array([result.average_delay for result in slot_results], dtype=float)
    slot_fidelity = np.array(
        [np.mean([summary_item.fidelity for summary_item in result.local_summaries]) for result in slot_results],
        dtype=float,
    )
    slot_solver_time = np.array(
        [sum(summary_item.solver_time for summary_item in result.local_summaries) for result in slot_results],
        dtype=float,
    )
    slot_throughput = np.array([len(result.assignments) for result in slot_results], dtype=float)
    slot_sync = np.array([result.sync_trigger_count for result in slot_results], dtype=float)

    return {
        "slot_delay": slot_delay,
        "cumulative_delay": np.cumsum(slot_delay) / np.arange(1, len(slot_delay) + 1),
        "slot_fidelity": slot_fidelity,
        "slot_solver_time": slot_solver_time,
        "slot_throughput": slot_throughput,
        "slot_sync": slot_sync,
        "avg_delay": float(np.mean(slot_delay)),
        "avg_fidelity": float(np.mean(slot_fidelity)),
        "avg_solver_time": float(np.mean(slot_solver_time)),
        "total_solver_time": float(np.sum(slot_solver_time)),
        "avg_throughput": float(np.mean(slot_throughput)),
        "avg_sync": float(np.mean(slot_sync)),
        "summary": summary,
    }


def aggregate_series(records: list[dict[str, object]], key: str) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.vstack([record[key] for record in records])
    return np.mean(stacked, axis=0), np.std(stacked, axis=0)


def smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    """Plot-only centered moving average with edge padding."""
    if window <= 1 or len(values) <= 2:
        return values
    window = min(window, len(values))
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_convergence_csv(path: Path, methods_by_seed: dict[str, list[dict[str, object]]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "seed_index",
                "slot",
                "average_delay",
                "cumulative_average_delay",
                "average_fidelity",
                "solver_time",
                "throughput",
                "sync_triggers",
            ]
        )
        for method_key, records in methods_by_seed.items():
            for seed_index, record in enumerate(records):
                slots = len(record["slot_delay"])
                for slot in range(slots):
                    writer.writerow(
                        [
                            method_key,
                            seed_index,
                            slot + 1,
                            round(float(record["slot_delay"][slot]), 6),
                            round(float(record["cumulative_delay"][slot]), 6),
                            round(float(record["slot_fidelity"][slot]), 6),
                            round(float(record["slot_solver_time"][slot]), 6),
                            round(float(record["slot_throughput"][slot]), 6),
                            round(float(record["slot_sync"][slot]), 6),
                        ]
                    )


def write_density_csv(path: Path, density_rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "num_uavs",
                "seed",
                "avg_delay",
                "avg_fidelity",
                "avg_solver_time",
                "total_solver_time",
                "avg_throughput",
                "avg_sync_triggers",
            ]
        )
        for row in density_rows:
            writer.writerow(
                [
                    row["method"],
                    row["num_uavs"],
                    row["seed"],
                    round(float(row["avg_delay"]), 6),
                    round(float(row["avg_fidelity"]), 6),
                    round(float(row["avg_solver_time"]), 6),
                    round(float(row["total_solver_time"]), 6),
                    round(float(row["avg_throughput"]), 6),
                    round(float(row["avg_sync"]), 6),
                ]
            )


def style_plot() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "figure.dpi": 220,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def plot_series(
    path: Path,
    x: np.ndarray,
    methods_by_seed: dict[str, list[dict[str, object]]],
    key: str,
    ylabel: str,
    title: str,
    smooth_window: int = 1,
    show_raw: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for method_key, records in methods_by_seed.items():
        mean_values, std_values = aggregate_series(records, key)
        plot_values = smooth_series(mean_values, smooth_window)
        plot_std = smooth_series(std_values, smooth_window)
        meta = METHODS[method_key]
        if show_raw:
            ax.plot(
                x,
                mean_values,
                color=meta["color"],
                linewidth=0.9,
                linestyle=meta["linestyle"],
                alpha=0.25,
            )
        ax.plot(
            x,
            plot_values,
            label=meta["label"],
            color=meta["color"],
            linewidth=1.8,
            linestyle=meta["linestyle"],
            marker=meta["marker"],
            markersize=3.5,
        )
        if len(records) > 1:
            ax.fill_between(x, plot_values - plot_std, plot_values + plot_std, color=meta["color"], alpha=0.15)
    ax.set_xlabel("Time Slot")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_density_metric(path: Path, densities: list[int], density_rows: list[dict[str, object]], value_key: str, ylabel: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for method_key, meta in METHODS.items():
        means = []
        stds = []
        seed_count = len({row["seed"] for row in density_rows if row["method"] == method_key})
        for density in densities:
            values = [row[value_key] for row in density_rows if row["method"] == method_key and row["num_uavs"] == density]
            means.append(float(np.mean(values)))
            stds.append(float(np.std(values)))
        density_array = np.array(densities, dtype=int)
        means_array = np.array(means, dtype=float)
        stds_array = np.array(stds, dtype=float)
        ax.plot(
            density_array,
            means_array,
            marker=meta["marker"],
            color=meta["color"],
            label=meta["label"],
            linewidth=1.8,
            linestyle=meta["linestyle"],
            markersize=4.0,
        )
        if seed_count > 1:
            ax.fill_between(density_array, means_array - stds_array, means_array + stds_array, color=meta["color"], alpha=0.15)
    ax.set_xlabel("Number of UAVs")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    densities = parse_int_list(args.densities)
    seeds = parse_int_list(args.seeds)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_dir = ensure_dir(Path(args.output_dir) / f"comparison_{timestamp}")
    figures_dir = ensure_dir(study_dir / "figures")
    style_plot()

    study_config = {
        "slots": args.slots,
        "densities": densities,
        "seeds": seeds,
        "focus_uavs": args.focus_uavs,
        "smooth_window": args.smooth_window,
        "show_raw_convergence": args.show_raw_convergence,
        "methods": METHODS,
    }
    (study_dir / "study_config.json").write_text(json.dumps(study_config, indent=2), encoding="utf-8")

    convergence_records: dict[str, list[dict[str, object]]] = defaultdict(list)
    print(f"Running convergence study at {args.focus_uavs} UAVs...")
    for seed in seeds:
        for method_key in METHODS:
            run_dir = ensure_dir(study_dir / "raw_runs" / f"focus_{args.focus_uavs}" / method_key / f"seed_{seed}")
            config = build_config(args.slots, seed, args.focus_uavs, str(run_dir))
            print(f"  {METHODS[method_key]['label']} | seed={seed}")
            convergence_records[method_key].append(run_case(method_key, config))

    write_convergence_csv(study_dir / "convergence_metrics.csv", convergence_records)
    x_slots = np.arange(1, args.slots + 1)
    plot_series(
        figures_dir / "convergence_cumulative_delay.png",
        x_slots,
        convergence_records,
        "cumulative_delay",
        "Cumulative Average Service Delay",
        "Convergence of Average Service Delay",
        smooth_window=args.smooth_window,
        show_raw=args.show_raw_convergence,
    )
    plot_series(
        figures_dir / "delay_over_slots.png",
        x_slots,
        convergence_records,
        "slot_delay",
        "Average Service Delay",
        "Average Service Delay Over Time Slots",
        smooth_window=args.smooth_window,
        show_raw=args.show_raw_convergence,
    )
    plot_series(
        figures_dir / "fidelity_over_slots.png",
        x_slots,
        convergence_records,
        "slot_fidelity",
        "Average Twin Fidelity",
        "Average Twin Fidelity Over Time Slots",
        smooth_window=args.smooth_window,
        show_raw=args.show_raw_convergence,
    )
    plot_series(
        figures_dir / "sync_triggers_over_slots.png",
        x_slots,
        convergence_records,
        "slot_sync",
        "Average Sync Triggers",
        "Average Sync Triggers Over Time Slots",
        smooth_window=args.smooth_window,
        show_raw=args.show_raw_convergence,
    )

    density_rows: list[dict[str, object]] = []
    print("Running user-density sweep...")
    for density in densities:
        for seed in seeds:
            for method_key in METHODS:
                run_dir = ensure_dir(study_dir / "raw_runs" / f"density_{density}" / method_key / f"seed_{seed}")
                config = build_config(args.slots, seed, density, str(run_dir))
                print(f"  {METHODS[method_key]['label']} | UAVs={density} | seed={seed}")
                result = run_case(method_key, config)
                density_rows.append(
                    {
                        "method": method_key,
                        "num_uavs": density,
                        "seed": seed,
                        "avg_delay": result["avg_delay"],
                        "avg_fidelity": result["avg_fidelity"],
                        "avg_solver_time": result["avg_solver_time"],
                        "total_solver_time": result["total_solver_time"],
                        "avg_throughput": result["avg_throughput"],
                        "avg_sync": result["avg_sync"],
                    }
                )

    write_density_csv(study_dir / "density_summary.csv", density_rows)
    plot_density_metric(
        figures_dir / "user_density_vs_delay.png",
        densities,
        density_rows,
        "avg_delay",
        "Average Service Delay",
        "User Density vs Delay",
    )
    plot_density_metric(
        figures_dir / "user_density_vs_solver_time.png",
        densities,
        density_rows,
        "avg_solver_time",
        "Average Solver Time per Slot (s)",
        "User Density vs Solver Time",
    )
    plot_density_metric(
        figures_dir / "user_density_vs_throughput.png",
        densities,
        density_rows,
        "avg_throughput",
        "Average Assigned Tasks per Slot",
        "User Density vs Throughput",
    )
    plot_density_metric(
        figures_dir / "user_density_vs_fidelity.png",
        densities,
        density_rows,
        "avg_fidelity",
        "Average Twin Fidelity",
        "User Density vs Fidelity",
    )

    print(f"\nStudy outputs saved in: {study_dir}")
    print(f"Figures saved in: {figures_dir}")


if __name__ == "__main__":
    main()
