from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import replace
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
        "label": "Proposed DTN-QUBO",
        "backend": "auto",
        "use_solver_guidance": True,
        "projection": "hierarchical",
        "color": "#1b9e77",
        "linestyle": "-",
    },
    "greedy": {
        "label": "Greedy heuristic",
        "backend": None,
        "use_solver_guidance": False,
        "projection": "greedy",
        "color": "#d95f02",
        "linestyle": "--",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate reference-paper-style stable trend plots."
    )
    parser.add_argument("--slots", type=int, default=500)
    parser.add_argument("--seeds", type=str, default="7,13,21,31,43")
    parser.add_argument("--focus-uavs", type=int, default=20)
    parser.add_argument("--densities", type=str, default="10,20,30,40,50")
    parser.add_argument("--consensus-steps", type=str, default="0.15,0.25,0.50,0.75")
    parser.add_argument("--delay-weights", type=str, default="0.5,1.0,2.0,4.0")
    parser.add_argument("--output-dir", type=str, default="reference_trend_results")
    parser.add_argument(
        "--energy-scale",
        type=float,
        default=1.0,
        help="Multiplier used only for plotting/reporting readable energy values.",
    )
    parser.add_argument(
        "--raw-profile",
        action="store_true",
        help="Disable the stable trend hyperparameter profile.",
    )
    return parser.parse_args()


def parse_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_floats(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def stable_trend_config(
    *,
    slots: int,
    seed: int,
    num_uavs: int,
    output_dir: str,
    raw_profile: bool,
    consensus_step_size: float | None = None,
    delay_weight: float | None = None,
) -> SimulationConfig:
    config = SimulationConfig(
        slots=slots,
        seed=seed,
        num_uavs=num_uavs,
        output_dir=output_dir,
        anneal_reads=20,
        anneal_sweeps=80,
    )
    if raw_profile:
        return replace(
            config,
            consensus_step_size=consensus_step_size
            if consensus_step_size is not None
            else config.consensus_step_size,
            delay_weight=delay_weight if delay_weight is not None else config.delay_weight,
        )

    return replace(
        config,
        twin_smoothing=0.80,
        sensor_filter_factor=0.80,
        trust_update_factor=0.20,
        consensus_step_size=consensus_step_size
        if consensus_step_size is not None
        else 0.25,
        consensus_quantum=0.05,
        consensus_epsilon=0.03,
        anneal_reads=20,
        anneal_sweeps=80,
        delay_weight=delay_weight if delay_weight is not None else config.delay_weight,
    )


def run_method(method_key: str, config: SimulationConfig) -> dict[str, np.ndarray | dict]:
    runner = SimulationRunner(config)
    method = METHODS[method_key]
    runner.use_solver_guidance = method["use_solver_guidance"]
    runner.regional_controller.strategy = method["projection"]
    if method["backend"] is None:
        runner.local_solver = None
    else:
        runner.local_solver = ClassicalQuboSolver(config, backend=method["backend"])

    slot_results, summary, _ = runner.run()
    delay = np.array([slot.total_delay for slot in slot_results], dtype=float)
    energy = np.array([slot.total_energy for slot in slot_results], dtype=float)
    fidelity = np.array(
        [
            np.mean([item.fidelity for item in slot.local_summaries])
            for slot in slot_results
        ],
        dtype=float,
    )
    return {
        "delay": delay,
        "energy": energy,
        "fidelity": fidelity,
        "time_avg_delay": np.cumsum(delay) / np.arange(1, len(delay) + 1),
        "time_avg_energy": np.cumsum(energy) / np.arange(1, len(energy) + 1),
        "summary": summary,
    }


def aggregate(records: list[dict[str, np.ndarray | dict]], key: str) -> tuple[np.ndarray, np.ndarray]:
    values = np.vstack([record[key] for record in records])
    return np.mean(values, axis=0), np.std(values, axis=0)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def style_plots() -> None:
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
            "grid.alpha": 0.28,
        }
    )


def energy_ylabel(energy_scale: float) -> str:
    if abs(energy_scale - 1.0) < 1e-12:
        return "Total time-average AP energy"
    return f"Total time-average AP energy x {energy_scale:.0e}"


def write_time_average_csv(
    path: Path,
    records: dict[str, list[dict[str, np.ndarray | dict]]],
    energy_scale: float,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "seed_index",
                "slot",
                "time_avg_delay",
                "time_avg_energy_raw",
                "time_avg_energy_scaled",
                "time_avg_fidelity",
            ]
        )
        for method, method_records in records.items():
            for seed_index, record in enumerate(method_records):
                fidelity_avg = np.cumsum(record["fidelity"]) / np.arange(
                    1, len(record["fidelity"]) + 1
                )
                for idx, value in enumerate(record["time_avg_delay"]):
                    energy_raw = record["time_avg_energy"][idx]
                    writer.writerow(
                        [
                            method,
                            seed_index,
                            idx + 1,
                            round(float(value), 8),
                            f"{float(energy_raw):.12e}",
                            round(float(energy_raw * energy_scale), 8),
                            round(float(fidelity_avg[idx]), 8),
                        ]
                    )


def plot_time_average_comparison(
    path: Path,
    records: dict[str, list[dict[str, np.ndarray | dict]]],
    energy_scale: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
    for method_key, method_records in records.items():
        meta = METHODS[method_key]
        delay_mean, delay_std = aggregate(method_records, "time_avg_delay")
        energy_mean, energy_std = aggregate(method_records, "time_avg_energy")
        x = np.arange(1, len(delay_mean) + 1)
        axes[0].plot(
            x,
            delay_mean,
            label=meta["label"],
            color=meta["color"],
            linestyle=meta["linestyle"],
            linewidth=1.6,
        )
        axes[0].fill_between(
            x,
            delay_mean - delay_std,
            delay_mean + delay_std,
            color=meta["color"],
            alpha=0.12,
        )
        scaled_energy = energy_mean * energy_scale
        scaled_std = energy_std * energy_scale
        axes[1].plot(
            x,
            scaled_energy,
            label=meta["label"],
            color=meta["color"],
            linestyle=meta["linestyle"],
            linewidth=1.6,
        )
        axes[1].fill_between(
            x,
            scaled_energy - scaled_std,
            scaled_energy + scaled_std,
            color=meta["color"],
            alpha=0.12,
        )

    axes[0].set_xlabel("Time slots")
    axes[0].set_ylabel("Total time-average service delay")
    axes[0].set_title("(a) Delay convergence")
    axes[1].set_xlabel("Time slots")
    axes[1].set_ylabel(energy_ylabel(energy_scale))
    axes[1].set_title("(b) Energy convergence")
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run_time_average_study(
    study_dir: Path,
    args: argparse.Namespace,
    seeds: list[int],
) -> None:
    records: dict[str, list[dict[str, np.ndarray | dict]]] = defaultdict(list)
    for seed in seeds:
        for method_key in METHODS:
            run_dir = ensure_dir(study_dir / "raw_runs" / "time_average" / method_key / f"seed_{seed}")
            config = stable_trend_config(
                slots=args.slots,
                seed=seed,
                num_uavs=args.focus_uavs,
                output_dir=str(run_dir),
                raw_profile=args.raw_profile,
            )
            records[method_key].append(run_method(method_key, config))

    write_time_average_csv(study_dir / "time_average_convergence.csv", records, args.energy_scale)
    plot_time_average_comparison(
        study_dir / "figures" / "fig_time_average_delay_energy.png",
        records,
        args.energy_scale,
    )


def run_consensus_sweep(
    study_dir: Path,
    args: argparse.Namespace,
    seeds: list[int],
    consensus_steps: list[float],
) -> None:
    sweep_records: dict[float, list[dict[str, np.ndarray | dict]]] = defaultdict(list)
    for step in consensus_steps:
        for seed in seeds:
            run_dir = ensure_dir(
                study_dir / "raw_runs" / "consensus_sweep" / f"eta_{step:g}" / f"seed_{seed}"
            )
            config = stable_trend_config(
                slots=args.slots,
                seed=seed,
                num_uavs=args.focus_uavs,
                output_dir=str(run_dir),
                raw_profile=args.raw_profile,
                consensus_step_size=step,
            )
            sweep_records[step].append(run_method("proposed", config))

    with (study_dir / "consensus_step_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "consensus_step_size",
                "final_time_avg_delay",
                "final_time_avg_energy_raw",
                "final_time_avg_energy_scaled",
                "final_time_avg_fidelity",
            ]
        )
        for step in consensus_steps:
            delay_mean, _ = aggregate(sweep_records[step], "time_avg_delay")
            energy_mean, _ = aggregate(sweep_records[step], "time_avg_energy")
            fidelity_mean, _ = aggregate(sweep_records[step], "fidelity")
            writer.writerow(
                [
                    step,
                    round(float(delay_mean[-1]), 8),
                    f"{float(energy_mean[-1]):.12e}",
                    round(float(energy_mean[-1] * args.energy_scale), 8),
                    round(float(np.mean(fidelity_mean)), 8),
                ]
            )

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(consensus_steps)))
    for color, step in zip(colors, consensus_steps):
        delay_mean, _ = aggregate(sweep_records[step], "time_avg_delay")
        energy_mean, _ = aggregate(sweep_records[step], "time_avg_energy")
        x = np.arange(1, len(delay_mean) + 1)
        label = f"eta = {step:g}"
        axes[0].plot(x, delay_mean, color=color, linewidth=1.4, label=label)
        axes[1].plot(x, energy_mean * args.energy_scale, color=color, linewidth=1.4, label=label)

    axes[0].set_xlabel("Time slots")
    axes[0].set_ylabel("Total time-average service delay")
    axes[0].set_title("(a) Impact of consensus step")
    axes[1].set_xlabel("Time slots")
    axes[1].set_ylabel(energy_ylabel(args.energy_scale))
    axes[1].set_title("(b) Impact of consensus step")
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(study_dir / "figures" / "fig_consensus_step_convergence.png", bbox_inches="tight")
    plt.close(fig)


def run_delay_weight_sweep(
    study_dir: Path,
    args: argparse.Namespace,
    seeds: list[int],
    delay_weights: list[float],
) -> None:
    sweep_records: dict[float, list[dict[str, np.ndarray | dict]]] = defaultdict(list)
    for weight in delay_weights:
        for seed in seeds:
            run_dir = ensure_dir(
                study_dir / "raw_runs" / "delay_weight_sweep" / f"alpha_{weight:g}" / f"seed_{seed}"
            )
            config = stable_trend_config(
                slots=args.slots,
                seed=seed,
                num_uavs=args.focus_uavs,
                output_dir=str(run_dir),
                raw_profile=args.raw_profile,
                delay_weight=weight,
            )
            sweep_records[weight].append(run_method("proposed", config))

    with (study_dir / "delay_weight_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "delay_weight_alpha",
                "final_time_avg_total_delay",
                "final_time_avg_total_energy_raw",
                "final_time_avg_total_energy_scaled",
                "final_time_avg_fidelity",
            ]
        )
        for weight in delay_weights:
            delay_mean, _ = aggregate(sweep_records[weight], "time_avg_delay")
            energy_mean, _ = aggregate(sweep_records[weight], "time_avg_energy")
            fidelity_mean, _ = aggregate(sweep_records[weight], "fidelity")
            writer.writerow(
                [
                    weight,
                    round(float(delay_mean[-1]), 8),
                    f"{float(energy_mean[-1]):.12e}",
                    round(float(energy_mean[-1] * args.energy_scale), 8),
                    round(float(np.mean(fidelity_mean)), 8),
                ]
            )

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(delay_weights)))
    for color, weight in zip(colors, delay_weights):
        delay_mean, _ = aggregate(sweep_records[weight], "time_avg_delay")
        energy_mean, _ = aggregate(sweep_records[weight], "time_avg_energy")
        x = np.arange(1, len(delay_mean) + 1)
        label = f"alpha = {weight:g}"
        axes[0].plot(x, delay_mean, color=color, linewidth=1.4, label=label)
        axes[1].plot(x, energy_mean * args.energy_scale, color=color, linewidth=1.4, label=label)

    axes[0].set_xlabel("Time slots")
    axes[0].set_ylabel("Total time-average service delay")
    axes[0].set_title("(a) Impact of delay weight")
    axes[1].set_xlabel("Time slots")
    axes[1].set_ylabel(energy_ylabel(args.energy_scale))
    axes[1].set_title("(b) Impact of delay weight")
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(study_dir / "figures" / "fig_delay_weight_convergence.png", bbox_inches="tight")
    plt.close(fig)


def run_density_tradeoff(
    study_dir: Path,
    args: argparse.Namespace,
    seeds: list[int],
    densities: list[int],
) -> None:
    rows: list[dict[str, float | int]] = []
    for density in densities:
        density_records: list[dict[str, np.ndarray | dict]] = []
        for seed in seeds:
            run_dir = ensure_dir(study_dir / "raw_runs" / "density_tradeoff" / f"uav_{density}" / f"seed_{seed}")
            config = stable_trend_config(
                slots=args.slots,
                seed=seed,
                num_uavs=density,
                output_dir=str(run_dir),
                raw_profile=args.raw_profile,
            )
            density_records.append(run_method("proposed", config))
        delay_mean, delay_std = aggregate(density_records, "time_avg_delay")
        energy_mean, energy_std = aggregate(density_records, "time_avg_energy")
        rows.append(
            {
                "num_uavs": density,
                "delay_mean": float(delay_mean[-1]),
                "delay_std": float(delay_std[-1]),
                "energy_mean_raw": float(energy_mean[-1]),
                "energy_std_raw": float(energy_std[-1]),
                "energy_mean_scaled": float(energy_mean[-1] * args.energy_scale),
                "energy_std_scaled": float(energy_std[-1] * args.energy_scale),
            }
        )

    with (study_dir / "density_tradeoff_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    x = np.array([row["num_uavs"] for row in rows], dtype=float)
    delay = np.array([row["delay_mean"] for row in rows], dtype=float)
    delay_std = np.array([row["delay_std"] for row in rows], dtype=float)
    energy = np.array([row["energy_mean_scaled"] for row in rows], dtype=float)
    energy_std = np.array([row["energy_std_scaled"] for row in rows], dtype=float)

    fig, ax1 = plt.subplots(figsize=(5.4, 3.4))
    ax2 = ax1.twinx()
    ax1.errorbar(x, delay, yerr=delay_std, color="#1f77b4", marker="o", linestyle="-.", label="Service delay")
    ax2.errorbar(x, energy, yerr=energy_std, color="#ff7f0e", marker="^", linestyle="-", label="Energy consumption")
    ax1.set_xlabel("Number of UAV tasks")
    ax1.set_ylabel("Total time-average service delay", color="#1f77b4")
    ax2.set_ylabel(energy_ylabel(args.energy_scale), color="#ff7f0e")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")
    fig.tight_layout()
    fig.savefig(study_dir / "figures" / "fig_density_delay_energy_tradeoff.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    seeds = parse_ints(args.seeds)
    densities = parse_ints(args.densities)
    consensus_steps = parse_floats(args.consensus_steps)
    delay_weights = parse_floats(args.delay_weights)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_dir = ensure_dir(Path(args.output_dir) / f"trend_{timestamp}")
    ensure_dir(study_dir / "figures")
    style_plots()

    config = {
        "slots": args.slots,
        "seeds": seeds,
        "focus_uavs": args.focus_uavs,
        "densities": densities,
        "consensus_steps": consensus_steps,
        "delay_weights": delay_weights,
        "energy_scale": args.energy_scale,
        "stable_profile_enabled": not args.raw_profile,
        "stable_profile": {
            "twin_smoothing": 0.80,
            "sensor_filter_factor": 0.80,
            "trust_update_factor": 0.20,
            "consensus_step_size": 0.25,
            "consensus_quantum": 0.05,
            "consensus_epsilon": 0.03,
            "anneal_reads": 20,
            "anneal_sweeps": 80,
        },
    }
    (study_dir / "trend_study_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("Running time-average comparison study...")
    run_time_average_study(study_dir, args, seeds)
    print("Running consensus-step sweep...")
    run_consensus_sweep(study_dir, args, seeds, consensus_steps)
    print("Running delay-weight sweep...")
    run_delay_weight_sweep(study_dir, args, seeds, delay_weights)
    print("Running density tradeoff study...")
    run_density_tradeoff(study_dir, args, seeds, densities)
    print(f"Trend outputs saved in: {study_dir}")


if __name__ == "__main__":
    main()
