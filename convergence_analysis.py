#!/usr/bin/env python
r"""
Convergence Analysis — Iterative QUBO Assignment (Equation 21)
================================================================

Demonstrates that the per-node iterative QUBO refinement from Algorithm 1
(lines 15-29) converges the NETWORK-WIDE objective J(t) from Eq. (21)
**without** regional-controller intervention.

Equation (21):
    J(t) = Σ_u Σ_{m∈M∪{0}} x_{u,m}(t) [α·T_{u,m}(t) + β·E_{u,m}(t) + γ·S_{u,m}(t)]
           + ω_s · Σ_m a^sync_m(t)
           + ω_f · Σ_m (1 − F_m(t))
           + ν   · Σ_m χ^trig_m(t)

╔═════════════════════════════════════════════════════════════════╗
║  WHAT IS  T_{u,m}(t)  ?                                       ║
║                                                                 ║
║  T_{u,m}(t) is the TOTAL END-TO-END SERVICE DELAY incurred     ║
║  when task u is offloaded to infrastructure node m at slot t.   ║
║                                                                 ║
║  It decomposes as:                                              ║
║                                                                 ║
║    T_{u,m} = T^tx_{u,m} + T^comp_{u,m}                         ║
║              + T^sync_m  + T^coord_m                            ║
║                                                                 ║
║  where                                                          ║
║    T^tx_{u,m}   = L_u / R_{u,m}(t)                              ║
║        Transmission delay: input data size L_u (bits) divided   ║
║        by the uplink channel rate R_{u,m} (bits/s).  R_{u,m}    ║
║        depends on channel gain, bandwidth, noise, and the       ║
║        number of co-scheduled tasks at node m.                  ║
║                                                                 ║
║    T^comp_{u,m} = D_u · L_u / f_{u,m}(t)                       ║
║        Computation delay: total CPU cycles (D_u cycles/bit      ║
║        × L_u bits) divided by the equal-share CPU allocation    ║
║        f_{u,m} = F_m^cpu / N_m(t), where F_m^cpu is the        ║
║        node's CPU capacity (Hz) and N_m(t) is the number        ║
║        of tasks assigned to m in this slot.                     ║
║                                                                 ║
║    T^sync_m     = tier-dependent DT synchronisation latency     ║
║        Only incurred when the twin-age / uncertainty trigger    ║
║        fires (a^sync_m = 1); value depends on tier:             ║
║        BS → 30 ms, HAP → 80 ms, LEO → 120 ms.                  ║
║                                                                 ║
║    T^coord_m    = tier-dependent regional coordination latency  ║
║        Only incurred when the trust / mismatch trigger fires    ║
║        (χ^trig_m = 1); value depends on tier:                   ║
║        BS → 10 ms, HAP → 30 ms, LEO → 50 ms.                   ║
║                                                                 ║
║  In the code:  evaluator.evaluate_remote(...).total_delay_s     ║
║  returns T_{u,m} in seconds.                                    ║
╚═════════════════════════════════════════════════════════════════╝

Usage:
    cd DTN_SAGIN_COMPLETE_CODE
    python convergence_analysis.py
"""
from __future__ import annotations

import copy
import sys
from collections import defaultdict
from math import exp
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from env.config import SimulationConfig
from env.evaluator import PhysicalEvaluator
from env.generator import SaginEnvironment
from env.models import (
    APNode, CandidateScore, FinalAssignment, LocalSummary, Task,
)
from twin.twin_logic import TwinManager
from opt.qubo_generator import LocalQuboBuilder
from opt.solver import ClassicalQuboSolver

# Windows PowerShell may use a legacy code page that cannot print the Greek
# symbols used in the explanation below. Keep the script runnable without
# requiring users to set PYTHONIOENCODING manually.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Equation (21)  –  Network-Wide Objective
# ---------------------------------------------------------------------------

def compute_J_eq21(
    config: SimulationConfig,
    evaluator: PhysicalEvaluator,
    assignment_mapping: dict[str, str],
    tasks: list[Task],
    ap_lookup: dict[str, APNode],
    sync_ap_ids: set[str],
    coord_ap_ids: set[str],
    local_summaries: list[LocalSummary],
) -> dict[str, float]:
    """
    Compute every term of Eq. (21) from a raw task→destination mapping.

    Returns a dict with decomposed cost components and the total J(t).
    """
    # Count tasks assigned per AP so we can compute equal-share f_{u,m}
    load_per_ap: dict[str, int] = defaultdict(int)
    for dest in assignment_mapping.values():
        if dest in ap_lookup:
            load_per_ap[dest] += 1

    # -------- Term 1: Σ_u x_{u,m} [α·T + β·E + γ·S] --------
    raw_delay = 0.0    # sum of T_{u,m}  (seconds)
    raw_energy = 0.0   # sum of E_{u,m}  (joules)
    raw_mission = 0.0  # sum of S_{u,m}  (freshness risk)
    for task in tasks:
        dest = assignment_mapping.get(task.task_id, "LOCAL")
        if dest not in ap_lookup:
            # Local execution  (destination 0 in the paper)
            metrics = evaluator.evaluate_local(task)
        else:
            ap = ap_lookup[dest]
            load = max(load_per_ap.get(dest, 1), 1)
            metrics = evaluator.evaluate_remote(
                task, ap, load,
                sync_active=dest in sync_ap_ids,
                coord_active=dest in coord_ap_ids,
            )
        raw_delay += metrics.total_delay_s
        raw_energy += metrics.total_energy_j
        raw_mission += metrics.freshness_risk

    alpha_T = config.delay_weight * evaluator.normalize_delay(raw_delay)
    beta_E  = config.energy_weight * evaluator.normalize_energy(raw_energy)
    gamma_S = config.mission_weight * evaluator.normalize_freshness_risk(raw_mission)

    # -------- Term 2: ω_s · Σ_m a^sync_m --------
    omega_s_sync = config.sync_cost_weight * len(sync_ap_ids)

    # -------- Term 3: ω_f · Σ_m (1 − F_m) --------
    omega_f_fidelity = config.fidelity_weight * sum(
        (1.0 - s.fidelity) for s in local_summaries
    )

    # -------- Term 4: ν · Σ_m χ^trig_m --------
    nu_coord = config.coord_overhead_weight * len(coord_ap_ids)

    J_total = alpha_T + beta_E + gamma_S + omega_s_sync + omega_f_fidelity + nu_coord
    return {
        "J_total": J_total,
        "alpha_T": alpha_T,
        "beta_E": beta_E,
        "gamma_S": gamma_S,
        "omega_s_sync": omega_s_sync,
        "omega_f_fidelity": omega_f_fidelity,
        "nu_coord": nu_coord,
        "raw_delay_s": raw_delay,
        "raw_energy_j": raw_energy,
    }


# ---------------------------------------------------------------------------
# Extract an assignment mapping from per-node QUBO solutions
#   (no regional controller — raw QUBO output only)
# ---------------------------------------------------------------------------

def qubo_solutions_to_mapping(
    per_node_samples: dict[str, dict[tuple[str, str], int]],
    tasks: list[Task],
    ap_lookup: dict[str, APNode],
    evaluator: PhysicalEvaluator,
) -> dict[str, str]:
    """
    Combine per-node binary solutions into a one-hot task→destination mapping.

    When two nodes both claim a task (both set b_{u,m}=1), pick the one with
    lower service cost — this mirrors what each node independently *proposes*
    without the regional controller resolving anything.
    """
    task_by_id = {task.task_id: task for task in tasks}
    candidates: dict[str, list[tuple[str, float]]] = defaultdict(list)

    for ap_id, sample in per_node_samples.items():
        ap = ap_lookup[ap_id]
        for (task_id, dest_id), value in sample.items():
            if value == 1 and dest_id == ap_id:
                # Score by service cost at load=1 (optimistic single-task view)
                if task_id in task_by_id:
                    metrics = evaluator.evaluate_remote(
                        task_by_id[task_id], ap, 1.0,
                    )
                    candidates[task_id].append(
                        (ap_id, evaluator.service_cost(metrics))
                    )

    mapping: dict[str, str] = {}
    for task in tasks:
        task_candidates = candidates.get(task.task_id, [])
        if task_candidates:
            mapping[task.task_id] = min(task_candidates, key=lambda x: x[1])[0]
        else:
            # No node accepted this task — fall back to first candidate AP
            mapping[task.task_id] = task.A_u_t[0] if task.A_u_t else "LOCAL"
    return mapping


# ---------------------------------------------------------------------------
# Run convergence experiment for ONE seed
# ---------------------------------------------------------------------------

def run_convergence_for_seed(
    seed: int,
    K_max: int = 15,
    warmup_slots: int = 5,
    test_slots: int = 10,
    num_uavs: int = 20,
) -> dict:
    """
    Returns per-iteration J(t) trajectories for multiple test slots under one
    random seed, proving the QUBO inner loop converges J(t).
    """
    config = SimulationConfig(
        seed=seed,
        slots=warmup_slots + test_slots,
        num_uavs=num_uavs,
        max_qubo_iterations=K_max,
        algorithm="proposed",
        output_dir=f"results/convergence_seed{seed}",
    )
    env = SaginEnvironment(config)
    twin_mgr = TwinManager(config)
    qubo_builder = LocalQuboBuilder(config)
    solver = ClassicalQuboSolver(config, backend="auto")
    evaluator = PhysicalEvaluator(config)
    ap_lookup = env.ap_by_id()

    all_trajectories: list[list[dict[str, float]]] = []
    # Keep per-node traces, then choose the most active node from each tier
    # for plotting (the _0 nodes need not receive any candidate tasks).
    representative_trajectories: dict[str, list[list[dict[str, float]]]] = {
        ap.ap_id: [] for ap in env.aps
    }

    for slot in range(1, config.slots + 1):
        env.step_mobility()
        tasks = env.create_tasks_for_slot(slot)
        if not tasks:
            continue
        grouped = env.group_tasks_by_owner(tasks)

        # ---- DT update (Algorithm 1, lines 4-11) ----
        local_summaries: list[LocalSummary] = []
        for ap in env.aps:
            queue = grouped.get(ap.ap_id, [])
            obs = env.build_observation(ap, slot, queue, len(tasks))
            req_fid = max((t.F_u_min for t in queue), default=0.5)
            sync_trig, coord_trig = twin_mgr.update(ap, obs, req_fid)
            local_summaries.append(LocalSummary(
                ap_id=ap.ap_id, slot=slot, queue_size=len(queue),
                local_load=float(ap.current_cpu_load / max(ap.cpu_capacity, 1.0)),
                sync_triggered=bool(sync_trig),
                coordination_triggered=bool(coord_trig),
                trust=float(ap.trust),
                twin_age=int(ap.twin_state.age),
                uncertainty=float(ap.twin_state.uncertainty),
                mismatch=float(ap.twin_state.mismatch),
                fidelity=float(ap.twin_state.fidelity),
                qubo_dimension=0, solver_time=0.0,
                selected_pairs=[], candidate_scores=[],
            ))

        sync_ap_ids = {s.ap_id for s in local_summaries if s.sync_triggered}
        coord_ap_ids = {s.ap_id for s in local_summaries if s.coordination_triggered}

        if slot <= warmup_slots:
            # Warmup — just let DT states settle, skip convergence tracking
            for ap in env.aps:
                ap.current_task_load = max(len(grouped.get(ap.ap_id, [])), 0)
            continue

        # ---- Freeze reference summaries for this slot ----
        frozen_refs = {
            ap_id: tuple(ap.coordination_state.reported_summary)
            for ap_id, ap in ap_lookup.items()
        }

        # ---- Per-node iterative QUBO (Algorithm 1, lines 13-31) ----
        # Track J(t) at EACH iteration k
        iteration_trajectory: list[dict[str, float]] = []

        # Per-node iteration state  {ap_id: (nbar, multiplier, b_prev)}
        node_state: dict[str, dict] = {}
        for ap in env.aps:
            eligible = [
                t for t in tasks
                if ap.ap_id in t.A_u_t and ap.twin_state.fidelity >= t.F_u_min
            ]
            node_state[ap.ap_id] = {
                "eligible": eligible,
                "nbar": max(1.0, float(ap.current_task_load)),
                "multiplier": tuple(ap.coordination_state.coordination_multiplier),
                "b_prev": {(t.task_id, ap.ap_id): 0 for t in eligible},
                "b_curr": {},
                "converged": False,
            }

        for k in range(K_max):
            per_node_samples: dict[str, dict[tuple[str, str], int]] = {}
            representative_iteration: dict[str, dict[str, float]] = {}
            nodes_converged = 0

            for ap in env.aps:
                ns = node_state[ap.ap_id]
                if ns["converged"]:
                    per_node_samples[ap.ap_id] = ns["b_curr"]
                    nodes_converged += 1
                    continue
                eligible = ns["eligible"]
                if not eligible:
                    per_node_samples[ap.ap_id] = {}
                    if ap.ap_id in representative_trajectories:
                        representative_iteration[ap.ap_id] = {
                            "iteration": float(k), "service_cost": 0.0,
                            "selected_tasks": 0.0, "candidate_tasks": 0.0,
                            "load_update_residual": 0.0,
                            "reference_load": ns["nbar"],
                        }
                    nodes_converged += 1
                    continue

                # Build QUBO with iteration-dependent N̄(k) and λ(k) — eq (30)-(35)
                problem = qubo_builder.build(
                    ap, tasks, ap_lookup, slot,
                    sync_ap_ids, coord_ap_ids,
                    reference_load=ns["nbar"],
                    reference_summary=frozen_refs[ap.ap_id],
                    multiplier=ns["multiplier"],
                )
                # Solve QUBO — eq (33), line 18
                if problem.variables:
                    result = solver.solve(problem)
                    sample = {
                        v: int(result.sample.get(v, 0))
                        for v in problem.variables
                    }
                else:
                    sample = {}
                ns["b_curr"] = sample
                per_node_samples[ap.ap_id] = sample

                # Track a representative node's actual local service cost.
                # QUBO energies are not comparable across k when coefficients
                # change, so the plot uses the physical service costs carried
                # by this iteration's candidate scores instead.
                if ap.ap_id in representative_trajectories:
                    score_by_task = {
                        score.task_id: score for score in problem.candidate_scores
                    }
                    selected_ids = [
                        task_id for (task_id, dest_id), value in sample.items()
                        if value and dest_id == ap.ap_id
                    ]
                    representative_iteration[ap.ap_id] = {
                        "iteration": float(k),
                        "service_cost": float(sum(
                            score_by_task[task_id].local_cost
                            for task_id in selected_ids if task_id in score_by_task
                        )),
                        "selected_tasks": float(len(selected_ids)),
                        "candidate_tasks": float(len(problem.candidate_scores)),
                        "load_update_residual": abs(
                            max(1.0, float(sum(sample.values())))
                            - float(problem.reference_load)
                        ),
                        "reference_load": float(problem.reference_load),
                    }

                # Update N̄(k+1) — eq (39), line 19
                ns["nbar"] = max(1.0, float(sum(sample.values())))

                # Update λ(k+1) — eq (40), lines 20-24
                summary_ap = next(
                    (s for s in local_summaries if s.ap_id == ap.ap_id), None
                )
                if summary_ap and summary_ap.coordination_triggered:
                    selected = [
                        t for t in eligible
                        if sample.get((t.task_id, ap.ap_id), 0)
                    ]
                    trust_risk = (1.0 - ap.trust) * sum(t.psi_u for t in selected)
                    app_pr = (
                        sum(t.F_u_min for t in selected) / len(selected)
                        if selected else 0.0
                    )
                    cand_summary = (
                        float(len(selected)),
                        float(trust_risk),
                        float(ap.twin_state.uncertainty),
                        float(1.0 - ap.twin_state.fidelity),
                        float(app_pr),
                    )
                    eta = config.coordination_multiplier_step
                    ns["multiplier"] = tuple(
                        lam + eta * (z - ref)
                        for lam, z, ref in zip(
                            ns["multiplier"], cand_summary,
                            frozen_refs[ap.ap_id],
                        )
                    )

                # Convergence check — line 26
                if sample == ns["b_prev"] and k > 0:
                    ns["converged"] = True
                    nodes_converged += 1
                ns["b_prev"] = dict(sample)

            # ---- Evaluate J(k) from Eq. (21) using RAW QUBO output ----
            mapping = qubo_solutions_to_mapping(
                per_node_samples, tasks, ap_lookup, evaluator,
            )
            J_components = compute_J_eq21(
                config, evaluator, mapping,
                tasks, ap_lookup,
                sync_ap_ids, coord_ap_ids,
                local_summaries,
            )
            J_components["iteration"] = k
            J_components["nodes_converged"] = nodes_converged
            J_components["total_nodes"] = len(env.aps)
            iteration_trajectory.append(J_components)
            for ap_id, values in representative_iteration.items():
                if len(representative_trajectories[ap_id]) <= len(all_trajectories):
                    representative_trajectories[ap_id].append([])
                representative_trajectories[ap_id][-1].append(values)

            if nodes_converged == len(env.aps):
                # Pad remaining iterations with the converged value
                for remaining_k in range(k + 1, K_max):
                    padded = dict(J_components)
                    padded["iteration"] = remaining_k
                    iteration_trajectory.append(padded)
                break

        all_trajectories.append(iteration_trajectory)

        # Update AP loads for next slot
        for ap in env.aps:
            ns = node_state[ap.ap_id]
            ap.current_task_load = max(1, int(ns["nbar"]))

    return {
        "seed": seed,
        "K_max": K_max,
        "trajectories": all_trajectories,
        "representative_trajectories": representative_trajectories,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_convergence(results_by_seed: list[dict], output_path: Path):
    """
    Plot J(t) vs QUBO iteration k showing convergence.

    Layout: two subplots
      (a) J_total vs k — individual slots (thin) + mean (bold)
      (b) Decomposed cost components vs k
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 12,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 350,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": ":",
        "mathtext.fontset": "stix",
    })

    # Collect all trajectories across seeds
    all_trajs: list[list[dict]] = []
    K_max = 15
    for res in results_by_seed:
        K_max = res["K_max"]
        all_trajs.extend(res["trajectories"])

    if not all_trajs:
        print("ERROR: No trajectories to plot.")
        return

    # Build arrays  [num_slots x K_max]
    num_slots = len(all_trajs)
    J_total = np.full((num_slots, K_max), np.nan)
    alpha_T = np.full((num_slots, K_max), np.nan)
    beta_E = np.full((num_slots, K_max), np.nan)
    gamma_S = np.full((num_slots, K_max), np.nan)
    conv_frac = np.full((num_slots, K_max), np.nan)

    for i, traj in enumerate(all_trajs):
        for entry in traj:
            k = int(entry["iteration"])
            if k < K_max:
                J_total[i, k] = entry["J_total"]
                alpha_T[i, k] = entry["alpha_T"]
                beta_E[i, k] = entry["beta_E"]
                gamma_S[i, k] = entry["gamma_S"]
                conv_frac[i, k] = (
                    entry["nodes_converged"] / max(entry["total_nodes"], 1)
                )

    ks = np.arange(K_max)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.0))

    # ── Panel (a): J_total convergence ──
    ax = axes[0]
    for i in range(num_slots):
        valid = ~np.isnan(J_total[i])
        if valid.any():
            ax.plot(ks[valid], J_total[i, valid],
                    color="#0072B2", alpha=0.12, linewidth=0.7)
    mean_J = np.nanmean(J_total, axis=0)
    std_J = np.nanstd(J_total, axis=0)
    valid_mean = ~np.isnan(mean_J)
    ax.fill_between(
        ks[valid_mean],
        (mean_J - std_J)[valid_mean],
        (mean_J + std_J)[valid_mean],
        color="#0072B2", alpha=0.18, label=r"$\pm 1\sigma$",
    )
    ax.plot(ks[valid_mean], mean_J[valid_mean],
            color="#0072B2", linewidth=2.4, marker="o", markersize=5,
            label=r"$\overline{J}(k)$ (mean across slots)")
    ax.set_xlabel(r"QUBO Iteration Index $k$")
    ax.set_ylabel(r"Network-Wide Objective $J(t)$ [Eq. (21)]")
    ax.set_title(r"(a) Convergence of $J(t)$ — No Regional Controller", fontsize=11)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xticks(ks)

    # ── Panel (b): Decomposed components ──
    ax = axes[1]
    mean_aT = np.nanmean(alpha_T, axis=0)
    mean_bE = np.nanmean(beta_E, axis=0)
    mean_gS = np.nanmean(gamma_S, axis=0)
    v = ~np.isnan(mean_aT)

    ax.plot(ks[v], mean_aT[v], color="#D55E00", linewidth=2, marker="s",
            markersize=4, label=r"$\alpha \cdot \overline{T}_{u,m}$ (delay)")
    ax.plot(ks[v], mean_bE[v], color="#009E73", linewidth=2, marker="^",
            markersize=4, label=r"$\beta \cdot \overline{E}_{u,m}$ (energy)")
    ax.plot(ks[v], mean_gS[v], color="#CC79A7", linewidth=2, marker="D",
            markersize=4, label=r"$\gamma \cdot \overline{S}_{u,m}$ (mission)")
    ax.set_xlabel(r"QUBO Iteration Index $k$")
    ax.set_ylabel("Mean Cost Component")
    ax.set_title(r"(b) Component Decomposition of $J(t)$", fontsize=11)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xticks(ks)

    # ── Panel (c): Fraction of converged nodes ──
    ax = axes[2]
    mean_conv = np.nanmean(conv_frac, axis=0)
    v = ~np.isnan(mean_conv)
    ax.plot(ks[v], mean_conv[v] * 100.0,
            color="#E69F00", linewidth=2.4, marker="o", markersize=5)
    ax.fill_between(
        ks[v],
        np.nanpercentile(conv_frac[:, v] * 100.0, 10, axis=0),
        np.nanpercentile(conv_frac[:, v] * 100.0, 90, axis=0),
        color="#E69F00", alpha=0.18, label="10th–90th percentile",
    )
    ax.set_xlabel(r"QUBO Iteration Index $k$")
    ax.set_ylabel(r"Converged Nodes (\%)")
    ax.set_title(r"(c) Node-Level Convergence $b^{(k+1)}_m = b^{(k)}_m$", fontsize=11)
    ax.set_ylim(-5, 108)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xticks(ks)

    fig.tight_layout(w_pad=2.5)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved convergence figure: {output_path}")


def plot_representative_tiers(results_by_seed: list[dict], output_path: Path):
    """Plot the most active node's local cost trajectory in each SAGIN tier."""
    tier_nodes = ["BS", "HAP", "LEO"]
    colors = {"BS": "#0072B2", "HAP": "#009E73", "LEO": "#D55E00"}
    max_iterations = max(result["K_max"] for result in results_by_seed)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    summaries = {}

    for ax, tier in zip(axes, tier_nodes):
        candidate_totals: dict[str, float] = {}
        for result in results_by_seed:
            for node_id, traces in result.get("representative_trajectories", {}).items():
                if node_id.startswith(f"{tier}_"):
                    candidate_totals[node_id] = candidate_totals.get(node_id, 0.0) + sum(
                        entry.get("candidate_tasks", 0.0)
                        for trace in traces for entry in trace
                    )
        if not candidate_totals:
            ax.text(0.5, 0.5, "No nodes in this tier", ha="center", va="center")
            ax.set_title(f"{tier} tier")
            continue
        ap_id = max(candidate_totals, key=candidate_totals.get)
        slot_traces: list[list[dict[str, float]]] = []
        for result in results_by_seed:
            slot_traces.extend(
                trace for trace in result.get("representative_trajectories", {}).get(ap_id, [])
                if trace
            )
        if not slot_traces:
            ax.text(0.5, 0.5, "No eligible tasks observed", ha="center", va="center")
            ax.set_title(f"{tier} node {ap_id}")
            continue

        values = np.full((len(slot_traces), max_iterations), np.nan)
        for row, trace in enumerate(slot_traces):
            for entry in trace:
                k = int(entry["iteration"])
                if k < max_iterations:
                    values[row, k] = entry["load_update_residual"]

        ks = np.arange(max_iterations)
        counts = np.sum(~np.isnan(values), axis=0)
        mean_cost = np.divide(np.nansum(values, axis=0), counts,
                              out=np.full(max_iterations, np.nan), where=counts > 0)
        std_cost = np.array([
            np.nanstd(values[:, k]) if counts[k] else np.nan
            for k in range(max_iterations)
        ])
        valid = ~np.isnan(mean_cost)
        for row in range(len(slot_traces)):
            row_valid = ~np.isnan(values[row])
            ax.plot(ks[row_valid], values[row, row_valid], color=colors[tier],
                    alpha=0.12, linewidth=0.8)
        ax.fill_between(ks[valid], (mean_cost - std_cost)[valid],
                        (mean_cost + std_cost)[valid], color=colors[tier], alpha=0.18,
                        label=r"$\pm 1\sigma$ across slots")
        ax.plot(ks[valid], mean_cost[valid], color=colors[tier], linewidth=2.4,
                marker="o", markersize=4, label="Mean load update residual")
        if np.nanmax(values) == 0.0:
            ax.text(0.5, 0.62, "Residual is zero throughout\n(no load-reference update)",
                    transform=ax.transAxes, ha="center", va="center", fontsize=9,
                    color=colors[tier])
        ax.set_title(f"{tier} representative node ({ap_id})")
        ax.set_xlabel("QUBO iteration index $k$")
        observed_ks = ks[valid]
        ax.set_xticks(observed_ks)
        ax.set_xlim(max(-0.2, observed_ks[0] - 0.2), observed_ks[-1] + 0.35)
        ax.set_ylim(bottom=0.0)
        ax.grid(True, alpha=0.25, linestyle=":")
        ax.legend(fontsize=8)
        first, final = mean_cost[valid][0], mean_cost[valid][-1]
        reduction = (1.0 - final / first) * 100.0 if first else 0.0
        summaries[tier] = (first, final, reduction, len(slot_traces))

    axes[0].set_ylabel(r"Load-reference update residual $|\bar N(k+1)-\bar N(k)|$")
    fig.suptitle("Representative Node Load-Update Convergence by SAGIN Tier\n"
                 "Per-node QUBO loop; regional assignment is not applied", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=350, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved tier-node convergence figure: {output_path}")
    for tier, (first, final, reduction, count) in summaries.items():
        print(f"    {tier}: mean load residual {first:.4f} -> {final:.4f} "
              f"({reduction:.1f}% change; {count} slot traces)")


# ---------------------------------------------------------------------------
# T_{u,m}  information  (printed to console)
# ---------------------------------------------------------------------------

def print_T_explanation(config: SimulationConfig):
    print("""
========================================================================
  T_{{u,m}}(t) -- Total End-to-End Service Delay (seconds)
========================================================================

  T_{{u,m}}(t) = T^tx + T^comp + T^sync + T^coord

  +--------------------+-------------------+---------------------------+
  | Component          | Formula           | Description               |
  +--------------------+-------------------+---------------------------+
  | T^tx_{{u,m}}       | L_u / R_{{u,m}}   | Transmission delay:       |
  |                    |                   | input bits / uplink rate   |
  +--------------------+-------------------+---------------------------+
  | T^comp_{{u,m}}     | D_u*L_u / f_{{u,m}}| Computation delay:       |
  |                    | f_{{u,m}}=F_m/N_m | CPU cycles / CPU share    |
  +--------------------+-------------------+---------------------------+
  | T^sync_m           | tier-dependent    | DT sync signaling latency |
  |                    | (only if sync)    |                           |
  +--------------------+-------------------+---------------------------+
  | T^coord_m          | tier-dependent    | Regional coordination     |
  |                    | (only if coord)   | signaling latency         |
  +--------------------+-------------------+---------------------------+

  Configured sync delays : BS=30ms, HAP=80ms, LEO=120ms
  Configured coord delays: BS=10ms, HAP=30ms, LEO=50ms

  In the code: evaluator.evaluate_remote().total_delay_s returns
  T_{{u,m}}(t) in seconds; the delay term in Eq. (21) is the weighted
  sum  alpha * sum_u x_{{u,m}} * T_{{u,m}}(t)  where alpha = delay_wt.
========================================================================
""")
    print(f"  Current objective weights from config:")
    print(f"    α (delay_weight)           = {config.delay_weight}")
    print(f"    β (energy_weight)          = {config.energy_weight}")
    print(f"    γ (mission_weight)         = {config.mission_weight}")
    print(f"    ω_s (sync_cost_weight)     = {config.sync_cost_weight}")
    print(f"    ω_f (fidelity_weight)      = {config.fidelity_weight}")
    print(f"    ν (coord_overhead_weight)  = {config.coord_overhead_weight}")
    print(f"    K_max (max_qubo_iterations)= {config.max_qubo_iterations}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    K_MAX = 15         # Max QUBO iterations to show
    WARMUP = 5         # Slots for DT state warm-up
    TEST_SLOTS = 10    # Slots to capture convergence data from
    SEEDS = [7, 13, 21, 31, 43]
    NUM_UAVS = 20

    output_dir = Path("convergence_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = SimulationConfig()
    print("=" * 70)
    print(" Convergence Analysis — Iterative QUBO Assignment (Eq. 21)")
    print("=" * 70)
    print_T_explanation(config)

    print(f"Running convergence experiment:")
    print(f"  Seeds      : {SEEDS}")
    print(f"  K_max      : {K_MAX}")
    print(f"  Warmup     : {WARMUP} slots")
    print(f"  Test slots : {TEST_SLOTS} per seed")
    print(f"  UAV tasks  : {NUM_UAVS}")
    print()

    results = []
    for seed in SEEDS:
        print(f"  Seed {seed}...", end=" ", flush=True)
        res = run_convergence_for_seed(
            seed, K_max=K_MAX,
            warmup_slots=WARMUP,
            test_slots=TEST_SLOTS,
            num_uavs=NUM_UAVS,
        )
        n_traj = len(res["trajectories"])
        print(f"collected {n_traj} slot trajectories.")
        results.append(res)

    fig_path = output_dir / "convergence_J_eq21_all_nodes.png"
    plot_convergence(results, fig_path)
    tier_fig_path = output_dir / "convergence_tier_load_residual_final.png"
    plot_representative_tiers(results, tier_fig_path)

    # Print summary statistics
    all_J0 = []
    all_Jfinal = []
    for res in results:
        for traj in res["trajectories"]:
            if traj:
                all_J0.append(traj[0]["J_total"])
                all_Jfinal.append(traj[-1]["J_total"])
    if all_J0 and all_Jfinal:
        mean_J0 = np.mean(all_J0)
        mean_Jf = np.mean(all_Jfinal)
        reduction_pct = (1.0 - mean_Jf / mean_J0) * 100.0 if mean_J0 != 0 else 0.0
        print(f"\n  ── Convergence Summary ──")
        print(f"    Mean J(k=0)    : {mean_J0:.4f}")
        print(f"    Mean J(k=final): {mean_Jf:.4f}")
        print(f"    Reduction      : {reduction_pct:.1f}%")
        print(f"    Trajectories   : {len(all_J0)} slots × {len(SEEDS)} seeds")

    print(f"\n  Network-wide figure saved to: {fig_path.resolve()}")
    print(f"  Tier-node figure saved to: {tier_fig_path.resolve()}")
    print("  Done.")


if __name__ == "__main__":
    main()
