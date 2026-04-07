from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class SimulationConfig:
    seed: int = 7
    slots: int = 8
    area_width: float = 100.0
    area_height: float = 100.0
    candidate_ap_limit: int = 3
    num_bs: int = 4
    num_haps: int = 2
    num_leos: int = 1
    num_uavs: int = 12
    task_arrival_probability: float = 0.8
    sync_age_threshold: int = 2
    sync_mismatch_threshold: float = 0.22
    sync_uncertainty_threshold: float = 0.18
    coordination_load_threshold: float = 0.75
    coordination_overlap_threshold: float = 0.50
    coordination_trust_threshold: float = 0.45
    twin_smoothing: float = 0.65
    qubo_penalty: float = 6.0
    qubo_load_coupling: float = 0.35
    consensus_step_size: float = 0.35
    consensus_quantum: float = 0.10
    consensus_epsilon: float = 0.08
    twin_gaussian_std: float = 0.04
    anneal_reads: int = 5
    anneal_sweeps: int = 20
    energy_weight: float = 0.8
    delay_weight: float = 1.0
    mission_weight: float = 1.2
    fidelity_weight: float = 0.9
    output_dir: str = "results"
    randomize_task_weights: bool = True
    cpu_capacity_by_tier: dict[str, float] = field(
        default_factory=lambda: {"BS": 18.0, "HAP": 30.0, "LEO": 40.0}
    )
    bandwidth_by_tier: dict[str, float] = field(
        default_factory=lambda: {"BS": 20.0, "HAP": 35.0, "LEO": 50.0}
    )
    communication_budget_by_tier: dict[str, float] = field(
        default_factory=lambda: {"BS": 24.0, "HAP": 42.0, "LEO": 60.0}
    )
    power_budget_by_tier: dict[str, float] = field(
        default_factory=lambda: {"BS": 18.0, "HAP": 28.0, "LEO": 36.0}
    )
    altitude_by_tier: dict[str, float] = field(
        default_factory=lambda: {"BS": 0.0, "HAP": 20.0, "LEO": 60.0}
    )
