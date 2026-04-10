from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class SimulationConfig:
    seed: int = 7
    slots: int = 8
    area_width: float = 100.0
    area_height: float = 100.0
    candidate_ap_limit: int = 10
    num_bs: int = 4
    num_haps: int = 2
    num_leos: int = 1
    num_uavs: int = 50
    task_arrival_probability: float = 0.4 #to be adjusted based on desired load
    sync_age_threshold: int = 2
    sync_mismatch_threshold: float = 0.22
    sync_uncertainty_threshold: float = 0.18
    coordination_load_threshold: float = 0.75
    coordination_overlap_threshold: float = 0.85
    coordination_trust_threshold: float = 0.45
    twin_smoothing: float = 0.65
    qubo_penalty: float = 40.0
    qubo_load_coupling: float = 1.0
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
    ap_trust_init_min: float = 0.70
    ap_trust_init_max: float = 0.95
    ap_initial_predicted_bandwidth: float = 0.75
    uav_altitude_min: float = 8.0
    uav_altitude_max: float = 20.0
    uav_max_altitude: float = 25.0
    uav_velocity_xy_min: float = -3.0
    uav_velocity_xy_max: float = 3.0
    uav_velocity_z_min: float = -0.8
    uav_velocity_z_max: float = 0.8
    observation_ratio_clip: float | None = 1.5
    cpu_capacity_by_tier: dict[str, float] = field(
        default_factory=lambda: {"BS": 10.0, "UAV": 10.0, "HAP": 20.0, "LEO": 30.0}
    )
    bandwidth_by_tier: dict[str, float] = field(
        default_factory=lambda: {"BS": 20.0, "UAV": 20.0, "HAP": 100.0, "LEO": 500.0}
    )
    communication_budget_by_tier: dict[str, float] = field(
        default_factory=lambda: {"BS": 24.0, "UAV": 15.0, "HAP": 42.0, "LEO": 60.0}
    )
    power_budget_by_tier: dict[str, float] = field(
        default_factory=lambda: {"BS": 18.0, "UAV": 10.0, "HAP": 28.0, "LEO": 36.0}
    )
    altitude_by_tier: dict[str, float] = field(
        default_factory=lambda: {"BS": 0.03,"UAV": 0.1, "HAP": 20.0, "LEO": 550.0}
    )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_report_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "simulation_horizon_slots": self.slots,
            "network_topology": {
                "num_bs": self.num_bs,
                "num_haps": self.num_haps,
                "num_leos": self.num_leos,
                "num_uavs": self.num_uavs,
                "candidate_ap_limit": self.candidate_ap_limit,
                "area_width": self.area_width,
                "area_height": self.area_height,
                "altitude_by_tier": self.altitude_by_tier,
            },
            "task_generation": {
                "task_arrival_probability": self.task_arrival_probability,
                "randomize_task_weights": self.randomize_task_weights,
                "uav_altitude_min": self.uav_altitude_min,
                "uav_altitude_max": self.uav_altitude_max,
                "uav_max_altitude": self.uav_max_altitude,
                "uav_velocity_xy_min": self.uav_velocity_xy_min,
                "uav_velocity_xy_max": self.uav_velocity_xy_max,
                "uav_velocity_z_min": self.uav_velocity_z_min,
                "uav_velocity_z_max": self.uav_velocity_z_max,
            },
            "tier_resources": {
                "bandwidth_by_tier": self.bandwidth_by_tier,
                "communication_budget_by_tier": self.communication_budget_by_tier,
                "cpu_capacity_by_tier": self.cpu_capacity_by_tier,
                "power_budget_by_tier": self.power_budget_by_tier,
            },
            "twin_and_trust": {
                "sync_age_threshold": self.sync_age_threshold,
                "sync_mismatch_threshold": self.sync_mismatch_threshold,
                "sync_uncertainty_threshold": self.sync_uncertainty_threshold,
                "coordination_load_threshold": self.coordination_load_threshold,
                "coordination_overlap_threshold": self.coordination_overlap_threshold,
                "coordination_trust_threshold": self.coordination_trust_threshold,
                "twin_smoothing": self.twin_smoothing,
                "twin_gaussian_std": self.twin_gaussian_std,
                "ap_trust_init_min": self.ap_trust_init_min,
                "ap_trust_init_max": self.ap_trust_init_max,
                "ap_initial_predicted_bandwidth": self.ap_initial_predicted_bandwidth,
                "observation_ratio_clip": self.observation_ratio_clip,
            },
            "qubo_and_solver": {
                "qubo_penalty": self.qubo_penalty,
                "qubo_load_coupling": self.qubo_load_coupling,
                "anneal_reads": self.anneal_reads,
                "anneal_sweeps": self.anneal_sweeps,
            },
            "regional_coordination": {
                "consensus_step_size": self.consensus_step_size,
                "consensus_quantum": self.consensus_quantum,
                "consensus_epsilon": self.consensus_epsilon,
            },
            "objective_weights": {
                "delay_weight": self.delay_weight,
                "energy_weight": self.energy_weight,
                "mission_weight": self.mission_weight,
                "fidelity_weight": self.fidelity_weight,
            },
        }

    def to_paper_parameter_view(self) -> dict[str, object]:
        return {
            "BS Bandwidth": self.bandwidth_by_tier["BS"],
            "HAP Bandwidth": self.bandwidth_by_tier["HAP"],
            "LEO Bandwidth": self.bandwidth_by_tier["LEO"],
            "BS CPU Capacity": self.cpu_capacity_by_tier["BS"],
            "HAP CPU Capacity": self.cpu_capacity_by_tier["HAP"],
            "LEO CPU Capacity": self.cpu_capacity_by_tier["LEO"],
            "Number of BSs": self.num_bs,
            "Number of HAPs": self.num_haps,
            "Number of LEO Satellites": self.num_leos,
            "Number of UAV Tasks": self.num_uavs,
            "AP Initial Trust Range": [self.ap_trust_init_min, self.ap_trust_init_max],
            "UAV Altitude Range": [self.uav_altitude_min, self.uav_altitude_max],
            "UAV Mobility Max Altitude": self.uav_max_altitude,
            "Observation Ratio Clip": self.observation_ratio_clip,
            "Twin-Age Sync Threshold": self.sync_age_threshold,
            "Sync Uncertainty Threshold": self.sync_uncertainty_threshold,
            "Sync Mismatch Threshold": self.sync_mismatch_threshold,
            "Trust Trigger Threshold": self.coordination_trust_threshold,
            "Application-Pressure Threshold": self.coordination_load_threshold,
            "Distributed Backend Calls per Slot": 1,
            "Centralized Reference Solver": "Classical QUBO solver",
            "Classical Annealing Reads per AP": self.anneal_reads,
            "Classical Annealing Sweeps per AP": self.anneal_sweeps,
            "Simulation Horizon": self.slots,
        }
