from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class UAVAgent:
    uav_id: str
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float


@dataclass(slots=True)
class TwinState:
    predicted_load: float = 0.0
    predicted_bandwidth: float = 0.0
    predicted_cpu_ratio: float = 0.0
    age: int = 0
    uncertainty: float = 0.0
    mismatch: float = 0.0
    fidelity: float = 1.0
    last_observation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    forecast_state: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(slots=True)
class CoordinationState:
    dual_price: float = 0.0
    quantized_dual: float = 0.0
    coupling_penalty: float = 0.0
    consensus_drift: float = 0.0
    last_broadcast_slot: int = -1


@dataclass(slots=True)
class APNode:
    ap_id: str
    tier: str
    x: float
    y: float
    z: float
    bandwidth: float
    cpu_capacity: float
    communication_budget: float
    power_budget: float
    trust: float
    sync_threshold: float
    coord_threshold: float
    twin_state: TwinState = field(default_factory=TwinState)
    coordination_state: CoordinationState = field(default_factory=CoordinationState)
    current_task_load: int = 0
    current_cpu_load: float = 0.0


@dataclass(slots=True)
class Task:
    task_id: str
    source_uav: str
    owner_ap_id: str
    x: float
    y: float
    z: float
    L_u: float
    D_u: float
    omega_u: float
    psi_u: float
    xi_u: float
    bandwidth_demand: float
    cpu_demand: float
    power_demand: float
    A_u_t: list[str]
    arrival_slot: int


@dataclass(slots=True)
class Observation:
    ap_id: str
    slot: int
    load_ratio: float
    bandwidth_ratio: float
    cpu_ratio: float
    queue_size: int
    candidate_overlap: float

    @property
    def vector(self) -> tuple[float, float, float, float, float]:
        return (
            self.load_ratio,
            self.bandwidth_ratio,
            self.cpu_ratio,
            self.queue_size,
            self.candidate_overlap,
        )


@dataclass(slots=True)
class CandidateScore:
    task_id: str
    owner_ap_id: str
    destination_id: str
    local_cost: float
    delay_cost: float
    energy_cost: float
    mission_cost: float
    fidelity_cost: float
    coupling_penalty: float
    projected_load: float
    required_bandwidth: float
    required_cpu: float
    required_power: float


@dataclass(slots=True)
class QuboProblem:
    ap_id: str
    slot: int
    local_load: float
    variables: list[tuple[str, str]]
    linear: dict[tuple[str, str], float]
    quadratic: dict[tuple[tuple[str, str], tuple[str, str]], float]
    penalty_mu: float
    penalty_by_task: dict[str, float]
    candidate_scores: list[CandidateScore]


@dataclass(slots=True)
class SolveResult:
    sample: dict[tuple[str, str], int]
    energy: float
    samples: list[dict[tuple[str, str], int]]
    solver_name: str
    solver_time: float


@dataclass(slots=True)
class LocalSummary:
    ap_id: str
    slot: int
    queue_size: int
    local_load: float
    sync_triggered: bool
    coordination_triggered: bool
    trust: float
    twin_age: int
    uncertainty: float
    mismatch: float
    fidelity: float
    qubo_dimension: int
    solver_time: float
    selected_pairs: list[tuple[str, str]]
    candidate_scores: list[CandidateScore]


@dataclass(slots=True)
class FinalAssignment:
    task_id: str
    owner_ap_id: str
    destination_id: str
    local_cost: float
    delay_cost: float
    energy_cost: float
    mission_cost: float
    fidelity_cost: float


@dataclass(slots=True)
class SlotResult:
    slot: int
    assignments: list[FinalAssignment]
    local_summaries: list[LocalSummary]
    average_delay: float
    average_energy: float
    mission_cost: float
    fidelity_cost: float
    sync_trigger_count: int
    coordination_trigger_count: int
    one_hot_valid: bool
    resource_feasible: bool
    tasks_per_ap: dict[str, int]
    local_execution_count: int
