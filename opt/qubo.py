from __future__ import annotations

from itertools import combinations

from env.config import SimulationConfig
from env.mobility import euclidean_distance_3d
from env.models import APNode, CandidateScore, QuboProblem, Task


class LocalQuboBuilder:
    def __init__(self, config: SimulationConfig):
        self.config = config

    def build(
        self,
        ap: APNode,
        tasks: list[Task],
        ap_lookup: dict[str, APNode],
        slot: int,
    ) -> QuboProblem:
        variables: list[tuple[str, str]] = []
        linear: dict[tuple[str, str], float] = {}
        quadratic: dict[tuple[tuple[str, str], tuple[str, str]], float] = {}
        candidate_scores: list[CandidateScore] = []

        by_task: dict[str, list[tuple[str, str]]] = {}
        by_destination: dict[str, list[tuple[str, str]]] = {}

        for task in tasks:
            task_variables: list[tuple[str, str]] = []
            for destination_id in task.A_u_t:
                destination = ap_lookup[destination_id]
                score = self._score_assignment(task, destination, ap_lookup)
                variable = (task.task_id, destination_id)
                variables.append(variable)
                linear[variable] = score.local_cost + score.coupling_penalty - self.config.qubo_penalty
                candidate_scores.append(score)
                task_variables.append(variable)
                by_destination.setdefault(destination_id, []).append(variable)
            by_task[task.task_id] = task_variables

        for task_variables in by_task.values():
            for left, right in combinations(task_variables, 2):
                quadratic[(left, right)] = quadratic.get((left, right), 0.0) + 2.0 * self.config.qubo_penalty

        for destination_id, destination_variables in by_destination.items():
            destination = ap_lookup[destination_id]
            coupling_scale = self.config.qubo_load_coupling / max(destination.cpu_capacity, 1.0)
            for left, right in combinations(destination_variables, 2):
                task_left = self._task_by_id(tasks, left[0])
                task_right = self._task_by_id(tasks, right[0])
                cpu_pressure = task_left.cpu_demand + task_right.cpu_demand
                bandwidth_pressure = task_left.bandwidth_demand + task_right.bandwidth_demand
                power_pressure = task_left.power_demand + task_right.power_demand
                coupling = coupling_scale * cpu_pressure
                coupling += 0.08 * bandwidth_pressure / max(destination.communication_budget, 1.0)
                coupling += 0.08 * power_pressure / max(destination.power_budget, 1.0)
                quadratic[(left, right)] = quadratic.get((left, right), 0.0) + coupling

        return QuboProblem(
            ap_id=ap.ap_id,
            slot=slot,
            variables=variables,
            linear=linear,
            quadratic=quadratic,
            penalty_mu=self.config.qubo_penalty,
            candidate_scores=candidate_scores,
        )

    def _score_assignment(
        self,
        task: Task,
        destination: APNode,
        ap_lookup: dict[str, APNode],
    ) -> CandidateScore:
        owner_ap = ap_lookup[task.owner_ap_id]
        air_distance = euclidean_distance_3d(task.x, task.y, task.z, destination.x, destination.y, destination.z)
        owner_distance = euclidean_distance_3d(task.x, task.y, task.z, owner_ap.x, owner_ap.y, owner_ap.z)

        uplink_delay = task.L_u / max(destination.bandwidth, 1.0)
        compute_delay = task.D_u / max(destination.cpu_capacity, 1.0)
        backbone_delay = 0.03 * abs(air_distance - owner_distance)
        delay_cost = self.config.delay_weight * (uplink_delay + compute_delay + backbone_delay)

        energy_cost = self.config.energy_weight * (
            0.04 * task.L_u + 0.012 * air_distance + 0.02 * task.D_u / max(destination.cpu_capacity, 1.0)
        )

        projected_load = (destination.current_load + task.D_u / 4.0) / max(destination.cpu_capacity, 1.0)
        bandwidth_ratio = task.bandwidth_demand / max(destination.communication_budget, 1.0)
        power_ratio = task.power_demand / max(destination.power_budget, 1.0)
        mission_cost = self.config.mission_weight * (
            task.omega_u * delay_cost + task.psi_u * energy_cost + 0.20 * projected_load + 0.10 * bandwidth_ratio
        )
        fidelity_cost = self.config.fidelity_weight * (
            task.xi_u * (1.0 - destination.trust + destination.twin_state.uncertainty + 0.2 * destination.twin_state.age)
        )
        coupling_penalty = destination.coordination_state.coupling_penalty * (
            projected_load + bandwidth_ratio + power_ratio
        )
        local_cost = mission_cost + fidelity_cost

        return CandidateScore(
            task_id=task.task_id,
            owner_ap_id=task.owner_ap_id,
            destination_id=destination.ap_id,
            local_cost=local_cost,
            delay_cost=delay_cost,
            energy_cost=energy_cost,
            mission_cost=mission_cost,
            fidelity_cost=fidelity_cost,
            coupling_penalty=coupling_penalty,
            projected_load=projected_load,
            required_bandwidth=task.bandwidth_demand,
            required_cpu=task.cpu_demand,
            required_power=task.power_demand,
        )

    @staticmethod
    def _task_by_id(tasks: list[Task], task_id: str) -> Task:
        for task in tasks:
            if task.task_id == task_id:
                return task
        raise KeyError(f"Missing task {task_id}")
