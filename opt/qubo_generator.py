from __future__ import annotations

from env.config import SimulationConfig
from env.evaluator import PhysicalEvaluator
from env.feasibility import FeasibilityManager
from env.models import APNode, CandidateScore, QuboProblem, Task


class LocalQuboBuilder:
    """Build the paper's local assignment QUBO over task-destination binaries."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.evaluator = PhysicalEvaluator(config)
        self.feasibility = FeasibilityManager(config)

    def build(
        self,
        ap: APNode,
        tasks: list[Task],
        ap_lookup: dict[str, APNode],
        slot: int,
        sync_ap_ids: set[str] | None = None,
        coord_ap_ids: set[str] | None = None,
        *,
        reference_load: float = 1.0,
        reference_summary: tuple[float, float, float, float, float] | None = None,
        multiplier: tuple[float, float, float, float, float] | None = None,
    ) -> QuboProblem:
        sync_ap_ids = sync_ap_ids or set()
        coord_ap_ids = coord_ap_ids or set()
        # A node's local binary vector contains one variable for each task
        # that may be served by this node and passes its fidelity gate.
        local_tasks = [
            task for task in tasks
            if ap.ap_id in task.A_u_t and ap.twin_state.fidelity >= task.F_u_min
        ]
        scores = self.score_candidates(
            ap,
            local_tasks,
            ap_lookup,
            sync_ap_ids=sync_ap_ids,
            coord_ap_ids=coord_ap_ids,
            load_override=reference_load,
        )
        scores = [score for score in scores if score.destination_id == ap.ap_id]
        variables = [
            (score.task_id, score.destination_id)
            for score in scores
        ]
        rho = self.config.qubo_penalty
        reference_summary = reference_summary or ap.coordination_state.reported_summary
        multiplier = multiplier or ap.coordination_state.coordination_multiplier
        # The active assignment terms in z-tilde are load and trust-weighted
        # mission pressure. Other coordinates are fixed for this slot.
        rho_trust = 1.0 - ap.trust
        zbar_load = reference_summary[0] - multiplier[0] / max(rho, 1e-12)
        zbar_risk = reference_summary[1] - multiplier[1] / max(rho, 1e-12)
        linear: dict[tuple[str, str], float] = {}
        quadratic: dict[
            tuple[tuple[str, str], tuple[str, str]],
            float,
        ] = {}

        # Eq. (30): first-order service-cost approximation at N-bar(k).
        # B is evaluated with a centered finite difference of the shared
        # physical objective; A = C(Nbar)-Nbar*C'(Nbar).
        by_task = {task.task_id: task for task in local_tasks}
        candidate_by_task = {score.task_id: score for score in scores}
        nbar = max(float(reference_load), 1.0)
        for task_id, task in by_task.items():
            score = candidate_by_task[task_id]
            derivative = self._service_cost_derivative(
                task, ap, nbar, sync_active=ap.ap_id in sync_ap_ids,
                coord_active=ap.ap_id in coord_ap_ids,
            )
            a_value = score.local_cost - nbar * derivative
            weight = rho_trust * task.psi_u
            variable = (task_id, ap.ap_id)
            linear[variable] = (
                a_value + derivative
                + 0.5 * rho
                - rho * zbar_load
                + 0.5 * rho * weight * weight
                - rho * zbar_risk * weight
            )

        task_variables = list(variables)
        for i, left in enumerate(task_variables):
            left_task = by_task[left[0]]
            left_b = self._service_cost_derivative(
                left_task, ap, nbar, sync_active=ap.ap_id in sync_ap_ids,
                coord_active=ap.ap_id in coord_ap_ids,
            )
            for right in task_variables[i + 1:]:
                right_task = by_task[right[0]]
                right_b = self._service_cost_derivative(
                    right_task, ap, nbar, sync_active=ap.ap_id in sync_ap_ids,
                    coord_active=ap.ap_id in coord_ap_ids,
                )
                right_weight = rho_trust * right_task.psi_u
                left_weight = rho_trust * left_task.psi_u
                # Eq. (34), using the symmetric Q convention Q_uu=0.
                quadratic[(left, right)] = (
                    0.5 * (left_b + right_b)
                    + 0.5 * rho
                    + 0.5 * rho * left_weight * right_weight
                )

        return QuboProblem(
            ap_id=ap.ap_id,
            slot=slot,
            local_load=nbar,
            variables=variables,
            linear=linear,
            quadratic=quadratic,
            penalty_mu=rho,
            penalty_by_task={task.task_id: rho for task in local_tasks},
            candidate_scores=scores,
            reference_load=nbar,
            reference_summary=reference_summary,
        )

    def _service_cost_derivative(
        self, task: Task, ap: APNode, load: float, *,
        sync_active: bool, coord_active: bool,
    ) -> float:
        # The sync/coordination terms are slot-fixed; their derivative is zero.
        return self.evaluator.service_cost_derivative(task, ap, load)

    def score_candidates(
        self,
        owner_ap: APNode,
        tasks: list[Task],
        ap_lookup: dict[str, APNode],
        preferred_pairs: set[tuple[str, str]] | None = None,
        sync_ap_ids: set[str] | None = None,
        coord_ap_ids: set[str] | None = None,
        load_override: float | None = None,
    ) -> list[CandidateScore]:
        sync_ap_ids = sync_ap_ids or set()
        coord_ap_ids = coord_ap_ids or set()
        scores: list[CandidateScore] = []
        for task in tasks:
            for destination_id in task.A_u_t:
                destination = ap_lookup[destination_id]
                load_count = max(float(load_override if load_override is not None else destination.current_task_load + 1), 1.0)
                metrics = self.evaluator.evaluate_remote(
                    task,
                    destination,
                    load_count,
                    sync_active=destination_id in sync_ap_ids,
                    coord_active=destination_id in coord_ap_ids,
                )
                guidance_penalty = (
                    self.config.qubo_guidance_penalty
                    if preferred_pairs is not None
                    and (task.task_id, destination_id) not in preferred_pairs
                    else 0.0
                )
                demand = self.feasibility.task_demand(task)
                scores.append(
                    CandidateScore(
                        task_id=task.task_id,
                        owner_ap_id=owner_ap.ap_id,
                        destination_id=destination_id,
                        local_cost=self.evaluator.service_cost(metrics),
                        delay_cost=metrics.total_delay_s,
                        energy_cost=metrics.total_energy_j,
                        mission_cost=metrics.freshness_risk,
                        fidelity_cost=metrics.fidelity_penalty,
                        coupling_penalty=(
                            destination.coordination_state.coupling_penalty
                            + guidance_penalty
                        ),
                        projected_load=load_count,
                        required_bandwidth=(
                            demand.communication_bits
                            / max(
                                self.config.slot_duration_seconds
                                * self.config.mbps_to_bits_per_second,
                                1e-12,
                            )
                        ),
                        required_cpu=(
                            demand.cpu_cycles
                            / max(
                                self.config.slot_duration_seconds
                                * self.config.ghz_to_hz,
                                1e-12,
                            )
                        ),
                        required_power=demand.power_watts,
                    )
                )
        return scores
