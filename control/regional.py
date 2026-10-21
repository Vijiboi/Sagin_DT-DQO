from __future__ import annotations

from collections import defaultdict

from env.config import SimulationConfig
from env.evaluator import PhysicalEvaluator
from env.models import APNode, CandidateScore, FinalAssignment, LocalSummary, Task


class RegionalController:
    """Task-wise fidelity-admissible regional selection from Eqs. (42)-(44)."""

    def __init__(
        self,
        strategy: str = "hierarchical",
        config: SimulationConfig | None = None,
    ):
        self.strategy = strategy
        self.config = config or SimulationConfig()
        self.evaluator = PhysicalEvaluator(self.config)

    def project(
        self,
        local_summaries: list[LocalSummary],
        ap_lookup: dict[str, APNode],
        tasks: list[Task],
    ) -> list[FinalAssignment]:
        scores_by_task: dict[str, list[CandidateScore]] = defaultdict(list)
        for summary in local_summaries:
            for score in summary.candidate_scores:
                scores_by_task[score.task_id].append(score)

        sync_ap_ids = {
            summary.ap_id for summary in local_summaries if summary.sync_triggered
        }
        coord_ap_ids = {
            summary.ap_id
            for summary in local_summaries
            if summary.coordination_triggered
        }
        # Eqs. (42)-(44): score each task at the final local positive load,
        # retain only fidelity-admissible APs, and select the deterministic
        # minimum. Local execution is always included as destination 0.
        selected: dict[str, str] = {}
        for task in tasks:
            eligible_scores = [
                score for score in scores_by_task.get(task.task_id, [])
                if score.destination_id in task.A_u_t
                and ap_lookup[score.destination_id].twin_state.fidelity >= task.F_u_min
            ]
            choices = [
                (float(score.local_cost), score.destination_id)
                for score in eligible_scores
            ]
            local_metrics = self.evaluator.evaluate_local(task)
            choices.append((
                self.evaluator.service_cost(local_metrics, include_mission=False),
                "LOCAL",
            ))
            selected[task.task_id] = min(choices, key=lambda item: (item[0], item[1]))[1]

        return self.materialize_mapping(
            selected,
            tasks,
            ap_lookup,
            sync_ap_ids,
            coord_ap_ids,
        )

    def materialize_mapping(
        self,
        mapping: dict[str, str],
        tasks: list[Task],
        ap_lookup: dict[str, APNode],
        sync_ap_ids: set[str] | None = None,
        coord_ap_ids: set[str] | None = None,
    ) -> list[FinalAssignment]:
        """Evaluate every algorithm's mapping under one full outcome model."""
        sync_ap_ids = sync_ap_ids or set()
        coord_ap_ids = coord_ap_ids or set()
        task_map = {task.task_id: task for task in tasks}
        assigned_per_ap: dict[str, int] = defaultdict(int)
        for task in tasks:
            destination = mapping.get(task.task_id, "LOCAL")
            if destination in ap_lookup:
                assigned_per_ap[destination] += 1

        assignments: list[FinalAssignment] = []
        for task in tasks:
            destination = mapping.get(task.task_id, "LOCAL")
            if destination not in ap_lookup:
                assignments.append(self._local_execution_assignment(task))
                continue

            ap = ap_lookup[destination]
            metrics = self.evaluator.evaluate_remote(
                task,
                ap,
                assigned_per_ap[destination],
                sync_active=destination in sync_ap_ids,
                coord_active=destination in coord_ap_ids,
            )
            assignments.append(
                FinalAssignment(
                    task_id=task.task_id,
                    owner_ap_id=task.owner_ap_id,
                    destination_id=destination,
                    local_cost=self.evaluator.proposed_cost(metrics),
                    delay_cost=metrics.total_delay_s,
                    energy_cost=metrics.total_energy_j,
                    mission_cost=metrics.freshness_risk,
                    fidelity_cost=metrics.fidelity_penalty,
                )
            )

        assert len(assignments) == len(task_map)
        return assignments

    def _local_execution_assignment(self, task: Task) -> FinalAssignment:
        metrics = self.evaluator.evaluate_local(task)
        return FinalAssignment(
            task_id=task.task_id,
            owner_ap_id=task.owner_ap_id,
            destination_id="LOCAL",
            local_cost=self.evaluator.service_cost(metrics, include_mission=False),
            delay_cost=metrics.total_delay_s,
            energy_cost=metrics.total_energy_j,
            mission_cost=metrics.freshness_risk,
            fidelity_cost=metrics.fidelity_penalty,
        )

