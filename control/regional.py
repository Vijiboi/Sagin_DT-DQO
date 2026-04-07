from __future__ import annotations

from collections import defaultdict

from env.baseline import greedy_one_hot_assignment
from env.models import CandidateScore, FinalAssignment, LocalSummary


class RegionalController:
    def project(
        self,
        local_summaries: list[LocalSummary],
        ap_lookup,
    ) -> list[FinalAssignment]:
        all_scores: list[CandidateScore] = []
        proposed_by_task: dict[str, list[CandidateScore]] = defaultdict(list)

        for summary in local_summaries:
            score_map = {(score.task_id, score.destination_id): score for score in summary.candidate_scores}
            all_scores.extend(summary.candidate_scores)
            for pair in summary.selected_pairs:
                if pair in score_map:
                    proposed_by_task[pair[0]].append(score_map[pair])

        all_scores_by_task: dict[str, list[CandidateScore]] = defaultdict(list)
        for score in all_scores:
            all_scores_by_task[score.task_id].append(score)

        final_assignments: list[FinalAssignment] = []
        resource_usage = {
            ap_id: {"cpu": 0.0, "bandwidth": 0.0, "power": 0.0}
            for ap_id in ap_lookup
        }
        for task_id, task_scores in all_scores_by_task.items():
            proposals = proposed_by_task.get(task_id) or []
            ranked = sorted(
                proposals if proposals else task_scores,
                key=lambda score: score.local_cost + score.coupling_penalty + 0.1 * score.projected_load,
            )
            if proposals:
                fallback = sorted(
                    task_scores,
                    key=lambda score: score.local_cost + score.coupling_penalty + 0.1 * score.projected_load,
                )
                for score in fallback:
                    if score not in ranked:
                        ranked.append(score)

            winner = self._first_feasible(ranked, ap_lookup, resource_usage)
            if winner is None:
                winner = min(task_scores, key=lambda score: score.local_cost + score.coupling_penalty)

            usage = resource_usage[winner.destination_id]
            usage["cpu"] += winner.required_cpu
            usage["bandwidth"] += winner.required_bandwidth
            usage["power"] += winner.required_power
            final_assignments.append(
                FinalAssignment(
                    task_id=winner.task_id,
                    owner_ap_id=winner.owner_ap_id,
                    destination_id=winner.destination_id,
                    local_cost=winner.local_cost,
                    delay_cost=winner.delay_cost,
                    energy_cost=winner.energy_cost,
                    mission_cost=winner.mission_cost,
                    fidelity_cost=winner.fidelity_cost,
                )
            )

        if not final_assignments and all_scores:
            return greedy_one_hot_assignment(all_scores)
        return final_assignments

    @staticmethod
    def validate_one_hot(assignments: list[FinalAssignment]) -> bool:
        task_ids = [assignment.task_id for assignment in assignments]
        return len(task_ids) == len(set(task_ids))

    @staticmethod
    def _first_feasible(ranked_scores, ap_lookup, resource_usage):
        for score in ranked_scores:
            ap = ap_lookup[score.destination_id]
            usage = resource_usage[score.destination_id]
            cpu_ok = usage["cpu"] + score.required_cpu <= ap.cpu_capacity
            bandwidth_ok = usage["bandwidth"] + score.required_bandwidth <= ap.communication_budget
            power_ok = usage["power"] + score.required_power <= ap.power_budget
            if cpu_ok and bandwidth_ok and power_ok:
                return score
        return None
