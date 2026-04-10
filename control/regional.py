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
        remaining_tasks = set(all_scores_by_task)
        while remaining_tasks:
            ranked_tasks = []
            for task_id in remaining_tasks:
                ranked_scores = self._rank_scores(
                    proposed_by_task.get(task_id, []),
                    all_scores_by_task[task_id],
                )
                feasible_scores = [
                    score for score in ranked_scores if self._is_feasible(score, ap_lookup, resource_usage)
                ]
                best_score = feasible_scores[0] if feasible_scores else ranked_scores[0]
                demand = best_score.required_cpu + best_score.required_bandwidth + best_score.required_power
                ranked_tasks.append(
                    (
                        len(feasible_scores),
                        -demand,
                        best_score.local_cost + best_score.coupling_penalty,
                        task_id,
                        feasible_scores,
                        ranked_scores,
                    )
                )

            _, _, _, task_id, feasible_scores, ranked_scores = min(ranked_tasks)
            winner = feasible_scores[0] if feasible_scores else ranked_scores[0]

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
            remaining_tasks.remove(task_id)

        if not final_assignments and all_scores:
            return greedy_one_hot_assignment(all_scores, ap_lookup)
        return final_assignments

    @staticmethod
    def validate_one_hot(assignments: list[FinalAssignment]) -> bool:
        task_ids = [assignment.task_id for assignment in assignments]
        return len(task_ids) == len(set(task_ids))

    @staticmethod
    def _rank_scores(proposals, task_scores):
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
        return ranked

    @staticmethod
    def _is_feasible(score, ap_lookup, resource_usage):
        ap = ap_lookup[score.destination_id]
        usage = resource_usage[score.destination_id]
        cpu_ok = usage["cpu"] + score.required_cpu <= ap.cpu_capacity
        bandwidth_ok = usage["bandwidth"] + score.required_bandwidth <= ap.communication_budget
        power_ok = usage["power"] + score.required_power <= ap.power_budget
        return cpu_ok and bandwidth_ok and power_ok
