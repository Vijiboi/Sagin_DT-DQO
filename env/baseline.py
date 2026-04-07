from __future__ import annotations

from collections import defaultdict

from .models import CandidateScore, FinalAssignment


def greedy_one_hot_assignment(
    scores: list[CandidateScore],
) -> list[FinalAssignment]:
    by_task: dict[str, list[CandidateScore]] = defaultdict(list)
    for score in scores:
        by_task[score.task_id].append(score)

    assignments: list[FinalAssignment] = []
    for task_scores in by_task.values():
        best = min(task_scores, key=lambda item: item.local_cost)
        assignments.append(
            FinalAssignment(
                task_id=best.task_id,
                owner_ap_id=best.owner_ap_id,
                destination_id=best.destination_id,
                local_cost=best.local_cost,
                delay_cost=best.delay_cost,
                energy_cost=best.energy_cost,
                mission_cost=best.mission_cost,
                fidelity_cost=best.fidelity_cost,
            )
        )
    return assignments
