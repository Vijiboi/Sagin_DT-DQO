from __future__ import annotations

from collections import defaultdict

from .models import CandidateScore, FinalAssignment, APNode

def greedy_one_hot_assignment(scores: list[CandidateScore], ap_lookup: dict[str, APNode]) -> list[FinalAssignment]:
    by_task: dict[str, list[CandidateScore]] = defaultdict(list)
    for score in scores:
        by_task[score.task_id].append(score)
    
    assignments: list[FinalAssignment] = []
    # Track usage locally during this fallback loop
    temp_usage = defaultdict(lambda: {"cpu": 0.0, "bandwidth": 0.0, "power": 0.0})

    for task_id, task_scores in by_task.items():
        # Sort candidates by cost
        sorted_candidates = sorted(task_scores, key=lambda item: item.local_cost)
        
        for best in sorted_candidates:
            dest = ap_lookup[best.destination_id]
            usage = temp_usage[best.destination_id]
            
            # Check if this AP can actually take the task
            cpu_ok = dest.current_cpu_load + usage["cpu"] + best.required_cpu <= dest.cpu_capacity
            if cpu_ok:
                usage["cpu"] += best.required_cpu
                assignments.append(FinalAssignment(
                    task_id=best.task_id,
                    owner_ap_id=best.owner_ap_id,
                    destination_id=best.destination_id,
                    local_cost=best.local_cost,
                    delay_cost=best.delay_cost,
                    energy_cost=best.energy_cost,
                    mission_cost=best.mission_cost,
                    fidelity_cost=best.fidelity_cost
                ))
                break 
    return assignments