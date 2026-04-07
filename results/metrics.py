from __future__ import annotations

from collections import defaultdict

from env.models import FinalAssignment, LocalSummary, SlotResult


def build_slot_result(
    slot: int,
    assignments: list[FinalAssignment],
    local_summaries: list[LocalSummary],
    resource_feasible: bool,
) -> SlotResult:
    count = max(len(assignments), 1)
    average_delay = sum(item.delay_cost for item in assignments) / count
    average_energy = sum(item.energy_cost for item in assignments) / count
    mission_cost = sum(item.mission_cost for item in assignments)
    fidelity_cost = sum(item.fidelity_cost for item in assignments)
    sync_trigger_count = sum(summary.sync_triggered for summary in local_summaries)
    coordination_trigger_count = sum(summary.coordination_triggered for summary in local_summaries)
    tasks_per_ap: dict[str, int] = defaultdict(int)
    local_execution_count = 0

    for assignment in assignments:
        tasks_per_ap[assignment.destination_id] += 1
        if assignment.destination_id == assignment.owner_ap_id:
            local_execution_count += 1

    one_hot_valid = len({item.task_id for item in assignments}) == len(assignments)
    return SlotResult(
        slot=slot,
        assignments=assignments,
        local_summaries=local_summaries,
        average_delay=average_delay,
        average_energy=average_energy,
        mission_cost=mission_cost,
        fidelity_cost=fidelity_cost,
        sync_trigger_count=sync_trigger_count,
        coordination_trigger_count=coordination_trigger_count,
        one_hot_valid=one_hot_valid,
        resource_feasible=resource_feasible,
        tasks_per_ap=dict(tasks_per_ap),
        local_execution_count=local_execution_count,
    )
