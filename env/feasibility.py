from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .config import SimulationConfig
from .models import APNode, Task


@dataclass(frozen=True, slots=True)
class ResourceUsage:
    cpu_cycles: float = 0.0
    communication_bits: float = 0.0
    power_watts: float = 0.0


class FeasibilityManager:
    """Applies one common per-slot feasible set to every decision algorithm."""

    def __init__(self, config: SimulationConfig):
        self.config = config

    def task_demand(self, task: Task) -> ResourceUsage:
        bits = task.L_u * self.config.mbits_to_bits
        return ResourceUsage(
            cpu_cycles=bits * task.D_u,
            communication_bits=bits,
            power_watts=task.power_demand,
        )

    def capacity(self, ap: APNode) -> ResourceUsage:
        slot_seconds = self.config.slot_duration_seconds
        return ResourceUsage(
            cpu_cycles=ap.cpu_capacity * self.config.ghz_to_hz * slot_seconds,
            communication_bits=(
                ap.communication_budget
                * self.config.mbps_to_bits_per_second
                * slot_seconds
            ),
            power_watts=ap.power_budget,
        )

    def can_add(
        self,
        task: Task,
        ap: APNode,
        usage: ResourceUsage,
    ) -> bool:
        demand = self.task_demand(task)
        capacity = self.capacity(ap)
        return (
            usage.cpu_cycles + demand.cpu_cycles <= capacity.cpu_cycles + 1e-9
            and usage.communication_bits + demand.communication_bits
            <= capacity.communication_bits + 1e-9
            and usage.power_watts + demand.power_watts <= capacity.power_watts + 1e-9
        )

    def add(self, task: Task, usage: ResourceUsage) -> ResourceUsage:
        demand = self.task_demand(task)
        return ResourceUsage(
            cpu_cycles=usage.cpu_cycles + demand.cpu_cycles,
            communication_bits=usage.communication_bits + demand.communication_bits,
            power_watts=usage.power_watts + demand.power_watts,
        )

    def assignment_is_feasible(
        self,
        mapping: dict[str, str],
        tasks: list[Task],
        ap_lookup: dict[str, APNode],
    ) -> bool:
        task_map = {task.task_id: task for task in tasks}
        usage: dict[str, ResourceUsage] = defaultdict(ResourceUsage)
        for task_id, destination in mapping.items():
            if destination == "LOCAL":
                continue
            task = task_map[task_id]
            if destination not in ap_lookup:
                return False
            if not self.can_add(task, ap_lookup[destination], usage[destination]):
                return False
            usage[destination] = self.add(task, usage[destination])
        return True

    def repair(
        self,
        mapping: dict[str, str],
        tasks: list[Task],
        ap_lookup: dict[str, APNode],
        ranked_candidates: dict[str, list[str]] | None = None,
    ) -> dict[str, str]:
        usage: dict[str, ResourceUsage] = defaultdict(ResourceUsage)
        repaired: dict[str, str] = {}
        for task in tasks:
            preferred = mapping.get(task.task_id, "LOCAL")
            candidates = [preferred]
            if ranked_candidates is not None:
                candidates.extend(ranked_candidates.get(task.task_id, []))
            candidates.extend(task.A_u_t)

            selected = "LOCAL"
            for destination in dict.fromkeys(candidates):
                if destination == "LOCAL" or destination not in ap_lookup:
                    continue
                if self.can_add(task, ap_lookup[destination], usage[destination]):
                    selected = destination
                    usage[destination] = self.add(task, usage[destination])
                    break
            repaired[task.task_id] = selected
        return repaired
