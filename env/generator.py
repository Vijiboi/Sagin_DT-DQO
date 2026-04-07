from __future__ import annotations

from collections import defaultdict
from random import Random

from .config import SimulationConfig
from .graph import build_communication_graph
from .mobility import ap_distance, bounce_update
from .models import APNode, Observation, Task, TwinState, UAVAgent


class SaginEnvironment:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.rng = Random(config.seed)
        self.aps = self._generate_aps()
        self.communication_graph = build_communication_graph(self.aps)
        self.uavs = self._generate_uavs()

    def _generate_aps(self) -> list[APNode]:
        aps: list[APNode] = []
        tier_counts = {
            "BS": self.config.num_bs,
            "HAP": self.config.num_haps,
            "LEO": self.config.num_leos,
        }
        for tier, count in tier_counts.items():
            for index in range(count):
                ap_id = f"{tier}_{index}"
                aps.append(
                    APNode(
                        ap_id=ap_id,
                        tier=tier,
                        x=self.rng.uniform(0.0, self.config.area_width),
                        y=self.rng.uniform(0.0, self.config.area_height),
                        z=self.config.altitude_by_tier[tier],
                        bandwidth=self.config.bandwidth_by_tier[tier],
                        cpu_capacity=self.config.cpu_capacity_by_tier[tier],
                        communication_budget=self.config.communication_budget_by_tier[tier],
                        power_budget=self.config.power_budget_by_tier[tier],
                        trust=self.rng.uniform(0.70, 0.95),
                        sync_threshold=self.config.sync_mismatch_threshold,
                        coord_threshold=self.config.coordination_load_threshold,
                        twin_state=TwinState(
                            predicted_load=0.0,
                            predicted_bandwidth=0.75,
                            predicted_cpu_ratio=0.0,
                        ),
                    )
                )
        return aps

    def _generate_uavs(self) -> list[UAVAgent]:
        uavs: list[UAVAgent] = []
        for index in range(self.config.num_uavs):
            uavs.append(
                UAVAgent(
                    uav_id=f"UAV_{index}",
                    x=self.rng.uniform(0.0, self.config.area_width),
                    y=self.rng.uniform(0.0, self.config.area_height),
                    z=self.rng.uniform(8.0, 20.0),
                    vx=self.rng.uniform(-3.0, 3.0),
                    vy=self.rng.uniform(-3.0, 3.0),
                    vz=self.rng.uniform(-0.8, 0.8),
                )
            )
        return uavs

    def step_mobility(self) -> None:
        for uav in self.uavs:
            bounce_update(uav, self.config.area_width, self.config.area_height, 25.0)

    def ap_by_id(self) -> dict[str, APNode]:
        return {ap.ap_id: ap for ap in self.aps}

    def candidate_aps_for_uav(self, uav: UAVAgent) -> list[str]:
        ordered = sorted(self.aps, key=lambda ap: ap_distance(uav, ap))
        return [ap.ap_id for ap in ordered[: self.config.candidate_ap_limit]]

    def serving_ap_for_uav(self, uav: UAVAgent) -> str:
        return self.candidate_aps_for_uav(uav)[0]

    def create_tasks_for_slot(self, slot: int) -> list[Task]:
        tasks: list[Task] = []
        for uav in self.uavs:
            if self.rng.random() > self.config.task_arrival_probability:
                continue
            tasks.append(
                Task(
                    task_id=f"T{slot}_{uav.uav_id}",
                    source_uav=uav.uav_id,
                    owner_ap_id=self.serving_ap_for_uav(uav),
                    x=uav.x,
                    y=uav.y,
                    z=uav.z,
                    L_u=self.rng.uniform(5.0, 18.0),
                    D_u=self.rng.uniform(4.0, 14.0),
                    omega_u=self.rng.uniform(0.8, 1.4) if self.config.randomize_task_weights else 1.0,
                    psi_u=self.rng.uniform(0.7, 1.3) if self.config.randomize_task_weights else 1.0,
                    xi_u=self.rng.uniform(0.5, 1.2) if self.config.randomize_task_weights else 1.0,
                    bandwidth_demand=self.rng.uniform(1.0, 5.0),
                    cpu_demand=self.rng.uniform(2.0, 8.0),
                    power_demand=self.rng.uniform(0.8, 3.5),
                    A_u_t=self.candidate_aps_for_uav(uav),
                    arrival_slot=slot,
                )
            )
        return tasks

    def group_tasks_by_owner(self, tasks: list[Task]) -> dict[str, list[Task]]:
        grouped: dict[str, list[Task]] = defaultdict(list)
        for task in tasks:
            grouped[task.owner_ap_id].append(task)
        return grouped

    def build_observation(
        self,
        ap: APNode,
        slot: int,
        queue: list[Task],
        total_tasks: int,
    ) -> Observation:
        queue_size = len(queue)
        load_ratio = min(queue_size / max(ap.cpu_capacity, 1.0), 1.5)
        bandwidth_ratio = min(ap.bandwidth / 50.0, 1.0)
        cpu_ratio = min((queue_size * 2.0) / max(ap.cpu_capacity, 1.0), 1.5)
        overlap = 0.0
        if total_tasks:
            overlap = sum(len(task.A_u_t) > 1 for task in queue) / max(queue_size, 1)
        return Observation(
            ap_id=ap.ap_id,
            slot=slot,
            load_ratio=load_ratio,
            bandwidth_ratio=bandwidth_ratio,
            cpu_ratio=cpu_ratio,
            queue_size=queue_size,
            candidate_overlap=overlap,
        )
