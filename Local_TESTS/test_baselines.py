from __future__ import annotations

import unittest

from env.baseline import BaselineController
from env.config import SimulationConfig
from env.models import APNode, Observation, Task
from twin.twin_logic import TwinManager


def make_ap(ap_id: str, bandwidth: float, cpu_capacity: float = 10.0) -> APNode:
    return APNode(
        ap_id=ap_id,
        tier="BS",
        x=0.0,
        y=0.0,
        z=0.03,
        bandwidth=bandwidth,
        cpu_capacity=cpu_capacity,
        communication_budget=100.0,
        power_budget=100.0,
        trust=0.8,
        sync_threshold=0.1,
        coord_threshold=0.1,
    )


def make_task(task_id: str, candidates: list[str], size: float = 1.0) -> Task:
    return Task(
        task_id=task_id,
        source_uav="UAV_0",
        owner_ap_id=candidates[0],
        x=0.0,
        y=0.0,
        z=10.0,
        L_u=size,
        D_u=1000.0,
        omega_u=1.0,
        psi_u=1.0,
        xi_u="imagery",
        bandwidth_demand=1.0,
        cpu_demand=1.0,
        power_demand=1.0,
        A_u_t=candidates,
        arrival_slot=1,
        AoI=2,
    )


class BaselineRuleTests(unittest.TestCase):
    def test_greedy_link_quality_selects_maximum_rate(self) -> None:
        config = SimulationConfig(slots=1)
        slow = make_ap("BS_slow", bandwidth=5.0)
        fast = make_ap("BS_fast", bandwidth=20.0)
        aps = {slow.ap_id: slow, fast.ap_id: fast}
        controller = BaselineController(config, aps)
        task = make_task("T1", [slow.ap_id, fast.ap_id])

        decision = controller.decide("greedy", [task], aps, slot=1)

        self.assertEqual(decision.assignment[task.task_id], fast.ap_id)

    def test_centralized_qubo_respects_document_capacity_term(self) -> None:
        config = SimulationConfig(slots=1, anneal_reads=8, anneal_sweeps=60)
        first = make_ap("BS_0", bandwidth=20.0, cpu_capacity=1.0)
        second = make_ap("BS_1", bandwidth=19.0, cpu_capacity=1.0)
        aps = {first.ap_id: first, second.ap_id: second}
        controller = BaselineController(config, aps)
        tasks = [
            make_task("T1", [first.ap_id, second.ap_id], size=1.0),
            make_task("T2", [first.ap_id, second.ap_id], size=1.0),
        ]

        decision = controller.decide("centralized_qubo", tasks, aps, slot=1)

        self.assertEqual(set(decision.assignment.values()), {first.ap_id, second.ap_id})

    def test_periodic_trigger_ignores_uncertainty_until_period(self) -> None:
        config = SimulationConfig(
            sync_trigger_policy="periodic",
            coord_trigger_policy="periodic",
            periodic_sync_period=3,
            periodic_coord_period=3,
        )
        manager = TwinManager(config)
        ap = make_ap("BS_0", bandwidth=20.0)

        triggers = []
        for slot in range(1, 4):
            observation = Observation(
                ap_id=ap.ap_id,
                slot=slot,
                load_ratio=1.0,
                bandwidth_ratio=1.0,
                cpu_ratio=1.0,
                queue_size=1,
                candidate_overlap=1.0,
                mobility_pressure=1.0,
            )
            triggers.append(manager.update(ap, observation))

        self.assertEqual(triggers[0], (False, False))
        self.assertEqual(triggers[1], (False, False))
        self.assertEqual(triggers[2], (True, True))

    def test_always_on_trigger_fires_every_slot(self) -> None:
        config = SimulationConfig(
            sync_trigger_policy="always",
            coord_trigger_policy="always",
        )
        manager = TwinManager(config)
        ap = make_ap("BS_0", bandwidth=20.0)
        observation = Observation(
            ap_id=ap.ap_id,
            slot=1,
            load_ratio=0.0,
            bandwidth_ratio=0.0,
            cpu_ratio=0.0,
            queue_size=0,
            candidate_overlap=0.0,
            mobility_pressure=0.0,
        )

        self.assertEqual(manager.update(ap, observation), (True, True))


if __name__ == "__main__":
    unittest.main()
