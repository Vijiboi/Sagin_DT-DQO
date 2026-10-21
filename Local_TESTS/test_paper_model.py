from __future__ import annotations

import unittest

from env.config import SimulationConfig
from env.evaluator import PhysicalEvaluator
from env.feasibility import FeasibilityManager, ResourceUsage
from env.models import APNode, Task
from opt.assignment_bqm import AssignmentBqmSolver
from opt.qubo_generator import LocalQuboBuilder
from sim.runner import SimulationRunner


def make_ap(ap_id: str = "BS_0", cpu_ghz: float = 10.0) -> APNode:
    return APNode(
        ap_id=ap_id,
        tier="BS",
        x=0.0,
        y=0.0,
        z=0.03,
        bandwidth=20.0,
        cpu_capacity=cpu_ghz,
        communication_budget=20.0,
        power_budget=10.0,
        trust=0.8,
        sync_threshold=0.1,
        coord_threshold=0.1,
    )


def make_task(task_id: str = "T1", candidates: list[str] | None = None) -> Task:
    candidates = candidates or ["BS_0"]
    return Task(
        task_id=task_id,
        source_uav="UAV_0",
        owner_ap_id=candidates[0],
        x=0.0,
        y=0.0,
        z=10.0,
        L_u=1.0,
        D_u=1000.0,
        omega_u=1.0,
        psi_u=1.0,
        xi_u="control",
        bandwidth_demand=1.0,
        cpu_demand=1.0,
        power_demand=1.0,
        A_u_t=candidates,
        arrival_slot=1,
        AoI=2,
    )


class PaperModelTests(unittest.TestCase):
    def test_physical_evaluator_uses_si_conversions(self) -> None:
        config = SimulationConfig()
        metrics = PhysicalEvaluator(config).evaluate_remote(
            make_task(),
            make_ap(),
            load_count=1,
        )

        self.assertAlmostEqual(metrics.transmission_delay_s, 0.05, places=9)
        self.assertAlmostEqual(metrics.computation_delay_s, 0.1, places=9)
        self.assertAlmostEqual(metrics.transmission_energy_j, 0.05, places=9)
        self.assertAlmostEqual(metrics.computation_energy_j, 10.0, places=9)

    def test_shared_feasibility_counts_cycles_bits_and_power(self) -> None:
        config = SimulationConfig()
        manager = FeasibilityManager(config)
        ap = make_ap(cpu_ghz=1.0)
        task = make_task()

        self.assertTrue(manager.can_add(task, ap, ResourceUsage()))
        used = manager.add(task, ResourceUsage())
        self.assertFalse(manager.can_add(task, ap, used))

    def test_proposed_qubo_has_one_variable_per_candidate(self) -> None:
        config = SimulationConfig()
        first = make_ap("BS_0")
        second = make_ap("BS_1")
        task = make_task(candidates=["BS_0", "BS_1"])
        problem = LocalQuboBuilder(config).build(
            first,
            [task],
            {"BS_0": first, "BS_1": second},
            slot=1,
        )

        self.assertEqual(
            set(problem.variables),
            {("T1", "BS_0"), ("T1", "BS_1")},
        )
        pair = (("T1", "BS_0"), ("T1", "BS_1"))
        self.assertEqual(problem.quadratic[pair], 2.0 * config.qubo_penalty)

    def test_binary_baseline_solver_returns_feasible_one_hot_mapping(self) -> None:
        config = SimulationConfig(anneal_reads=2, anneal_sweeps=5)
        first = make_ap("BS_0", cpu_ghz=1.0)
        second = make_ap("BS_1", cpu_ghz=1.0)
        tasks = [
            make_task("T1", ["BS_0", "BS_1"]),
            make_task("T2", ["BS_0", "BS_1"]),
        ]
        costs = {
            "T1": {"BS_0": 0.0, "BS_1": 1.0},
            "T2": {"BS_0": 0.0, "BS_1": 1.0},
        }
        aps = {"BS_0": first, "BS_1": second}

        result = AssignmentBqmSolver(config).solve(tasks, aps, costs)

        self.assertEqual(set(result.mapping), {"T1", "T2"})
        self.assertTrue(result.feasible)
        self.assertEqual(set(result.mapping.values()), {"BS_0", "BS_1"})

    def test_runner_records_decision_only_runtime(self) -> None:
        config = SimulationConfig(
            slots=2,
            num_uavs=3,
            num_bs=1,
            num_haps=1,
            num_leos=0,
            candidate_ap_limit=2,
            algorithm="greedy",
            output_dir="results/test_runs",
        )
        slots, _, _ = SimulationRunner(config).run()

        self.assertTrue(all(slot.decision_runtime_seconds > 0.0 for slot in slots))
        self.assertTrue(all(slot.one_hot_valid for slot in slots))
        self.assertTrue(all(slot.resource_feasible for slot in slots))


if __name__ == "__main__":
    unittest.main()
