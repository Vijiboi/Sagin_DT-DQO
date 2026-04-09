from __future__ import annotations

from collections import defaultdict

from control.consensus import QuantizedConsensusCoordinator
from control.regional import RegionalController
from env.config import SimulationConfig
from env.generator import SaginEnvironment
from env.models import CandidateScore, LocalSummary
from opt.qubo import LocalQuboBuilder
from opt.solver import ClassicalQuboSolver
from results.io import write_run_outputs
from results.metrics import build_slot_result
from sim.checks import validate_pre_quantum_checks
from twin.twin_logic import TwinManager


class SimulationRunner:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.environment = SaginEnvironment(config)
        self.twin_manager = TwinManager(config)
        self.qubo_builder = LocalQuboBuilder(config)
        self.local_solver = ClassicalQuboSolver(config)
        self.consensus_coordinator = QuantizedConsensusCoordinator(config)
        self.regional_controller = RegionalController()

    def run(self) -> tuple[list, dict[str, object], str]:
        slot_results = []
        ap_lookup = self.environment.ap_by_id()

        for slot in range(self.config.slots):
            self.environment.step_mobility()
            tasks = self.environment.create_tasks_for_slot(slot)
            grouped_tasks = self.environment.group_tasks_by_owner(tasks)
            local_summaries: list[LocalSummary] = []

            for ap in self.environment.aps:
                queue = grouped_tasks.get(ap.ap_id, [])
                observation = self.environment.build_observation(ap, slot, queue, len(tasks))
                sync_triggered, coordination_triggered = self.twin_manager.update(ap, observation)
                selected_pairs: list[tuple[str, str]] = []
                solver_time = 0.0
                candidate_scores: list[CandidateScore] = []
                qubo_dimension = 0

                if queue:
                    problem = self.qubo_builder.build(ap, queue, ap_lookup, slot)
                    solve_result = self.local_solver.solve(problem)
                    selected_pairs = self._decode_sample(problem.candidate_scores, solve_result.sample)
                    solver_time = solve_result.solver_time
                    candidate_scores = problem.candidate_scores
                    qubo_dimension = len(problem.variables)

                local_summaries.append(
                    LocalSummary(
                        ap_id=ap.ap_id,
                        slot=slot,
                        queue_size=len(queue),
                        local_load=problem.local_load if queue else ap.current_cpu_load / max(ap.cpu_capacity, 1.0),
                        sync_triggered=sync_triggered,
                        coordination_triggered=coordination_triggered,
                        trust=ap.trust,
                        twin_age=ap.twin_state.age,
                        uncertainty=ap.twin_state.uncertainty,
                        mismatch=ap.twin_state.mismatch,
                        fidelity=ap.twin_state.fidelity,
                        qubo_dimension=qubo_dimension,
                        solver_time=solver_time,
                        selected_pairs=selected_pairs,
                        candidate_scores=candidate_scores,
                    )
                )

            self.consensus_coordinator.update(
                ap_lookup,
                self.environment.communication_graph,
                local_summaries,
                slot,
            )
            assignments = self.regional_controller.project(local_summaries, ap_lookup)
            resource_feasible = self._validate_resource_feasibility(assignments, local_summaries, ap_lookup)
            self._refresh_ap_loads(assignments, local_summaries, ap_lookup)
            slot_results.append(build_slot_result(slot, assignments, local_summaries, resource_feasible))

        summary = self._build_summary(slot_results)
        output_path = str(write_run_outputs(self.config.output_dir, slot_results, summary))
        return slot_results, summary, output_path

    @staticmethod
    def _decode_sample(candidate_scores, sample: dict[tuple[str, str], int]) -> list[tuple[str, str]]:
        selected = [pair for pair, active in sample.items() if active == 1]
        if selected:
            return selected

        best_by_task: dict[str, tuple[str, str, float]] = {}
        for score in candidate_scores:
            current = best_by_task.get(score.task_id)
            if current is None or score.local_cost < current[2]:
                best_by_task[score.task_id] = (score.task_id, score.destination_id, score.local_cost)
        return [(task_id, destination_id) for task_id, destination_id, _ in best_by_task.values()]

    @staticmethod
    def _refresh_ap_loads(assignments, local_summaries, ap_lookup) -> None:
        score_map = {}
        for summary in local_summaries:
            for score in summary.candidate_scores:
                score_map[(score.task_id, score.destination_id)] = score

        task_load_counter: dict[str, int] = defaultdict(int)
        cpu_load_counter: dict[str, float] = defaultdict(float)
        for assignment in assignments:
            score = score_map.get((assignment.task_id, assignment.destination_id))
            task_load_counter[assignment.destination_id] += 1
            if score is not None:
                cpu_load_counter[assignment.destination_id] += score.required_cpu
        for ap_id, ap in ap_lookup.items():
            ap.current_task_load = task_load_counter.get(ap_id, 0)
            ap.current_cpu_load = cpu_load_counter.get(ap_id, 0.0)

    @staticmethod
    def _validate_resource_feasibility(assignments, local_summaries, ap_lookup) -> bool:
        score_map = {}
        for summary in local_summaries:
            for score in summary.candidate_scores:
                score_map[(score.task_id, score.destination_id)] = score

        usage: dict[str, dict[str, float]] = defaultdict(lambda: {"cpu": 0.0, "bandwidth": 0.0, "power": 0.0})
        for assignment in assignments:
            score = score_map.get((assignment.task_id, assignment.destination_id))
            if score is None:
                return False
            usage[assignment.destination_id]["cpu"] += score.required_cpu
            usage[assignment.destination_id]["bandwidth"] += score.required_bandwidth
            usage[assignment.destination_id]["power"] += score.required_power

        for ap_id, ap in ap_lookup.items():
            ap_usage = usage[ap_id]
            if ap_usage["cpu"] > ap.cpu_capacity + 1e-9:
                return False
            if ap_usage["bandwidth"] > ap.communication_budget + 1e-9:
                return False
            if ap_usage["power"] > ap.power_budget + 1e-9:
                return False
        return True

    def _build_summary(self, slot_results) -> dict[str, object]:
        all_assignments = [assignment for slot in slot_results for assignment in slot.assignments]
        all_local_summaries = [summary for slot in slot_results for summary in slot.local_summaries]

        slot_count = max(len(slot_results), 1)
        assignment_count = max(len(all_assignments), 1)
        solver_time_per_ap = {
            summary.ap_id: round(
                sum(item.solver_time for item in all_local_summaries if item.ap_id == summary.ap_id) / slot_count,
                6,
            )
            for summary in all_local_summaries
        }

        return {
            "simulation_parameters": {
                "runtime_config": self.config.to_report_dict(),
                "paper_parameter_view": self.config.to_paper_parameter_view(),
            },
            "slots": len(slot_results),
            "average_delay": round(sum(item.delay_cost for item in all_assignments) / assignment_count, 6),
            "average_energy": round(sum(item.energy_cost for item in all_assignments) / assignment_count, 6),
            "mission_cost": round(sum(item.mission_cost for item in all_assignments), 6),
            "fidelity_cost": round(sum(item.fidelity_cost for item in all_assignments), 6),
            "sync_trigger_count": int(sum(item.sync_trigger_count for item in slot_results)),
            "coordination_trigger_count": int(sum(item.coordination_trigger_count for item in slot_results)),
            "solver_time_per_ap": solver_time_per_ap,
            "average_local_load": round(
                sum(summary.local_load for summary in all_local_summaries) / max(len(all_local_summaries), 1),
                6,
            ),
            "final_one_hot_assignment_valid": all(slot.one_hot_valid for slot in slot_results),
            "communication_graph_edges": len(self.environment.communication_graph.edges),
            "quantized_dual_by_ap": {
                ap_id: ap.coordination_state.quantized_dual for ap_id, ap in self.environment.ap_by_id().items()
            },
            "tasks_per_ap_last_slot": slot_results[-1].tasks_per_ap if slot_results else {},
            "local_execution_count_last_slot": slot_results[-1].local_execution_count if slot_results else 0,
            "pre_quantum_checks": validate_pre_quantum_checks(slot_results),
        }
