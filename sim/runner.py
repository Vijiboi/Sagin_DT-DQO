from __future__ import annotations
from collections import defaultdict
import numpy as np
from time import perf_counter

from env.config import SimulationConfig
from env.baseline import BaselineController
from env.feasibility import FeasibilityManager
from env.generator import SaginEnvironment
from env.models import CandidateScore, LocalSummary, APNode, Task, SlotResult
from twin.twin_logic import TwinManager
from opt.qubo_generator import LocalQuboBuilder
from opt.solver import ClassicalQuboSolver
from opt.hybrid import DWaveHybridSolver
from control.regional import RegionalController
from results.io import write_run_outputs 
from results.metrics import build_slot_result 

class SimulationRunner:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.environment = SaginEnvironment(config)
        self.twin_manager = TwinManager(config)
        self.qubo_builder = LocalQuboBuilder(config)
        self.local_solver = self._build_solver(config)
        self.regional_controller = RegionalController(config=config)
        self.use_solver_guidance = config.use_solver_guidance
        self.baseline_controller = BaselineController(config, self.environment.ap_by_id())
        self.feasibility = FeasibilityManager(config)

    def run(self) -> tuple[list, dict[str, object], str]:
        slot_results = []
        ap_lookup = self.environment.ap_by_id()
        solver_counts = defaultdict(int)
        total_solver_time = 0.0
        
        for slot in range(1, self.config.slots + 1):
            self.environment.step_mobility()
            tasks = self.environment.create_tasks_for_slot(slot)
            grouped_tasks = self.environment.group_tasks_by_owner(tasks)
            local_summaries = []

            for ap in self.environment.aps:
                queue = grouped_tasks.get(ap.ap_id, [])
                observation = self.environment.build_observation(ap, slot, queue, len(tasks))
                required_fidelity = max((task.F_u_min for task in queue), default=0.5)
                sync_triggered, coord_triggered = self.twin_manager.update(ap, observation, required_fidelity)
                local_summaries.append(LocalSummary(
                    ap_id=ap.ap_id, slot=slot, queue_size=len(queue),
                    local_load=float(ap.current_cpu_load / max(ap.cpu_capacity, 1.0)),
                    sync_triggered=bool(sync_triggered), 
                    coordination_triggered=bool(coord_triggered),
                    trust=float(ap.trust), 
                    twin_age=int(ap.twin_state.age),
                    uncertainty=float(ap.twin_state.uncertainty), 
                    mismatch=float(ap.twin_state.mismatch),
                    fidelity=float(ap.twin_state.fidelity), 
                    qubo_dimension=0,
                    solver_time=0.0,
                    selected_pairs=[],
                    candidate_scores=[],
                ))

            sync_ap_ids = {
                summary.ap_id for summary in local_summaries if summary.sync_triggered
            }
            coord_ap_ids = {
                summary.ap_id for summary in local_summaries if summary.coordination_triggered
            }
            self.baseline_controller.record_synchronizations(ap_lookup, sync_ap_ids)
            decision_start = perf_counter()

            algorithm = self.config.algorithm.lower()
            use_proposed_solver = algorithm in {"proposed", "always_on"} or (
                algorithm == "uncertainty_unaware" and bool(sync_ap_ids)
            )

            if use_proposed_solver:
                # Freeze the latest available report as z_ref(t) for this slot.
                # Reports are refreshed only after execution, and are retained
                # at the coordinator until another event-triggered report.
                for summary in local_summaries:
                    if summary.coordination_triggered:
                        state = ap_lookup[summary.ap_id].coordination_state
                        state.reported_summary = state.realized_summary
                frozen_references = {
                    ap_id: tuple(ap.coordination_state.reported_summary)
                    for ap_id, ap in ap_lookup.items()
                }
                iteration_limit = max(int(self.config.max_qubo_iterations), 1)
                for local_summary in local_summaries:
                    ap = ap_lookup[local_summary.ap_id]
                    eligible = [
                        task for task in tasks
                        if ap.ap_id in task.A_u_t and ap.twin_state.fidelity >= task.F_u_min
                    ]
                    state = ap.coordination_state
                    nbar = max(1.0, float(ap.current_task_load))
                    b_previous: dict[tuple[str, str], int] = {
                        (task.task_id, ap.ap_id): 0 for task in eligible
                    }
                    final_sample = b_previous.copy()
                    for _k in range(iteration_limit):
                        local_summary.qubo_iterations += 1
                        problem = self.qubo_builder.build(
                            ap, tasks, ap_lookup, slot, sync_ap_ids, coord_ap_ids,
                            reference_load=nbar,
                            reference_summary=frozen_references[ap.ap_id],
                            multiplier=state.coordination_multiplier,
                        )
                        if self.local_solver is not None and problem.variables:
                            solve_result = self.local_solver.solve(problem)
                            solver_counts[solve_result.solver_name] += 1
                            total_solver_time += solve_result.solver_time
                            local_summary.solver_time += float(solve_result.solver_time)
                            final_sample = {
                                variable: int(solve_result.sample.get(variable, 0))
                                for variable in problem.variables
                            }
                            local_summary.qubo_dimension = len(problem.variables)
                        else:
                            final_sample = {variable: 0 for variable in problem.variables}

                        local_summary.selected_pairs = [
                            variable for variable, value in final_sample.items() if value
                        ]
                        nbar = max(1.0, float(sum(final_sample.values())))
                        candidate_summary = self._candidate_coordination_summary(
                            ap, eligible, final_sample,
                        )
                        if local_summary.coordination_triggered:
                            eta = self.config.coordination_multiplier_step
                            state.coordination_multiplier = tuple(
                                lam + eta * (z - ref)
                                for lam, z, ref in zip(
                                    state.coordination_multiplier,
                                    candidate_summary,
                                    frozen_references[ap.ap_id],
                                )
                            )
                        mismatch = float(np.linalg.norm(
                            np.asarray(candidate_summary) - np.asarray(frozen_references[ap.ap_id])
                        ))
                        local_summary.coordination_mismatch = mismatch
                        if final_sample == b_previous:
                            break
                        b_previous = final_sample.copy()

                    # Send task/service scores evaluated at the final positive
                    # reference load. Regional selection consumes these scores.
                    local_summary.candidate_scores = self.qubo_builder.score_candidates(
                        ap, eligible, ap_lookup,
                        sync_ap_ids=sync_ap_ids,
                        coord_ap_ids=coord_ap_ids,
                        load_override=nbar,
                    )
                    local_summary.candidate_scores = [
                        score for score in local_summary.candidate_scores
                        if score.destination_id == ap.ap_id
                    ]
                    local_summary.reference_load_final = nbar
                    local_summary.multiplier_load = state.coordination_multiplier[0]
                    local_summary.multiplier_risk = state.coordination_multiplier[1]

                assignments = self.regional_controller.project(
                    local_summaries, ap_lookup, tasks,
                )
            else:
                baseline_algorithm = (
                    "uncertainty_unaware_fallback"
                    if algorithm == "uncertainty_unaware"
                    else algorithm
                )
                decision = self.baseline_controller.decide(
                    baseline_algorithm,
                    tasks,
                    ap_lookup,
                    slot,
                )
                solver_counts[decision.solver_name] += 1
                total_solver_time += decision.solver_time
                repaired_mapping = self.feasibility.repair(
                    decision.assignment,
                    tasks,
                    ap_lookup,
                )
                assignments = self.regional_controller.materialize_mapping(
                    repaired_mapping,
                    tasks,
                    ap_lookup,
                    sync_ap_ids,
                    coord_ap_ids,
                )

            decision_runtime_seconds = perf_counter() - decision_start
            self._refresh_ap_loads(assignments, local_summaries, ap_lookup)
            self._record_realized_summaries(assignments, tasks, ap_lookup)
            resource_feasible = self._is_resource_feasible(assignments, tasks, ap_lookup)
            slot_results.append(
                build_slot_result(
                    slot,
                    assignments,
                    local_summaries,
                    resource_feasible,
                    decision_runtime_seconds,
                )
            )

        summary = self._build_summary(slot_results, solver_counts, total_solver_time)
        # Convert output_path to string and write outputs [cite: 534]
        output_path = str(write_run_outputs(self.config.output_dir, slot_results, summary))
        
        return slot_results, summary, output_path

    @staticmethod
    def _build_solver(config: SimulationConfig):
        backend = config.local_solver_backend.lower()
        if backend in {"dwave", "dwave_hybrid", "hybrid"}:
            return DWaveHybridSolver()
        if backend in {"none", "off", "disabled"}:
            return None
        classical_backend = "auto" if backend == "classical" else backend
        return ClassicalQuboSolver(config, backend=classical_backend)

    def _candidate_coordination_summary(self, ap, tasks, sample):
        selected = [task for task in tasks if sample.get((task.task_id, ap.ap_id), 0)]
        trust_risk = (1.0 - ap.trust) * sum(task.psi_u for task in selected)
        app_pressure = (
            sum(task.F_u_min for task in selected) / len(selected)
            if selected else 0.0
        )
        return (
            float(len(selected)),
            float(trust_risk),
            float(ap.twin_state.uncertainty),
            float(1.0 - ap.twin_state.fidelity),
            float(app_pressure),
        )

    @staticmethod
    def _record_realized_summaries(assignments, tasks, ap_lookup):
        task_by_id = {task.task_id: task for task in tasks}
        for ap_id, ap in ap_lookup.items():
            selected = [
                task_by_id[item.task_id]
                for item in assignments
                if item.destination_id == ap_id and item.task_id in task_by_id
            ]
            ap.coordination_state.realized_summary = (
                float(len(selected)),
                float((1.0 - ap.trust) * sum(task.psi_u for task in selected)),
                float(ap.twin_state.uncertainty),
                float(1.0 - ap.twin_state.fidelity),
                float(sum(task.F_u_min for task in selected) / len(selected)) if selected else 0.0,
            )

    def _refresh_ap_loads(self, assignments, summaries, ap_lookup):
        cpu_load_counter = defaultdict(float)
        task_counter = defaultdict(int)
        score_map = {(s.task_id, s.destination_id): s for summ in summaries for s in summ.candidate_scores}

        for assign in assignments:
            task_counter[assign.destination_id] += 1
            score = score_map.get((assign.task_id, assign.destination_id))
            if score:
                cpu_load_counter[assign.destination_id] += float(score.required_cpu)

        for ap_id, ap in ap_lookup.items():
            ap.current_task_load = int(task_counter.get(ap_id, 0))
            ap.current_cpu_load = float(cpu_load_counter.get(ap_id, 0.0))

    def _is_resource_feasible(self, assignments, tasks, ap_lookup) -> bool:
        mapping = {
            assignment.task_id: assignment.destination_id
            for assignment in assignments
        }
        return self.feasibility.assignment_is_feasible(
            mapping,
            tasks,
            ap_lookup,
        )

    def _build_summary(self, results, solver_counts, total_solver_time: float) -> dict[str, object]:
        """Ensures all metrics are native Python types for JSON serialization [cite: 600-610]."""
        solved_problems = sum(solver_counts.values())
        return {
            "simulation_horizon": int(len(results)),
            "sync_triggers": int(sum(r.sync_trigger_count for r in results)),
            "coord_triggers": int(sum(r.coordination_trigger_count for r in results)),
            "one_hot_validity": bool(all(r.one_hot_valid for r in results)),
            "avg_fidelity": round(float(np.mean([s.fidelity for r in results for s in r.local_summaries])), 4),
            "decision_runtime_seconds": {
                "total": round(
                    float(sum(result.decision_runtime_seconds for result in results)),
                    6,
                ),
                "average_per_slot": round(
                    float(
                        np.mean(
                            [result.decision_runtime_seconds for result in results]
                        )
                    ),
                    6,
                ),
            },
            "solver_statistics": {
                "solved_problems": int(solved_problems),
                "solver_counts": dict(solver_counts),
                "total_solver_time": round(float(total_solver_time), 6),
                "avg_solver_time": round(float(total_solver_time / max(solved_problems, 1)), 6),
            },
            "simulation_parameters": self.config.to_report_dict(),
        }
