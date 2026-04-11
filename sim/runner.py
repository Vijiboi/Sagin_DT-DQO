from __future__ import annotations
from collections import defaultdict
import numpy as np
import os

from env.config import SimulationConfig
from env.generator import SaginEnvironment
from env.models import CandidateScore, LocalSummary, APNode, Task, SlotResult
from twin.twin_logic import TwinManager
from opt.qubo_generator import LocalQuboBuilder
from opt.solver import ClassicalQuboSolver
from control.regional import RegionalController
from results.io import write_run_outputs 
from results.metrics import build_slot_result 

class SimulationRunner:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.environment = SaginEnvironment(config)
        self.twin_manager = TwinManager(config)
        self.qubo_builder = LocalQuboBuilder(config)
        self.local_solver = ClassicalQuboSolver(config) 
        self.regional_controller = RegionalController()

    def run(self) -> tuple[list, dict[str, object], str]:
        slot_results = []
        ap_lookup = self.environment.ap_by_id()
        
        for slot in range(1, self.config.slots + 1):
            self.environment.step_mobility()
            tasks = self.environment.create_tasks_for_slot(slot)
            grouped_tasks = self.environment.group_tasks_by_owner(tasks)
            local_summaries = []

            for ap in self.environment.aps:
                queue = grouped_tasks.get(ap.ap_id, [])
                observation = self.environment.build_observation(ap, slot, queue, len(tasks))
                sync_triggered, coord_triggered = self.twin_manager.update(ap, observation)
                
                solver_time = 0.0
                candidate_scores = []
                if queue:
                    problem = self.qubo_builder.build(ap, queue, ap_lookup, slot)
                    solve_result = self.local_solver.solve(problem)
                    solver_time = solve_result.solver_time
                    candidate_scores = problem.candidate_scores
                
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
                    qubo_dimension=len(queue),
                    solver_time=float(solver_time), 
                    selected_pairs=[], 
                    candidate_scores=candidate_scores
                ))

            assignments = self.regional_controller.project(local_summaries, ap_lookup, tasks)
            self._refresh_ap_loads(assignments, local_summaries, ap_lookup)
            slot_results.append(build_slot_result(slot, assignments, local_summaries, True))

        summary = self._build_summary(slot_results)
        # Convert output_path to string and write outputs [cite: 534]
        output_path = str(write_run_outputs(self.config.output_dir, slot_results, summary))
        
        return slot_results, summary, output_path

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

    def _build_summary(self, results) -> dict[str, object]:
        """Ensures all metrics are native Python types for JSON serialization [cite: 600-610]."""
        return {
            "simulation_horizon": int(len(results)),
            "sync_triggers": int(sum(r.sync_trigger_count for r in results)),
            "coord_triggers": int(sum(r.coordination_trigger_count for r in results)),
            "one_hot_validity": bool(all(r.one_hot_valid for r in results)),
            "avg_fidelity": round(float(np.mean([s.fidelity for r in results for s in r.local_summaries])), 4)
        }