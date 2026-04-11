from __future__ import annotations
from collections import defaultdict
import numpy as np
from env.baseline import greedy_one_hot_assignment
from env.models import CandidateScore, FinalAssignment, LocalSummary, APNode, Task

class RegionalController:
    def project(
        self,
        local_summaries: list[LocalSummary],
        ap_lookup: dict[str, APNode],
        tasks: list[Task]
    ) -> list[FinalAssignment]:
        """
        Enforces global one-hot task-assignment across all APs and local execution [cite: 1505-1509].
        Selection Rule: m* = arg min { J_hat + lambda_f * I(F < F_min) }[cite: 430].
        """
        all_scores: list[CandidateScore] = []
        # Group scores by task_id to evaluate competition for the same task 
        all_scores_by_task: dict[str, list[CandidateScore]] = defaultdict(list)
        
        for summary in local_summaries:
            all_scores.extend(summary.candidate_scores)
            for score in summary.candidate_scores:
                all_scores_by_task[score.task_id].append(score)

        final_assignments: list[FinalAssignment] = []
        # Track hard resource usage per AP during projection [cite: 1522-1525, 1592-1597]
        resource_usage = {
            ap_id: {"cpu": 0.0, "bandwidth": 0.0, "power": 0.0}
            for ap_id in ap_lookup
        }

        # Match tasks to generated scores
        task_map = {t.task_id: t for t in tasks}
        remaining_tasks = set(all_scores_by_task.keys())

        # Enforce Hierarchical Projection Rule [cite: 437, 1527-1550]
        while remaining_tasks:
            ranked_tasks = []
            for task_id in list(remaining_tasks):
                task_obj = task_map.get(task_id)
                if not task_obj: continue
                
                # Rank candidates for this task based on cost + fidelity penalty [cite: 1530-1533]
                ranked_scores = self._rank_scores_with_fidelity(
                    all_scores_by_task[task_id], 
                    task_obj, 
                    ap_lookup
                )
                
                # Check hardware feasibility (CPU/BW/Power) [cite: 1534-1536, 1592-1597]
                feasible_scores = [
                    s for s in ranked_scores 
                    if self._is_feasible(s, ap_lookup, resource_usage)
                ]
                
                if feasible_scores:
                    best_s = feasible_scores[0]
                    # Priority: favor tasks with fewer feasible options or higher cost [cite: 1540-1549]
                    ranked_tasks.append((
                        len(feasible_scores), 
                        best_s.local_cost + best_s.coupling_penalty, 
                        task_id, 
                        feasible_scores
                    ))

            if not ranked_tasks: break
            
            # Select the most constrained task to assign next [cite: 1550]
            _, _, winner_task_id, winner_feasible = min(ranked_tasks)
            winner_score = winner_feasible[0]
            
            # Update physical resource consumption [cite: 1552-1555]
            usage = resource_usage[winner_score.destination_id]
            usage["cpu"] += winner_score.required_cpu
            usage["bandwidth"] += winner_score.required_bandwidth
            usage["power"] += winner_score.required_power
            
            # Build final one-hot result [cite: 1556-1567]
            final_assignments.append(FinalAssignment(
                task_id=winner_score.task_id,
                owner_ap_id=winner_score.owner_ap_id,
                destination_id=winner_score.destination_id,
                local_cost=winner_score.local_cost,
                delay_cost=winner_score.delay_cost,
                energy_cost=winner_score.energy_cost,
                mission_cost=winner_score.mission_cost,
                fidelity_cost=winner_score.fidelity_cost
            ))
            remaining_tasks.remove(winner_task_id)

        # Fallback to greedy if projection failed to assign all [cite: 1569-1570]
        if not final_assignments and all_scores:
            return greedy_one_hot_assignment(all_scores)
            
        return final_assignments

    def _rank_scores_with_fidelity(self, scores: list[CandidateScore], task: Task, ap_lookup: dict):
        """Applies lambda_f penalty for fidelity violations (Eq. 102) ."""
        lambda_f = 1e6 # Large infeasibility penalty [cite: 432]
        return sorted(
            scores,
            key=lambda s: s.local_cost + s.coupling_penalty + 
            (lambda_f if ap_lookup[s.destination_id].twin_state.fidelity < task.F_u_min else 0)
        )

    @staticmethod
    def _is_feasible(score: CandidateScore, ap_lookup, resource_usage):
        """Hard constraint check for node capacity [cite: 1592-1597]."""
        ap = ap_lookup[score.destination_id]
        usage = resource_usage[score.destination_id]
        cpu_ok = usage["cpu"] + score.required_cpu <= ap.cpu_capacity
        bandwidth_ok = usage["bandwidth"] + score.required_bandwidth <= ap.communication_budget
        power_ok = usage["power"] + score.required_power <= ap.power_budget
        return cpu_ok and bandwidth_ok and power_ok