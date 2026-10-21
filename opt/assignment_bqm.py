from __future__ import annotations

from dataclasses import dataclass
from math import exp
from random import Random
from time import perf_counter

import dimod

from env.config import SimulationConfig
from env.feasibility import FeasibilityManager
from env.models import APNode, Task


@dataclass(frozen=True, slots=True)
class AssignmentBqmResult:
    mapping: dict[str, str]
    solver_name: str
    solver_time: float
    feasible: bool


class AssignmentBqmSolver:
    """Local classical solver for the paper's binary assignment model."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.feasibility = FeasibilityManager(config)

    def solve(
        self,
        tasks: list[Task],
        ap_lookup: dict[str, APNode],
        costs: dict[str, dict[str, float]],
    ) -> AssignmentBqmResult:
        if not tasks:
            return AssignmentBqmResult({}, "empty_bqm", 0.0, True)
        variable_count = sum(len(task.A_u_t) for task in tasks)
        if variable_count > self.config.cqm_to_bqm_variable_limit:
            return self._solve_direct_qubo(tasks, ap_lookup, costs)

        start = perf_counter()
        cqm = dimod.ConstrainedQuadraticModel()
        variables = {
            (task.task_id, ap_id): dimod.Binary((task.task_id, ap_id))
            for task in tasks
            for ap_id in task.A_u_t
        }
        cqm.set_objective(
            sum(
                costs[task_id][ap_id] * variable
                for (task_id, ap_id), variable in variables.items()
            )
        )

        for task in tasks:
            cqm.add_constraint(
                sum(
                    variables[(task.task_id, ap_id)]
                    for ap_id in task.A_u_t
                )
                == 1,
                label=f"one_hot::{task.task_id}",
            )

        for ap_id, ap in ap_lookup.items():
            relevant = [
                task for task in tasks if (task.task_id, ap_id) in variables
            ]
            if not relevant:
                continue
            capacity = self.feasibility.capacity(ap)
            scale = max(int(self.config.resource_constraint_scale), 1)
            resource_coefficients = {
                "cpu": [
                    max(
                        1,
                        round(
                            scale
                            * self.feasibility.task_demand(task).cpu_cycles
                            / max(capacity.cpu_cycles, 1e-12)
                        ),
                    )
                    for task in relevant
                ],
                "communication": [
                    max(
                        1,
                        round(
                            scale
                            * self.feasibility.task_demand(task).communication_bits
                            / max(capacity.communication_bits, 1e-12)
                        ),
                    )
                    for task in relevant
                ],
                "power": [
                    max(
                        1,
                        round(
                            scale
                            * self.feasibility.task_demand(task).power_watts
                            / max(capacity.power_watts, 1e-12)
                        ),
                    )
                    for task in relevant
                ],
            }
            for resource_name, coefficients in resource_coefficients.items():
                if sum(coefficients) <= scale:
                    continue
                expression = sum(
                    coefficient * variables[(task.task_id, ap_id)]
                    for task, coefficient in zip(relevant, coefficients)
                )
                cqm.add_constraint(
                    expression <= scale,
                    label=f"{resource_name}::{ap_id}",
                )

        bqm, inverter = dimod.cqm_to_bqm(
            cqm,
            lagrange_multiplier=self.config.qubo_penalty,
        )
        sampler = dimod.SimulatedAnnealingSampler()
        response = sampler.sample(
            bqm,
            num_reads=max(self.config.anneal_reads, 1),
            num_sweeps=max(self.config.anneal_sweeps, 1),
        )

        best_mapping: dict[str, str] | None = None
        best_objective = float("inf")
        for row in response.data(fields=["sample", "energy"]):
            decoded = inverter(row.sample)
            mapping = self._decode(tasks, decoded, costs)
            if not self.feasibility.assignment_is_feasible(
                mapping,
                tasks,
                ap_lookup,
            ):
                continue
            objective = sum(
                costs[task_id][destination]
                for task_id, destination in mapping.items()
            )
            if objective < best_objective:
                best_mapping = mapping
                best_objective = objective

        if best_mapping is None:
            preferred = self._decode(tasks, inverter(response.first.sample), costs)
            best_mapping = self.feasibility.repair(
                preferred,
                tasks,
                ap_lookup,
                ranked_candidates={
                    task.task_id: sorted(
                        task.A_u_t,
                        key=lambda ap_id: costs[task.task_id][ap_id],
                    )
                    for task in tasks
                },
            )

        elapsed = perf_counter() - start
        return AssignmentBqmResult(
            mapping=best_mapping,
            solver_name="dimod_cqm_to_bqm_sa",
            solver_time=elapsed,
            feasible=self.feasibility.assignment_is_feasible(
                best_mapping,
                tasks,
                ap_lookup,
            ),
        )

    def _solve_direct_qubo(
        self,
        tasks: list[Task],
        ap_lookup: dict[str, APNode],
        costs: dict[str, dict[str, float]],
    ) -> AssignmentBqmResult:
        start = perf_counter()
        ranked_candidates = {
            task.task_id: sorted(
                task.A_u_t,
                key=lambda ap_id: costs[task.task_id][ap_id],
            )
            for task in tasks
        }
        rng = Random(self.config.seed + len(tasks) * 997)
        best_mapping = self.feasibility.repair(
            {task.task_id: ranked_candidates[task.task_id][0] for task in tasks},
            tasks,
            ap_lookup,
            ranked_candidates=ranked_candidates,
        )
        best_energy = self._assignment_energy(best_mapping, tasks, ap_lookup, costs)

        for read in range(max(self.config.anneal_reads, 1)):
            if read == 0:
                state = dict(best_mapping)
            else:
                state = {
                    task.task_id: rng.choice(task.A_u_t)
                    for task in tasks
                }
            energy = self._assignment_energy(state, tasks, ap_lookup, costs)
            temperature = max(abs(energy) / max(len(tasks), 1), 1.0)
            for _ in range(max(self.config.anneal_sweeps, 1)):
                order = list(tasks)
                rng.shuffle(order)
                for task in order:
                    current = state[task.task_id]
                    candidates = [ap_id for ap_id in task.A_u_t if ap_id != current]
                    if not candidates:
                        continue
                    trial = dict(state)
                    trial[task.task_id] = rng.choice(candidates)
                    candidate_energy = self._assignment_energy(
                        trial,
                        tasks,
                        ap_lookup,
                        costs,
                    )
                    delta = candidate_energy - energy
                    if delta <= 0.0 or rng.random() < exp(
                        -delta / max(temperature, 1e-12)
                    ):
                        state = trial
                        energy = candidate_energy
                temperature *= 0.92

            repaired = self.feasibility.repair(
                state,
                tasks,
                ap_lookup,
                ranked_candidates=ranked_candidates,
            )
            repaired_energy = self._assignment_energy(
                repaired,
                tasks,
                ap_lookup,
                costs,
            )
            if repaired_energy < best_energy:
                best_mapping = repaired
                best_energy = repaired_energy

        elapsed = perf_counter() - start
        return AssignmentBqmResult(
            mapping=best_mapping,
            solver_name="onehot_assignment_qubo_sa",
            solver_time=elapsed,
            feasible=self.feasibility.assignment_is_feasible(
                best_mapping,
                tasks,
                ap_lookup,
            ),
        )

    def _assignment_energy(
        self,
        mapping: dict[str, str],
        tasks: list[Task],
        ap_lookup: dict[str, APNode],
        costs: dict[str, dict[str, float]],
    ) -> float:
        task_by_id = {task.task_id: task for task in tasks}
        total = 0.0
        usage: dict[str, list[Task]] = {}
        for task_id, destination in mapping.items():
            if destination == "LOCAL" or destination not in ap_lookup:
                total += self.config.qubo_penalty
                continue
            total += costs[task_id][destination]
            usage.setdefault(destination, []).append(task_by_id[task_id])

        for ap_id, assigned in usage.items():
            capacity = self.feasibility.capacity(ap_lookup[ap_id])
            cpu = sum(
                self.feasibility.task_demand(task).cpu_cycles
                for task in assigned
            ) / max(capacity.cpu_cycles, 1e-12)
            communication = sum(
                self.feasibility.task_demand(task).communication_bits
                for task in assigned
            ) / max(capacity.communication_bits, 1e-12)
            power = sum(
                self.feasibility.task_demand(task).power_watts
                for task in assigned
            ) / max(capacity.power_watts, 1e-12)
            overload = (
                max(0.0, cpu - 1.0) ** 2
                + max(0.0, communication - 1.0) ** 2
                + max(0.0, power - 1.0) ** 2
            )
            total += self.config.qubo_penalty * overload
        return float(total)

    @staticmethod
    def _decode(
        tasks: list[Task],
        sample,
        costs: dict[str, dict[str, float]],
    ) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for task in tasks:
            selected = [
                ap_id
                for ap_id in task.A_u_t
                if sample.get((task.task_id, ap_id), 0) > 0
            ]
            mapping[task.task_id] = min(
                selected or task.A_u_t,
                key=lambda ap_id: costs[task.task_id][ap_id],
            )
        return mapping
