from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import exp
from random import Random
from time import perf_counter

from .config import SimulationConfig
from .evaluator import PhysicalEvaluator
from .feasibility import FeasibilityManager
from .models import APNode, Task
from opt.assignment_bqm import AssignmentBqmSolver


@dataclass(slots=True)
class NodeSnapshot:
    load_reference: float
    risk: float
    trust: float
    fidelity: float


@dataclass(slots=True)
class BaselineDecision:
    assignment: dict[str, str]
    solver_name: str
    solver_time: float


class BaselineController:
    """Decision rules for the eleven baselines in the supplied specification."""

    def __init__(self, config: SimulationConfig, ap_lookup: dict[str, APNode]):
        self.config = config
        self.rng = Random(config.seed + 701)
        self.evaluator = PhysicalEvaluator(config)
        self.feasibility = FeasibilityManager(config)
        self.assignment_bqm = AssignmentBqmSolver(config)
        self.last_received_state = {
            ap_id: self._snapshot(ap) for ap_id, ap in ap_lookup.items()
        }
        self.static_risk = {
            ap_id: self.evaluator.clip01(1.0 - ap.trust)
            for ap_id, ap in ap_lookup.items()
        }
        self.rate_history: dict[tuple[str, str], list[float]] = defaultdict(list)

    def record_synchronizations(
        self,
        ap_lookup: dict[str, APNode],
        synchronized_ap_ids: set[str],
    ) -> None:
        for ap_id in synchronized_ap_ids:
            self.last_received_state[ap_id] = self._snapshot(ap_lookup[ap_id])

    def decide(
        self,
        algorithm: str,
        tasks: list[Task],
        ap_lookup: dict[str, APNode],
        slot: int,
    ) -> BaselineDecision:
        start = perf_counter()
        normalized = algorithm.lower()
        if normalized in {"greedy", "greedy_link_quality"}:
            mapping = self._rank_and_assign(
                tasks,
                ap_lookup,
                lambda task, ap: (
                    -self._metrics(task, ap).rate_mbps,
                    ap.current_task_load,
                    ap.ap_id,
                ),
            )
            name = "greedy_link_quality"
        elif normalized == "centralized_qubo":
            mapping = self._centralized_qubo(tasks, ap_lookup, slot)
            name = "centralized_qubo_classical"
        elif normalized == "dt_free":
            mapping = self._rank_and_assign(
                tasks,
                ap_lookup,
                self._dt_free_cost,
            )
            name = "dt_free_distributed"
        elif normalized == "channel_only":
            mapping = self._channel_only(tasks, ap_lookup)
            name = "channel_only_predictive"
        elif normalized == "static_risk":
            mapping = self._rank_and_assign(
                tasks,
                ap_lookup,
                lambda task, ap: self.evaluator.baseline_cost(
                    self._metrics(task, ap),
                    risk_value=self.static_risk[ap.ap_id],
                    include_trust=False,
                ),
            )
            name = "static_risk_distributed"
        elif normalized == "aoi_unaware":
            midpoint = 0.5 * (self.config.task_aoi_min + self.config.task_aoi_max)
            mapping = self._rank_and_assign(
                tasks,
                ap_lookup,
                lambda task, ap: self.evaluator.proposed_cost(
                    self._metrics(task, ap, aoi_override=midpoint)
                ),
            )
            name = "aoi_unaware_distributed"
        elif normalized == "trust_unaware":
            mapping = self._rank_and_assign(
                tasks,
                ap_lookup,
                lambda task, ap: self.evaluator.baseline_cost(
                    self._metrics(task, ap),
                    include_trust=False,
                ),
            )
            name = "trust_unaware_distributed"
        elif normalized == "local_qubo":
            mapping = self._local_qubo(tasks, ap_lookup, slot)
            name = "local_qubo_classical"
        elif normalized == "local_search":
            mapping = self._local_search(tasks, ap_lookup)
            name = "local_search"
        elif normalized in {"uncertainty_fallback", "uncertainty_unaware_fallback"}:
            mapping = self._rank_and_assign(
                tasks,
                ap_lookup,
                lambda task, ap: (
                    -self._metrics(task, ap).rate_mbps,
                    ap.current_task_load,
                    ap.ap_id,
                ),
            )
            name = "periodic_trigger_greedy_fallback"
        else:
            raise ValueError(f"Unsupported baseline algorithm: {algorithm}")

        mapping = self.feasibility.repair(mapping, tasks, ap_lookup)
        return BaselineDecision(
            assignment=mapping,
            solver_name=name,
            solver_time=perf_counter() - start,
        )

    def _rank_and_assign(
        self,
        tasks: list[Task],
        ap_lookup: dict[str, APNode],
        cost,
    ) -> dict[str, str]:
        ranked = {
            task.task_id: sorted(
                task.A_u_t,
                key=lambda ap_id: cost(task, ap_lookup[ap_id]),
            )
            for task in tasks
        }
        preferred = {
            task.task_id: ranked[task.task_id][0]
            for task in tasks
        }
        return self.feasibility.repair(
            preferred,
            tasks,
            ap_lookup,
            ranked_candidates=ranked,
        )

    def _channel_only(
        self,
        tasks: list[Task],
        ap_lookup: dict[str, APNode],
    ) -> dict[str, str]:
        observed: dict[tuple[str, str], float] = {}

        def cost(task: Task, ap: APNode) -> float:
            current = self._metrics(task, ap).rate_mbps
            key = (task.source_uav, ap.ap_id)
            history = self.rate_history[key][-3:]
            predicted = sum(history + [current]) / (len(history) + 1)
            observed[key] = current
            metrics = self._metrics(task, ap, rate_override_mbps=predicted)
            return (
                self.config.delay_weight
                * self.evaluator.normalize_delay(metrics.total_delay_s)
                + self.config.energy_weight
                * self.evaluator.normalize_energy(metrics.total_energy_j)
            )

        mapping = self._rank_and_assign(tasks, ap_lookup, cost)
        for key, rate in observed.items():
            self.rate_history[key].append(rate)
        return mapping

    def _centralized_qubo(
        self,
        tasks: list[Task],
        ap_lookup: dict[str, APNode],
        slot: int,
    ) -> dict[str, str]:
        costs = self._cost_table(tasks, ap_lookup)
        return self.assignment_bqm.solve(tasks, ap_lookup, costs).mapping

    def _local_qubo(
        self,
        tasks: list[Task],
        ap_lookup: dict[str, APNode],
        slot: int,
    ) -> dict[str, str]:
        groups: dict[str, list[Task]] = defaultdict(list)
        for task in tasks:
            groups[task.owner_ap_id].append(task)
        mapping: dict[str, str] = {}
        for index, origin in enumerate(sorted(groups)):
            group = groups[origin]
            costs = self._cost_table(group, ap_lookup)
            mapping.update(
                self.assignment_bqm.solve(group, ap_lookup, costs).mapping
            )
        return self.feasibility.repair(mapping, tasks, ap_lookup)

    def _local_search(
        self,
        tasks: list[Task],
        ap_lookup: dict[str, APNode],
    ) -> dict[str, str]:
        costs = self._cost_table(tasks, ap_lookup)
        current = self._greedy_assignment(tasks, costs, ap_lookup)
        current_cost = self._linear_cost(current, costs)
        best = dict(current)
        best_cost = current_cost
        for _ in range(max(self.config.local_search_iterations, 1)):
            task = self.rng.choice(tasks) if tasks else None
            if task is None:
                break
            destinations = [
                ap_id
                for ap_id in task.A_u_t
                if ap_id != current.get(task.task_id)
            ] + ["LOCAL"]
            trial = dict(current)
            trial[task.task_id] = self.rng.choice(destinations)
            if not self.feasibility.assignment_is_feasible(
                trial,
                tasks,
                ap_lookup,
            ):
                continue
            trial_cost = self._linear_cost(trial, costs)
            if trial_cost < current_cost:
                current, current_cost = trial, trial_cost
                if trial_cost < best_cost:
                    best, best_cost = dict(trial), trial_cost
        return best

    def _anneal_assignment(
        self,
        tasks: list[Task],
        costs: dict[str, dict[str, float]],
        ap_lookup: dict[str, APNode],
        salt: int,
    ) -> dict[str, str]:
        if not tasks:
            return {}
        rng = Random(self.config.seed + salt)
        best = self._greedy_assignment(tasks, costs, ap_lookup)
        best_energy = self._penalized_energy(best, tasks, costs, ap_lookup)
        for read in range(max(self.config.anneal_reads, 1)):
            state = dict(best)
            if read:
                for task in tasks:
                    state[task.task_id] = rng.choice(task.A_u_t)
            energy = self._penalized_energy(state, tasks, costs, ap_lookup)
            temperature = max(abs(energy) / max(len(tasks), 1), 1.0)
            for _ in range(max(self.config.anneal_sweeps, 1)):
                order = list(tasks)
                rng.shuffle(order)
                for task in order:
                    trial = dict(state)
                    trial[task.task_id] = rng.choice(task.A_u_t)
                    candidate = self._penalized_energy(
                        trial,
                        tasks,
                        costs,
                        ap_lookup,
                    )
                    delta = candidate - energy
                    if delta <= 0.0 or rng.random() < exp(
                        -delta / max(temperature, 1e-12)
                    ):
                        state, energy = trial, candidate
                temperature *= 0.92
            if energy < best_energy:
                best, best_energy = dict(state), energy
        return self.feasibility.repair(best, tasks, ap_lookup)

    def _greedy_assignment(
        self,
        tasks: list[Task],
        costs: dict[str, dict[str, float]],
        ap_lookup: dict[str, APNode],
    ) -> dict[str, str]:
        ranked = {
            task.task_id: sorted(
                task.A_u_t,
                key=lambda ap_id: costs[task.task_id][ap_id],
            )
            for task in tasks
        }
        return self.feasibility.repair(
            {task.task_id: ranked[task.task_id][0] for task in tasks},
            tasks,
            ap_lookup,
            ranked_candidates=ranked,
        )

    def _cost_table(
        self,
        tasks: list[Task],
        ap_lookup: dict[str, APNode],
    ) -> dict[str, dict[str, float]]:
        return {
            task.task_id: {
                ap_id: self.evaluator.baseline_cost(
                    self._metrics(task, ap_lookup[ap_id])
                )
                for ap_id in task.A_u_t
            }
            for task in tasks
        }

    def _dt_free_cost(self, task: Task, ap: APNode) -> float:
        snapshot = self.last_received_state[ap.ap_id]
        metrics = self.evaluator.evaluate_remote(
            task,
            ap,
            snapshot.load_reference,
            risk_override=snapshot.risk,
            trust_override=snapshot.trust,
            fidelity_override=snapshot.fidelity,
        )
        return (
            self.config.delay_weight
            * self.evaluator.normalize_delay(metrics.total_delay_s)
            + self.config.energy_weight
            * self.evaluator.normalize_energy(metrics.total_energy_j)
        )

    def _metrics(self, task: Task, ap: APNode, **overrides):
        return self.evaluator.evaluate_remote(
            task,
            ap,
            max(float(ap.current_task_load + 1), 1.0),
            **overrides,
        )

    def _snapshot(self, ap: APNode) -> NodeSnapshot:
        return NodeSnapshot(
            load_reference=max(float(ap.current_task_load + 1), 1.0),
            risk=self.evaluator.dynamic_risk(ap),
            trust=ap.trust,
            fidelity=ap.twin_state.fidelity,
        )

    def _penalized_energy(
        self,
        mapping: dict[str, str],
        tasks: list[Task],
        costs: dict[str, dict[str, float]],
        ap_lookup: dict[str, APNode],
    ) -> float:
        energy = self._linear_cost(mapping, costs)
        by_destination: dict[str, list[Task]] = defaultdict(list)
        task_by_id = {task.task_id: task for task in tasks}
        for task_id, destination in mapping.items():
            if destination in ap_lookup:
                by_destination[destination].append(task_by_id[task_id])
        for destination, assigned in by_destination.items():
            capacity = self.feasibility.capacity(ap_lookup[destination])
            cpu = sum(
                self.feasibility.task_demand(task).cpu_cycles
                for task in assigned
            )
            communication = sum(
                self.feasibility.task_demand(task).communication_bits
                for task in assigned
            )
            power = sum(task.power_demand for task in assigned)
            overload = (
                max(0.0, cpu / max(capacity.cpu_cycles, 1e-12) - 1.0) ** 2
                + max(
                    0.0,
                    communication / max(capacity.communication_bits, 1e-12)
                    - 1.0,
                )
                ** 2
                + max(0.0, power / max(capacity.power_watts, 1e-12) - 1.0)
                ** 2
            )
            energy += self.config.qubo_penalty * overload
        return float(energy)

    def _linear_cost(
        self,
        mapping: dict[str, str],
        costs: dict[str, dict[str, float]],
    ) -> float:
        total = 0.0
        for task_id, destination in mapping.items():
            if destination == "LOCAL":
                # Local is only introduced as a feasibility fallback.
                total += self.config.qubo_penalty
            else:
                total += costs[task_id][destination]
        return float(total)
