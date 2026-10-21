from __future__ import annotations

from dataclasses import dataclass
from math import exp

from .config import SimulationConfig
from .link_model import free_space_path_loss_db, large_scale_gain_from_path_loss, predicted_uplink_rate
from .models import APNode, Task


@dataclass(frozen=True, slots=True)
class CandidateMetrics:
    rate_mbps: float
    transmission_delay_s: float
    computation_delay_s: float
    synchronization_delay_s: float
    coordination_delay_s: float
    total_delay_s: float
    transmission_energy_j: float
    computation_energy_j: float
    synchronization_energy_j: float
    total_energy_j: float
    dynamic_risk: float
    freshness_risk: float
    trust: float
    fidelity_penalty: float
    synchronization_cost: float
    coordination_cost: float


class PhysicalEvaluator:
    """Single unit-safe physical and objective evaluator used by every method."""

    def __init__(self, config: SimulationConfig):
        self.config = config

    def evaluate_remote(
        self,
        task: Task,
        ap: APNode,
        load_count: float,
        *,
        rate_override_mbps: float | None = None,
        risk_override: float | None = None,
        trust_override: float | None = None,
        fidelity_override: float | None = None,
        aoi_override: float | None = None,
        sync_active: bool = False,
        coord_active: bool = False,
    ) -> CandidateMetrics:
        load = max(float(load_count), 1.0)
        rate_mbps = (
            float(rate_override_mbps)
            if rate_override_mbps is not None
            else predicted_uplink_rate(task, ap, self.config, load)
        )
        rate_mbps = max(rate_mbps, 1e-12)

        input_bits = task.L_u * self.config.mbits_to_bits
        rate_bits_per_second = rate_mbps * self.config.mbps_to_bits_per_second
        cycles = input_bits * task.D_u
        cpu_share_hz = (
            ap.cpu_capacity
            * self.config.ghz_to_hz
            / load
        )

        transmission_delay_s = input_bits / rate_bits_per_second
        computation_delay_s = cycles / max(cpu_share_hz, 1e-12)
        synchronization_delay_s = (
            float(self.config.sync_delay_by_tier.get(ap.tier, 0.0))
            if sync_active
            else 0.0
        )
        coordination_delay_s = (
            float(self.config.coord_signal_delay_by_tier.get(ap.tier, 0.0))
            if coord_active
            else 0.0
        )
        total_delay_s = (
            transmission_delay_s
            + computation_delay_s
            + synchronization_delay_s
            + coordination_delay_s
        )

        transmission_energy_j = (
            self.config.uav_transmit_power * transmission_delay_s
        )
        computation_energy_j = (
            self.config.kappa_m * cycles * cpu_share_hz**2
        )
        synchronization_energy_j = (
            float(self.config.sync_energy_by_tier.get(ap.tier, 0.0))
            if sync_active
            else 0.0
        )
        total_energy_j = (
            transmission_energy_j
            + computation_energy_j
            + synchronization_energy_j
        )

        dynamic_risk = (
            self.dynamic_risk(ap)
            if risk_override is None
            else self.clip01(risk_override)
        )
        trust = ap.trust if trust_override is None else self.clip01(trust_override)
        fidelity = (
            ap.twin_state.fidelity
            if fidelity_override is None
            else self.clip01(fidelity_override)
        )
        decision_aoi = task.AoI if aoi_override is None else float(aoi_override)
        freshness = 1.0 - exp(-self.config.eta_u * decision_aoi)
        freshness_risk = (
            task.omega_u
            * task.psi_u
            * (1.0 - trust)
            * freshness
        )

        return CandidateMetrics(
            rate_mbps=float(rate_mbps),
            transmission_delay_s=float(transmission_delay_s),
            computation_delay_s=float(computation_delay_s),
            synchronization_delay_s=float(synchronization_delay_s),
            coordination_delay_s=float(coordination_delay_s),
            total_delay_s=float(total_delay_s),
            transmission_energy_j=float(transmission_energy_j),
            computation_energy_j=float(computation_energy_j),
            synchronization_energy_j=float(synchronization_energy_j),
            total_energy_j=float(total_energy_j),
            dynamic_risk=float(dynamic_risk),
            freshness_risk=float(freshness_risk),
            trust=float(trust),
            fidelity_penalty=float(1.0 - fidelity),
            synchronization_cost=float(sync_active),
            coordination_cost=float(coord_active),
        )

    def evaluate_local(self, task: Task) -> CandidateMetrics:
        input_bits = task.L_u * self.config.mbits_to_bits
        cycles = input_bits * task.D_u
        cpu_hz = self.config.uav_local_cpu_capacity * self.config.ghz_to_hz
        delay_s = cycles / max(cpu_hz, 1e-12)
        energy_j = self.config.kappa_local * cycles * cpu_hz**2
        trust = self.config.local_execution_trust
        freshness = 1.0 - exp(-self.config.eta_u * task.AoI)
        freshness_risk = (
            task.omega_u
            * task.psi_u
            * (1.0 - trust)
            * freshness
        )
        return CandidateMetrics(
            rate_mbps=0.0,
            transmission_delay_s=0.0,
            computation_delay_s=float(delay_s),
            synchronization_delay_s=0.0,
            coordination_delay_s=0.0,
            total_delay_s=float(delay_s),
            transmission_energy_j=0.0,
            computation_energy_j=float(energy_j),
            synchronization_energy_j=0.0,
            total_energy_j=float(energy_j),
            dynamic_risk=float(1.0 - trust),
            freshness_risk=float(freshness_risk),
            trust=float(trust),
            fidelity_penalty=0.0,
            synchronization_cost=0.0,
            coordination_cost=0.0,
        )

    def baseline_cost(
        self,
        metrics: CandidateMetrics,
        *,
        risk_value: float | None = None,
        include_trust: bool = True,
    ) -> float:
        risk = metrics.dynamic_risk if risk_value is None else self.clip01(risk_value)
        cost = (
            self.config.delay_weight * self.normalize_delay(metrics.total_delay_s)
            + self.config.energy_weight * self.normalize_energy(metrics.total_energy_j)
            + self.config.mission_weight * self.normalize_risk(risk)
        )
        if include_trust:
            cost -= self.config.trust_reward_weight * self.normalize_trust(metrics.trust)
        return float(cost)

    def proposed_cost(
        self,
        metrics: CandidateMetrics,
        *,
        freshness_risk_override: float | None = None,
        include_fidelity: bool = True,
        include_sync: bool = True,
        include_coordination: bool = True,
    ) -> float:
        freshness_risk = (
            metrics.freshness_risk
            if freshness_risk_override is None
            else float(freshness_risk_override)
        )
        return float(
            self.config.delay_weight * self.normalize_delay(metrics.total_delay_s)
            + self.config.energy_weight * self.normalize_energy(metrics.total_energy_j)
            + self.config.mission_weight
            * self.normalize_freshness_risk(freshness_risk)
            + (
                self.config.fidelity_weight * metrics.fidelity_penalty
                if include_fidelity
                else 0.0
            )
            + (
                self.config.sync_cost_weight * metrics.synchronization_cost
                if include_sync
                else 0.0
            )
            + (
                self.config.coord_overhead_weight * metrics.coordination_cost
                if include_coordination
                else 0.0
            )
            - self.config.trust_reward_weight
            * self.normalize_trust(metrics.trust)
        )

    def service_cost(
        self, metrics: CandidateMetrics, *, include_mission: bool = True,
    ) -> float:
        """Load-dependent C_u,m(N;t) used by paper Eqs. (28)-(44)."""
        return float(
            self.config.delay_weight * self.normalize_delay(metrics.total_delay_s)
            + self.config.energy_weight * self.normalize_energy(metrics.total_energy_j)
            + (
                self.config.mission_weight
                * self.normalize_freshness_risk(metrics.freshness_risk)
                + self.config.sync_cost_weight * metrics.synchronization_cost
                + self.config.coord_overhead_weight * metrics.coordination_cost
                if include_mission else 0.0
            )
        )

    def service_cost_derivative(self, task: Task, ap: APNode, load_count: float) -> float:
        """Analytic dC_u,m(N;t)/dN for the simulator's equal-share model."""
        from math import log, log2

        n = max(float(load_count), 1.0)
        input_bits = task.L_u * self.config.mbits_to_bits
        cycles = input_bits * task.D_u
        allocated_cpu_hz = ap.cpu_capacity * self.config.ghz_to_hz

        if ap.tier == "BS":
            delay_tx_derivative = input_bits / max(
                ap.bandwidth * self.config.mbps_to_bits_per_second, 1e-12
            )
        else:
            path_loss = free_space_path_loss_db(task, ap, self.config)
            gain = large_scale_gain_from_path_loss(path_loss)
            a = self.config.uav_transmit_power * gain / max(
                self.config.noise_spectral_density * ap.bandwidth, 1e-18
            )
            log_term = log2(1.0 + a * n)
            delay_scale = input_bits / max(
                ap.bandwidth * self.config.mbps_to_bits_per_second, 1e-12
            )
            rate_shape_derivative = (
                log_term - (a * n / ((1.0 + a * n) * log(2.0)))
            ) / max(log_term * log_term, 1e-18)
            delay_tx_derivative = delay_scale * rate_shape_derivative

        delay_derivative = delay_tx_derivative + cycles / max(allocated_cpu_hz, 1e-12)
        energy_derivative = (
            self.config.uav_transmit_power * delay_tx_derivative
            - 2.0 * self.config.kappa_m * cycles * allocated_cpu_hz**2 / n**3
        )
        return float(
            self.config.delay_weight * delay_derivative / (
                max(self.config.delay_reference_s, 1e-12)
                if self.config.normalize_objective else 1.0
            )
            + self.config.energy_weight * energy_derivative / (
                max(self.config.energy_reference_j, 1e-12)
                if self.config.normalize_objective else 1.0
            )
        )
    def normalize_delay(self, delay_s: float) -> float:
        return self._normalize(delay_s, self.config.delay_reference_s)

    def normalize_energy(self, energy_j: float) -> float:
        return self._normalize(energy_j, self.config.energy_reference_j)

    def normalize_risk(self, risk: float) -> float:
        return self._normalize(risk, self.config.risk_reference)

    def normalize_freshness_risk(self, risk: float) -> float:
        return self._normalize(risk, self.config.freshness_risk_reference)

    def normalize_trust(self, trust: float) -> float:
        return self._normalize(trust, self.config.trust_reference)

    @staticmethod
    def dynamic_risk(ap: APNode) -> float:
        return PhysicalEvaluator.clip01(
            0.5 * ap.twin_state.mismatch
            + 0.5 * min(ap.twin_state.uncertainty, 1.0)
        )

    @staticmethod
    def clip01(value: float) -> float:
        return min(1.0, max(0.0, float(value)))

    def _normalize(self, value: float, reference: float) -> float:
        if not self.config.normalize_objective:
            return float(value)
        return float(value) / max(float(reference), 1e-12)
