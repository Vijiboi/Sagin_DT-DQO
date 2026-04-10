from __future__ import annotations

from math import exp
from random import Random

from env.config import SimulationConfig
from env.models import APNode, Observation


class TwinManager:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.rng = Random(config.seed + 101)

    def update(self, ap: APNode, observation: Observation) -> tuple[bool, bool]:
        twin = ap.twin_state
        alpha = self.config.twin_smoothing
        observed = (
            observation.load_ratio,
            observation.bandwidth_ratio,
            observation.cpu_ratio,
        )
        forecast = self._forecast_state(twin)
        deltas = [abs(obs - pred) for obs, pred in zip(observed, forecast)]
        mismatch = sum(deltas) / len(deltas)
        uncertainty = max(deltas)
        fidelity = exp(-2.5 * mismatch)

        sync_trigger = (
            twin.age >= self.config.sync_age_threshold
            or mismatch >= self.config.sync_mismatch_threshold
            or uncertainty >= self.config.sync_uncertainty_threshold
        )

        if sync_trigger:
            posterior = observed
            twin.age = 0
        else:
            posterior = tuple(
                alpha * prior + (1.0 - alpha) * obs for prior, obs in zip(forecast, observed)
            )
            twin.age += 1

        twin.predicted_load = posterior[0]
        twin.predicted_bandwidth = posterior[1]
        twin.predicted_cpu_ratio = posterior[2]
        twin.uncertainty = uncertainty
        twin.mismatch = mismatch
        twin.fidelity = fidelity
        twin.last_observation = observed
        twin.forecast_state = forecast

        ap.trust = self._updated_trust(ap.trust, fidelity, uncertainty, twin.age)
        coordination_trigger = (
            observation.load_ratio >= self.config.coordination_load_threshold
            or observation.candidate_overlap >= self.config.coordination_overlap_threshold
            or ap.trust <= self.config.coordination_trust_threshold
        )
        return sync_trigger, coordination_trigger

   
    def _updated_trust(self, current_trust: float, fidelity: float, uncertainty: float, age: int) -> float:
        age_penalty = (age / self.config.sync_age_threshold) * 0.05
        next_trust = 0.70 * current_trust + 0.30 * fidelity - 0.10 * uncertainty - age_penalty
        return min(0.99, max(0.10, next_trust))

    def _forecast_state(self, twin) -> tuple[float, float, float]:
        base = (
            twin.predicted_load,
            twin.predicted_bandwidth,
            twin.predicted_cpu_ratio,
        )
        forecast = []
        for value in base:
            innovation = self.rng.gauss(0.0, self.config.twin_gaussian_std)
            forecast.append(min(1.5, max(0.0, value + innovation)))
        return tuple(forecast)
