from __future__ import annotations
from math import exp, sqrt
from random import Random
import numpy as np
from env.config import SimulationConfig
from env.models import APNode, Observation

class TwinManager:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.rng = Random(config.seed + 101)

    def update(self, ap: APNode, observation: Observation) -> tuple[bool, bool]:
        """
        Closes the loop between physical observations and twin evolution[cite: 7].
        Returns: (sync_triggered, coordination_triggered)
        """
        twin = ap.twin_state
        beta = self.config.twin_smoothing  # Corresponding to beta parameters in [cite: 119-127]
        
        # 1. Physical Observation Vector s_m(t) [cite: 115]
        # Components: observed channel/load/reliability/anomaly/mission-pressure
        s_obs = np.array([
            observation.load_ratio,
            observation.bandwidth_ratio,
            observation.cpu_ratio,
            # Placeholder for anomaly/reliability from observation model
            0.0, 
            observation.candidate_overlap 
        ])

        # 2. Extract current Twin Predicted State s_hat_m(t) [cite: 136]
        s_hat = np.array([
            twin.predicted_load,
            twin.predicted_bandwidth,
            twin.predicted_cpu_ratio,
            0.0, # Reliability prediction
            twin.predicted_cpu_ratio # Mission pressure summary prediction
        ])

        # 3. Evaluate Twin Mismatch and Uncertainty [cite: 141, 135]
        # epsilon_m(t) = ||s_m(t) - s_hat_m(t)||_2
        mismatch = np.linalg.norm(s_obs - s_hat)
        normalized_mismatch = min(1.0, mismatch / self.config.observation_ratio_clip)
        
        # Sigma_m(t+1) = beta*Sigma_m(t) + (1-beta)*||innovation||^2 
        innovation_energy = np.sum((s_obs - s_hat)**2)
        uncertainty = (beta * twin.uncertainty) + (1.0 - beta) * innovation_energy

        # 4. Evaluate Twin Fidelity F_m(t) 
        # F_m(t) = exp(-kappa_eps * epsilon_bar - kappa_A * age)
        # Using paper coefficients: kappa_eps=1.0, kappa_A=0.1 (implied by thresholds)
        fidelity = exp(-1.0 * normalized_mismatch - 0.1 * twin.age)

        # 5. Evaluate Synchronization Trigger a_m^sync(t) 
        # Triggered by Age, Uncertainty, Mismatch, or Fidelity Requirements
        sync_trigger = (
            twin.age > self.config.sync_age_threshold or
            uncertainty > self.config.sync_uncertainty_threshold or
            normalized_mismatch > self.config.sync_mismatch_threshold or
            fidelity < 0.5 # Baseline fidelity requirement
        )

        # 6. Perform State Update [cite: 119-132]
        if sync_trigger:
            # Refresh using newly observed physical information
            twin.predicted_load = s_obs[0]
            twin.predicted_bandwidth = s_obs[1]
            twin.predicted_cpu_ratio = s_obs[2]
            twin.age = 1 # Reset age to 1 
        else:
            # Passive evolution (Age increases, state remains constant/drifts)
            twin.age += 1 

        # Update remaining twin metrics
        twin.uncertainty = uncertainty
        twin.mismatch = normalized_mismatch
        twin.fidelity = fidelity
        twin.last_observation = tuple(s_obs[:3])

        # 7. Update Trust State tau_m(t+1) [cite: 157]
        # tau_m(t+1) = projection((1-zeta)*tau_m(t) + zeta*[a1*q + a2*(1-eps) + a3*(1-chi)])
        ap.trust = self._update_trust(ap, s_obs, normalized_mismatch)

        # 8. Evaluate Regional Coordination Trigger chi_m^trig(t) 
        coordination_trigger = (
            abs(ap.trust - twin.mismatch) > self.config.coord_trust_threshold or # Trust variation
            mismatch > self.config.coord_state_threshold or
            uncertainty > self.config.coord_uncertainty_threshold or
            normalized_mismatch > self.config.coord_mismatch_threshold
        )

        return sync_trigger, coordination_trigger

    def _update_trust(self, ap: APNode, s_obs: np.ndarray, norm_mismatch: float) -> float:
        """Implements the Trust Evolution Equation[cite: 157]."""
        zeta = self.config.trust_update_factor
        a1, a2, a3 = self.config.trust_weights
        
        # Reliability (q), mismatch-inverse (1-eps), anomaly-inverse (1-chi)
        q_obs = 1.0 - (s_obs[0] * 0.1) # Simplified reliability mapping
        innovation = (a1 * q_obs) + (a2 * (1.0 - norm_mismatch)) + (a3 * 1.0)
        
        new_trust = (1.0 - zeta) * ap.trust + zeta * innovation
        return min(1.0, max(0.0, new_trust)) # Projection onto [0,1]