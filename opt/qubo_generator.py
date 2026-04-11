from __future__ import annotations
import numpy as np
from env.config import SimulationConfig
from env.models import APNode, Task, QuboProblem, CandidateScore

class LocalQuboBuilder: 
    def __init__(self, config: SimulationConfig):
        self.config = config

    def build(self, ap: APNode, tasks: list[Task], ap_lookup: dict[str, APNode], slot: int) -> QuboProblem:
        """
        Constructs the backend-compatible local quadratic surrogate (Eq. 93).
        """
        n_ref = self._compute_local_load(ap, tasks)  
        
        linear_mu = {}
        quadratic_Q = {}
        candidate_scores = []
        rho_m = 1.0 - ap.trust  
        nu = self.config.coord_overhead_weight  
        qubo_penalty = self.config.qubo_penalty

        for i, u in enumerate(tasks):
            b_u = self._calculate_b_coefficient(ap, u, n_ref) 
            a_u = self._calculate_a_coefficient(ap, u, n_ref, b_u) 
            
            var_u = (u.task_id, ap.ap_id)
            
            # Unary effects mu_u (Eq. 96)
            z_bar_1 = ap.coordination_state.dual_price  
            mu_u = (a_u + b_u + (nu / 2.0) - (nu * z_bar_1) + 
                    (nu / 2.0) * (rho_m**2) * (u.psi_u**2) - 
                    (nu * ap.coordination_state.quantized_dual * rho_m * u.psi_u))
            
            linear_mu[var_u] = mu_u - qubo_penalty

            for j in range(i + 1, len(tasks)):
                v = tasks[j]
                b_v = self._calculate_b_coefficient(ap, v, n_ref)
                var_v = (v.task_id, ap.ap_id)
                q_uv = 0.5 * (b_u + b_v) + nu + (nu * (rho_m**2) * u.psi_u * v.psi_u)
                quadratic_Q[(var_u, var_v)] = q_uv

            # Pass individual cost components to the score generator [cite: 323]
            candidate_scores.append(self._generate_candidate_score(ap, u, mu_u, n_ref))

        return QuboProblem(
            ap_id=ap.ap_id,
            slot=slot,
            local_load=n_ref,
            variables=[(t.task_id, ap.ap_id) for t in tasks],
            linear=linear_mu,
            quadratic=quadratic_Q,
            penalty_mu=qubo_penalty,
            penalty_by_task={t.task_id: u.psi_u for t in tasks},
            candidate_scores=candidate_scores
        )

    def _compute_local_load(self, ap: APNode, tasks: list[Task]) -> float:
        queued_cpu = sum(t.cpu_demand for t in tasks)
        return (ap.current_cpu_load + queued_cpu) / max(ap.cpu_capacity, 1.0)

    def _calculate_b_coefficient(self, ap: APNode, task: Task, n_ref: float) -> float:
        delay_grad = (task.L_u * task.D_u) / max(ap.cpu_capacity, 1.0) 
        energy_grad = -2.0 * self.config.kappa_m * task.L_u * task.D_u * (ap.cpu_capacity**2) / (n_ref**3 if n_ref > 0 else 1.0) 
        return self.config.delay_weight * delay_grad + self.config.energy_weight * energy_grad

    def _calculate_a_coefficient(self, ap: APNode, task: Task, n_ref: float, b_u: float) -> float:
        # Re-using local nominal cost calculation 
        delay = (task.L_u / 10.0) + (task.L_u * task.D_u * n_ref) / max(ap.cpu_capacity, 1.0)
        energy = self.config.kappa_m * task.L_u * task.D_u * (ap.cpu_capacity / max(n_ref, 1.0))**2
        gamma_n = self.config.delay_weight * delay + self.config.energy_weight * energy
        return gamma_n - (n_ref * b_u)

    def _generate_candidate_score(self, ap: APNode, task: Task, mu_u: float, n_ref: float) -> CandidateScore:
        """
        Calculates individual physical costs with high precision for analytics [cite: 336-342].
        """
        # 1. Physical Cost Components [cite: 340-342]
        # Delay calculation
        raw_delay = (task.L_u / 10.0) + (task.L_u * task.D_u * n_ref) / max(ap.cpu_capacity, 1.0)
        delay_cost = self.config.delay_weight * raw_delay
        
        # Energy calculation: Ensure kappa_m is non-zero in config.py
        # Energy = kappa * L * D * f^2 
        # Here we use the normalized frequency/capacity squared
        f_m = ap.cpu_capacity 
        raw_energy = self.config.kappa_m * task.L_u * task.D_u * (f_m**2)
        
        # If the value is too small for standard floats, we can apply a 
        # scaling factor for visualization purposes, but for strict 
        # paper adherence, we keep the raw value:
        energy_cost = self.config.energy_weight * raw_energy

        # 2. Twin-based Risk/Freshness Components [cite: 109, 223-227]
        phi_AoI = (1.0 - np.exp(-self.config.eta_u * task.AoI))
        mission_cost = task.psi_u * (1.0 - ap.trust) * phi_AoI
        fidelity_cost = 1.0 - ap.twin_state.fidelity 
        
        return CandidateScore(
            task_id=task.task_id,
            owner_ap_id=task.owner_ap_id,
            destination_id=ap.ap_id,
            local_cost=float(mu_u),
            delay_cost=float(delay_cost),
            energy_cost=float(energy_cost), # Ensure this isn't getting clipped
            mission_cost=float(mission_cost),
            fidelity_cost=float(fidelity_cost),
            coupling_penalty=float(ap.coordination_state.coupling_penalty),
            projected_load=float(ap.current_cpu_load),
            required_bandwidth=task.bandwidth_demand,
            required_cpu=task.cpu_demand,
            required_power=task.power_demand
        )