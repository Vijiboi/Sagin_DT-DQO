# Digital Twin Driven SAGIN-Quantum Simulation

## Overview

This repository implements a high-fidelity discrete-time simulation framework for task orchestration in **Mobile Edge Computing (MEC)** enabled **Satellite-Aerial-Ground Integrated Networks (SAGIN)**.

Unlike conventional Digital Twin (DT) architectures where the twin acts only as a passive estimator, this framework treats the Digital Twin as an **active control object**. Runtime health metrics such as:

* Age of Twin: $A_m^{tw}$
* Predictive Uncertainty: $\Sigma_m$
* Twin Mismatch: $\epsilon_m$
* Fidelity Score: $F_m$

directly influence:

* Task scheduling decisions
* Event-triggered synchronization
* Regional consensus coordination
* Resource allocation policies

---

# 1. Theoretical Architecture

The framework follows a hierarchical two-timescale optimization architecture for stochastic mixed-integer task allocation under communication, computation, and fidelity constraints.

---

## 1.1 Physical Layer and Multi-Tier Channel Model

The SAGIN infrastructure consists of:

* Terrestrial Base Stations: $\mathcal{B}$
* High Altitude Platforms (HAPs): $\mathcal{H}$
* Low Earth Orbit Satellites (LEOs): $\mathcal{S}$

Drone agents:

$$
\mathcal{U}
$$

generate computational tasks characterized by:

* Data size: $L_u$ (bits)
* Computational density: $D_u$ (CPU cycles/bit)

### Free Space Path Loss

$$ PL_{u,m}^{FS}(t) = 32.45 + 20 \log_{10}(f_c) + 20 \log_{10}(d_{u,m}(t)) $$

where:

* $f_c$ = carrier frequency (GHz)
* $d_{u,m}(t)$ = UAV-node distance (km)

### Achievable Uplink Capacity

$$ R_{u,m}(t) = b_{u,m}(t)\log_2\left(1+\frac{P_u g_{u,m}(t)}{N_0 b_{u,m}(t)}\right) $$

with

$$
b_{u,m}(t)
\approx
\frac{B_m}{N_m(t)}
$$

where:

* $P_u$ = UAV transmit power
* $g_{u,m}(t)$ = channel gain
* $N_0$ = noise power density
* $B_m$ = node bandwidth
* $N_m(t)$ = active task count

---

## 1.2 Digital Twin State Evolution

Each infrastructure node maintains a virtual state estimate:

$$
\hat{s}_m(t)
$$

using EWMA tracking.

### State Update

$$ \hat{l}_m(t+1) = \beta_l l_m^{obs}(t) + (1-\beta_l)\hat{l}_m(t) $$

### Twin Mismatch

$$ \epsilon_m^{tw}(t) = \left| s_m(t)-\hat{s}_m(t) \right|_2 $$

### Twin Fidelity

$$ F_m(t) = \exp \left( -\kappa_{\epsilon}\overline{\epsilon}_m^{tw}( \kappa_A A_m^{tw}(t) \right) $$

Mission-critical tasks require

$$
F_m(t)\ge F_u^{min}
$$

before offloading is permitted.

---

## 1.3 Outer Loop: Consensus and Event Triggers

Synchronization and coordination are activated only when thresholds on:

* Twin age
* Uncertainty
* Fidelity degradation

are exceeded.

### Dual Price Update

$$ \lambda_m^{(k+1)} = \max \left( 0, \lambda_m^{(k)} + \eta_k \Delta_m \right) $$

where:

* $\lambda_m$ = congestion price
* $\eta_k$ = learning rate
* $\Delta_m$ = local resource imbalance

---

## 1.4 Inner Loop: QUBO Transformation

To construct a QUBO formulation, load-dependent delay and energy functions are linearized around an operating point:

$$
\overline{N}_m^{(k)}
$$

### First-Order Taylor Approximation

$$
\Gamma_{u,m}(N)
\approx
\Gamma_{u,m}
\left(
\overline{N}*m^{(k)}
\right)
+
\Gamma'*{u,m}
\left(
\overline{N}_m^{(k)}
\right)
\left(
N_m(t)-\overline{N}_m^{(k)}
\right)
$$

which simplifies to:

$$ \Gamma_{u,m}(N) = A_{u,m}^{(k)} + B_{u,m}^{(k)}N_m(t) $$

with

$$ N_m(t) = \sum_{v\in\mathcal{U}} b_{v,m}(t) $$

### QUBO Coupling Matrix

$$ [Q_m^{(k)}]_{u,v} = \frac{1}{2} \left( B_{u,m}^{(k)} + B_{v,m}^{(k)} \right) + \nu + \nu \rho_m^2(t)\psi_u\psi_v $$

---

# 2. Directory Structure

```text
├── env/
│   ├── config.py
│   ├── models.py
│   ├── link_model.py
│   ├── mobility.py
│   ├── graph.py
│   └── generator.py
│
├── twin/
│   ├── __init__.py
│   └── twin_logic.py
│
├── opt/
│   ├── __init__.py
│   ├── qubo_generator.py
│   ├── solver.py
│   └── hybrid.py
│
├── control/
│   ├── __init__.py
│   ├── consensus.py
│   └── regional.py
│
├── sim/
│   ├── checks.py
│   └── main.py
│
├── results/
│
└── analyze_algorithm_performance.py
```

---

# 3. Execution Pipeline

```text
[1. Mobility & Task Generation]
                |
                v
[2. Twin Health Assessment]
                |
                v
[3. Regional Coordination Mesh]
                |
                v
[4. Local Matrix Compilation]
                |
                v
[5. Optimization Execution]
                |
                v
[6. Hierarchical Projection]
```

---

# 4. Installation

## Dependencies

```bash
pip install numpy matplotlib dwave-neal dimod dwave-system
```

---

## Unit Testing

```bash
python -m unittest test_env.py

python -m unittest test_twin.py

python -m unittest test_opt.py

python -m unittest test_control.py
```

---

# 5. Running Simulations

## Classical Backend

```bash
python -m sim.main \
    --slots 50 \
    --seed 7 \
    --solver-backend classical \
    --output-dir results
```

---

## Long-Horizon Trend Analysis

```bash
python generate_reference_trend_plots.py \
  --slots 500 \
  --seeds 7,13,21,31,43 \
  --focus-uavs 20 \
  --densities 10,20,30,40,50 \
  --consensus-steps 0.15,0.25,0.50,0.75 \
  --delay-weights 0.5,1.0,2.0,4.0 \
  --output-dir reference_trend_results
```

### Generated Figures

* `fig_time_average_delay_energy.png`
* `fig_consensus_step_convergence.png`
* `fig_delay_weight_convergence.png`
* `fig_density_delay_energy_tradeoff.png`

---

# 6. Quantum Hardware Integration

The Digital Twin and Consensus layers remain classical.

Only the inner-loop QUBO optimization is executed on quantum hardware.

---

## D-Wave Authentication

```bash
dwave config create
```

Enter your Leap API token when prompted.

---

## Quantum Backend Execution

```bash
python -m sim.main \
    --slots 50 \
    --seed 7 \
    --solver-backend dwave_hybrid \
    --output-dir results
```
