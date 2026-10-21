# DTN-SAGIN Closed-Loop Assignment Simulator

This repository implements the hierarchical digital-twin network controller described in `Emergency-Aware Closed-Loop Digital Twin Networks for MEC-Enabled SAGINs `. The proposed path now runs the local QUBO iterations independently at each infrastructure node, updates the positive load reference and coordination multiplier inside that loop, and then applies the paper's fidelity-aware regional score selection.

The implementation is a classical simulation of the paper's QUBO-based controller. `classical` uses the configured local classical solver (exact enumeration for small problems and simulated annealing otherwise); `dwave_hybrid` is an optional external hybrid backend. The simulation does not require a quantum computer.

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
├── iterative_results/
│
├── results/
│
├── reference_measurements/
│
├── Convergence_analysis.py
│
├── convergence_results
│
└── Local_TESTS 

Note:- The files under LocalTESTS are for testing each aspect of model (like sim, env, twin etc.) seperately
```
---

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

# 4. Set up

Open the folder containing this README in VS Code, then open a PowerShell terminal in that folder. Python 3.10 or newer is recommended.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` in that terminal, then activate again.

# 5. Run one simulation

```powershell
python -m sim.main --slots 30 --seed 7 --reads 2 --sweeps 5 --max-qubo-iterations 10 --output-dir results\seed_7
```

The single-run CLI also accepts `--solver-backend classical|exact|annealing|dimod|neal|dwave_hybrid|none`. `classical` is the default. `--reads` and `--sweeps` configure classical annealing; they do not control the QUBO inner-iteration limit. The proposed method's per-node inner-loop limit is `max_qubo_iterations` in `env/config.py` (default 10).

Each run writes a timestamped directory containing `summary.json`, `simulation_parameters.json`, per-slot metrics, assignments, and digital-twin metrics. Do not reuse an output directory when you want an unambiguous experiment record.

# 6. Reproduce the full sweep study

From the project root, run:

```powershell
python generate_measured_baseline_plots.py `
  --rerun-simulations `
  --slots 30 `
  --seeds 7,13,21,31,43 `
  --anneal-reads 2 `
  --anneal-sweeps 5 `
  --max-qubo-iterations 10 `
  --tasks 20,40,60,80,100 `
  --aoi-values 1,3,5,7,9,11 `
  --speeds 5,10,15,20,25,30 `
  --sync-thresholds 1,2,3,4,5,6 `
  --coord-thresholds 0.05,0.10,0.15,0.20,0.25 `
  --tradeoff-thresholds 1,2,3,4,5,6,7,8 `
  --output-dir iterative_results_algorithm1
```

The backtick at the end of each PowerShell line continues the command. You can put the whole command on one line if preferred. The study uses 30 simulation slots per run, five fixed seeds, two annealing reads, five annealing sweeps, and ten local QUBO iterations per infrastructure node/slot (unless `--max-qubo-iterations` is changed). The first 10 slots are excluded from the reported averages when the run is longer than 10 slots; all slots remain in the saved raw run files.

### Sweep points and methods

| Sweep CSV | X values | Methods | Runs |
|---|---:|---:|---:|
| `uav_task_sweep_raw.csv` | 20, 40, 60, 80, 100 UAVs | 12 methods | 300 |
| `aoi_sweep_raw.csv` | 1, 3, 5, 7, 9, 11 | 6 methods | 180 |
| `mobility_sweep_raw.csv` | 5, 10, 15, 20, 25, 30 m/s | 6 methods | 180 |
| `sync_threshold_sweep_raw.csv` | 1, 2, 3, 4, 5, 6 slots | 3 methods | 90 |
| `coord_threshold_sweep_raw.csv` | 0.05, 0.10, 0.15, 0.20, 0.25 | 3 methods | 75 |
| `tradeoff_sweep_raw.csv` | 1, 2, 3, 4, 5, 6, 7, 8 | 3 methods | 120 |
| **Total** |  |  | **945** |

The 12 methods in the task-count sweep are `proposed`, `greedy`, `centralized_qubo`, `channel_only`, `dt_free`, `static_risk`, `aoi_unaware`, `trust_unaware`, `always_on`, `local_qubo`, `local_search`, and `uncertainty_unaware`. The AoI, mobility, synchronization, coordination, and tradeoff method subsets are defined in `generate_measured_baseline_plots.py` and are saved into the output `study_config.json`.

The experiment creates one row per method, x-value, and seed: 945 rows with the current method list and defaults. An older archived bundle under `reference_measurements` contains 920 rows from an earlier method configuration. It is historical output from the prior code and should not be treated as output from the updated controller.

### What the study writes

- `study_config.json`: all run arguments, sweep values, and method settings.
- `*_raw.csv`: measured metrics, one row per method/x-value/seed.
- `raw_runs/<sweep>/x_<value>/<method>/seed_<seed>/run_<timestamp>/`: complete simulation output for each row.
- `figures/`: nine PNG plots generated from the raw measured CSV values.
- `paste_ready_figures/`: copies of the figures with presentation-friendly names.

The sweep is computationally intensive: each run includes 30 slots, and every proposed/iterative method performs one local QUBO solve per active infrastructure node and inner iteration. Keep the terminal open until the script reports that measurements and figures are complete. Use a new `--output-dir` for each full rerun so run artifacts and CSVs do not mix.

# 7. Faster iteration

For a fast implementation pass, use fewer slots, one seed, and a small subset of x-values. For example:

```powershell
python generate_measured_baseline_plots.py --rerun-simulations --slots 12 --seeds 7 --anneal-reads 1 --anneal-sweeps 2 --max-qubo-iterations 3 --tasks 20,40 --aoi-values 1,5 --speeds 5,15 --sync-thresholds 1,3 --coord-thresholds 0.05,0.15 --tradeoff-thresholds 1,4 --output-dir iterative_smoke
```

This is for checking that the workflow runs; it is not the full five-seed study.

# 8. Replot saved measurements without simulation

```powershell
python generate_measured_baseline_plots.py --replot-from iterative_results_algorithm1 --output-dir replotted_figures
```

This reads the six `*_raw.csv` files and produces figures. It does not launch any simulator runs. Plot rendering applies Savitzky-Golay smoothing to the displayed line where enough x-points exist; the raw CSV values are not smoothed or modified.

# 9. How the proposed controller follows Algorithm 

At the start of each slot, the simulator updates mobility, generates tasks, updates each twin, and evaluates synchronization and coordination triggers. A triggered node reports its latest realized coordination summary; the coordinator retains each report and freezes its reference vector for the current slot.

For every infrastructure node, the controller initializes a positive reference load and starts with the previous slot's coordination multiplier. For each `k` up to `max_qubo_iterations` it computes the first-order cost coefficients around the current reference load (Eq. 30), constructs the local QUBO (Eqs. 33-35), solves for the node's task candidate vector, updates the positive reference load (Eq. 39), conditionally updates the multiplier when coordination is triggered (Eq. 40), measures the coordination residual (Eq. 41), and stops when the candidate vector is unchanged. It then sends candidate/service scores to the regional controller.

The regional controller applies the task-wise minimum in Eqs. 42-44 over infrastructure candidates that pass the task's fidelity requirement, with local execution available as the fallback. It recomputes destination loads, evaluates the executed assignments, and stores their realized coordination summaries for future trigger-based reports. Regional selection uses the final local reference-load service scores; outcome metrics are recomputed from the selected mapping after execution.

# 10. Important reproducibility notes

- Seeds are set per run and recorded in each run's `simulation_parameters.json`; the sweep seed list is also saved in `study_config.json`.
- Re-running with the same code, Python/dependency versions, parameters, and seed reproduces the generated environment and stochastic inputs. Annealing backends and library/platform versions can still affect solver samples.
- The updated local QUBO/controller changes experiment outcomes. Existing CSVs and figures are not regenerated automatically; run the full sweep command to create new measurements.
- `reference_measurements/` and the pre-existing `iterative_results_full/` files are archived measurements from earlier implementation states. They are useful for comparison and replotting, but they are not evidence that the revised algorithm produced those values.



PS:- An extension to this project (where the outer loop of the architecture is managed by Transformer and LSTM based learners) is here:
https://github.com/Vijiboi/Agentic_DTN-Transformer
