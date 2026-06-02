# SAGIN-Quantum Simulation Framework

This repository implements the hierarchical, closed-loop Digital Twin Network (DTN) control architecture for emergency-response task orchestration in MEC-enabled Satellite-Aerial-Ground Integrated Networks (SAGIN) . The framework treats the digital twin as an active control object rather than a passive estimator, where twin fidelity directly influences task admissibility

## Folder Layout

- `env/`: Core SAGIN environment. Handles 3D mobility, task generation with randomized weights, and a directed communication graph for the 50-UAV swarm..
- `twin/`: The Digital Twin (DT) manager. Implements Gaussian innovation models to track state fidelity, uncertainty, and mismatch triggers for autonomous synchronization.
- `opt/`: The Optimization engine. Translates offloading costs into QUBO (Quadratic Unconstrained Binary Optimization) forms, featuring multi-tier coupling penalties to prevent resource hotspots.
- `control/`: Quantized consensus state, regional projection, and final one-hot assignment enforcement with resource budgets.
- `sim/`: End-to-end simulator runner and CLI entrypoint.
- `results/`: metrics aggregation and CSV/JSON writers.
- `test_result` & `final_simulation_output`: These directories were utilized for Phase 1 stress-testing and parameter tuning. They contain historical data verifying the transition from resource-blind to resource-aware offloading.

## Run

```powershell
python -m sim.main --slots 50 --seed 7 --output-dir results
```

To select a local backend explicitly:

```powershell
python -m sim.main --slots 50 --seed 7 --solver-backend classical --output-dir results
```
## Testing & Validation
The repository includes a comprehensive unit-testing suite to verify the logic of individual tiers before full integration:
`test_env.py`: Validates SAGIN node generation and mobility boundaries.
`test_twin.py`: Tests the Digital Twin's ability to trigger sync events based on state mismatch.
`test_opt.py`: Verifies that the QUBO builder generates valid "one-hot" assignments.
`test_control.py`: Confirms that the Quantized Consensus correctly increases "dual prices" under high resource pressure.
`test_sim.py`: A lightweight end-to-end runner used for rapid logic verification.

# Reference-Style Trend Plots

Use `generate_reference_trend_plots.py` when the goal is to produce stable,
paper-style trend figures rather than raw per-slot fluctuation plots.

The script does not hard-code values. It improves trend visibility by using:

- longer simulation horizons,
- multi-seed averaging,
- cumulative time-average delay and energy,
- total AP energy rather than mislabeled per-task average energy,
- a stable hyperparameter profile,
- a sweep over the consensus learning rate.
- a sweep over the SAGIN objective delay weight `alpha`.

The stable profile uses:

- `twin_smoothing = 0.80`
- `sensor_filter_factor = 0.80`
- `trust_update_factor = 0.20`
- `consensus_step_size = 0.25`
- `consensus_quantum = 0.05`
- `consensus_epsilon = 0.03`
- `anneal_reads = 20`
- `anneal_sweeps = 80`

These choices reduce slot-to-slot oscillation by slowing the DT/trust updates,
using a smaller consensus learning rate, and making the classical QUBO backend
more stable.

Recommended command for final plots:

```powershell
python generate_reference_trend_plots.py --slots 500 --seeds 7,13,21,31,43 --focus-uavs 20 --densities 10,20,30,40,50 --consensus-steps 0.15,0.25,0.50,0.75 --delay-weights 0.5,1.0,2.0,4.0 --output-dir reference_trend_results
```

Faster validation command:

```powershell
python generate_reference_trend_plots.py --slots 100 --seeds 7,13,21 --focus-uavs 20 --densities 10,20,30,40,50 --consensus-steps 0.15,0.25,0.50,0.75 --delay-weights 0.5,1.0,2.0,4.0 --output-dir reference_trend_results_quick
```

Generated plots:

- `fig_time_average_delay_energy.png`
- `fig_consensus_step_convergence.png`
- `fig_delay_weight_convergence.png`
- `fig_density_delay_energy_tradeoff.png`

Raw CSV values are saved beside the figures for traceability.
Because the SAGIN cost model now includes uplink transmission energy and
synchronization energy, the default energy plotting scale is `1.0`. Use
`--energy-scale` only if you deliberately change the physical units.


## Phase 2 Quantum Integration

1. Obtain an API token from D-Wave Leap.
2. Configure your environment:

```powershell
dwave config create
# Paste your token when prompted
```

3. Run the simulator with the D-Wave hybrid backend:

```powershell
python -m sim.main --slots 50 --seed 7 --solver-backend dwave_hybrid --output-dir results
```

The architecture supports seamless transition to Quantum hardware. The DTN logic remains unchanged; only the local QUBO/BQM backend is swapped to `DWaveHybridSolver`.
