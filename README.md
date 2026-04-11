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
## Testing & Validation
The repository includes a comprehensive unit-testing suite to verify the logic of individual tiers before full integration:
`test_env.py`: Validates SAGIN node generation and mobility boundaries.
`test_twin.py`: Tests the Digital Twin's ability to trigger sync events based on state mismatch.
`test_opt.py`: Verifies that the QUBO builder generates valid "one-hot" assignments.
`test_control.py`: Confirms that the Quantized Consensus correctly increases "dual prices" under high resource pressure.
`test_sim.py`: A lightweight end-to-end runner used for rapid logic verification.

## Phase 2 Quantum Integration

1. Obtain an API token from D-Wave Leap.
2. Configure your environment:

```powershell
dwave config create
# Paste your token when prompted
```

3. In `sim/runner.py`, uncomment `self.local_solver = DWaveHybridSolver()`

The architecture supports seamless transition to Quantum hardware. To move beyond classical heuristics, replace the local solver call in `sim/runner.py` with the `DWaveHybridSolver` hook to utilize the D-Wave Leap Hybrid Sampler.
