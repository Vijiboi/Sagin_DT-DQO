# SAGIN DT-DQO Simulator

This scaffold implements the paper workflow in two stages:

1. A fully classical hierarchical digital-twin pipeline.
2. A clean hook to replace only the local QUBO solver with a hybrid quantum backend.

## Folder Layout

- `env/`: SAGIN environment generation, mobility, task creation, and directed communication graph.
- `twin/`: local twin forecast, Gaussian innovation, mismatch, fidelity, trust, sync, and coordination triggers.
- `opt/`: local QUBO construction with coupling penalties, classical solver, and D-Wave hybrid hook.
- `control/`: quantized consensus state, regional projection, and final one-hot assignment enforcement with resource budgets.
- `sim/`: end-to-end simulator runner and CLI entrypoint.
- `results/`: metrics aggregation and CSV/JSON writers.

## Run

```powershell
python -m sim.main --slots 8 --seed 7 --reads 5 --sweeps 20
```

## Phase 2

Keep the DTN logic unchanged and replace only the local solver call with `opt.hybrid.DWaveHybridSolver`.
