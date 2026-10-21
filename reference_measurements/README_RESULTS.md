# DTN-SAGIN Measured Results Package

This folder contains the final measured DTN-SAGIN baseline study generated on 2026-06-16.
The figures are PNG only. No EPS files were produced.

## What Is Included

- `figures/`: nine generated PNG plots with compact filenames.
- `paste_ready_figures/`: the same nine PNG plots renamed as `Fig_1_...` through `Fig_9_...`.
- `*_raw.csv`: one row per measured method/seed/sweep point.
- `raw_runs/`: simulator outputs for every individual run, including slot metrics, assignments, twin metrics, and summary JSON.
- `code_snapshot/`: source code used to generate these results.
- `source_documents/`: supplied PDFs used as the baseline/formulation reference plus the DeepSeek audit material.
- `study_config.json`: experiment arguments and method/baseline mapping.
- `symbol_key.txt`: legend symbols and baseline names.
- `verification_audit.json`: row-count, trend, and validation summary.

## Reproduction Command

Command:

```powershell
python generate_measured_baseline_plots.py --slots 30 --seeds 7,13,21,31,43 --anneal-reads 2 --anneal-sweeps 5 --output-dir paper_results_final_20260616 --dpi 350
```

## Main Code Changes

- Added one shared physical evaluator for delay, energy, freshness risk, trust, fidelity, sync cost, and coordination cost.
- Converted all delay/energy calculations to consistent units: Mbit to bit, Mbps to bit/s, GHz to Hz, seconds, and joules.
- Added one shared feasibility manager for CPU cycles, communication bits, and power constraints.
- Refactored all baselines so each uses its documented decision rule but final metrics are evaluated by the same physical model.
- Implemented large-case binary assignment QUBO/BQM solving without hardcoded result values.
- Kept strict `dimod` CQM-to-BQM for small/verification cases and scalable one-hot assignment QUBO annealing for large sweeps.
- Changed plots to PNG with symbol-form legends and no shaded regions.
- Recorded decision-only runtime per slot instead of whole-script wall time.

## Baselines Used

- `P_DTN-HQ`: proposed closed-loop DTN with local QUBO guidance and hierarchical projection.
- `B_GLQ`: greedy link-quality baseline.
- `B_CQ`: centralized QUBO.
- `B_DTF`: DT-free distributed baseline.
- `B_CH`: channel-only predictive baseline.
- `B_SR`: static-risk distributed baseline.
- `B_A^-`: AoI-unaware distributed baseline.
- `B_tau^-`: trust-unaware distributed baseline.
- `B_AO`: always-on synchronization/coordination.
- `B_LQ`: classical same-objective local QUBO/BQM.
- `B_LS`: classical same-objective local search.
- `B_UUT`: uncertainty-unaware periodic trigger.

## Validation Summary

- Raw measured rows: `920 / 920` expected rows present.
- Individual run summaries: `920 / 920` present.
- Slot-level checks: every slot has one-hot assignment validity and resource feasibility.
- Tests: `python -m unittest discover -v` passed, `9` tests OK.
- Figures: `9` PNGs in `figures/` and `9` paste-ready PNGs in `paste_ready_figures/`.
- Plot shading: removed; the script contains no `fill_between` shading call.

## Result Interpretation Notes

- Synchronization overhead decreases as the twin-age threshold `delta_A` increases.
- Coordination overhead decreases as the trust trigger threshold `delta_tau` increases.
- Digital twin fidelity decreases as UAV speed increases.
- Freshness-risk cost increases with AoI. The proposed method is lower than AoI-unaware, static-risk, and channel-only baselines. Trust-unaware is close to the proposed method in this measured run; that is an honest simulator outcome, not a hardcoded curve.
- Runtime increases with the number of UAV tasks and is measured as decision runtime per slot after warm-up.
