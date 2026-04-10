import json
import os
from env.config import SimulationConfig
from sim.runner import SimulationRunner

def run_final_simulation():
    # 1. Initialize with your finalized 50-UAV config
    # Ensure you've made the 'UAV' key changes in config.py as we discussed!
    config = SimulationConfig(
        slots=10, 
        num_uavs=50, 
        output_dir="test_results"
    )
    
    print(f"--- Starting SAGIN Simulation ---")
    print(f"Nodes: 4 BS, 2 HAP, 1 LEO, 50 UAVs")
    
    # 2. Run the simulation
    runner = SimulationRunner(config)
    slot_results, summary, output_path = runner.run()
    
    # 3. Analyze the output
    print("\n--- Simulation Summary ---")
    print(f"Average Delay: {summary['average_delay']}")
    print(f"Average Energy: {summary['average_energy']}")
    print(f"Sync Triggers: {summary['sync_trigger_count']}")
    print(f"Coordination Triggers: {summary['coordination_trigger_count']}")
    
    # Check the validation results
    checks = summary['pre_quantum_checks']
    if checks['regional_projection_feasible']:
        print("\nSUCCESS: All resource constraints were respected.")
    else:
        print("\nWARNING: Resource violation detected in one or more slots.")

    print(f"\nDetailed logs saved to: {output_path}")

if __name__ == "__main__":
    run_final_simulation()