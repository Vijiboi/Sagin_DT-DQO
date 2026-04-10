from env.config import SimulationConfig
from env.models import APNode, Observation, TwinState
from twin.twin_logic import TwinManager

def test_twin_sync():
    config = SimulationConfig()
    manager = TwinManager(config)
    
    # Setting up a dummy AP with an initial trust of 0.9
    ap = APNode(ap_id="UAV_0", tier="UAV", x=0, y=0, z=0.1, 
                bandwidth=10, cpu_capacity=5, communication_budget=15, 
                power_budget=10, trust=0.9, sync_threshold=0.22, 
                coord_threshold=0.75, twin_state=TwinState())

    # Mock an observation with high load (forcing a mismatch)
    obs = Observation(ap_id="UAV_0", slot=1, load_ratio=0.9, 
                      bandwidth_ratio=0.8, cpu_ratio=0.8, 
                      queue_size=5, candidate_overlap=0.4)

    sync, coord = manager.update(ap, obs)
    
    print(f"Sync Triggered: {sync}") # Should be True due to mismatch
    print(f"Coordination Triggered: {coord}") # Should be True due to load > 0.75
    print(f"New AP Trust: {ap.trust:.4f}")

if __name__ == "__main__":
    test_twin_sync()