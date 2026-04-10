from env.config import SimulationConfig
from env.graph import CommunicationGraph
from env.models import APNode, LocalSummary, CoordinationState
from control.consensus import QuantizedConsensusCoordinator

def test_consensus_logic():
    config = SimulationConfig()
    coordinator = QuantizedConsensusCoordinator(config)
    
    # 1. Setup two connected APs
    ap1 = APNode(ap_id="UAV_0", tier="UAV", x=0, y=0, z=0.1, bandwidth=10, cpu_capacity=5, 
                 communication_budget=15, power_budget=10, trust=0.9, 
                 sync_threshold=0.22, coord_threshold=0.75)
    ap2 = APNode(ap_id="UAV_1", tier="UAV", x=0.05, y=0.05, z=0.1, bandwidth=10, cpu_capacity=5, 
                 communication_budget=15, power_budget=10, trust=0.9, 
                 sync_threshold=0.22, coord_threshold=0.75)
    
    ap_lookup = {"UAV_0": ap1, "UAV_1": ap2}
    graph = CommunicationGraph(adjacency={"UAV_0": ["UAV_1"], "UAV_1": ["UAV_0"]}, edges=[])

    # 2. Simulate high pressure on UAV_0
    summary0 = LocalSummary(ap_id="UAV_0", slot=1, queue_size=10, local_load=0.9, 
                            sync_triggered=False, coordination_triggered=True, 
                            trust=0.9, twin_age=0, uncertainty=0.1, mismatch=0.1, 
                            fidelity=0.9, qubo_dimension=2, solver_time=0.01, 
                            selected_pairs=[], candidate_scores=[])
    
    # 3. Update consensus
    coordinator.update(ap_lookup, graph, [summary0], slot=1)
    
    print(f"UAV_0 Dual Price: {ap1.coordination_state.dual_price:.4f}")
    print(f"UAV_0 Quantized Dual (Penalty): {ap1.coordination_state.quantized_dual:.4f}")
    
    if ap1.coordination_state.dual_price > 0:
        print("RESULT: Consensus logic correctly increased price due to high load.")
    else:
        print("RESULT: Price did not increase. Check pressure calculation logic.")

if __name__ == "__main__":
    test_consensus_logic()