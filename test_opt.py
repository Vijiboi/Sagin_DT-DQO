from env.config import SimulationConfig
from env.models import APNode, Task, TwinState
from opt.qubo import LocalQuboBuilder
from opt.solver import ClassicalQuboSolver

def test_optimization_flow():
    config = SimulationConfig()
    builder = LocalQuboBuilder(config)
    solver = ClassicalQuboSolver(config, backend="auto")

    # 1. Setup APs and a Task
    ap_bs = APNode(ap_id="BS_0", tier="BS", x=0, y=0, z=0.03, bandwidth=20, cpu_capacity=10, 
                   communication_budget=24, power_budget=18, trust=0.9, 
                   sync_threshold=0.22, coord_threshold=0.75)
    
    ap_uav = APNode(ap_id="UAV_0", tier="UAV", x=10, y=10, z=0.1, bandwidth=10, cpu_capacity=5, 
                    communication_budget=15, power_budget=10, trust=0.8, 
                    sync_threshold=0.22, coord_threshold=0.75)

    task = Task(task_id="T1_U0", source_uav="UAV_0", owner_ap_id="UAV_0", x=12, y=12, z=0.1,
                L_u=10.0, D_u=8.0, omega_u=1.0, psi_u=1.0, xi_u=1.0,
                bandwidth_demand=2.0, cpu_demand=4.0, power_demand=1.0,
                A_u_t=["BS_0", "UAV_0"], arrival_slot=1)

    ap_lookup = {"BS_0": ap_bs, "UAV_0": ap_uav}

    # 2. Build QUBO
    problem = builder.build(ap_uav, [task], ap_lookup, slot=1)
    print(f"QUBO Variables: {problem.variables}")

    # 3. Solve
    result = solver.solve(problem)
    print(f"Solver Name: {result.solver_name}")
    print(f"Best Sample: {result.sample}")

    # Validation: Sum of binary variables for a task should be 1
    assignment_sum = sum(result.sample.values())
    if assignment_sum == 1:
        print("RESULT: Valid one-hot assignment found.")
    else:
        print(f"RESULT: Invalid assignment (Sum={assignment_sum}). Increase qubo_penalty.")

if __name__ == "__main__":
    test_optimization_flow()