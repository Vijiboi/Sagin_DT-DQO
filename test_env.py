from env.config import SimulationConfig
from env.generator import SaginEnvironment

config = SimulationConfig(num_uavs=50) # Matching res paper's variable load
env = SaginEnvironment(config)

# Check if APs are placed in the right "tiers"
for ap in env.aps:
    print(f"AP {ap.ap_id} at {ap.z}km altitude")

# Test task generation
tasks = env.create_tasks_for_slot(slot=1)
print(f"Generated {len(tasks)} tasks for the first slot.")