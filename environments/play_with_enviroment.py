from environments.cube3_layer1 import Cube3Layer1
from environments.cube3_layer2 import Cube3Layer2

env = Cube3Layer2()

states = env.generate_states(10, [0, 1])
print("states", states[0])
env.is_solved(states[0])