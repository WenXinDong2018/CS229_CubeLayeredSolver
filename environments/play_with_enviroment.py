from environments.cube3 import Cube3
import numpy as np
from environments.cube3_layer1 import Cube3Layer1
from environments.cube3_layer2 import Cube3Layer2
from environments.cube3_layer3 import Cube3Layer3
env3 = Cube3Layer3()
env2 = Cube3Layer2()
states2 = env2.generate_states(10, [0, 1])
print("states", states2[0])
env2.is_solved(states2[0])
states3, _ = env3.generate_states_using_fixed_moves(10, [1, 10])
env2.is_solved(states3)
