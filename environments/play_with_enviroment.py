import numpy as np
from environments.cube3_layer1 import Cube3Layer1
from environments.cube3_layer2 import Cube3Layer2
from environments.cube3_layer3 import Cube3Layer3
env3 = Cube3Layer3()
env2 = Cube3Layer2()
env1 = Cube3Layer1()
states2, _ = env3.generate_states(1000, [0, 1], fixed_difficulty=False, random=True)
env2.is_solved(states2)
