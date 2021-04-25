from environments.cube3 import Cube3
import numpy as np
from environments.cube3_layer3 import Cube3Layer3
env = Cube3Layer3()
states, _ = env.generate_states_using_fixed_moves(4, [1, 10])

# env3.generate_states(5, [1, 10])
a = {'a': 1}
#print(54 - len(a))
