import pickle5 as pickle
from environments.cube3_layer2 import Cube3State

results = pickle.load(open("data/cube3/test/data_0.pkl", "rb"))

print(results['states'][0])