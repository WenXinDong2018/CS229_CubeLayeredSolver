import pickle
from utils import search_utils, env_utils
env1 = env_utils.get_environment("cube3_layer1")
env2 = env_utils.get_environment("cube3_layer2")
env3 = env_utils.get_environment("cube3_layer3")

moves= ["%s%i" % (f, n) for f in ['U', 'D', 'L', 'R', 'B', 'F'] for n in [-1, 1]]

results = pickle.load(open("results/cube3_sequential/results.pkl", "rb"))
solved = 0
averageTime = 0
averageSolLen = 0
averageNodes = 0
for solution, time, nodes, path in zip(results["solutions"], results["times"], results["num_nodes_generated"], results["paths"]):

    if len(solution):
        assert(env1.is_solved([path[-1]])[0])
        assert(env2.is_solved([path[-1]])[0])
        assert(env3.is_solved([path[-1]])[0])
        solved+=1
        averageNodes+= nodes
        averageSolLen += len(solution)
        averageTime += time
print("solved", solved, "problems")
print("average time is", averageTime/solved, "s")
print("average node generated is", averageNodes/solved)
print("average solution length  is", averageSolLen/solved)
