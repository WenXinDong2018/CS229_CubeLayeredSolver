from typing import List, Tuple
import numpy as np
from utils import nnet_utils, misc_utils
from environments.environment_abstract import Environment, State
from search_methods.gbfs import GBFS
from search_methods.astar import AStar, Node
from torch.multiprocessing import Queue, get_context
from environments.cube3_layer1 import Cube3Layer1
from environments.cube3_layer2 import Cube3Layer2
from environments.cube3_layer3 import Cube3Layer3
import time


def gbfs_update(states: List[State], env: Environment, num_steps: int, heuristic_fn, eps_max: float):
    eps: List[float] = list(np.random.rand(len(states)) * eps_max)

    gbfs = GBFS(states, env, eps=eps)
    for _ in range(num_steps):
        gbfs.step(heuristic_fn)

    trajs: List[List[Tuple[State, float]]] = gbfs.get_trajs()

    trajs_flat: List[Tuple[State, float]]
    trajs_flat, _ = misc_utils.flatten(trajs)

    is_solved: np.ndarray = np.array(gbfs.get_is_solved())

    states_update: List = []
    cost_to_go_update_l: List[float] = []
    for traj in trajs_flat:
        states_update.append(traj[0])
        cost_to_go_update_l.append(traj[1])

    cost_to_go_update = np.array(cost_to_go_update_l)
    return states_update, cost_to_go_update, is_solved


def astar_update(states: List[State], env: Environment, num_steps: int, heuristic_fn):
    weights: List[float] = list(np.random.rand(len(states)))
    astar1 = AStar(states, env[0], heuristic_fn[0], weights)
    astar2 = AStar(states, env[1], heuristic_fn[1], weights)
    astar3 = AStar(states, env[2], heuristic_fn[2], weights)
    for _ in range(num_steps):
        astar1.step(heuristic_fn[0], 1, verbose=False)
        astar2.step(heuristic_fn[1], 1, verbose=False)
        astar3.step(heuristic_fn[2], 1, verbose=False)

    nodes_popped_layer1: List[List[Node]] = astar1.get_popped_nodes()
    nodes_popped_layer2: List[List[Node]] = astar2.get_popped_nodes()
    nodes_popped_layer3: List[List[Node]] = astar3.get_popped_nodes()
    nodes_popped_flat_layer1: List[Node]
    nodes_popped_flat_layer2: List[Node]
    nodes_popped_flat_layer3: List[Node]
    nodes_popped_flat_layer1, _ = misc_utils.flatten(nodes_popped_layer1)
    nodes_popped_flat_layer2, _ = misc_utils.flatten(nodes_popped_layer2)
    nodes_popped_flat_layer3, _ = misc_utils.flatten(nodes_popped_layer3)

    for node in nodes_popped_flat_layer1 + nodes_popped_flat_layer2 + nodes_popped_flat_layer3:
        node.compute_bellman()

    states_update_layer1: List[State] = [node.state for node in nodes_popped_flat_layer1]
    states_update_layer2: List[State] = [node.state for node in nodes_popped_flat_layer2]
    states_update_layer3: List[State] = [node.state for node in nodes_popped_flat_layer3]
    cost_to_go_update_layer1: np.array = np.array([node.bellman for node in nodes_popped_flat_layer1])
    cost_to_go_update_layer2: np.array = np.array([node.bellman for node in nodes_popped_flat_layer2])
    cost_to_go_update_layer3: np.array = np.array([node.bellman for node in nodes_popped_flat_layer3])

    is_solved_layer1: np.array = np.array(astar1.has_found_goal())
    is_solved_layer2: np.array = np.array(astar2.has_found_goal())
    is_solved_layer3: np.array = np.array(astar3.has_found_goal())

    return [states_update_layer1, states_update_layer2, states_update_layer3], [cost_to_go_update_layer1, cost_to_go_update_layer2, cost_to_go_update_layer3], [is_solved_layer1, is_solved_layer2, is_solved_layer3]

'''update_batch_size is not important, it controls for how many random cubes gbfs performs one-step-look-ahead for at once'''
'''num_states is important, it controls how many random cubes we generate in total '''
def update_runner(num_states: int, back_max: int, update_batch_size: int, heur_fn_i_q, heur_fn_o_q,
                  proc_id: int, env: Environment, result_queue: Queue, num_steps: int, update_method: str,
                  eps_max: float, fixed_difficulty:bool, random: bool, normal_dist:bool):
    heuristic_fn = nnet_utils.heuristic_fn_queue(heur_fn_i_q, heur_fn_o_q, proc_id, env)

    start_idx: int = 0
    while start_idx < num_states:
        end_idx: int = min(start_idx + update_batch_size, num_states)
        states_itr, _ = env.generate_states(end_idx - start_idx, (0, back_max), fixed_difficulty=fixed_difficulty, random=random, normal_dist = normal_dist)

        if update_method.upper() == "GBFS":
            states_update, cost_to_go_update, is_solved = gbfs_update(states_itr, [Cube3Layer1, Cube3Layer2, Cube3Layer3], num_steps, heuristic_fn, eps_max)
        elif update_method.upper() == "ASTAR":
            states_update, cost_to_go_update, is_solved = astar_update(states_itr, env, num_steps, heuristic_fn)
        else:
            raise ValueError("Unknown update method %s" % update_method)

        states_update_nnet_layer1: List[np.ndaray] = Cube3Layer1.state_to_nnet_input(states_update[0])
        states_update_nnet_layer2: List[np.ndaray] = Cube3Layer2.state_to_nnet_input(states_update[1])
        states_update_nnet_layer3: List[np.ndaray] = Cube3Layer3.state_to_nnet_input(states_update[2])

        result_queue.put(([states_update_nnet_layer1, states_update_nnet_layer2, states_update_nnet_layer3], cost_to_go_update, is_solved))

        start_idx: int = end_idx

    result_queue.put(None)

class Updater_Multihead:
    def __init__(self, env: Environment, num_states: int, back_max: int, heur_fn_i_q, heur_fn_o_qs,
                 num_steps: int, update_method: str, update_batch_size: int = 1000, eps_max: float = 0.0, fixed_difficulty = False, random=False, normal_dist = False):
        super().__init__()
        ctx = get_context("spawn")
        self.num_steps = num_steps
        num_procs = len(heur_fn_o_qs)

        # initialize queues
        self.result_queue: ctx.Queue = ctx.Queue()

        # num states per process
        num_states_per_proc: List[int] = misc_utils.split_evenly(num_states, num_procs)

        self.num_batches: int = int(np.ceil(np.array(num_states_per_proc)/update_batch_size).sum())

        # initialize processes
        self.procs: List[ctx.Process] = []
        for proc_id in range(len(heur_fn_o_qs)):
            num_states_proc: int = num_states_per_proc[proc_id]
            if num_states_proc == 0:
                continue

            proc = ctx.Process(target=update_runner, args=(num_states_proc, back_max, update_batch_size,
                                                           heur_fn_i_q, heur_fn_o_qs[proc_id], proc_id, env,
                                                           self.result_queue, num_steps, update_method, eps_max, fixed_difficulty, random, normal_dist))
            proc.daemon = True
            proc.start()
            self.procs.append(proc)

    def update(self):
        states_update_nnet: List[np.ndarray]
        cost_to_go_update: np.ndarray
        is_solved: np.ndarray
        states_update_nnet, cost_to_go_update, is_solved = self._update()

        output_update = np.expand_dims(cost_to_go_update, 1)
        return states_update_nnet, output_update, is_solved

    def _update(self) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
        # process results
        states_update_nnet_l_layer1: List[List[np.ndarray]] = []
        states_update_nnet_l_layer2: List[List[np.ndarray]] = []
        states_update_nnet_l_layer3: List[List[np.ndarray]] = []
        cost_to_go_update_l_layer1: List = []
        cost_to_go_update_l_layer2: List = []
        cost_to_go_update_l_layer3: List = []
        is_solved_l_layer1: List = []
        is_solved_l_layer2: List = []
        is_solved_l_layer3: List = []

        none_count: int = 0
        result_count: int = 0
        display_counts: List[int] = list(np.linspace(1, self.num_batches, 10, dtype=np.int))

        start_time = time.time()

        while none_count < len(self.procs):
            result = self.result_queue.get()
            if result is None:
                none_count += 1
                continue
            result_count += 1

            states_nnet_q: List[np.ndarray]
            states_nnet_q, cost_to_go_q, is_solved_q = result
            states_update_nnet_l_layer1.append(states_nnet_q[0])
            states_update_nnet_l_layer2.append(states_nnet_q[1])
            states_update_nnet_l_layer3.append(states_nnet_q[2])

            cost_to_go_update_l_layer1.append(cost_to_go_q[0])
            cost_to_go_update_l_layer2.append(cost_to_go_q[1])
            cost_to_go_update_l_layer3.append(cost_to_go_q[2])

            is_solved_l_layer1.append(is_solved_q[0])
            is_solved_l_layer2.append(is_solved_q[1])
            is_solved_l_layer3.append(is_solved_q[2])

            if result_count in display_counts:
                print("%.2f%% (Total time: %.2f)" % (100 * result_count/self.num_batches, time.time() - start_time))

        num_states_nnet_np_layer1: int = len(states_update_nnet_l_layer1[0])
        num_states_nnet_np_layer2: int = len(states_update_nnet_l_layer2[0])
        num_states_nnet_np_layer3: int = len(states_update_nnet_l_layer3[0])
        states_update_nnet: List[np.ndarray] = []
        for np_idx in range(num_states_nnet_np_layer1):
            states_nnet_idx: np.ndarray = np.concatenate([x[np_idx] for x in states_update_nnet_l_layer1], axis=0)
            states_update_nnet.append(states_nnet_idx)

        cost_to_go_update: np.ndarray = np.concatenate(cost_to_go_update_l, axis=0)
        is_solved: np.ndarray = np.concatenate(is_solved_l, axis=0)

        for proc in self.procs:
            proc.join()

        return states_update_nnet, cost_to_go_update, is_solved
