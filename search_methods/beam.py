from typing import List, Tuple, Set, Callable, Optional
from environments.environment_abstract import Environment, State
import numpy as np
from utils import search_utils, env_utils, nnet_utils, misc_utils
from argparse import ArgumentParser
import pickle
import torch


class BEAM_Instance:

    def __init__(self, state: State, eps: float, k:int):
        self.curr_states: List[State] = [state]
        self.is_solved: bool = False
        self.num_steps: int = 0
        self.trajs: List[Tuple[State, float]] = []
        self.seen_states: Set[State] = set()
        self.eps = eps

    def add_to_traj(self, state: State, cost_to_go: float):
        self.trajs.append((state, cost_to_go))
        # self.seen_states.add(state)

    def next_state(self, states:  List[State] ):
        self.curr_states = states
        self.num_steps += 1


class BEAM_SEARCH:
    def __init__(self, states: List[State], env: Environment, k = 5, eps: Optional[List[float]] = None):
        self.curr_states: List[State] = states
        self.env: Environment = env
        self.k = k
        if eps is None:
            eps = [0] * len(self.curr_states)

        self.instances: List[BEAM_Instance] = []
        for state, eps_inst in zip(states, eps):
            instance: BEAM_Instance = BEAM_Instance(state, eps_inst, k)
            self.instances.append(instance)

    def step(self, heuristic_fn: Callable) -> None:
        # check which are solved
        self._record_solved()

        # take a step for unsolved states
        self._move(heuristic_fn)

    def get_trajs(self) -> List[List[Tuple[State, float]]]:
        trajs_all: List[List[Tuple[State, float]]] = []
        for instance in self.instances:
            trajs_all.append(instance.trajs)

        return trajs_all

    def get_is_solved(self) -> List[bool]:
        is_solved: List[bool] = [x.is_solved for x in self.instances]

        return is_solved

    def get_num_steps(self) -> List[int]:
        num_steps: List[int] = [x.num_steps for x in self.instances]

        return num_steps

    def _record_solved(self) -> None:
        # get unsolved instances
        instances: List[BEAM_Instance] = self._get_unsolved_instances()
        if len(instances) == 0:
            return

        states: List[List[State]] = [instance.curr_states for instance in instances]

        is_solved: np.ndarray = np.asarray([np.sum(self.env.is_solved(states_i)) for states_i in states], dtype=np.int32)

        solved_idxs: List[int] = list(np.where(is_solved)[0])
        if len(solved_idxs) > 0:
            instances_solved: List[BEAM_Instance] = [instances[idx] for idx in solved_idxs]
            states_solved: List[List[State]] = [instance.curr_states for instance in instances_solved]

            for instance, state in zip(instances_solved, states_solved):
                instance.add_to_traj(state, 0.0)
                instance.is_solved = True

    def _move(self, heuristic_fn: Callable) -> None:
        # get unsolved instances
        instances: List[BEAM_Instance] = self._get_unsolved_instances()
        if len(instances) == 0:
            return
        states: List[List[State]]= [instance.curr_states for instance in instances]

        # get backed-up ctg and cost of each move
        # ctg_backups: np.ndarray
        # ctg_next_p_tcs: List[np.ndarray]
        # states_exp: List[List[State]]
        # ctg_backups, ctg_next_p_tcs, states_exp = search_utils.bellman(states, heuristic_fn, self.env)

        # make move
        for idx in range(len(instances)):
            # add state to trajectory
            instance: Instance = instances[idx]
            state: List[State] = states[idx]
            # print("current beam states", len(state))
            # ctg_backup: float = ctg_backups[idx]

            # instance.add_to_traj(state, ctg_backup)
            _, ctg_next_p_tcs, states_exp = search_utils.bellman(state, heuristic_fn, self.env)
            # get next state
            ctg_next_p_tc = []
            state_exp = []
            for i in range(len(states_exp)):
                # print("ctg_next_p_tcs[i]", i, ctg_next_p_tcs[i])
                ctg_next_p_tc.extend(ctg_next_p_tcs[i])
                state_exp.extend(states_exp[i])
            # print("ctg_next_p_tc ", idx, ctg_next_p_tc)
            # print("state_exp", state_exp)
            # state_next: State = state_exp[int(np.argmin(ctg_next_p_tc))]
            min_k = min(self.k, len(ctg_next_p_tc)-1)
            states_next: List[State] = [state_exp[i] for i in np.argpartition(np.array(ctg_next_p_tc, dtype=np.int8), min_k)][0:min_k]
            # print("states_next", len(states_next), np.argpartition(np.array(ctg_next_p_tc, dtype=np.int8), self.k))
            # state_exp[int(np.argmin(ctg_next_p_tc))]
            print("best", min(ctg_next_p_tc))
            # make random move with probability eps
            # eps_rand_move = np.random.random(1)[0] < instance.eps
            # seen_state: bool = state_next in instance.seen_states
            # if eps_rand_move or seen_state:
            #     rand_state_idx = np.random.choice(len(state_exp))
            #     state_next = state_exp[rand_state_idx]

            instance.next_state(states_next)

    def _get_unsolved_instances(self) -> List[BEAM_Instance]:
        instances_unsolved: List[BEAM_Instance] = [instance for instance in self.instances if not instance.is_solved]
        return instances_unsolved





def gbfs_test(num_states: int, back_max: int, env: Environment, heuristic_fn: Callable,
              args, max_solve_steps: Optional[int] = None):
    # get data
    # back_steps: List[int] = list(np.linspace(0, back_max, 30, dtype=np.int))
    # num_states_per_back_step: List[int] = misc_utils.split_evenly(num_states, len(back_steps))
    input_data = pickle.load(open(args.data_dir, "rb"))
    states: List[State] = input_data['states'][0:1]

    # states: List[State] = []
    # state_back_steps_l: List[int] = []

    # for back_step, num_states_i in zip(back_steps, num_states_per_back_step):
    #     if num_states_i > 0:
    #         states_i, back_steps_i = env.generate_states(num_states_i, (back_step, back_step))
    #         states.extend(states_i)
    #         state_back_steps_l.extend(back_steps_i)

    # state_back_steps: np.ndarray = np.array(state_back_steps_l)
    # if max_solve_steps is None:
    #     max_solve_steps = max(np.max(state_back_steps), 1)

    # Do GBFS for each back step
    print("Solving %i states with BEAM with %i steps" % (len(states), max_solve_steps))

    # Solve with GBFS
    gbfs = BEAM_SEARCH(states, env, k=1000, eps=None)
    for _ in range(max_solve_steps):
        # print("step-------~")
        gbfs.step(heuristic_fn)

    is_solved_all: np.ndarray = np.array(gbfs.get_is_solved())
    num_steps_all: np.ndarray = np.array(gbfs.get_num_steps())
    print("num_steps_all", num_steps_all)
    print("is_solved_all", is_solved_all)
    # # Get state cost-to-go
    # state_ctg_all: np.ndarray = heuristic_fn(states)

    # for back_step_test in np.unique(state_back_steps):
    #     # Get states
    #     step_idxs = np.where(state_back_steps == back_step_test)[0]
    #     if len(step_idxs) == 0:
    #         continue

    #     is_solved: np.ndarray = is_solved_all[step_idxs]
    #     num_steps: np.ndarray = num_steps_all[step_idxs]
    #     state_ctg: np.ndarray = state_ctg_all[step_idxs]

    #     # Get stats
    #     per_solved = 100 * float(sum(is_solved)) / float(len(is_solved))
    #     avg_solve_steps = 0.0
    #     if per_solved > 0.0:
    #         avg_solve_steps = np.mean(num_steps[is_solved])

    #     # Print results
    #     print("Back Steps: %i, %%Solved: %.2f, avgSolveSteps: %.2f, CTG Mean(Std/Min/Max): %.2f("
    #           "%.2f/%.2f/%.2f)" % (
    #               back_step_test, per_solved, avg_solve_steps, float(np.mean(state_ctg)),
    #               float(np.std(state_ctg)), np.min(state_ctg),
    #               np.max(state_ctg)))


def main():
    # parse arguments
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True, help="Directory of nnet model")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory of data")

    parser.add_argument('--env', type=str, required=True, help="Environment: cube3, 15-puzzle, 24-puzzle")
    parser.add_argument('--max_steps', type=int, default=None, help="Maximum number ofsteps to take when solving "
                                                                    "with GBFS. If none is given, then this "
                                                                    "is set to the maximum number of "
                                                                    "backwards steps taken to create the "
                                                                    "data")

    args = parser.parse_args()

    # environment
    env: Environment = env_utils.get_environment(args.env)

    # get device and nnet
    on_gpu: bool
    device: torch.device
    device, devices, on_gpu = nnet_utils.get_device()
    print("device: %s, devices: %s, on_gpu: %s" % (device, devices, on_gpu))

    heuristic_fn = nnet_utils.load_heuristic_fn(args.model_dir, device, on_gpu, env.get_nnet_model(),
                                                env, clip_zero=False)

    gbfs_test(1,300, env, heuristic_fn, args, max_solve_steps=args.max_steps)


if __name__ == "__main__":
    main()
