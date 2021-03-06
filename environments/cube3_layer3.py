from typing import List, Dict, Tuple, Union
import numpy as np
from sympy.combinatorics.permutations import Permutation
from torch import nn
from random import randrange

from utils.pytorch_models import ResnetModel
from .environment_abstract import Environment, State

'''Ignore this file'''
'''Ignore this file'''
'''Ignore this file'''

class Cube3State(State):
    __slots__ = ['colors', 'hash']

    def __init__(self, colors: np.ndarray):
        self.colors: np.ndarray = colors
        self.hash = None
        # print("colors", len(colors), colors)

    def __hash__(self):
        if self.hash is None:
            self.hash = hash(self.colors.tostring())

        return self.hash

    def __eq__(self, other):
        return np.array_equal(self.colors, other.colors)


class Cube3Layer3(Environment):
    moves: List[str] = ["%s%i" % (f, n) for f in ['U', 'D', 'L', 'R', 'B', 'F'] for n in [-1, 1]]
    moves_rev: List[str] = ["%s%i" % (f, n) for f in ['U', 'D', 'L', 'R', 'B', 'F'] for n in [1, -1]]
    # print("moves", moves)
    # print("moves_rev", moves_rev)
    def __init__(self):
        super().__init__()
        self.dtype = np.uint8
        self.cube_len = 3

        # solved state
        self.goal_colors: np.ndarray = np.arange(0, (self.cube_len ** 2) * 6, 1, dtype=self.dtype)
        # print("goal_colors", self.goal_colors)
        # get idxs changed for moves
        self.rotate_idxs_new: Dict[str, np.ndarray]
        self.rotate_idxs_old: Dict[str, np.ndarray]

        self.adj_faces: Dict[int, np.ndarray]
        self._get_adj()

        self.rotate_idxs_new, self.rotate_idxs_old = self._compute_rotation_idxs(self.cube_len, self.moves)

    def next_state(self, states: List[Cube3State], action: int) -> Tuple[List[Cube3State], List[float]]:
        states_np = np.stack([x.colors for x in states], axis=0)
        # print("states_np", states_np)
        states_next_np, transition_costs = self._move_np(states_np, action)

        states_next: List[Cube3State] = [Cube3State(x) for x in list(states_next_np)]

        return states_next, transition_costs

    def prev_state(self, states: List[Cube3State], action: int) -> List[Cube3State]:
        move: str = self.moves[action]
        move_rev_idx: int = np.where(np.array(self.moves_rev) == np.array(move))[0][0]

        return self.next_state(states, move_rev_idx)[0]

    def generate_goal_states(self, num_states: int, np_format: bool = False) -> Union[List[Cube3State], np.ndarray]:
        if np_format:
            goal_np: np.ndarray = np.expand_dims(self.goal_colors.copy(), 0)
            solved_states: np.ndarray = np.repeat(goal_np, num_states, axis=0)
        else:
            solved_states: List[Cube3State] = [Cube3State(self.goal_colors.copy()) for _ in range(num_states)]
        return solved_states

    def is_solved(self, states: List[Cube3State]) -> np.ndarray:
        states_np = np.stack([state.colors for state in states], axis=0)
        is_equal = np.equal(states_np, np.expand_dims(self.goal_colors, 0))

        return np.all(is_equal, axis=1)



    def state_to_nnet_input(self, states: List[Cube3State]) -> List[np.ndarray]:
        states_np = np.stack([state.colors for state in states], axis=0)

        representation_np: np.ndarray = states_np / (self.cube_len ** 2)
        representation_np: np.ndarray = representation_np.astype(self.dtype)

        representation: List[np.ndarray] = [representation_np]

        return representation

    def get_num_moves(self) -> int:
        return len(self.moves)

    def get_nnet_model(self) -> nn.Module:
        state_dim: int = (self.cube_len ** 2) * 6
        nnet = ResnetModel(state_dim, 6, 5000, 1000, 4, 1, True)

        return nnet
    """
    This part is the added part that generates states that fix layer2 and 1.
    """
    def get_fixed_moves(self) -> List[str]:
        moves_with_top_two_layers_fixed = ['edge_perm, edge_twist, corner_perm, corner_twist']

    def edge_permutation(self, states: List[np.ndarray], choice_of_edges: int, sign: int) -> np.ndarray:
        # edges in the last layers are 21, 30, 39, 48. We only can permutes three of them at a time
        # choice_of_edges can take either 0, 1, 2, 3. Each one represents a set of three edges that we will permute.
        # sign: given a set of edges that we want to permute, there's two ways to permute them.
        set_to_choose: Dict[int, np.ndarray] = {0: np.array([21, 30, 39]), 1: np.array([21, 39, 48]), 2: np.array([21, 30, 48]), 3: np.array([30, 39, 48])}
        output_states_np = np.stack([state for state in states])
        indices = set_to_choose[choice_of_edges]
        values = output_states_np[:, indices]

        # if sign == 1, we send the first element in perm_set to the second, second to last, then the last to first
        # if sign == 0, we send the first element in perm_set to the last, second to the first, then the last to the second
        perm_arr = np.array([1, 2, 0]) if sign == 1 else np.array([2, 0, 1])
        output_states_np[:, indices[perm_arr]] = values
        return output_states_np

    def edge_twist(self, states: List[np.ndarray], choice_of_edges: int, sign: int) -> np.ndarray:
        # edges in the last layers are 21, 30, 39, 48. We only can twist two of them at a time
        # choice_of_edges can take either 0, 1, 2, 3, 4, 5. Each one represents a set of two edges that we will twist.
        set_to_choose: Dict[int, np.ndarray] = {0: np.array([21, 30]), 1: np.array([21, 39]), 2: np.array([21, 48]), 3: np.array([30, 39]), 4: np.array([30, 48]), 5: np.array([39, 48])}
        correspondence: Dict[int, np.ndarray] = {21: np.array([10, 21]), 30: np.array([16, 30]), 39: np.array([12, 39]), 48: np.array([14, 48])}
        output_states_np = np.stack([state for state in states])
        perm_set = set_to_choose[choice_of_edges]

        edge_indices = np.concatenate((correspondence[perm_set[0]], correspondence[perm_set[1]]))
        edge_values = output_states_np[:, edge_indices]
        output_states_np[:, edge_indices[np.array([1, 0, 3, 2])]] = edge_values
        return output_states_np
    def corner_permutation(self, states: List[np.ndarray], choice_of_corners: int, sign: int) ->  np.ndarray:
        # edges in the last layers are 21, 30, 39, 48. We only can permutes three of them at a time
        # choice_of_edges can take either 0, 1, 2, 3. Each one represents a set of three edges that we will permute.
        # sign: given a set of edges that we want to permute, there's two ways to permute them.
        set_to_choose: Dict[int, np.ndarray] = {0: np.array([9, 11, 15]), 1: np.array([9, 15, 17]), 2: np.array([9, 11, 17]), 3: np.array([11, 15, 17])}
        correspondence: Dict[int, np.ndarray] = {9: np.array([9, 42, 18]), 11: np.array([11, 24, 45]), 15: np.array([15, 33, 36]), 17: np.array([17, 51, 21])}
        output_states_np = np.stack([state for state in states])
        D_idx = set_to_choose[choice_of_corners]
        indices = np.concatenate((D_idx, [correspondence[D_idx[0]][1], correspondence[D_idx[1]][1], correspondence[D_idx[2]][1]], [correspondence[D_idx[0]][2], correspondence[D_idx[1]][2], correspondence[D_idx[2]][2]]))
        values = output_states_np[:, indices]
        # if sign == 1, we send the first element in perm_set to the last, second to the first, then the last to the second with each corner twisted clockwise
        # if sign == 0, we send the first element in perm_set to the second, second to last, then the last to first with each corner twisted counter-clockwise
        perm_arr: np.ndarray = np.array([1, 2, 0, 4, 5, 3, 7, 8, 6]) if sign == 1 else np.array([2, 0, 1, 5, 3, 4, 8, 6, 7])
        output_states_np[:, indices[perm_arr]] = values
        return output_states_np

    def corner_twist(self, states: List[np.ndarray], choice_of_edges: int, sign: int) -> np.ndarray:
        # edges in the last layers are 21, 30, 39, 48. We only can twist two of them at a time
        # choice_of_edges can take either 0, 1, 2, 3. Each one represents a set of two on one edge.
        set_to_choose: Dict[int, np.ndarray] = {0: np.array([9, 11]), 1: np.array([9, 15]), 2: np.array([11, 17]), 3: np.array([15, 17])}
        correspondence: Dict[int, np.ndarray] = {9: np.array([9, 42, 18]), 11: np.array([11, 24, 45]), 15: np.array([15, 33, 36]), 17: np.array([17, 51, 21])}
        output_states_np = np.stack([state for state in states])
        D_idx = set_to_choose[choice_of_edges]
        corner_indices = np.concatenate((correspondence[D_idx[0]], correspondence[D_idx[1]]))
        corner_values = output_states_np[:, corner_indices]
        # if sign == 1, then we assume that we are doing a counter-clock wise twist on the (D_idx[0], D_idx[1], D_idx[2]) corner, and a clock wise twist on the other corner.
        # if sign == 0, then the opposite twist applies
        perm_arr = np.array([2, 0, 1, 4, 5, 0]) if sign == 1 else np.array([1, 2, 0, 5, 3, 4])
        output_states_np[:, corner_indices[perm_arr]] = corner_values
        return output_states_np

    def get_all_possible_fixed_moves(self) -> List[str]:
        output: List[str]
        # edge_perm: total 8 choices
        edge_perms = ['0 %i %i' % (c, s) for c in range(4) for s in range(2)]
        # edge_twist
        edge_twists = ['1 %i %i' % (c, 1) for c in range(6)]
        # corner_perm
        corner_perms = ['2 %i %i' % (c, s) for c in range(4) for s in range(2)]
        # corner_twist
        corner_twists = ['3 %i %i' % (c, s) for c in range(4) for s in range(2)]
        return edge_perms + edge_twists + corner_perms + corner_twists

    def fixed_move_dict(self):
        d = {0: self.edge_permutation, 1: self.edge_twist, 2: self.corner_permutation, 3: self.corner_twist}
        return d

    def get_corners(self) -> np.ndarray:
        # 8 corners oriented clockwise.
        return np.array([[0, 47, 26], [2, 20, 44], [8, 38, 35], [6, 29, 53], [11, 24, 45], [9, 42, 18], [15, 33, 36], [17, 51, 27]])

    def get_edges(self) -> np.ndarray:
        # 12 oriented corners.
        return np.array([[3, 50], [1, 23], [5, 41], [7, 32], [46, 25], [19, 43], [37, 34], [28, 52], [14, 48], [10, 21], [12, 31], [16, 30]])

    def generate_config(self, states: List[np.ndarray], corners_perm: np.ndarray, edge_perm: np.ndarray, corner_signs: np.ndarray, edge_signs: np.ndarray) -> np.ndarray:
        # corner_perm is a permutation of 0 to 7, with list[i] being that position that i-th corner will be sent to.
        # edge_perm is a permutation of 0 to 11, with list[i] being that position that i-th edge will be sent to.
        # corner_sign is a length 8 ordered list of number 0 to 2, with i-th number indicating a orientation of the i-th corner
        # edge_sign is a length 12 ordered list of number 0 to 1, with i-th number indicating a orientation of the i-th edge
        output_states_np = np.stack([state for state in states])
        corners = self.get_corners()
        edges = self.get_edges()
        corner_perm_map = np.array([[0, 1, 2], [2, 0, 1], [1, 2, 0]])
        edge_perm_map = np.array([[0, 1], [1, 0]])
        corner_values = np.stack([np.concatenate([state[corners[i]][corner_perm_map[corner_signs[i]]] for i in range(8)]) for state in states])
        edge_values = np.stack([np.concatenate([state[edges[i]][edge_perm_map[edge_signs[i]]] for i in range(12)]) for state in states])
        output_states_np[:, corners.flatten()] = corner_values
        output_states_np[:, edges.flatten()] = edge_values
        return output_states_np

    def generate_random_config(self, fix: int) -> List[np.ndarray]:
        corners_perm: np.ndarray
        edges_perm: np.ndarray
        corner_signs: np.ndarray
        edge_signs: np.ndarray
        while True:
            if fix != 1:
                corners_perm = np.concatenate((np.arange(4), np.random.permutation(4) + 4))
                edges_perm = np.concatenate((np.arange(4), np.random.permutation(8) + 4)) if fix == 2 else np.concatenate((np.arange(8), np.random.permutation(4) + 8))
            else:
                # fixes nothing
                corners_perm = np.random.permutation(8)
                edges_perm = np.random.permutation(12)

            c_perm = Permutation(corners_perm.tolist())
            e_perm = Permutation(edges_perm.tolist())
            if c_perm.signature() == e_perm.signature():
                break
        while True:
            corner_signs = np.concatenate((np.zeros(4, dtype=int), np.random.randint(3, size=4))) if fix != 1 else   np.random.randint(3, size=8)
            if np.sum(corner_signs) % 3 == 0:
                break
        while True:
            if fix != 1:
                edge_signs = np.concatenate((np.zeros(4, dtype=int), np.random.randint(2, size=8))) if fix == 2 else np.concatenate((np.zeros(8, dtype=int), np.random.randint(2, size=4)))
            else:
                edge_signs = np.random.randint(2, size=12)
            if np.sum(edge_signs) % 2 == 0:
                break
        return [corners_perm, edges_perm, corner_signs, edge_signs]

    def generate_states(self, num_states: int, backwards_range: Tuple[int, int], fixed_difficulty:bool = False, random:bool = False) -> Tuple[List[Cube3State], List[int]]:
        assert (num_states > 0)
        assert (backwards_range[0] >= 0)
        assert self.fixed_actions, "Environments without fixed actions must implement their own method"

        if random:
            # no random walk
            print("layer3 generating samples randomly")
            states_np: np.ndarray = self.generate_goal_states(num_states, np_format=True)
            for i in range(num_states):
                args = self.generate_random_config(fix=3)
                states_np[i] = self.generate_config([states_np[i]], args[0], args[1], args[2], args[3])[0]
            states: List[Cube3State] = [Cube3State(x) for x in list(states_np)]
            return states, [0 for _ in range(num_states)]
        # Initialize
        if fixed_difficulty:
            #generate examples with "backwards_range[1]" number of scrambles
            #if look at function calls, backwards_range[1] is the same as back_max
            scrambs = list(range(backwards_range[1], backwards_range[1] + 1))
            print("layer3 generating scrambles of fixed difficulty", scrambs)
        else:
            scrambs = list(range(backwards_range[0], backwards_range[1] + 1))
        num_env_moves: int = self.get_num_moves()
        #fixed_moves: List[str] = self.get_all_possible_fixed_moves()
        #function_map = self.fixed_move_dict()
        #num_fixed_moves: int = len(fixed_moves)
        # print("scrambs",scrambs, "num_env_moves", num_env_moves)
        # Get goal states
        states_np: np.ndarray = self.generate_goal_states(num_states, np_format=True)
        # print("states_np", states_np)
        # Scrambles
        scramble_nums: np.array = np.random.choice(scrambs, num_states)
        # print("scramble_nums: {}".format(scramble_nums))
        num_back_moves: np.array = np.zeros(num_states)

        # Go backward from goal state
        moves_lt = num_back_moves < scramble_nums
        # print('moves_lt: {}'.format(moves_lt))
        while np.any(moves_lt):
            
            idxs: np.ndarray = np.where(moves_lt)[0]
            subset_size: int = int(max(len(idxs) / num_env_moves, 1))
            idxs: np.ndarray = np.random.choice(idxs, subset_size)
            """
            move: int = randrange(num_fixed_moves)
            fixed_move: List[str] = fixed_moves[move].split(' ')
            states_np[idxs] = function_map[int(fixed_move[0])](states_np[idxs], int(fixed_move[1]), int(fixed_move[2]))
            # print("move states_np", states_np[idxs])
            num_back_moves[idxs] = num_back_moves[idxs] + 1
            moves_lt[idxs] = num_back_moves[idxs] < scramble_nums[idxs]
            """

            move: int = randrange(num_env_moves)
            states_np[idxs], _ = self._move_np(states_np[idxs], move)
            # print("move states_np", states_np[idxs])
            num_back_moves[idxs] = num_back_moves[idxs] + 1
            moves_lt[idxs] = num_back_moves[idxs] < scramble_nums[idxs]

        states: List[Cube3State] = [Cube3State(x) for x in list(states_np)]
        return states, scramble_nums.tolist()




    def expand(self, states: List[State], options: List[List[str]]= []) -> Tuple[List[List[State]], List[np.ndarray]]:
        assert self.fixed_actions, "Environments without fixed actions must implement their own method"

        # initialize
        num_states: int = len(states)
        num_env_moves: int = self.get_num_moves()
        num_options = len(options)

        states_exp: List[List[State]] = [[] for _ in range(len(states))]

        tc: np.ndarray = np.empty([num_states, num_env_moves+ num_options])

        # numpy states
        states_np: np.ndarray = np.stack([state.colors for state in states])

        # for each move, get next states, transition costs, and if solved
        move_idx: int
        move: int
        for move_idx in range(num_env_moves):
            # next state
            states_next_np: np.ndarray
            tc_move: List[float]
            states_next_np, tc_move = self._move_np(states_np, move_idx)

            # transition cost
            tc[:, move_idx] = np.array(tc_move)

            for idx in range(len(states)):
                states_exp[idx].append(Cube3State(states_next_np[idx]))

        for move_idx in range(num_options):
            # next state
            states_next_np: np.ndarray
            tc_move: List[float]
            states_next_np, tc_move = self._move_np_option(states_np, options[move_idx])

            # transition cost
            tc[:, move_idx+num_env_moves] = np.array(tc_move)

            for idx in range(len(states)):
                states_exp[idx].append(Cube3State(states_next_np[idx]))


        # make lists
        tc_l: List[np.ndarray] = [tc[i] for i in range(num_states)]

        return states_exp, tc_l

    def _move_np(self, states_np: np.ndarray, action: int):
        action_str: str = self.moves[action]
        # print("action_str", action_str)
        # print("states_np before move", states_np)
        states_next_np: np.ndarray = states_np.copy()
        states_next_np[:, self.rotate_idxs_new[action_str]] = states_np[:, self.rotate_idxs_old[action_str]]
        # print("self.rotate_idxs_new[action_str]", self.rotate_idxs_new[action_str])
        # print("self.rotate_idxs_old[action_str]", self.rotate_idxs_old[action_str])

        # print("states_next_np after move", states_next_np)
        transition_costs: List[float] = [1.0 for _ in range(states_np.shape[0])]

        return states_next_np, transition_costs
    def _move_np_option(self, states_np: np.ndarray, option: List[str]):

        states_next_np: np.ndarray = states_np.copy()
        for action_str in option:
            states_next_np[:, self.rotate_idxs_new[action_str]] = states_next_np[:, self.rotate_idxs_old[action_str]]
        #transition cost = length of option
        transition_costs: List[float] = [len(option) for _ in range(states_np.shape[0])]

        return states_next_np, transition_costs


    def _get_adj(self) -> None:
        # WHITE:0, YELLOW:1, BLUE:2, GREEN:3, ORANGE: 4, RED: 5
        self.adj_faces: Dict[int, np.ndarray] = {0: np.array([2, 5, 3, 4]),
                                                 1: np.array([2, 4, 3, 5]),
                                                 2: np.array([0, 4, 1, 5]),
                                                 3: np.array([0, 5, 1, 4]),
                                                 4: np.array([0, 3, 1, 2]),
                                                 5: np.array([0, 2, 1, 3])
                                                 }

    def _compute_rotation_idxs(self, cube_len: int,
                               moves: List[str]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        rotate_idxs_new: Dict[str, np.ndarray] = dict()
        rotate_idxs_old: Dict[str, np.ndarray] = dict()

        for move in moves:
            f: str = move[0]
            sign: int = int(move[1:])

            rotate_idxs_new[move] = np.array([], dtype=int)
            rotate_idxs_old[move] = np.array([], dtype=int)

            colors = np.zeros((6, cube_len, cube_len), dtype=np.int64)
            colors_new = np.copy(colors)

            # WHITE:0, YELLOW:1, BLUE:2, GREEN:3, ORANGE: 4, RED: 5

            adj_idxs = {0: {2: [range(0, cube_len), cube_len - 1], 3: [range(0, cube_len), cube_len - 1],
                            4: [range(0, cube_len), cube_len - 1], 5: [range(0, cube_len), cube_len - 1]},
                        1: {2: [range(0, cube_len), 0], 3: [range(0, cube_len), 0], 4: [range(0, cube_len), 0],
                            5: [range(0, cube_len), 0]},
                        2: {0: [0, range(0, cube_len)], 1: [0, range(0, cube_len)],
                            4: [cube_len - 1, range(cube_len - 1, -1, -1)], 5: [0, range(0, cube_len)]},
                        3: {0: [cube_len - 1, range(0, cube_len)], 1: [cube_len - 1, range(0, cube_len)],
                            4: [0, range(cube_len - 1, -1, -1)], 5: [cube_len - 1, range(0, cube_len)]},
                        4: {0: [range(0, cube_len), cube_len - 1], 1: [range(cube_len - 1, -1, -1), 0],
                            2: [0, range(0, cube_len)], 3: [cube_len - 1, range(cube_len - 1, -1, -1)]},
                        5: {0: [range(0, cube_len), 0], 1: [range(cube_len - 1, -1, -1), cube_len - 1],
                            2: [cube_len - 1, range(0, cube_len)], 3: [0, range(cube_len - 1, -1, -1)]}
                        }
            face_dict = {'U': 0, 'D': 1, 'L': 2, 'R': 3, 'B': 4, 'F': 5}
            face = face_dict[f]

            faces_to = self.adj_faces[face]
            if sign == 1:
                faces_from = faces_to[(np.arange(0, len(faces_to)) + 1) % len(faces_to)]
            else:
                faces_from = faces_to[(np.arange(len(faces_to) - 1, len(faces_to) - 1 + len(faces_to))) % len(faces_to)]

            cubes_idxs = [[0, range(0, cube_len)], [range(0, cube_len), cube_len - 1],
                          [cube_len - 1, range(cube_len - 1, -1, -1)], [range(cube_len - 1, -1, -1), 0]]
            cubes_to = np.array([0, 1, 2, 3])
            if sign == 1:
                cubes_from = cubes_to[(np.arange(len(cubes_to) - 1, len(cubes_to) - 1 + len(cubes_to))) % len(cubes_to)]
            else:
                cubes_from = cubes_to[(np.arange(0, len(cubes_to)) + 1) % len(cubes_to)]

            for i in range(4):
                idxs_new = [[idx1, idx2] for idx1 in np.array([cubes_idxs[cubes_to[i]][0]]).flatten() for idx2 in
                            np.array([cubes_idxs[cubes_to[i]][1]]).flatten()]
                idxs_old = [[idx1, idx2] for idx1 in np.array([cubes_idxs[cubes_from[i]][0]]).flatten() for idx2 in
                            np.array([cubes_idxs[cubes_from[i]][1]]).flatten()]
                for idxNew, idxOld in zip(idxs_new, idxs_old):
                    flat_idx_new = np.ravel_multi_index((face, idxNew[0], idxNew[1]), colors_new.shape)
                    flat_idx_old = np.ravel_multi_index((face, idxOld[0], idxOld[1]), colors.shape)
                    rotate_idxs_new[move] = np.concatenate((rotate_idxs_new[move], [flat_idx_new]))
                    rotate_idxs_old[move] = np.concatenate((rotate_idxs_old[move], [flat_idx_old]))

            # Rotate adjacent faces
            face_idxs = adj_idxs[face]
            for i in range(0, len(faces_to)):
                face_to = faces_to[i]
                face_from = faces_from[i]
                idxs_new = [[idx1, idx2] for idx1 in np.array([face_idxs[face_to][0]]).flatten() for idx2 in
                            np.array([face_idxs[face_to][1]]).flatten()]
                idxs_old = [[idx1, idx2] for idx1 in np.array([face_idxs[face_from][0]]).flatten() for idx2 in
                            np.array([face_idxs[face_from][1]]).flatten()]
                for idxNew, idxOld in zip(idxs_new, idxs_old):
                    flat_idx_new = np.ravel_multi_index((face_to, idxNew[0], idxNew[1]), colors_new.shape)
                    flat_idx_old = np.ravel_multi_index((face_from, idxOld[0], idxOld[1]), colors.shape)
                    rotate_idxs_new[move] = np.concatenate((rotate_idxs_new[move], [flat_idx_new]))
                    rotate_idxs_old[move] = np.concatenate((rotate_idxs_old[move], [flat_idx_old]))

        # print("rotate_idxs_new[U1]", rotate_idxs_new["U1"])
        # print("rotate_idxs_old[U1]", rotate_idxs_old["U1"])

        # print("rotate_idxs_new[F1]", rotate_idxs_new["F1"])
        # print("rotate_idxs_old[F1]", rotate_idxs_old["F1"])

        # print("rotate_idxs_new[B1]", rotate_idxs_new["B1"])
        # print("rotate_idxs_old[B1]", rotate_idxs_old["B1"])

        # print("rotate_idxs_new[L1]", rotate_idxs_new["L1"])
        # print("rotate_idxs_old[L1]", rotate_idxs_old["L1"])

        # print("rotate_idxs_new[R1]", rotate_idxs_new["R1"])
        # print("rotate_idxs_old[R1]", rotate_idxs_old["R1"])

        # print("rotate_idxs_new[D1]", rotate_idxs_new["D1"])
        # print("rotate_idxs_old[D1]", rotate_idxs_old["D1"])

        return rotate_idxs_new, rotate_idxs_old
