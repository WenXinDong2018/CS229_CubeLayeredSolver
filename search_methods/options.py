from typing import List, Dict
from utils.search_utils import create_options
import pickle
import os
human_options = [["R1", "U1", "R-1", "U-1"], ["L-1", "U-1", "L1", "U1"]] # left hand move and right hand move.
# 2-look OLL algorithms (for matching the third layer)
str_OLLs = ["L' D2 L2 B' L' B D2 L B' L' B", #Dot shape (edge)
            "B' L' D' L D B", #I-shape (edge)
            "F' D' R' D R F", #L-shape (edge)
            "R D2 R' D' R D' R'", #Anti-sune (corner)
            "L' D' L D' L' D L D' L' D2 L", #H (corner)
            "B' L B R' B' L' B R", #L (corner)
            "L' D2 L2 D L2 D L2 D2 L'", #Pi (corner)
            "L' D2 L D L' D L", #Sune (corner)
            "R' B' L B R B' L' B", #T (corner)
            "L2 U' L D2 L' U L D2 L" #U (corner)
            ]
str_PLLs = ["D' L' R L' U L' D2 L U' L' D2 L2 R' L D", #Aa
            "D L R' L U' L D2 L' U L D2 L2 R L' D'", #Ab
            "F2 B2 L U L' U L U' L' U' L2 F L' U' L' U L F' U' L' B2 F2", #F
            "U' R2 D R' D R' D' R D' R2 U D' R' D R", #Ga
            "U R' D' R U' D R2 D R' D R D' R D' R2", #Gb
            "U R2 D' R D' R D R' D R2 U' D R D' R'", #Gc
            "U' R D R' U D' R2 D' R D' R' D R' D R2", #Gd
            "B2 L' D' L B2 R' U R' U' R2", #Ja
            "D B2 R D R' B2 L U' L U L2 D'", #Jb
            "D' R D2 R U R' D R U' R' D' R' D R D R' D", #Ra
            "D' L' D2 L' U' L D' L' U L D L D' L' D' L D", #Rb
            "D2 B' L' D L D' L' D' L2 B L' D' L' D L D2", #T
            "B' L' F' L B L' F L B' L' F L B L' F' L", #E
            "R2 L2 R U R' U2 R U R2 F' R U R U' R' F R U' R' U' R U' R' L2 R2", #Na
            "R2 L2 R' U R' F R F' R U' R' F' U F R U R' U' R L2 R2", #Nb
            "R' F' R' F D' F D F2 R F D R D' R", #V
            "B R' B' R D R D' R' B R D' R' D R D R' B'", #Y
            "D R2 L2 U' L R' F2 R L' U' L2 R2 D' R2 L2 U' L R' F2 R L' U' L2 R2", #H
            "D2 L2 R2 U R' L F2 L' R U R2 L2 D2", #Ua
            "D2 R2 L2 U' L R' F2 R L' U' L2 R2 D2", #Ub
            "D' R2 L2 U' L R' F2 R L' U' L2 R2 D R2 L2 U' L R' F2 R L' U' L2 R2" #H
             ]
def get_OLLs():
    # call this method to convert the OLL_str to options usable for search
    return create_options(str_OLLs)
def get_PLLs():
    # call this method to convert the PLL_str to options usable for search
    return create_options(str_PLLs)

def get_expert_options(length: int, top: int):
    options = pickle.load(open("search_methods/options_data/options_of_length{}.pkl".format(length), "rb"))
    op_list = list(options.keys())[:top]
    return [op_list[i].split(' ') for i in range(len(op_list))]


def generate_options(human: bool, length: int = 4, top: int = 50) -> List[List[str]]:
    # call this method to combine the human options if human flag is set to True, 
    # else it will return the top #top# experts options gathered from Kaggle data set of length #length#.
    options = human_options
    if human: 
        return human_options + get_OLLs() + get_PLLs()
    else: 
        return get_expert_options(length, top)
