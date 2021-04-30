from utils.search_utils import create_options
human_options = [["R1", "U1", "R-1", "U-1"], ["L-1", "U-1", "L1", "U1"]]
# 2-look OLL algorithms (for matching the third layer)
str_OLLs = ["F R U R' U' F' B U L U' L' B'", "F R U R' U' F'", "B U L U' L' B'", "R U2 R' U' R U' R'", "R U R' U R U' R' U R U2 R'", "F R' F' L F R F' L'", "R U2 R2 U' R2 U' R2 U2 R", "R U R' U R U2 R'", "L F R F' L' F R F'", "R2 D R' U2 R D' R' U2 R'"]
str_PLLs = ["F R U' R' U' R U R' F' R U R' U' R' F R F'", "R U R' U' R' F R2 U' R' U' R U R' F'", "M2 D M2 U2 M2 D M2", "R U' R U R U R U' R' U' R2", "R2 U R U R' U' R' U' R' U R'", "M2 D M2 U M' F2 M2 B2 M'"]
def get_OLLs():
    return create_options(str_OLLs)
def get_PLLs():
    return create_options(str_PLLs)



def generate_options(human: bool) -> List[List[str]]:
    options = human_options
    if human: 
        return human_options + get_OLLs() + get_PLLs()
    else 