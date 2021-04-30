from utils.search_utils import create_options
human_options = [["F1", "U1", "F-1"], ["F1", "B1", "F-1"]]
# 2-look OLL algorithms (for matching the third layer)
str_OLLs = ["F R U R' U' F' f R U R' U' f'", "F R U R' U' F'", "f R U R' U' f'", "R U2 R' U' R U' R'", "R U R' U R U' R' U R U2 R'", "F R' F' r U R U' r'", "R U2 R2 U' R2 U' R2 U2 R", "R U R' U R U2 R'", "r U R' U' r' F R F'", "R2 D R' U2 R D' R' U2 R'"]
str_PLLs = ["F R U' R' U' R U R' F' R U R' U' R' F R F'", "R U R' U' R' F R2 U' R' U' R U R' F'", "M2 U M2 U2 M2 U M2", "R U' R U R U R U' R' U' R2", "R2 U R U R' U' R' U' R' U R'", "M' U M2 U M2 U M' U2 M2"]
def get_OLLs():
    return create_options(str_OLLs)
def get_PLLs():
    return create_options(str_PLLs)
print(get_PLLs()[2])