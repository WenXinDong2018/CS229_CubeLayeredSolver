from utils.search_utils import create_options
human_options = [["F1", "U1", "F-1"], ["F1", "B1", "F-1"]]
# 2-look OLL algorithms (for matching the third layer)
OLLs = [
    ['F1', 'R1', 'U1', 'R-1', 'U-1', 'F-1', 'B1', 'R1', 'U1', 'R-1', 'U-1', 'B-1'], 
    ['F1', 'R1', 'U1', 'R-1', 'U-1', 'F-1'],
    ['B1', 'R1', 'U1', 'R-1', 'U-1', 'B-1'],
    ['R1', 'U1', 'U1', 'R-1', 'U-1', 'R1', 'U-1', 'R-1'], 
    ['R1', 'U1', 'R-1', 'U1', 'R1', 'U-1', 'R-1', 'U1', 'R', 'U', 'U', 'R-1']
]
def get_OLLs():
    return create_options(str_OLLs)

