from typing import Callable
import numpy as np
import numpy.typing as npt

    

def get_alignment_coordinates(version):
    if version <= 1:
        return []
    intervals = (version // 7) + 1  # Number of gaps between alignment patterns
    distance = 4 * version + 4  # Distance between first and last alignment pattern
    step = round(distance / intervals)  # Round equal spacing to nearest integer
    step += step & 0b1  # Round step to next even number
    coordinates = [6]  # First coordinate is always 6 (can't be calculated with step)
    for i in range(1, intervals + 1):
        coordinates.append(6 + distance - step * (intervals - i))  # Start right/bottom and go left/up by step*k
    return coordinates

def get_finder_mask(version: int) -> npt.NDArray[np.bool_]:
    num_row_cols = (version-1) * 4 + 21 
    match version:
        case 1:
            mask = np.full((num_row_cols,num_row_cols), False)
            mask[:,6] = True
            mask[6,:] = True
            mask[:9,:9] = True
            mask[13:,:9] = True
            mask[:9,13:] = True
        case 2: 
            mask = np.full((num_row_cols,num_row_cols), False)
            mask[:,6] = True
            mask[6,:] = True
            mask[:9,:9] = True
            mask[num_row_cols - 8:,:9] = True
            mask[:9,num_row_cols - 8:] = True

            coords = get_alignment_coordinates(version)
            for i in coords:
                for j in coords:
                    print(i-2,i+3, j-2, j+3)
                    mask[i-2:i+3,j-2:j+3] = True

        case _:
            print(f"unsupported finder mask version {version}")
            exit()
    return mask

def get_mask_function(mask_pattern: list[bool]) -> Callable[[int,int],int]:
        match mask_pattern:
            case [True, True, True]:
                return mask_0
            case [True, False, False]:
                return mask_4
            case _:
                print(f"unkown mask{mask_pattern}")
                raise Exception

def mask_0(x: int, y: int) -> bool:
    return x % 3 == 0

def mask_4(x: int, y: int) -> bool:
    return y % 2  == 0

