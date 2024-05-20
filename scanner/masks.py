import numpy as np
import numpy.typing as npt

def function_mask(version: int) -> npt.NDArray[np.bool_]:
    mask = np.full((21,21), False)
    match version:
        case 1:
            mask = np.full((21,21), False)
            mask[:,6] = True
            mask[6,:] = True
            mask[:9,:9] = True
            mask[13:,:9] = True
            mask[:9,13:] = True
        case _:
            print(f"unsupported version {version}")

    return mask

def mask_0(x: int, y: int) -> bool:
    return x % 3 == 0

