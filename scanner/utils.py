import numpy.typing as npt
import numpy as np
def sample_around(gray_image: npt.NDArray[np.float_], x: int, y: int) -> list[bool]:
    bin_image = gray_image < 127
    offsets = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 0),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
    # True if black False if white
    colors: list[bool] = []
    for xoffset, yoffset in offsets:
        colors.append(bin_image[y + yoffset][x + xoffset])

    return colors
