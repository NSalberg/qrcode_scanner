from typing import NamedTuple
from math import isclose


class Rect(NamedTuple):
    cx: float
    cy: float
    width: float
    height: float

def rect_centers_approx_equal(rect0: Rect, rect2: Rect, tolerance: float = 1) -> bool:

    return isclose(rect0.cx, rect2.cx, abs_tol=tolerance) and isclose(
        rect0.cy, rect2.cy, abs_tol=tolerance
    )

def avg_rects(rects: list[Rect]) -> Rect:
    num_rects = len(rects)
    cx = -1
    cy = -1
    width = -1
    height = -1

    for rect in rects:
        cx += rect.cx
        cy += rect.cy
        width += rect.width
        height += rect.height

    cx = cx / num_rects
    cy = cy / num_rects
    width = width / num_rects
    height = height / num_rects
    return Rect(cx, cy, width, height)