from vis import Visualizer 
import imageio as iio
import numpy as np
from collections.abc import Iterable, Iterator 
from typing import List, Tuple, NamedTuple
from dataclasses import dataclass 
from math import isclose
from collections import Counter


def encode_line(iterable: Iterable[float]) -> List[int]:
    last_seen = iterable[0] > 127.5
    seen_count = 1
    encoded_line = []

    for value in iterable[1:]:
        current = value > 127.5
        if last_seen == current:
            seen_count += 1
        else:
            encoded_line.append(seen_count)
            last_seen = current
            seen_count = 1
    encoded_line.append(seen_count)

    return encoded_line


class Candidate(NamedTuple):
    length: int
    start: int

class Rect(NamedTuple):
    cx: float
    cy: float
    width: float
    height: float

def rect_centers_approx_equal(rect1: Rect, rect2: Rect, tolerance: float = 1) -> bool:
    return isclose(rect1.cx, rect2.cx, abs_tol=tolerance) and isclose(rect1.cy, rect2.cy, abs_tol=tolerance)

def avg_rects(rects:List[Rect]) -> Rect:
    num_rects = len(rects)
    cx = 0
    cy = 0
    width = 0
    height = 0

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

def add_rect_to_bucket(buckets: List[List[Rect]], rect: Rect):
    added = False
    for bucket in buckets:
        for bucket_rect in bucket:
            if rect_centers_approx_equal(bucket_rect, rect, 10):
                bucket.append(rect)
                added = True
                break

    if added == False:
        buckets.append([rect])
    return

def find_candidates(encodings: List[int]) -> List[Candidate]:
    if len(encodings) <= 5:
        return []
    candidates = []
    start = encodings[1] * -1 
    for i in range(len(encodings) - 5):
        length = encodings[i]
        start += length
        if encodings[i+1] != length:
            continue
        elif encodings[i+3] != length:
            continue
        elif encodings[i+4] != length:
            continue
        elif encodings[i+2] != length * 3:
            continue
        candidates.append(Candidate(length * 7, start))

    return candidates


def get_avg_timing_width(encodings, X, top_left):
    timing_x = int(4 * X + top_left.cx)
    timing_y = int(3 * X + top_left.cy)
    loc = 0
    sum = 0
    for enc in encodings[timing_y]:
        loc += enc
        if loc > timing_x and loc < timing_x + 8 * X:
            sum += enc
    avg = sum / 7 
    return avg

def get_avg_timing_height(encodings, X, top_left):
    timing_x = int(3 * X + top_left.cx)
    timing_y = int(4 * X + top_left.cy)
    loc = 0
    sum = 0
    for enc in encodings[timing_x]:
        loc += enc
        if loc > timing_y and loc < timing_y + 8 * X:
            sum += enc
            v.draw_circle(timing_x, loc, 0.5, "green")
            
    avg = sum / 7 
    return avg

class Line:
    def __init__(self, x1: float, y1: float, x2: float, y2: float):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def __str__(self):
        return f"Line: {(self.x1, self.y1, self.x2, self.y2)}" 

    def __repr__(self):
        return f"Line: {(self.x1, self.y1, self.x2, self.y2)}" 

    def find_intersect(self, other_line) -> Tuple[float, float]:
       # Unpack the coordinates of the first line
        x1, y1 = self.x1, self.y1
        x2, y2 = self.x2, self.y2
        
        # Unpack the coordinates of the second line
        x3, y3 = other_line.x1, other_line.y1
        x4, y4 = other_line.x2, other_line.y2
        
        # Calculate the slopes of the lines
        m1 = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')  # Avoid division by zero
        m2 = (y4 - y3) / (x4 - x3) if x4 != x3 else float('inf')  # Avoid division by zero
        
        # Check if lines are parallel
        if m1 == m2:
            return None  # Parallel lines don't intersect
        
        # Calculate the intersection point
        if m1 == float('inf'):  # Line 1 is vertical
            x_intersect = x1
            y_intersect = m2 * (x_intersect - x3) + y3
        elif m2 == float('inf'):  # Line 2 is vertical
            x_intersect = x3
            y_intersect = m1 * (x_intersect - x1) + y1
        else:
            x_intersect = (m1 * x1 - m2 * x3 + y3 - y1) / (m1 - m2)
            y_intersect = m1 * (x_intersect - x1) + y1
        
        return x_intersect, y_intersect

def sample_around(gray_image: np.array, x: int, y: int) -> List[bool]:
    bin_image = gray_image < 127
    offsets = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1), (0, 0), (0, 1),
               (1, -1), (1, 0), (1, 1)]
    # True if black False if white
    colors: List[bool] = []
    for xoffset, yoffset in offsets:
        colors.append(bin_image[y + yoffset][x + xoffset])

    return colors


v = Visualizer("test2.svg", 100, 100)
if __name__ == "__main__":
    v.add_image("test.gif")
    image = iio.v3.imread("test.gif")
    gray_image = np.dot(image, [0.2989, 0.5870, 0.1140])
    gray_image = np.round(gray_image).astype(np.uint8) 
    gray_image = np.squeeze(gray_image)



    horizontal_scan_encodings: List[List[int]] = [encode_line(row) for row in gray_image]
    vertical_scan_encodings: List[List[int]] = [encode_line(row) for row in gray_image.T]
    for y in gray_image:
        print("")
        for x in y:
            print("X" if x > 127 else " ", end="")

    final_candidates_vertical: List[Rect] = []
    final_candidates_horizontal: List[Rect] = []
    #could this be abstracted away?
    for y, encoding in enumerate(horizontal_scan_encodings):
        candidates = find_candidates(encoding)
        if candidates:
            for candidate in candidates:
                cx = candidate.length/2 + candidate.start
                cy = y + 0.5
                final_candidates_horizontal.append(Rect(cx, cy, candidate[0], candidate[0]))
                v.draw_circle(cx=cx, cy=cy, r=.5, color="red")

    for x, encoding in enumerate(vertical_scan_encodings):
        candidates = find_candidates(encoding)
        if candidates:
            for candidate in candidates:
                cx = x + 0.5
                cy = candidate.length/2 + candidate.start
                final_candidates_vertical.append(Rect(cx, cy, candidate[0], candidate[0]))
                v.draw_circle(cx=cx, cy=cy, r=.5, color="blue")
    
    buckets: List[List[Rect]] = []
    for candidate_vert in final_candidates_vertical:
        for candidate_horiz in final_candidates_horizontal:
            if rect_centers_approx_equal(candidate_vert, candidate_horiz):
                rect = avg_rects([candidate_vert, candidate_horiz])
                add_rect_to_bucket(buckets, rect) 

    centers: List[Rect] = []
    for bucket in buckets:
        center = avg_rects(bucket) 
        centers.append(center)
        v.draw_circle(center.cx, center.cy, 1, "yellow")
    print(centers)
    

    if len(centers) != 3:
        print("number of centers is not 3")
        exit()

    top_left: Rect = Rect(float("inf"),float("inf"),0,0)
    for center in centers:
        if center.cx * center.cy < top_left.cx * top_left.cy:
            top_left = center
    centers.remove(top_left)

    top_right: Rect 
    rightmost = 0
    for center in centers:
        if center.cx - top_left.cx > rightmost:
            top_right = center
    centers.remove(top_right)
    bottom_left = centers[0]
    print(top_right, top_left, bottom_left)

    D = top_right.cx - top_left.cx 
    X = (top_right.width + top_left.width) / 14
    CPul = top_left.width / 7
    version = ((D/X)-10)/4
    print(D,X,version, CPul)
    

    # FOR VERSION 1 COMPUTE X,Y AS AVERAGE OF TIMING WIDTHS  
    if version == 1:
        X = get_avg_timing_width(horizontal_scan_encodings, X, top_left)
        Y = get_avg_timing_height(vertical_scan_encodings, X, top_left)
        top_left_square_y = top_left.cy - 3 * Y 
        top_left_square_x = top_left.cx - 3 * X

        vertical_lines: List[Line] = []
        for x in np.arange(top_left_square_x, top_left_square_x + X * 21, X):
            vertical_lines.append(Line(x, top_left_square_y, x, top_left_square_y + Y * 20))

        horizontal_lines: List[Line] = []
        for y in np.arange(top_left_square_y, top_left_square_y + Y * 21, Y):
            horizontal_lines.append(Line(top_left_square_x, y, top_left_square_x + X * 20, y))
        """
        for line in vertical_lines:
           v.draw_line(line.x1, line.y1, line.x2, line.y2, width=0.5, color="green")
        for line in horizontal_lines:
           v.draw_line(line.x1, line.y1, line.x2, line.y2, width=0.5, color="blue")
          """      
        
        final_matrix: List[List[bool]] = [] 
        for vline in vertical_lines:
            for hline in horizontal_lines:
                xinter, yinter = vline.find_intersect(hline)
                color = Counter(sample_around(gray_image, int(xinter), int(yinter))).most_common(1)[0][0]
                final_matrix.append(color)
                c = "yellow" if color else "purple"
                print(color, xinter, yinter, vline, hline)
                v.draw_circle(xinter, yinter, r=1, color=c)
                

                


        
    del v
