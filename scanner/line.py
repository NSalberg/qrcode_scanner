from typing import List, Tuple, NamedTuple
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
