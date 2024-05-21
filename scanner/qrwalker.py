from .vis.visualizer import Visualizer
import numpy as np
import numpy.typing as npt

class QRWalker:
    def __init__(
        self,
        functional_mask: npt.NDArray[np.bool_],
    ):
        self.functional_mask = functional_mask
        self.x = len(functional_mask) - 1
        self.y = len(functional_mask) - 1
        self.vertical_move: bool = False
        self.moving_up = -1
        self.moving_left = -1
        self.count = 0

    def __iter__(self):
        self.count = 0
        self.x = len(self.functional_mask) - 1
        self.y = len(self.functional_mask) - 1
        return self

    def __next__(self) -> tuple[int, int]:
        return self.get_next_index()

    def get_next_index(self) -> tuple[int, int]:
        if self.count == 0:
            self.count += 1
            return (self.x, self.y)

        if self.x < 0 or self.y < 0:
            raise StopIteration

        self.count += 1
        self.y += self.vertical_move * self.moving_up
        self.x += self.moving_left
        if self.y < 0 or self.y > len(self.functional_mask) - 1:
            self.y -= self.vertical_move * self.moving_up
            self.x -= self.moving_left * 2
            self.moving_up = self.moving_up * -1
        # skip the seventh row because timing patterns
        if self.x == 6:
            self.x += -1

        self.vertical_move = not self.vertical_move
        self.moving_left = self.moving_left * -1

        if self.functional_mask[self.x, self.y] == True:
            index = self.get_next_index()
        else:
            index = (self.x, self.y)

        return index
