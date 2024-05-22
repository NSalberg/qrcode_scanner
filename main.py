from scanner.vis.visualizer import Visualizer
from scanner.qrwalker import QRWalker
import imageio as iio
import numpy as np
import numpy.typing as npt
from typing import NamedTuple
from math import isclose, ceil
from scanner.qrdecoder import QRDecoder

class Candidate(NamedTuple):
    length: int
    start: int

class QRFinder:
    def find_qr_code(self, image):
        return

class QRReader:
    def __init__(self):
        self.qr_finder = QRFinder()
        #self.qr_decoder = QRDecoder()
    def read_qr_image(self, image):
        #TODO: identify image format and process it 

        qr_region = self.qr_finder.find_qr_code(image)
        
        #TODO: if rotated or skewed undo rotation and skew 

        #QRDecoder.decode_image()

        return

if __name__ == "__main__":
    image_path = "test.gif"
    image = iio.v3.imread(image_path).squeeze()

    dims = len(image[0])
    v = Visualizer("test2.svg", width=dims, height=dims)
    v.add_image(image_path)


    decoder = QRDecoder(v)
    print(decoder.decode_image(image))

    del v

    image_path = "Hello,world!123v2.png"
    image = iio.v3.imread(image_path)

    dims = len(image[0])
    image = image[:,:,:3]
    v = Visualizer("test3.svg", width=dims, height=dims)
    v.add_image(image_path)

    decoder = QRDecoder(v)
    print(decoder.decode_image(image))
    del v
