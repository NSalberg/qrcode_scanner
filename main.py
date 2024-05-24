from scanner.vis import visualizer
from scanner.vis.visualizer import Visualizer
from scanner.qrwalker import QRWalker
import imageio as iio
import numpy as np
import numpy.typing as npt
from typing import NamedTuple
from scanner.qrdecoder import QRDecoder
import os

class Candidate(NamedTuple):
    length: int
    start: int

class QRFinder:
        def find_qr_code(self, image: npt.NDArray[np.float_] ):
            return

class QRReader:
    def __init__(self):
        self.qr_finder = QRFinder()
        #self.qr_decoder = QRDecoder()
    def read_qr_image(self, image: npt.NDArray[np.float_], vis: Visualizer) -> str:

        image = image[:,:,:3]
        print(f"image shape: {np.shape(image)}")

        #TODO: identify image format and process it 

        #qr_region = self.qr_finder.find_qr_code(image)
        
        #TODO: if rotated or skewed undo rotation and skew 

        dims = len(image[0])

        decoder = QRDecoder(vis)
        qr_data = decoder.decode_image(image)

        return qr_data

if __name__ == "__main__":
    directory = "./test_images/"
    reader = QRReader()
    for root, directories, filenames in os.walk(directory):
        for i, filename in enumerate(filenames):
            image_path = os.path.join(root, filename)
            print(image_path)
            image = iio.v3.imread(image_path).squeeze()
            dims = len(image[0])
            v = Visualizer(f"test{i}.svg", width=dims, height=dims)
            v.add_image(image_path)
            qr_data = reader.read_qr_image(image,v)
            print(f"{qr_data} \n")
            del v


