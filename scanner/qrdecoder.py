from .vis import Visualizer
from .qrwalker import QRWalker
from collections import Counter
from scanner.line import Line
from scanner.masks import get_finder_mask, get_mask_function
from scanner.rect import Rect, rect_centers_approx_equal, avg_rects
from scanner.utils import sample_around
from typing import NamedTuple
import numpy as np
import numpy.typing as npt
from math import isclose

Array2D = npt.NDArray[np.float_]
BinaryMatrix = npt.NDArray[np.bool_]

class Candidate(NamedTuple):
    length: int
    start: int

class QRDecoder:
    def __init__(self, visualizer: Visualizer):
        self.v = visualizer

    def decode_image(self, image: Array2D):
        gray_image = self.convert_to_grayscale(image)
        print(np.shape(gray_image))

        horizontal_scan_encodings: list[list[int]] = [self.encode_line(row) for row in gray_image]
        vertical_scan_encodings: list[list[int]] = [self.encode_line(row) for row in gray_image.T]

        centers = self.find_centers(horizontal_scan_encodings, vertical_scan_encodings)

        top_left_center, top_right_center, _ = self.localize_centers(centers)
        x_spacing, y_spacing, version = self.calculate_spacing_and_version(horizontal_scan_encodings, vertical_scan_encodings, top_left_center, top_right_center)
        
        qr_matrix = self.create_QR_matrix(gray_image, top_left_center, x_spacing, y_spacing, version)
        finder_mask = get_finder_mask(version)
        qr_matrix = self.apply_mask(qr_matrix, finder_mask, version)

        # walk qrcode
        walker = QRWalker(finder_mask)

        #print walk
#        for i, (x,y) in enumerate(walker):
#            self.v.draw_text(x * x_spacing + top_left_x, y * y_spacing + top_left_y, str(i), "blue", 5)

        data_bits = [qr_matrix[x,y] for x,y in walker]
        return self.parse_data(data_bits, version)

    def draw_qr_matrix(self, qr_matrix: Array2D, top_left_center: Rect, x_spacing: float, y_spacing: float) -> None:
        top_left_x = top_left_center.cx - (top_left_center.width/7 * 3)
        top_left_y = top_left_center.cy - (top_left_center.height / 7 * 3)

        for x in range(len(qr_matrix)):
            for y in range(len(qr_matrix)):
                c = "yellow" if qr_matrix[x, y] else "purple"
                self.v.draw_circle(x * x_spacing + top_left_x, y * y_spacing + top_left_y, r=1, color=c)

    def apply_mask(self, qr_matrix: BinaryMatrix, finder_mask: BinaryMatrix, version: int) -> BinaryMatrix:
        format = self.read_format(qr_matrix)
        mask = format[2:5]
        mask_function = get_mask_function(mask.tolist())

        for y in range(len(qr_matrix)):
            for x in range(len(qr_matrix)):
                if finder_mask[x, y] == False and mask_function(x, y):
                    qr_matrix[x, y] = not qr_matrix[x, y]

        return qr_matrix


    def calculate_spacing_and_version(self, horizontal_scan_encodings: list[list[int]], vertical_scan_encodings: list[list[int]], top_left_center: Rect, top_right_center: Rect) -> tuple[float, float, int]:
        x_spacing: float = (top_right_center.width + top_left_center.width) / 14
        y_spacing: float = x_spacing
        d: float = top_right_center.cx - top_left_center.cx
        version: int = self.identify_version(x_spacing, d)
        print("x, y spacing: ", x_spacing, y_spacing)

        # the standare suggests that we change our sample grid dependent upon the mini finder patterns but we ignore that for now
        x_spacing = self.get_avg_timing_width(horizontal_scan_encodings, x_spacing, top_left_center)
        y_spacing = self.get_avg_timing_height(vertical_scan_encodings, x_spacing, top_left_center)

        return x_spacing, y_spacing, version


    def convert_to_grayscale(self, image: Array2D) -> Array2D:
        gray_image: Array2D = np.dot(image, [0.2989, 0.5870, 0.1140])
        gray_image = np.round(gray_image).astype(np.uint8)
        gray_image = np.squeeze(gray_image)
        return gray_image


    # TODO: change to incorportate version info
    def get_avg_timing_width(self, encodings: list[list[int]], X: float, top_left: Rect):
        timing_x = int(4 * X + top_left.cx)
        timing_y = int(3 * X + top_left.cy)
        loc = 0
        sum = 0
        num = 0
        for enc in encodings[timing_y]:
            loc += enc
            if loc > timing_x and loc < timing_x + 8 * X:
                num += 1
                sum += enc
        avg = sum / num
        return avg


    # TODO: change to incorportate version info
    def get_avg_timing_height(self, encodings: list[list[int]], X: float, top_left: Rect,):
        timing_x = int(3 * X + top_left.cx)
        timing_y = int(4 * X + top_left.cy)
        loc = 0
        sum = 0
        num = 0
        for enc in encodings[timing_x]:
            loc += enc
            if loc > timing_y and loc < timing_y + 8 * X:
                sum += enc
                num += 1
                self.v.draw_circle(timing_x, loc, 0.5, "green")

        avg = sum / num
        return avg
    
    def parse_data(self, data_bits: list[bool], version: int) -> str:
        enc_bits = data_bits[:4]
        encoding = self.identify_encoding(enc_bits)

        num_bits_in_len_field = 0
        match encoding:
            case "Numeric":
                if version <= 9:
                    num_bits_in_len_field = 10
                elif version >= 10 and version <= 26:
                    num_bits_in_len_field = 12
                else:
                    num_bits_in_len_field = 14
                raise Exception
            case "Alphanumeric":
                if version <= 9:
                    num_bits_in_len_field = 9
                elif version >= 10 and version <= 26:
                    num_bits_in_len_field = 11
                else:
                    num_bits_in_len_field = 13
                raise Exception
            case "Byte":
                if version <= 9:
                    num_bits_in_len_field = 8
                elif version >= 10 and version <= 26:
                    num_bits_in_len_field = 16
                else:
                    num_bits_in_len_field = 16
            case "Kanji":
                if version <= 9:
                    num_bits_in_len_field = 8
                elif version >= 10 and version <= 26:
                    num_bits_in_len_field = 16
                else:
                    num_bits_in_len_field = 16
                raise Exception
            case _: 
                raise Exception


        bin_string = "".join(
            ["1" if bit else "0" for bit in data_bits[4 : 4 + num_bits_in_len_field]]
        )
        num_characters = int(bin_string, 2)
        print("num_bits in len field: ",num_bits_in_len_field)
        print("num characters: ", num_characters)

        word = ""
        byte_size = 8
        for i in range(num_characters):
            offset = i * byte_size
            start = 4 + num_bits_in_len_field + offset
            end = 4 + num_bits_in_len_field + byte_size + offset
            byte = "".join("1" if bit else "0" for bit in data_bits[start:end])
            word += chr(int(byte, 2))
            print(byte, chr(int(byte, 2)))
        return word

    def identify_encoding(self, encoding_bits: list[bool]):
        match encoding_bits:
            case [0, 0, 0, 1]:
                return "Numeric"
            case [0, 0, 1, 0]:
                return "Alphanumeric"
            case [0, 1, 0, 0]:
                return "Byte"
            case [1, 0, 0, 0]:
                return "Kanji"
            case [0, 0, 1, 1]:
                return "Structured Append"
            case [0, 1, 0, 1]:
                return "FNC1 first pos"
            case [1, 0, 0, 1]:
                return "FNC1 second pos"
            case [0, 0, 0, 0]:
                return "End of Message"
            case _: 
                pass

    def read_format(self, qr_matrix: BinaryMatrix) -> BinaryMatrix:
        s1: BinaryMatrix = qr_matrix[:6, 8]
        s2: BinaryMatrix = qr_matrix[7:9, 8]
        s3: BinaryMatrix = qr_matrix[14:, 8]
        return np.concatenate((s1, s2, s3))

    def create_QR_matrix(self, gray_image: Array2D, top_left_center: Rect, x_spacing: float, y_spacing: float, version: int) -> BinaryMatrix:
        top_left_square_y = top_left_center.cy - 3 * y_spacing
        top_left_square_x = top_left_center.cx - 3 * x_spacing
        print("top_left_x, x_spacing: ",top_left_square_x, x_spacing)
        num_row_cols = (version - 1) * 4 + 21
        print(f"numrowcols: {num_row_cols}")


        vertical_lines: list[Line] = []
        for x in np.arange(top_left_square_x, top_left_square_x + x_spacing * num_row_cols, x_spacing):
            vertical_lines.append(
                Line(x, top_left_square_y, x, top_left_square_y + y_spacing * num_row_cols - 1)
            )

        horizontal_lines: list[Line] = []
        for y in np.arange( top_left_square_y, top_left_square_y + y_spacing * num_row_cols, y_spacing):
            horizontal_lines.append(
                Line(top_left_square_x, y, top_left_square_x + x_spacing * num_row_cols - 1, y)
            )

        final_matrix: list[list[bool]] = []
        for vline in vertical_lines:
            row: list[bool] = []
            for hline in horizontal_lines:
                xinter, yinter = vline.find_intersect(hline)
                if xinter == None:
                    continue
                color = Counter(
                    sample_around(gray_image, int(xinter), int(yinter))
                ).most_common(1)[0][0]
                row.append(bool(color))
                #c = "yellow" if color else "purple"
            final_matrix.append(row)
        return np.array(final_matrix, dtype=np.bool_)

    def identify_version(self, x: float, d: float) -> int:
        version = int(((d / x) - 10) / 4)
        if version > 7:
            # TODO implement reading higher version number
            print(f"Unsupported version{version}")
            exit()
        return version

    def find_centers(self, horizontal_scan_encodings: list[list[int]], vertical_scan_encodings: list[list[int]]) -> list[Rect]:
        final_candidates_horizontal: list[Rect] = self.find_finder_candidates(
            horizontal_scan_encodings, is_vert=False
        )
        final_candidates_vertical: list[Rect] = self.find_finder_candidates(
            vertical_scan_encodings, is_vert=True
        )

        buckets: list[list[Rect]] = []
        for candidate_vert in final_candidates_vertical:
            for candidate_horiz in final_candidates_horizontal:
                if rect_centers_approx_equal(candidate_vert, candidate_horiz):
                    rect = avg_rects([candidate_vert, candidate_horiz])
                    self.add_rect_to_bucket(buckets, rect)

        centers: list[Rect] = []
        for bucket in buckets:
            center = avg_rects(bucket)
            centers.append(center)
            self.v.draw_circle(center.cx, center.cy, 1, "yellow")
        if len(centers) != 3:
            print("number of centers is not 3")
            exit()
        return centers

    def add_rect_to_bucket(self, buckets: list[list[Rect]], rect: Rect):
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

    def localize_centers(self, centers: list[Rect]) -> tuple[Rect, Rect, Rect]:
        top_left_center: Rect = Rect(float("inf"), float("inf"), 0, 0)
        for center in centers:
            if center.cx * center.cy < top_left_center.cx * top_left_center.cy:
                top_left_center = center
        centers.remove(top_left_center)

        top_right_center: Rect = Rect(float("inf"), float("inf"), 0, 0)
        rightmost = 0
        for center in centers:
            if center.cx - top_left_center.cx > rightmost:
                top_right_center = center
        centers.remove(top_right_center)
        bottom_left_center = centers[0]
        return top_left_center, top_right_center, bottom_left_center

    def find_finder_candidates(self, encodings: list[list[int]], is_vert: bool) -> list[Rect]:
        finder_candidates: list[Rect] = []
        for i, encoding in enumerate(encodings):
            candidates = self.find_candidates(encoding)
            if candidates:
                for candidate in candidates:
                    if is_vert:
                        ci = i + 0.5
                        cj = candidate.length / 2 + candidate.start
                    else:
                        ci = candidate.length / 2 + candidate.start
                        cj = i + 0.5
                    self.v.draw_circle(cx=ci, cy=cj, r=0.5, color="red")
                    finder_candidates.append(Rect(ci, cj, candidate[0], candidate[0]))
        return finder_candidates

    def find_candidates(self, encodings: list[int]) -> list[Candidate]:
        if len(encodings) <= 5:
            return []
        candidates: list[Candidate] = []
        start = encodings[1] * -1
        tol = .2
        for i in range(len(encodings) - 5):
            length = encodings[i]
            start += length
            if not isclose(encodings[i + 1], length, rel_tol=tol):
                continue
            elif not isclose(encodings[i + 3], length, rel_tol=tol):
                continue
            elif not isclose(encodings[i + 4], length, rel_tol=tol):
                continue
            elif not isclose(encodings[i + 2], length * 3, rel_tol=tol):
                continue
            candidates.append(Candidate(length * 7, start))

        return candidates

    def encode_line(self, iterable: Array2D) -> list[int]:
        last_seen = iterable[0] > 127.5
        seen_count = 1
        encoded_line: list[int] = []

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
