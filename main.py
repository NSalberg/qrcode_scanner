from scanner.vis.visualizer import Visualizer
import imageio as iio
import numpy as np
import numpy.typing as npt
from collections.abc import Iterable
from typing import NamedTuple
from math import isclose
from collections import Counter
from scanner.line import Line
from scanner.masks import mask_0, function_mask


class Candidate(NamedTuple):
    length: int
    start: int


class Rect(NamedTuple):
    cx: float
    cy: float
    width: float
    height: float


class QRWalker:
    def __init__(
        self,
        qr_code: npt.NDArray[np.bool_],
        functional_mask: npt.NDArray[np.bool_],
        visualizer: Visualizer,
    ):
        self.qr_code: npt.NDArray[np.bool_] = qr_code
        self.functional_mask = functional_mask
        self.v = visualizer
        self.x = len(qr_code) - 1
        self.y = len(qr_code) - 1
        self.vertical_move: bool = False
        self.moving_up = -1
        self.moving_left = -1
        self.count = 0

    def __iter__(self):
        self.count = 0
        self.x = len(self.qr_code) - 1
        self.y = len(self.qr_code) - 1
        return self

    def __next__(self) -> bool:
        return self.get_next_bit()

    def get_next_bit(self) -> bool:
        if self.count == 0:
            self.count += 1
            return self.qr_code[self.x, self.y]

        if self.x < 0 or self.y < 0:
            raise StopIteration

        # v.draw_text(self.x * 4 + 10, self.y * 4 + 10, text=f"{self.x} {self.y}", color = "blue", size = 2, font_weight = "bold")

        self.count += 1
        self.y += self.vertical_move * self.moving_up
        self.x += self.moving_left
        if self.y < 0 or self.y > len(self.qr_code) - 1:
            self.y -= self.vertical_move * self.moving_up
            self.x -= self.moving_left * 2
            self.moving_up = self.moving_up * -1
        # skip the seventh row because timing patterns
        if self.x == 6:
            self.x += -1

        self.vertical_move = not self.vertical_move
        self.moving_left = self.moving_left * -1

        if self.functional_mask[self.x, self.y] == True:
            v.draw_text(
                self.x * 4 + 10,
                self.y * 4 + 10,
                text=str(self.count),
                color="blue",
                size=2,
                font_weight="bold",
            )
            bit = self.get_next_bit()
        else:
            v.draw_text(
                self.x * 4 + 10,
                self.y * 4 + 10,
                text=str(self.count),
                color="green",
                size=2,
                font_weight="bold",
            )
            bit = self.qr_code[self.x, self.y]

        return bit


class QRDecoder:
    def __init__(self, visualizer: Visualizer):
        self.v = visualizer

    def decode_image(self, image: npt.NDArray[np.float_]):
        gray_image: npt.NDArray[np.float_] = np.dot(image, [0.2989, 0.5870, 0.1140])
        gray_image = np.round(gray_image).astype(np.uint8)
        gray_image = np.squeeze(gray_image)

        horizontal_scan_encodings: list[list[int]] = [
            self.encode_line(row) for row in gray_image
        ]
        vertical_scan_encodings: list[list[int]] = [
            self.encode_line(row) for row in gray_image.T
        ]

        centers = self.find_centers(horizontal_scan_encodings, vertical_scan_encodings)
        if len(centers) != 3:
            print("number of centers is not 3")
            exit()
        top_left_center, top_right_center, bottom_left_center = self.localize_centers(
            centers
        )
        d: float = top_right_center.cx - top_left_center.cx
        x_spacing: float = (top_right_center.width + top_left_center.width) / 14
        y_spacing: float = x_spacing
        version: int = self.identify_version(x_spacing, d)

        # FOR VERSION 1 COMPUTE X,Y AS AVERAGE OF TIMING WIDTHS
        qr_matrix = []
        if version == 1:
            x_spacing = get_avg_timing_width(horizontal_scan_encodings, x_spacing, top_left_center)
            y_spacing = get_avg_timing_height(vertical_scan_encodings, x_spacing, top_left_center)
            qr_matrix = self.create_QR_matrix(gray_image, top_left_center, x_spacing, y_spacing)
        else:
            print(f"Unsupported version: {version}")
            return

        # identify mask function
        format = self.read_format(qr_matrix)
        f_mask = function_mask(1)
        mask = format[2:5]
        match mask.tolist():
            case [True, True, True]:
                mask_function = mask_0
                pass
            case _:
                print("unkown mask")
                return

        # print masked values
        for y in range(len(qr_matrix)):
            for x in range(len(qr_matrix.T)):
                if f_mask[x, y] == False and mask_function(x, y):
                    qr_matrix[x, y] = not qr_matrix[x, y]
                c = "yellow" if qr_matrix[x, y] else "purple"
                v.draw_circle(x * x_spacing + 10, y * y_spacing + 10, r=0.5, color=c)

        # walk qrcode
        walker = QRWalker(qr_matrix, f_mask, self.v)
        data_bits = [bit for bit in walker]
        self.parse_data(data_bits, version)

    def parse_data(self, data_bits: list[bool], version: int):
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
            case "Alphanumeric":
                if version <= 9:
                    num_bits_in_len_field = 9
                elif version >= 10 and version <= 26:
                    num_bits_in_len_field = 11
                else:
                    num_bits_in_len_field = 13
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

        bin_string = "".join(
            ["1" if bit else "0" for bit in data_bits[4 : 4 + num_bits_in_len_field]]
        )
        num_characters = int(bin_string, 2)
        print(num_bits_in_len_field)
        print(num_characters)

        word = ""
        byte_size = 8
        for i in range(num_characters + 1):
            offset = i * byte_size
            start = 4 + offset
            end = 4 + num_bits_in_len_field + offset
            byte = "".join("1" if bit else "0" for bit in data_bits[start:end])
            word += chr(int(byte, 2))
            print(byte, chr(int(byte, 2)))
        print(word)

        pass

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

    def read_format(self, qr_matrix: npt.NDArray[np.bool_]) -> npt.NDArray[np.np.bool_]:
        s1: npt.NDArray[np.bool_] = qr_matrix[:6, 8]
        s2: npt.NDArray[np.bool_] = qr_matrix[7:9, 8]
        s3: npt.NDArray[np.bool_] = qr_matrix[14:, 8]
        return np.concatenate((s1, s2, s3))

    def create_QR_matrix(self, gray_image: npt.NDArray[np.float_], top_left_center: Rect, x_spacing: float, y_spacing: float) -> npt.NDArray[np.bool_]:
        top_left_square_y = top_left_center.cy - 3 * y_spacing
        top_left_square_x = top_left_center.cx - 3 * x_spacing
        print(top_left_square_x, x_spacing)

        vertical_lines: list[Line] = []
        for x in np.arange(top_left_square_x, top_left_square_x + x_spacing * 21, x_spacing):
            vertical_lines.append(
                Line(x, top_left_square_y, x, top_left_square_y + y_spacing * 20)
            )

        horizontal_lines: list[Line] = []
        for y in np.arange( top_left_square_y, top_left_square_y + y_spacing * 21, y_spacing):
            horizontal_lines.append(
                Line(top_left_square_x, y, top_left_square_x + x_spacing * 20, y)
            )

        final_matrix: list[list[bool]] = []
        for vline in vertical_lines:
            row = []
            for hline in horizontal_lines:
                xinter, yinter = vline.find_intersect(hline)
                color = Counter(
                    sample_around(gray_image, int(xinter), int(yinter))
                ).most_common(1)[0][0]
                row.append(color)
                #c = "yellow" if color else "purple"
            final_matrix.append(row)
        return np.array(final_matrix, dtype=np.bool_)

    def identify_version(self, x: float, d: float) -> int:
        version = int(((d / x) - 10) / 4)
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
            v.draw_circle(center.cx, center.cy, 1, "yellow")
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

    def find_finder_candidates(self, encodings: list[list[int]], is_vert: bool):
        finder_candidates = []
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
                    v.draw_circle(cx=ci, cy=cj, r=0.5, color="red")
                    rect = Rect(ci, cj, candidate[0], candidate[0])
                    finder_candidates.append(Rect(ci, cj, candidate[0], candidate[0]))
        return finder_candidates

    def find_candidates(self, encodings: list[int]) -> list[Candidate]:
        if len(encodings) <= 5:
            return []
        candidates = []
        start = encodings[1] * -1
        for i in range(len(encodings) - 5):
            length = encodings[i]
            start += length
            if encodings[i + 1] != length:
                continue
            elif encodings[i + 3] != length:
                continue
            elif encodings[i + 4] != length:
                continue
            elif encodings[i + 2] != length * 3:
                continue
            candidates.append(Candidate(length * 7, start))

        return candidates

    def encode_line(self, iterable: Iterable[float]) -> list[int]:
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


def rect_centers_approx_equal(rect1: Rect, rect2: Rect, tolerance: float = 1) -> bool:
    return isclose(rect1.cx, rect2.cx, abs_tol=tolerance) and isclose(
        rect1.cy, rect2.cy, abs_tol=tolerance
    )


def avg_rects(rects: list[Rect]) -> Rect:
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


def get_avg_timing_width(encodings: list[list[int]], X: float, top_left):
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


def sample_around(gray_image: np.array, x: int, y: int) -> list[bool]:
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


def detect_qr_codes():
    pass


if __name__ == "__main__":
    image_path = "bigtest.gif"
    image = iio.v3.imread(image_path)

    dims = len(image[0])
    v = Visualizer("test2.svg", width=dims, height=dims)
    v.add_image(image_path)

    qrcodes = detect_qr_codes()

    decoder = QRDecoder(v)
    decoder.decode_image(image)

    del v
