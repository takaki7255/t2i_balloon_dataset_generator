"""Python implementation of the manga panel detector inspired by frame_separation.cpp.

This module replicates the major processing stages of the original C++ pipeline:

1. Pre-process the input page (grayscale conversion, smoothing).
2. Detect and suppress speech balloons that may occlude panel borders.
3. Extract strong line evidence through binarisation, Canny edge detection, and
   Hough transform.
4. Complement panel candidates with bounding boxes and filter by area.
5. Refine panel corners and align them to the page edges.
6. Output cropped RGBA panel images with transparent backgrounds.

The script can be used as a library (see ``FrameDetector``) or as a command line tool::

    python frame_separation.py --input path/to/page.png --output panels/

Requirements: ``opencv-python`` and ``numpy``.
"""
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

Point = Tuple[int, int]


@dataclass
class Line:
    """Support class mirroring the C++ ``Line`` structure.

    When ``y2x`` is True the line is defined as ``x = a*y + b`` (vertical-ish edges),
    otherwise ``y = a*x + b``.  The ``judge_area`` method allows checking on which
    side of the line a point lies; it mimics the behaviour of ``Line::judgeArea``
    in the original code.
    """

    p1: Point
    p2: Point
    y2x: bool
    a: float = 0.0
    b: float = 0.0

    def __post_init__(self) -> None:
        self._calc()

    def _calc(self) -> None:
        if self.y2x:
            dy = self.p2[1] - self.p1[1]
            if dy == 0:
                self.a = 0.0
                self.b = float(min(self.p1[0], self.p2[0]))
            else:
                self.a = (self.p2[0] - self.p1[0]) / dy
                self.b = self.p1[0] - self.p1[1] * self.a
        else:
            dx = self.p2[0] - self.p1[0]
            if dx == 0:
                self.a = 0.0
                self.b = float(min(self.p1[1], self.p2[1]))
            else:
                self.a = (self.p2[1] - self.p1[1]) / dx
                self.b = self.p1[1] - self.p1[0] * self.a

    def judge_area(self, point: Point) -> int:
        if self.y2x:
            x, y = point[1], point[0]
        else:
            x, y = point
        value = self.a * x + self.b
        if y > value:
            return 0
        if y < value:
            return 1
        return 2


@dataclass
class PanelQuad:
    """Stores the four panel corner points and derived line equations."""

    lt: Point
    lb: Point
    rt: Point
    rb: Point
    top_line: Line | None = None
    bottom_line: Line | None = None
    left_line: Line | None = None
    right_line: Line | None = None

    def renew_lines(self) -> None:
        self.top_line = Line(self.lt, self.rt, False)
        self.bottom_line = Line(self.lb, self.rb, False)
        self.left_line = Line(self.lb, self.lt, True)
        self.right_line = Line(self.rb, self.rt, True)

    def outside(self, p: Point) -> bool:
        assert self.top_line and self.bottom_line and self.left_line and self.right_line
        return (
            self.top_line.judge_area(p) == 1
            or self.right_line.judge_area(p) == 0
            or self.bottom_line.judge_area(p) == 0
            or self.left_line.judge_area(p) == 1
        )


class FrameDetector:
    """Python port of ``Framedetect`` from the C++ implementation."""

    def __init__(self) -> None:
        self.page_corners = PanelQuad((0, 0), (0, 0), (0, 0), (0, 0))
        self.page_corners.renew_lines()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def frame_detect(self, src_page: np.ndarray) -> List[np.ndarray]:
        """Detect and extract panel images from ``src_page``.

        Returns a list of RGBA images cropped to each panel.
        """

        if src_page.ndim == 2:
            gray_img = src_page.copy()
            color_page = cv2.cvtColor(src_page, cv2.COLOR_GRAY2BGR)
        else:
            color_page = src_page.copy()
            gray_img = cv2.cvtColor(src_page, cv2.COLOR_BGR2GRAY)

        img_size = (gray_img.shape[1], gray_img.shape[0])
        panel_images: List[np.ndarray] = []

        # --- Speech balloon suppression -------------------------------------------------
        bin_balloon = cv2.threshold(gray_img, 230, 255, cv2.THRESH_BINARY)[1]
        bin_balloon = cv2.erode(bin_balloon, None, iterations=1)
        bin_balloon = cv2.dilate(bin_balloon, None, iterations=1)
        contours, hierarchy = cv2.findContours(bin_balloon, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        gaussian_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
        self.extract_speech_balloon(contours, hierarchy, gaussian_img)

        # --- Panel candidate extraction --------------------------------------------------
        inverse_bin = cv2.threshold(gaussian_img, 210, 255, cv2.THRESH_BINARY_INV)[1]
        self.find_frame_existence_area(inverse_bin)

        canny_img = cv2.Canny(gray_img, 120, 130, apertureSize=3)
        lines_broad = self._detect_hough_lines(canny_img, rho=1, theta=np.pi / 180.0, threshold=50)
        lines_fine = self._detect_hough_lines(canny_img, rho=1, theta=np.pi / 360.0, threshold=50)
        lines_img = np.zeros_like(inverse_bin)
        self.draw_hough_lines(lines_broad, lines_img)
        self.draw_hough_lines(lines_fine, lines_img)

        and_img = cv2.bitwise_and(inverse_bin, lines_img)
        tmp_img = and_img.copy()
        contours_and, _ = cv2.findContours(tmp_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        complement = self.create_and_img_with_bounding_box(and_img.copy(), contours_and, inverse_bin)
        contours_rect, _ = cv2.findContours(complement, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bounding_boxes: List[cv2.Rect] = []
        for cnt in contours_rect:
            rect = cv2.boundingRect(cnt)
            if self.judge_area_of_bounding_box(rect, complement.shape[0] * complement.shape[1]):
                bounding_boxes.append(rect)

        # --- Final panel extraction ------------------------------------------------------
        for cnt in contours_rect:
            brect = cv2.boundingRect(cnt)
            if not self.judge_area_of_bounding_box(brect, src_page.shape[0] * src_page.shape[1]):
                continue

            is_overlap = self.judge_bounding_box_overlap(bounding_boxes, brect)
            if not is_overlap:
                continue

            approx = cv2.approxPolyDP(cnt, 6, True)
            quad = self._define_panel_quad(approx, brect, inverse_bin)
            if quad is None:
                continue

            alpha_image = self.create_alpha_image(src_page, quad)
            x, y, w, h = brect
            panel = alpha_image[y : y + h, x : x + w].copy()
            panel_images.append(panel)

            # Optional: draw detected panel on a copy (for debugging)
            cv2.polylines(
                color_page,
                [np.array([quad.lt, quad.rt, quad.rb, quad.lb], dtype=np.int32)],
                True,
                (0, 0, 255),
                thickness=2,
                lineType=cv2.LINE_AA,
            )

        return panel_images

    # ------------------------------------------------------------------
    # Helper functions replicating C++ logic
    # ------------------------------------------------------------------
    def extract_speech_balloon(
        self,
        contours: Sequence[np.ndarray],
        hierarchy: Iterable[Tuple[int, int, int, int]] | None,
        gaussian_img: np.ndarray,
    ) -> None:
        img_area = gaussian_img.shape[0] * gaussian_img.shape[1]
        hierarchy = list(hierarchy) if hierarchy is not None else [(-1, -1, -1, -1)] * len(contours)
        for contour, hier in zip(contours, hierarchy):
            area = cv2.contourArea(contour)
            length = cv2.arcLength(contour, closed=True)
            if length == 0:
                continue
            if img_area * 0.008 <= area < img_area * 0.03:
                circularity = 4.0 * math.pi * area / (length * length)
                if circularity > 0.4:
                    cv2.drawContours(gaussian_img, [contour], 0, color=0, thickness=-1, lineType=cv2.LINE_AA)

    def find_frame_existence_area(self, inverse_bin: np.ndarray) -> None:
        histogram = np.sum(inverse_bin[:, 3:-3] > 0, axis=0)
        min_x = 0
        max_x = inverse_bin.shape[1] - 1
        for idx, val in enumerate(histogram, start=3):
            if val > 0:
                min_x = idx
                break
        for offset, val in enumerate(reversed(histogram), start=3):
            if val > 0:
                max_x = inverse_bin.shape[1] - offset
                break

        if min_x < 6:
            min_x = 0
        if max_x > inverse_bin.shape[1] - 6:
            max_x = inverse_bin.shape[1]

        self.page_corners = PanelQuad(
            (min_x, 0),
            (min_x, inverse_bin.shape[0]),
            (max_x, 0),
            (max_x, inverse_bin.shape[0]),
        )
        self.page_corners.renew_lines()

    def draw_hough_lines(self, lines: Sequence[Tuple[float, float]], canvas: np.ndarray) -> None:
        h, w = canvas.shape[:2]
        diag = int(math.hypot(w, h))
        for rho, theta in lines[:100]:
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 - diag * b), int(y0 + diag * a))
            pt2 = (int(x0 + diag * b), int(y0 - diag * a))
            cv2.line(canvas, pt1, pt2, 255, 1, lineType=cv2.LINE_AA)

    def create_and_img_with_bounding_box(
        self,
        src_img: np.ndarray,
        contours: Sequence[np.ndarray],
        inverse_bin: np.ndarray,
    ) -> np.ndarray:
        page_area = src_img.shape[0] * src_img.shape[1]
        for cnt in contours:
            rect = cv2.boundingRect(cnt)
            if not self.judge_area_of_bounding_box(rect, page_area):
                continue
            cv2.rectangle(src_img, rect, 255, thickness=3)
        return cv2.bitwise_and(src_img, inverse_bin)

    @staticmethod
    def judge_area_of_bounding_box(rect: Tuple[int, int, int, int], page_area: int) -> bool:
        x, y, w, h = rect
        return (w * h) >= page_area * 0.048

    @staticmethod
    def judge_bounding_box_overlap(
        bounding_boxes: Sequence[Tuple[int, int, int, int]],
        brect: Tuple[int, int, int, int],
    ) -> bool:
        x, y, w, h = brect
        brect_box = (x, y, x + w, y + h)
        for bx, by, bw, bh in bounding_boxes:
            other = (bx, by, bx + bw, by + bh)
            if other == brect_box:
                continue
            # Check for containment (brect fully inside other)
            if (
                brect_box[0] >= other[0]
                and brect_box[1] >= other[1]
                and brect_box[2] <= other[2]
                and brect_box[3] <= other[3]
            ):
                return False
        return True

    def _define_panel_quad(
        self,
        approx: np.ndarray,
        brect: Tuple[int, int, int, int],
        inverse_bin: np.ndarray,
    ) -> PanelQuad | None:
        x, y, w, h = brect
        xmin, ymin = x, y
        xmax, ymax = x + w, y + h
        if xmin < 6:
            xmin = 0
        if ymin < 6:
            ymin = 0
        if xmax > inverse_bin.shape[1] - 6:
            xmax = inverse_bin.shape[1]
        if ymax > inverse_bin.shape[0] - 6:
            ymax = inverse_bin.shape[0]

        bb_points = PanelQuad((xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax))
        bb_points.renew_lines()

        quad = PanelQuad(bb_points.lt, bb_points.lb, bb_points.rt, bb_points.rb)
        flags = {"lt": False, "lb": False, "rt": False, "rb": False}
        min_dist = {
            "lt": float(inverse_bin.shape[0]),
            "lb": float(inverse_bin.shape[0]),
            "rt": float(inverse_bin.shape[0]),
            "rb": float(inverse_bin.shape[0]),
        }

        points = approx.reshape(-1, 2)
        for px, py in points:
            self.define_panel_corner("lt", flags, quad, min_dist, (px, py), bb_points, self.page_corners.lt)
            self.define_panel_corner("lb", flags, quad, min_dist, (px, py), bb_points, self.page_corners.lb)
            self.define_panel_corner("rt", flags, quad, min_dist, (px, py), bb_points, self.page_corners.rt)
            self.define_panel_corner("rb", flags, quad, min_dist, (px, py), bb_points, self.page_corners.rb)

        self.align_to_edge(quad, inverse_bin.shape[1], inverse_bin.shape[0])
        quad.renew_lines()
        return quad

    def define_panel_corner(
        self,
        key: str,
        flags: dict,
        quad: PanelQuad,
        min_dist: dict,
        point: Point,
        bb_points: PanelQuad,
        page_corner: Point,
    ) -> None:
        if flags[key]:
            return
        bb_point = getattr(bb_points, key)
        if np.linalg.norm(np.array(bb_point) - np.array(page_corner)) < 8:
            setattr(quad, key, page_corner)
            flags[key] = True
        else:
            dist = np.linalg.norm(np.array(bb_point) - np.array(point))
            if dist < min_dist[key]:
                min_dist[key] = dist
                setattr(quad, key, point)

    def align_to_edge(self, quad: PanelQuad, width: int, height: int) -> None:
        threshold = 6
        if quad.lt[0] < threshold:
            quad.lt = (0, quad.lt[1])
        if quad.lt[1] < threshold:
            quad.lt = (quad.lt[0], 0)
        if quad.rt[0] > width - threshold:
            quad.rt = (width, quad.rt[1])
        if quad.rt[1] < threshold:
            quad.rt = (quad.rt[0], 0)
        if quad.lb[0] < threshold:
            quad.lb = (0, quad.lb[1])
        if quad.lb[1] > height - threshold:
            quad.lb = (quad.lb[0], height)
        if quad.rb[0] > width - threshold:
            quad.rb = (width, quad.rb[1])
        if quad.rb[1] > height - threshold:
            quad.rb = (quad.rb[0], height)

    def create_alpha_image(self, src_page: np.ndarray, quad: PanelQuad) -> np.ndarray:
        if src_page.ndim == 2:
            rgba = cv2.cvtColor(src_page, cv2.COLOR_GRAY2BGRA)
        else:
            rgba = cv2.cvtColor(src_page, cv2.COLOR_BGR2BGRA)

        mask = np.zeros(rgba.shape[:2], dtype=np.uint8)
        polygon = np.array([quad.lt, quad.rt, quad.rb, quad.lb], dtype=np.int32)
        cv2.fillPoly(mask, [polygon], 255)
        rgba[:, :, 3] = mask
        return rgba

    @staticmethod
    def _detect_hough_lines(
        image: np.ndarray,
        rho: float,
        theta: float,
        threshold: int,
    ) -> List[Tuple[float, float]]:
        lines = cv2.HoughLines(image, rho, theta, threshold)
        if lines is None:
            return []
        return [tuple(line[0]) for line in lines]


# ----------------------------------------------------------------------
# Command line interface
# ----------------------------------------------------------------------

def _load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _save_panels(panels: Sequence[np.ndarray], base_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = base_path.stem
    for idx, panel in enumerate(panels, start=1):
        out_path = output_dir / f"{stem}_panel_{idx:02d}.png"
        cv2.imwrite(str(out_path), panel)


def run_cli(args: argparse.Namespace) -> None:
    detector = FrameDetector()
    input_path = Path(args.input)
    output_dir = Path(args.output)

    image_paths: List[Path]
    if input_path.is_dir():
        image_paths = sorted(
            [p for p in input_path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".bmp"}]
        )
    else:
        image_paths = [input_path]

    if not image_paths:
        raise FileNotFoundError(f"No input images found at {input_path}")

    for image_path in image_paths:
        image = _load_image(image_path)
        panels = detector.frame_detect(image)
        if not panels:
            print(f"[WARN] No panels detected for {image_path}")
            continue
        _save_panels(panels, image_path, output_dir)
        print(f"[INFO] Saved {len(panels)} panels for {image_path}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect manga panels and export them as RGBA crops.")
    parser.add_argument("--input", required=True, help="Input image file or directory of images.")
    parser.add_argument("--output", required=True, help="Directory to store extracted panel PNG files.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_cli(parse_args())
