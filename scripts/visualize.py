import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import interp1d, splev, splrep
from ultralytics import YOLO


CLASS_NAMES = {
    0: "2",
    1: "4",
    2: "6",
    3: "8",
    4: "M",
    5: "7",
    6: "9",
    7: "3",
    8: "1",
    9: "0",
    10: "5",
    11: "Waterline",
}

WATERLINE_CLASS_ID = 11


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".jpg"
    success, encoded = cv2.imencode(suffix, image)
    if not success:
        raise RuntimeError(f"Failed to encode output image: {path}")
    encoded.tofile(str(path))


def as_float(value: np.ndarray | float) -> float:
    return float(np.asarray(value).reshape(-1)[0])


class DraftMeasurementVisualizer:
    def __init__(
        self,
        model_path: str | Path,
        conf: float = 0.25,
        imgsz: int = 1024,
        roi_width: float = 150.0,
    ) -> None:
        self.model = YOLO(str(model_path))
        self.conf = conf
        self.imgsz = imgsz
        self.roi_width = roi_width

    @staticmethod
    def arc_length(y_start: float, y_end: float, poly_coef: np.ndarray) -> float:
        y1, y2 = min(y_start, y_end), max(y_start, y_end)
        ys = np.linspace(y1, y2, num=int(max(2, y2 - y1 + 1)))
        xs = np.polyval(poly_coef, ys)
        points = np.column_stack((xs, ys))
        return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())

    def predict(self, image_path: str | Path) -> tuple[np.ndarray, list[dict], np.ndarray]:
        result = self.model.predict(
            str(image_path),
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False,
        )[0]

        image = result.orig_img.copy()
        characters = []
        waterline_points = []

        for box in result.boxes:
            class_id = int(box.cls[0])
            xyxy = box.xyxy[0].cpu().numpy()
            center = np.array(
                [
                    (xyxy[0] + xyxy[2]) / 2.0,
                    (xyxy[1] + xyxy[3]) / 2.0,
                ],
                dtype=float,
            )

            if class_id == WATERLINE_CLASS_ID:
                waterline_points.append(center)
                continue

            characters.append(
                {
                    "name": CLASS_NAMES[class_id],
                    "bbox": xyxy,
                    "center": center,
                    "w": float(xyxy[2] - xyxy[0]),
                    "h": float(xyxy[3] - xyxy[1]),
                }
            )

        return image, characters, np.asarray(waterline_points, dtype=float)

    @staticmethod
    def select_axis_characters(characters: list[dict]) -> list[dict]:
        meter_marks = [item for item in characters if item["name"] == "M"]
        axis_characters = []

        for item in characters:
            if item["name"] not in {"2", "4", "6", "8", "M"}:
                continue

            is_meter_prefix = False
            for meter_mark in meter_marks:
                if meter_mark is item:
                    continue

                y_diff = abs(item["center"][1] - meter_mark["center"][1])
                x_diff = meter_mark["center"][0] - item["center"][0]

                if y_diff < item["h"] * 0.8 and 0 < x_diff < item["w"] * 3.0:
                    is_meter_prefix = True
                    break

            if not is_meter_prefix:
                axis_characters.append(item)

        return axis_characters

    @staticmethod
    def fit_axis(axis_characters: list[dict]) -> tuple[interp1d, np.ndarray]:
        points = np.array([item["center"] for item in axis_characters], dtype=float)
        points = points[points[:, 1].argsort()]

        unique_y = np.unique(points[:, 1])
        unique_x = np.array([points[points[:, 1] == y_value, 0].mean() for y_value in unique_y])

        interp_kind = "quadratic" if len(unique_y) >= 3 else "linear"
        interp_func = interp1d(unique_y, unique_x, kind=interp_kind, fill_value="extrapolate")

        poly_degree = 2 if len(unique_y) >= 4 else 1
        poly_coef = np.polyfit(unique_y, unique_x, poly_degree)

        return interp_func, poly_coef

    def fit_waterline(
        self,
        waterline_points: np.ndarray,
        axis_x,
    ) -> tuple[float, float, tuple | None, np.ndarray | None, np.ndarray | None]:
        expected_x = axis_x(waterline_points[:, 1])
        roi_mask = np.abs(waterline_points[:, 0] - expected_x) < self.roi_width
        roi_points = waterline_points[roi_mask]

        if len(roi_points) <= 4:
            raise RuntimeError("Not enough waterline points inside the draft-mark ROI.")

        roi_points = roi_points[roi_points[:, 0].argsort()]
        unique_x, indices = np.unique(roi_points[:, 0], return_index=True)
        unique_y = roi_points[indices, 1]

        tck = splrep(unique_x, unique_y, s=len(unique_x) * 2, k=3)
        probe_y = np.linspace(unique_y.min() - 50.0, unique_y.max() + 50.0, 1000)
        probe_x = axis_x(probe_y)
        valid = (probe_x >= unique_x.min()) & (probe_x <= unique_x.max())

        if not valid.any():
            raise RuntimeError("The fitted waterline does not intersect the draft-mark axis.")

        valid_x = probe_x[valid]
        valid_y = probe_y[valid]
        best_idx = int(np.argmin(np.abs(valid_y - splev(valid_x, tck))))

        return float(valid_y[best_idx]), float(valid_x[best_idx]), tck, unique_x, unique_y

    @staticmethod
    def collect_meter_anchors(safe_characters: list[dict]) -> tuple[list[dict], set[int]]:
        meter_anchors = []
        used_as_prefix = set()

        for index, item in enumerate(safe_characters):
            if item["name"] != "M":
                continue

            prefixes = []
            for prefix_index, prefix in enumerate(safe_characters):
                if not prefix["name"].isdigit():
                    continue

                y_diff = abs(prefix["center"][1] - item["center"][1])
                x_diff = item["center"][0] - prefix["center"][0]

                if y_diff < item["h"] * 0.8 and 0 < x_diff < item["w"] * 3.5:
                    prefixes.append((prefix_index, prefix))

            if not prefixes:
                continue

            prefixes.sort(key=lambda pair: pair[1]["center"][0])
            value = float("".join(prefix["name"] for _, prefix in prefixes))

            if 0 < value <= 30:
                meter_anchors.append(
                    {
                        "val": value,
                        "y_c": float(item["center"][1]),
                        "pt": item["center"].copy(),
                    }
                )
                used_as_prefix.update(prefix_index for prefix_index, _ in prefixes)

        meter_anchors.sort(key=lambda anchor: anchor["y_c"])

        for index in range(1, len(meter_anchors)):
            expected_value = meter_anchors[index - 1]["val"] - 1.0
            if abs(meter_anchors[index]["val"] - expected_value) > 0.1:
                meter_anchors[index]["val"] = expected_value

        return meter_anchors, used_as_prefix

    @staticmethod
    def is_meter_row_digit(candidate: dict, meter_anchors: list[dict]) -> bool:
        for anchor in meter_anchors:
            y_diff = abs(candidate["center"][1] - anchor["y_c"])
            x_diff = anchor["pt"][0] - candidate["center"][0]
            if y_diff < candidate["h"] * 0.8 and 0 < x_diff < candidate["w"] * 3.5:
                return True
        return False

    def map_reference_points(
        self,
        safe_characters: list[dict],
        meter_anchors: list[dict],
        used_as_prefix: set[int],
        axis_x,
    ) -> list[dict]:
        mapped_points = [
            {
                "name": "M",
                "y_pix": anchor["y_c"],
                "y_phys": anchor["val"] + 0.05,
                "pt": np.asarray(anchor["pt"], dtype=float),
            }
            for anchor in meter_anchors
        ]

        for index, item in enumerate(safe_characters):
            if index in used_as_prefix or not item["name"].isdigit():
                continue

            if abs(item["center"][0] - as_float(axis_x(item["center"][1]))) > item["w"] * 1.5:
                continue

            ref_anchor = min(meter_anchors, key=lambda anchor: abs(anchor["y_c"] - item["center"][1]))
            meter_base = ref_anchor["val"] - 1.0 if item["center"][1] > ref_anchor["y_c"] else ref_anchor["val"]
            is_meter_row = self.is_meter_row_digit(item, meter_anchors)

            if item["name"] in {"2", "4", "6", "8"} and not is_meter_row:
                physical_value = meter_base + float(item["name"]) / 10.0 + 0.05
            else:
                physical_value = meter_base + 0.05

            mapped_points.append(
                {
                    "name": item["name"],
                    "y_pix": float(item["center"][1]),
                    "y_phys": physical_value,
                    "pt": item["center"].copy(),
                }
            )

        return mapped_points

    def estimate_local_scale(self, mapped_points: list[dict], poly_coef: np.ndarray) -> float:
        decimeter_points = []

        for item in mapped_points:
            if item["name"] not in {"2", "4", "6", "8"}:
                continue
            fractional = round(item["y_phys"] - int(item["y_phys"]), 3)
            if fractional != 0.05:
                decimeter_points.append(item)

        decimeter_points.sort(key=lambda item: item["y_pix"])

        pixel_distances = []
        physical_distances = []

        for index in range(len(decimeter_points) - 1):
            first = decimeter_points[index]
            second = decimeter_points[index + 1]
            arc = self.arc_length(first["y_pix"], second["y_pix"], poly_coef)
            physical_diff = abs(first["y_phys"] - second["y_phys"])

            if arc > 10.0 and physical_diff > 0.1:
                pixel_distances.append(arc)
                physical_distances.append(physical_diff)

        if not pixel_distances:
            return 0.002

        return float(np.median(np.asarray(physical_distances) / np.asarray(pixel_distances)))

    def render(
        self,
        image: np.ndarray,
        axis_x,
        waterline_model,
        waterline_x: np.ndarray | None,
        intersection: tuple[float, float],
        anchor: dict,
        draft_m: float,
    ) -> np.ndarray:
        output = image.copy()
        height = output.shape[0]
        pi_y, pi_x = intersection

        axis_y = np.linspace(0, height, 500)
        axis_points = np.column_stack((axis_x(axis_y), axis_y)).astype(np.int32)
        cv2.polylines(output, [axis_points.reshape(-1, 1, 2)], False, (255, 255, 255), 2)

        if waterline_model is not None and waterline_x is not None:
            plot_x = np.linspace(waterline_x.min(), waterline_x.max(), 200)
            plot_y = splev(plot_x, waterline_model)
            waterline_curve = np.column_stack((plot_x, plot_y)).astype(np.int32)
            cv2.polylines(output, [waterline_curve.reshape(-1, 1, 2)], False, (255, 255, 0), 3)

        cv2.circle(output, (int(pi_x), int(pi_y)), 10, (0, 0, 255), -1)

        segment_y = np.linspace(anchor["y_pix"], pi_y, int(max(2, abs(pi_y - anchor["y_pix"]))))
        segment_points = np.column_stack((axis_x(segment_y), segment_y)).reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(output, [segment_points], False, (255, 0, 255), 4)

        anchor_point = anchor["pt"]
        cv2.circle(output, (int(anchor_point[0]), int(anchor_point[1])), 15, (0, 255, 255), 3)
        cv2.rectangle(output, (20, 20), (580, 150), (0, 0, 0), -1)
        cv2.putText(
            output,
            f"Draft: {draft_m:.3f} m",
            (40, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.2,
            (0, 255, 0),
            3,
        )

        return output

    def process(self, image_path: str | Path) -> tuple[np.ndarray, dict]:
        image, characters, waterline_points = self.predict(image_path)

        if len(waterline_points) < 5 or len(characters) < 2:
            raise RuntimeError("Insufficient draft-mark or waterline detections.")

        axis_characters = self.select_axis_characters(characters)
        if len(axis_characters) < 2:
            raise RuntimeError("Insufficient valid draft-axis characters.")

        axis_interp, poly_coef = self.fit_axis(axis_characters)

        def axis_x(y_value):
            return axis_interp(np.atleast_1d(y_value))

        pi_y, pi_x, waterline_model, waterline_x, _ = self.fit_waterline(waterline_points, axis_x)

        median_height = float(np.median([item["h"] for item in characters]))
        safe_characters = [
            item
            for item in characters
            if not ((item["bbox"][3] >= pi_y - 5.0) or (item["h"] < median_height * 0.75))
        ]

        if not safe_characters:
            raise RuntimeError("No reliable draft-mark characters remain above the waterline.")

        meter_anchors, used_as_prefix = self.collect_meter_anchors(safe_characters)
        if not meter_anchors:
            raise RuntimeError("No valid meter anchor was detected.")

        mapped_points = self.map_reference_points(safe_characters, meter_anchors, used_as_prefix, axis_x)
        local_scale = self.estimate_local_scale(mapped_points, poly_coef)

        points_above_water = [item for item in mapped_points if item["y_pix"] < pi_y]
        if not points_above_water:
            raise RuntimeError("No mapped reference point is located above the waterline.")

        meter_points = [item for item in points_above_water if item["name"] == "M"]
        candidates = meter_points if meter_points else points_above_water
        candidates.sort(key=lambda item: pi_y - item["y_pix"])
        anchor = candidates[0]

        arc = self.arc_length(anchor["y_pix"], pi_y, poly_coef)
        draft_m = anchor["y_phys"] - arc * local_scale

        visualization = self.render(
            image=image,
            axis_x=axis_x,
            waterline_model=waterline_model,
            waterline_x=waterline_x,
            intersection=(pi_y, pi_x),
            anchor=anchor,
            draft_m=draft_m,
        )

        metadata = {
            "image": str(image_path),
            "draft_m": float(draft_m),
            "anchor_value_m": float(anchor["y_phys"]),
            "anchor_point_px": [float(anchor["pt"][0]), float(anchor["pt"][1])],
            "intersection_px": [float(pi_x), float(pi_y)],
            "arc_length_px": float(arc),
            "local_scale_m_per_px": float(local_scale),
            "axis_polynomial": [float(value) for value in poly_coef],
        }

        return visualization, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate and visualize ship draft from one image.")
    parser.add_argument("--model", required=True, type=Path, help="Path to the trained YOLO model weights.")
    parser.add_argument("--image", required=True, type=Path, help="Path to the input image.")
    parser.add_argument("--output", default=Path("draft_visualization.jpg"), type=Path, help="Path to save the visualization.")
    parser.add_argument("--conf", default=0.25, type=float, help="Detection confidence threshold.")
    parser.add_argument("--imgsz", default=1024, type=int, help="Inference image size.")
    parser.add_argument("--roi-width", default=150.0, type=float, help="Horizontal ROI width around the draft-mark axis.")
    parser.add_argument("--show", action="store_true", help="Display the visualization in an OpenCV window.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    visualizer = DraftMeasurementVisualizer(
        model_path=args.model,
        conf=args.conf,
        imgsz=args.imgsz,
        roi_width=args.roi_width,
    )

    visualization, metadata = visualizer.process(args.image)
    save_image(args.output, visualization)

    print(f"Draft: {metadata['draft_m']:.3f} m")
    print(f"Saved visualization to: {args.output}")
    print(f"Anchor point: ({metadata['anchor_point_px'][0]:.1f}, {metadata['anchor_point_px'][1]:.1f})")
    print(f"Waterline intersection: ({metadata['intersection_px'][0]:.1f}, {metadata['intersection_px'][1]:.1f})")
    print(f"Local scale: {metadata['local_scale_m_per_px']:.6f} m/px")

    if args.show:
        window = "Draft measurement visualization"
        cv2.imshow(window, visualization)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
