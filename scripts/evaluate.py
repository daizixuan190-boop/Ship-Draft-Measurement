"""Evaluate MD-YOLO draft-reading results on a labeled test split.

The script runs a trained detector, reconstructs physical draft readings, and
prints scenario-wise metrological metrics. It is intended for reproducible
evaluation, not for generating manuscript-specific tables.

Expected test CSV columns:
    image_name, manual_draft_m, scenarios

Example:
    python evaluate.py \
        --model runs/mdyolo/weights/best.pt \
        --test-list data/test.csv \
        --image-dir data/images \
        --predictions-out outputs/predictions.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable

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
DEFAULT_SCENARIO_ORDER = (
    "Calm water",
    "Wave disturbance",
    "Inclined draft marks",
    "Hull curvature",
)


@dataclass(frozen=True)
class TestRecord:
    image_name: str
    manual_draft_m: float
    scenario_raw: str
    scenario_group: str


@dataclass
class PredictionRecord:
    image_name: str
    manual_draft_m: float
    scenario_raw: str
    scenario_group: str
    pred_draft_m: float | None
    status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MD-YOLO draft-reading performance."
    )
    parser.add_argument("--model", type=Path, required=True, help="Path to model weights.")
    parser.add_argument(
        "--test-list",
        type=Path,
        required=True,
        help="CSV file with image_name, manual_draft_m, and scenarios columns.",
    )
    parser.add_argument("--image-dir", type=Path, required=True, help="Image directory.")
    parser.add_argument(
        "--predictions-out",
        type=Path,
        default=None,
        help="Optional CSV path for per-image predictions.",
    )
    parser.add_argument("--imgsz", type=int, default=1024, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence.")
    parser.add_argument(
        "--roi-width",
        type=float,
        default=150.0,
        help="ROI half-width around the fitted draft axis in pixels.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print one line per image during evaluation.",
    )
    return parser.parse_args()


def round_to(value: float | None, digits: int) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def format_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def normalize_scenario(raw_scenario: str) -> str:
    scenario = raw_scenario.strip()
    if scenario in {"Small waves", "Large waves"}:
        return "Wave disturbance"
    return scenario


def parse_manual_draft(value: str) -> float:
    return float(value.strip().lower().replace("m", "").replace(",", "."))


def load_test_records(test_list: Path) -> list[TestRecord]:
    records: list[TestRecord] = []
    with test_list.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"image_name", "manual_draft_m", "scenarios"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in {test_list}: {sorted(missing)}")

        for row in reader:
            image_name = Path(row["image_name"].strip()).name
            manual_draft = parse_manual_draft(row["manual_draft_m"])
            scenario_raw = row["scenarios"].strip()
            records.append(
                TestRecord(
                    image_name=image_name,
                    manual_draft_m=manual_draft,
                    scenario_raw=scenario_raw,
                    scenario_group=normalize_scenario(scenario_raw),
                )
            )
    return records


class DraftReadingEvaluator:
    def __init__(
        self,
        model_path: Path,
        image_size: int = 1024,
        confidence: float = 0.25,
        roi_width: float = 150.0,
    ) -> None:
        self.model = YOLO(str(model_path))
        self.image_size = image_size
        self.confidence = confidence
        self.roi_width = roi_width

    @staticmethod
    def arc_length(y_start: float, y_end: float, polynomial: np.ndarray) -> float:
        y_min, y_max = sorted((float(y_start), float(y_end)))
        sample_count = int(max(2, y_max - y_min + 1))
        ys = np.linspace(y_min, y_max, num=sample_count)
        xs = np.polyval(polynomial, ys)
        points = np.column_stack((xs, ys))
        diffs = np.diff(points, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))

    def predict_draft(self, image_path: Path) -> tuple[float | None, str]:
        results = self.model.predict(
            str(image_path),
            conf=self.confidence,
            imgsz=self.image_size,
            verbose=False,
        )[0]

        draft_marks: list[dict] = []
        waterline_points: list[list[float]] = []

        for box in results.boxes:
            class_id = int(box.cls[0])
            xyxy = box.xyxy[0].cpu().numpy()
            center = [(xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2]

            if class_id == WATERLINE_CLASS_ID:
                waterline_points.append(center)
                continue

            draft_marks.append(
                {
                    "name": CLASS_NAMES[class_id],
                    "bbox": xyxy,
                    "center": center,
                    "w": xyxy[2] - xyxy[0],
                    "h": xyxy[3] - xyxy[1],
                }
            )

        if len(waterline_points) < 5 or len(draft_marks) < 2:
            return None, "insufficient_features"

        axis_marks = self._select_axis_marks(draft_marks)
        if len(axis_marks) < 2:
            return None, "insufficient_axis_marks"

        axis_interp, axis_poly = self._fit_draft_axis(axis_marks)
        intersection_y = self._estimate_waterline_intersection(
            np.asarray(waterline_points), axis_interp
        )
        if intersection_y is None:
            return None, "waterline_intersection_failed"

        safe_marks = self._select_marks_above_water(draft_marks, intersection_y)
        if not safe_marks:
            return None, "no_safe_marks"

        mapped_marks = self._assign_physical_values(safe_marks, axis_interp)
        if not mapped_marks:
            return None, "no_physical_anchor"

        scale = self._estimate_local_scale(mapped_marks, axis_poly)
        anchors_above_water = [mark for mark in mapped_marks if mark["y_pix"] < intersection_y]
        if not anchors_above_water:
            return None, "no_anchor_above_water"

        meter_anchors = [mark for mark in anchors_above_water if mark["name"] == "M"]
        if meter_anchors:
            final_anchor = min(meter_anchors, key=lambda mark: intersection_y - mark["y_pix"])
        else:
            final_anchor = min(anchors_above_water, key=lambda mark: intersection_y - mark["y_pix"])

        distance_pix = self.arc_length(final_anchor["y_pix"], intersection_y, axis_poly)
        draft_value = final_anchor["y_phys"] - distance_pix * scale
        return round_to(draft_value, 3), "ok"

    @staticmethod
    def _select_axis_marks(draft_marks: list[dict]) -> list[dict]:
        meter_marks = [mark for mark in draft_marks if mark["name"] == "M"]
        axis_marks = []

        for mark in draft_marks:
            if mark["name"] not in {"2", "4", "6", "8", "M"}:
                continue

            is_meter_prefix = False
            for meter_mark in meter_marks:
                if meter_mark is mark:
                    continue
                y_diff = abs(mark["center"][1] - meter_mark["center"][1])
                x_diff = meter_mark["center"][0] - mark["center"][0]
                if y_diff < mark["h"] * 0.8 and 0 < x_diff < mark["w"] * 3.0:
                    is_meter_prefix = True
                    break

            if not is_meter_prefix:
                axis_marks.append(mark)

        return axis_marks

    @staticmethod
    def _fit_draft_axis(axis_marks: list[dict]):
        points = np.array([mark["center"] for mark in axis_marks])
        points = points[points[:, 1].argsort()]

        unique_y = np.unique(points[:, 1])
        unique_x = np.array([points[points[:, 1] == y, 0].mean() for y in unique_y])

        interpolation_kind = "quadratic" if len(unique_y) >= 3 else "linear"
        axis_interp = interp1d(
            unique_y,
            unique_x,
            kind=interpolation_kind,
            fill_value="extrapolate",
        )

        polynomial_degree = 2 if len(unique_y) >= 4 else 1
        axis_poly = np.polyfit(unique_y, unique_x, polynomial_degree)
        return axis_interp, axis_poly

    def _estimate_waterline_intersection(self, waterline_points: np.ndarray, axis_interp) -> float | None:
        expected_x = axis_interp(waterline_points[:, 1])
        in_roi = np.abs(waterline_points[:, 0] - expected_x) < self.roi_width
        roi_points = waterline_points[in_roi]

        if len(roi_points) <= 4:
            return None

        roi_points = roi_points[roi_points[:, 0].argsort()]
        unique_x, unique_indices = np.unique(roi_points[:, 0], return_index=True)
        unique_y = roi_points[unique_indices, 1]

        try:
            spline = splrep(unique_x, unique_y, s=len(unique_x) * 2, k=3)
            candidate_y = np.linspace(unique_y.min() - 50, unique_y.max() + 50, 1000)
            axis_x = axis_interp(candidate_y)
            valid = (axis_x >= unique_x.min()) & (axis_x <= unique_x.max())
            if not valid.any():
                return None
            residual = np.abs(candidate_y[valid] - splev(axis_x[valid], spline))
            return float(candidate_y[valid][np.argmin(residual)])
        except Exception:
            return None

    @staticmethod
    def _select_marks_above_water(draft_marks: list[dict], intersection_y: float) -> list[dict]:
        median_height = np.median([mark["h"] for mark in draft_marks])
        safe_marks = []
        for mark in draft_marks:
            is_submerged = mark["bbox"][3] >= intersection_y - 5
            is_unreliable = mark["h"] < median_height * 0.75
            if not is_submerged and not is_unreliable:
                safe_marks.append(mark)
        return safe_marks

    @staticmethod
    def _assign_physical_values(safe_marks: list[dict], axis_interp) -> list[dict]:
        meter_anchors = []
        used_prefix_indices = set()

        for mark_index, mark in enumerate(safe_marks):
            if mark["name"] != "M":
                continue

            prefixes = []
            for prefix_index, prefix in enumerate(safe_marks):
                if not prefix["name"].isdigit():
                    continue
                y_diff = abs(prefix["center"][1] - mark["center"][1])
                x_diff = mark["center"][0] - prefix["center"][0]
                if y_diff < mark["h"] * 0.8 and 0 < x_diff < mark["w"] * 3.5:
                    prefixes.append((prefix_index, prefix))

            if not prefixes:
                continue

            prefixes.sort(key=lambda item: item[1]["center"][0])
            meter_value = float("".join(prefix["name"] for _, prefix in prefixes))
            if 0 < meter_value <= 30:
                meter_anchors.append(
                    {"val": meter_value, "y_c": mark["center"][1], "pt": mark["center"]}
                )
                for prefix_index, _ in prefixes:
                    used_prefix_indices.add(prefix_index)

        if not meter_anchors:
            return []

        meter_anchors.sort(key=lambda item: item["y_c"])
        for index in range(1, len(meter_anchors)):
            expected_value = meter_anchors[index - 1]["val"] - 1.0
            if abs(meter_anchors[index]["val"] - expected_value) > 0.1:
                meter_anchors[index]["val"] = expected_value

        mapped_marks = [
            {
                "name": "M",
                "y_pix": anchor["y_c"],
                "y_phys": anchor["val"] + 0.05,
                "pt": np.array(anchor["pt"]),
            }
            for anchor in meter_anchors
        ]

        for mark_index, mark in enumerate(safe_marks):
            if mark_index in used_prefix_indices or mark["name"] not in {"2", "4", "6", "8"}:
                continue
            if abs(mark["center"][0] - axis_interp(mark["center"][1])) > mark["w"] * 1.5:
                continue

            nearest_meter = min(meter_anchors, key=lambda item: abs(item["y_c"] - mark["center"][1]))
            meter_base = nearest_meter["val"] - 1.0 if mark["center"][1] > nearest_meter["y_c"] else nearest_meter["val"]
            physical_value = meter_base + float(mark["name"]) / 10.0 + 0.05

            mapped_marks.append(
                {
                    "name": mark["name"],
                    "y_pix": mark["center"][1],
                    "y_phys": physical_value,
                    "pt": np.array(mark["center"]),
                }
            )

        return mapped_marks

    def _estimate_local_scale(self, mapped_marks: list[dict], axis_poly: np.ndarray) -> float:
        decimeter_marks = [mark for mark in mapped_marks if mark["name"] in {"2", "4", "6", "8"}]
        decimeter_marks.sort(key=lambda mark: mark["y_pix"])

        pixel_distances = []
        physical_distances = []
        for first, second in zip(decimeter_marks[:-1], decimeter_marks[1:]):
            pixel_distance = self.arc_length(first["y_pix"], second["y_pix"], axis_poly)
            physical_distance = abs(first["y_phys"] - second["y_phys"])
            if pixel_distance > 10 and physical_distance > 0.1:
                pixel_distances.append(pixel_distance)
                physical_distances.append(physical_distance)

        if not pixel_distances:
            return 0.002

        return float(np.median(np.asarray(physical_distances) / np.asarray(pixel_distances)))


def exact_match_rate(predictions: Iterable[float], references: Iterable[float]) -> float:
    pairs = list(zip(predictions, references))
    if not pairs:
        return 0.0
    matches = sum(1 for pred, ref in pairs if round(pred, 2) == round(ref, 2))
    return 100.0 * matches / len(pairs)


def tolerance_rate(errors: Iterable[float], tolerance: float) -> float:
    errors = list(errors)
    if not errors:
        return 0.0
    return 100.0 * sum(abs(error) <= tolerance for error in errors) / len(errors)


def format_mean_sd(values: list[float]) -> str:
    if not values:
        return "-"
    avg = mean(values)
    sd = stdev(values) if len(values) > 1 else 0.0
    return f"{avg:.3f} ± {sd:.3f}"


def summarize(records: list[PredictionRecord], scenario: str) -> dict[str, str | int | float]:
    if not records:
        return {}

    successful = [record for record in records if record.pred_draft_m is not None]
    references = [record.manual_draft_m for record in records]

    if not successful:
        return {
            "Scenario": scenario,
            "Images": len(records),
            "Successful": 0,
            "Draft range (m)": f"{min(references):.2f}-{max(references):.2f}",
            "Bias (mean ± SD, m)": "-",
            "Exact@0.01 m (%)": 0.0,
            "Within ±0.01 m (%)": 0.0,
            "Within ±0.02 m (%)": 0.0,
        }

    successful_refs = [record.manual_draft_m for record in successful]
    predictions = [float(record.pred_draft_m) for record in successful]
    errors = [ref - pred for ref, pred in zip(successful_refs, predictions)]

    return {
        "Scenario": scenario,
        "Images": len(records),
        "Successful": len(successful),
        "Draft range (m)": f"{min(references):.2f}-{max(references):.2f}",
        "Bias (mean ± SD, m)": format_mean_sd(errors),
        "Exact@0.01 m (%)": round(exact_match_rate(predictions, successful_refs), 1),
        "Within ±0.01 m (%)": round(tolerance_rate(errors, 0.01), 1),
        "Within ±0.02 m (%)": round(tolerance_rate(errors, 0.02), 1),
    }


def print_summary(summary_rows: list[dict[str, str | int | float]]) -> None:
    headers = [
        "Scenario",
        "Images",
        "Successful",
        "Draft range (m)",
        "Bias (mean ± SD, m)",
        "Exact@0.01 m (%)",
        "Within ±0.01 m (%)",
        "Within ±0.02 m (%)",
    ]

    print("\nEvaluation summary")
    print(",".join(headers))
    for row in summary_rows:
        print(",".join(str(row.get(header, "")) for header in headers))

    print("\nMarkdown summary")
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in summary_rows:
        print("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")


def save_predictions(records: list[PredictionRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_name",
        "manual_draft_m",
        "scenario_raw",
        "scenario_group",
        "pred_draft_m",
        "status",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "image_name": record.image_name,
                    "manual_draft_m": record.manual_draft_m,
                    "scenario_raw": record.scenario_raw,
                    "scenario_group": record.scenario_group,
                    "pred_draft_m": record.pred_draft_m,
                    "status": record.status,
                }
            )


def evaluate(args: argparse.Namespace) -> list[PredictionRecord]:
    records = load_test_records(args.test_list)
    evaluator = DraftReadingEvaluator(
        model_path=args.model,
        image_size=args.imgsz,
        confidence=args.conf,
        roi_width=args.roi_width,
    )

    predictions: list[PredictionRecord] = []
    for record in records:
        image_path = args.image_dir / record.image_name
        if not image_path.exists():
            prediction = PredictionRecord(
                image_name=record.image_name,
                manual_draft_m=record.manual_draft_m,
                scenario_raw=record.scenario_raw,
                scenario_group=record.scenario_group,
                pred_draft_m=None,
                status="image_not_found",
            )
        else:
            pred_draft, status = evaluator.predict_draft(image_path)
            prediction = PredictionRecord(
                image_name=record.image_name,
                manual_draft_m=record.manual_draft_m,
                scenario_raw=record.scenario_raw,
                scenario_group=record.scenario_group,
                pred_draft_m=pred_draft,
                status=status,
            )

        predictions.append(prediction)
        if args.verbose:
            print(
                record.image_name,
                record.scenario_group,
                format_float(record.manual_draft_m, 3),
                prediction.pred_draft_m,
                prediction.status,
            )

    return predictions


def main() -> None:
    args = parse_args()
    predictions = evaluate(args)

    summary_rows = []
    for scenario in DEFAULT_SCENARIO_ORDER:
        scenario_records = [
            record for record in predictions if record.scenario_group == scenario
        ]
        if scenario_records:
            summary_rows.append(summarize(scenario_records, scenario))

    summary_rows.append(summarize(predictions, "Overall"))
    print_summary(summary_rows)

    if args.predictions_out is not None:
        save_predictions(predictions, args.predictions_out)
        print(f"\nSaved per-image predictions to: {args.predictions_out}")


if __name__ == "__main__":
    main()
