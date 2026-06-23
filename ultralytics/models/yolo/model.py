from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import ROOT


class YOLO(Model):
    """Detect-only MD-YOLO wrapper."""

    def __init__(
        self,
        model: str | Path = ROOT / "cfg/models/mdyolo/mdyolo.yaml",
        task: str | None = "detect",
        verbose: bool = False,
    ):
        super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        return {
            "detect": {
                "model": DetectionModel,
                "trainer": yolo.detect.DetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            }
        }
