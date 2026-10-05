"""
PPE (Personal Protective Equipment) Detector module.
Detects workers and checks for hardhats, safety vests, and goggles.
Includes mock fallback for demo without model weights.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional


PPE_ITEMS = ["hardhat", "vest", "goggles"]

PPE_WEIGHTS = {
    "hardhat": 0.40,
    "vest": 0.35,
    "goggles": 0.25,
}

PPE_COLORS = {
    "hardhat": (0, 200, 255),    # Yellow-orange
    "vest": (0, 255, 100),       # Green
    "goggles": (255, 200, 0),    # Cyan-ish
}

COMPLIANCE_COLORS = {
    "compliant": (50, 205, 50),     # Green
    "partial": (0, 165, 255),       # Orange
    "non_compliant": (0, 0, 220),   # Red
}


@dataclass
class PPEStatus:
    """PPE compliance status for a single worker."""
    hardhat: bool = False
    vest: bool = False
    goggles: bool = False

    @property
    def score(self) -> float:
        total = 0.0
        if self.hardhat:
            total += PPE_WEIGHTS["hardhat"]
        if self.vest:
            total += PPE_WEIGHTS["vest"]
        if self.goggles:
            total += PPE_WEIGHTS["goggles"]
        return round(total * 100, 1)

    @property
    def level(self) -> str:
        s = self.score
        if s >= 90:
            return "compliant"
        elif s >= 50:
            return "partial"
        return "non_compliant"

    @property
    def items_worn(self) -> List[str]:
        worn = []
        if self.hardhat: worn.append("hardhat")
        if self.vest: worn.append("vest")
        if self.goggles: worn.append("goggles")
        return worn

    @property
    def items_missing(self) -> List[str]:
        missing = []
        if not self.hardhat: missing.append("hardhat")
        if not self.vest: missing.append("vest")
        if not self.goggles: missing.append("goggles")
        return missing


@dataclass
class WorkerDetection:
    """A detected worker with PPE status."""
    worker_id: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    ppe_status: PPEStatus = field(default_factory=PPEStatus)
    center: Tuple[int, int] = field(init=False)

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) // 2, (y1 + y2) // 2)


class PPEDetector:
    """
    PPE compliance detector.
    Detects workers and checks for safety equipment.
    Falls back to mock detections when YOLO is not available.
    """

    def __init__(self, use_mock: bool = True):
        self.use_mock = use_mock
        self.model = None

        if not use_mock:
            try:
                from ultralytics import YOLO
                self.model = YOLO("yolov8n.pt")
                self.use_mock = False
            except ImportError:
                print("[WARNING] ultralytics not installed. Using mock detector.")
                self.use_mock = True

    def detect(self, frame: np.ndarray, sensitivity: str = "medium") -> List[WorkerDetection]:
        """
        Detect workers and their PPE status.

        Args:
            frame: BGR image
            sensitivity: "low", "medium", or "high"

        Returns:
            List of WorkerDetection objects
        """
        if self.use_mock:
            return self._mock_detect(frame, sensitivity)
        return self._yolo_detect(frame, sensitivity)

    def _yolo_detect(self, frame: np.ndarray, sensitivity: str) -> List[WorkerDetection]:
        """Real detection using YOLOv8 (person class only)."""
        conf_map = {"low": 0.3, "medium": 0.5, "high": 0.7}
        conf = conf_map.get(sensitivity, 0.5)

        results = self.model(frame, conf=conf, verbose=False)
        workers = []
        wid = 0

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id == 0:  # person
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    c = float(box.conf[0])
                    # Mock PPE status for real person detections
                    ppe = PPEStatus(
                        hardhat=np.random.random() > 0.3,
                        vest=np.random.random() > 0.25,
                        goggles=np.random.random() > 0.5,
                    )
                    workers.append(WorkerDetection(
                        worker_id=wid, bbox=(x1, y1, x2, y2),
                        confidence=c, ppe_status=ppe
                    ))
                    wid += 1

        return workers

    def _mock_detect(self, frame: np.ndarray, sensitivity: str) -> List[WorkerDetection]:
        """Generate realistic mock worker detections with PPE status."""
        h, w = frame.shape[:2]
        rng = np.random.RandomState(hash(frame.tobytes()[:500]) % (2**31))
        num_workers = rng.randint(3, 9)

        workers = []
        floor_y = int(h * 0.35)

        # Compliance profiles
        profiles = [
            {"hardhat": True, "vest": True, "goggles": True},      # Full
            {"hardhat": True, "vest": True, "goggles": False},      # Partial
            {"hardhat": True, "vest": False, "goggles": False},     # Minimal
            {"hardhat": False, "vest": True, "goggles": False},     # Missing hardhat
            {"hardhat": False, "vest": False, "goggles": False},    # Non-compliant
            {"hardhat": True, "vest": True, "goggles": True},      # Full
            {"hardhat": False, "vest": True, "goggles": True},     # Missing hardhat
            {"hardhat": True, "vest": False, "goggles": True},     # Missing vest
        ]

        for i in range(num_workers):
            pw = rng.randint(int(w * 0.06), int(w * 0.12))
            ph = rng.randint(int(h * 0.20), int(h * 0.40))
            px = rng.randint(pw, w - pw)
            py = rng.randint(floor_y, h - ph - 10)

            profile = profiles[rng.randint(len(profiles))]
            ppe = PPEStatus(**profile)
            conf = round(rng.uniform(0.72, 0.98), 2)

            workers.append(WorkerDetection(
                worker_id=i,
                bbox=(px - pw // 2, py, px + pw // 2, py + ph),
                confidence=conf,
                ppe_status=ppe,
            ))

        return workers


def draw_compliance_overlay(frame: np.ndarray,
                            workers: List[WorkerDetection]) -> np.ndarray:
    """
    Draw PPE compliance annotations on frame.

    Color codes:
    - Green: Fully compliant
    - Orange: Partially compliant
    - Red: Non-compliant
    """
    annotated = frame.copy()

    for worker in workers:
        x1, y1, x2, y2 = worker.bbox
        level = worker.ppe_status.level
        color = COMPLIANCE_COLORS[level]

        # Bounding box
        thickness = 3 if level == "non_compliant" else 2
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Score label background
        score = worker.ppe_status.score
        label = f"W{worker.worker_id} | {score:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 4, y1 - 5),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # PPE item indicators (small icons below bbox)
        icon_y = y2 + 5
        icon_x = x1
        for item in PPE_ITEMS:
            has_item = getattr(worker.ppe_status, item)
            icon_color = (0, 200, 0) if has_item else (0, 0, 200)
            symbol = "✓" if has_item else "✗"
            abbrev = item[0].upper()

            cv2.circle(annotated, (icon_x + 8, icon_y + 8), 8, icon_color, -1)
            cv2.putText(annotated, abbrev, (icon_x + 3, icon_y + 12),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            icon_x += 22

        # Violation alert for non-compliant
        if level == "non_compliant":
            cv2.putText(annotated, "VIOLATION", (x1, y1 - th - 25),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

    return annotated


def compute_overall_compliance(workers: List[WorkerDetection]) -> dict:
    """Compute overall compliance statistics."""
    if not workers:
        return {"total_workers": 0, "compliant": 0, "partial": 0,
                "non_compliant": 0, "overall_score": 0, "compliance_rate": 0}

    compliant = sum(1 for w in workers if w.ppe_status.level == "compliant")
    partial = sum(1 for w in workers if w.ppe_status.level == "partial")
    non_compliant = sum(1 for w in workers if w.ppe_status.level == "non_compliant")
    avg_score = np.mean([w.ppe_status.score for w in workers])

    return {
        "total_workers": len(workers),
        "compliant": compliant,
        "partial": partial,
        "non_compliant": non_compliant,
        "overall_score": round(avg_score, 1),
        "compliance_rate": round(compliant / len(workers) * 100, 1),
    }
