"""solution.py — the interface the organizers' harness imports.

Part A (detect_events) is owned by the event-detection teammate.
Part B (RiskEstimator) lives in src/risk.py and src/tracking.py.
"""
from __future__ import annotations

import os
import random

import numpy as np

from src.risk import Calibrator, RiskConfig, RiskModel, load_flow_prior
from src.tracking import CausalTracker

random.seed(0)
np.random.seed(0)

CLASSES: list[str] = [
    "accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
    "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
    "solid_line_crossing", "stop_line", "congestion", "road_obstacle", "fire_smoke",
]

RISK_HORIZON_SEC = 5.0
# process ~8 frames per second in Part B; the other frames repeat the last score
RISK_TARGET_FPS = float(os.environ.get("RISK_TARGET_FPS", 8))


def detect_events(video_path: str) -> list[list]:
    """Part A — placeholder until the event-detection module is merged."""
    return []


class RiskEstimator:
    """Part B — causal accident anticipation. Sees frames in order and nothing else."""

    def reset(self, meta: dict) -> None:
        self.meta = meta
        fps = float(meta.get("fps") or 25.0)
        self.stride = max(1, int(round(fps / RISK_TARGET_FPS)))
        cal = Calibrator.load()
        cfg = RiskConfig.from_dict(cal.config if cal else None)
        self.tracker = CausalTracker(fps=fps, stride=self.stride)
        size = (int(meta.get("width") or 0), int(meta.get("height") or 0))
        self.model = RiskModel(cfg, cal, frame_size=size if all(size) else None, flow_prior=load_flow_prior())
        self.idx = 0
        self.last_score = 0.0
        self.log: list[dict] = []   # per processed frame features (for dev tools)

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        i = self.idx
        self.idx += 1
        if i % self.stride:
            return self.last_score
        tracks = self.tracker.update(frame, t_sec)
        self.last_score, f = self.model.step(tracks, t_sec)
        self.log.append(f)
        return self.last_score
