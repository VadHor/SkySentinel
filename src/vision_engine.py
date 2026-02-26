import cv2
import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class VisionConfig:
    min_contour_area: int = 200     
    mask_sky_percentage: float = 0.6 

@dataclass
class FrameAnalysis:
    timestamp: float
    motion_detected: bool
    centroids: List[Tuple[int, int]] = field(default_factory=list)
    frame: Optional[np.ndarray] = None

class VisionEngine:
    def __init__(self, cfg: VisionConfig):
        self.cfg = cfg
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

    def process_frame(self, frame: np.ndarray) -> FrameAnalysis:
        height, width = frame.shape[:2]
        timestamp = time.time()
        
        # Masquage du sol (on ne garde que le ciel)
        sky_limit = int(height * self.cfg.mask_sky_percentage)
        roi = frame[0:sky_limit, 0:width]
        
        blurred = cv2.GaussianBlur(roi, (5, 5), 0)
        fgmask = self.fgbg.apply(blurred)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        centroids = []
        for c in contours:
            if cv2.contourArea(c) > self.cfg.min_contour_area:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centroids.append((cx, cy))
                    cv2.circle(frame, (cx, cy), 10, (0, 0, 255), 2)

        return FrameAnalysis(
            timestamp=timestamp,
            motion_detected=len(centroids) > 0,
            centroids=centroids,
            frame=frame if len(centroids) > 0 else None
        )
