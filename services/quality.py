from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass
class QualityResult:
    passed: bool
    reasons: list[str]
    blur_score: float
    brightness: float
    width: int
    height: int


class QualityChecker:
    def __init__(self, min_size: int = 96, min_blur_score: float = 20.0, min_brightness: float = 35.0):
        self.min_size = min_size
        self.min_blur_score = min_blur_score
        self.min_brightness = min_brightness

    def check(self, image: Image.Image) -> QualityResult:
        rgb = np.array(image.convert('RGB'))
        height, width = rgb.shape[:2]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        reasons: list[str] = []

        if width < self.min_size or height < self.min_size:
            reasons.append('image_too_small')
        if blur_score < self.min_blur_score:
            reasons.append('image_too_blurry')
        if brightness < self.min_brightness:
            reasons.append('image_too_dark')

        return QualityResult(
            passed=not reasons,
            reasons=reasons,
            blur_score=blur_score,
            brightness=brightness,
            width=width,
            height=height,
        )
