from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass
class FeatureOutput:
    vector: np.ndarray
    preview_rgb: np.ndarray


def pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    return np.array(image.convert('RGB'))


class FeatureExtractor:
    def __init__(self, image_size: int = 128):
        self.image_size = image_size

    @property
    def signature(self) -> str:
        return f'feature-extractor:v1:image_size={self.image_size}'

    def extract(self, image: Image.Image) -> FeatureOutput:
        rgb = pil_to_rgb_array(image)
        resized = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)

        # Color histograms
        h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
        v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
        l_hist = cv2.calcHist([lab], [0], None, [8], [0, 256]).flatten()
        a_hist = cv2.calcHist([lab], [1], None, [8], [0, 256]).flatten()
        b_hist = cv2.calcHist([lab], [2], None, [8], [0, 256]).flatten()

        rgb_stats = np.concatenate([resized.mean(axis=(0, 1)), resized.std(axis=(0, 1))])
        gray_stats = np.array([gray.mean(), gray.std(), gray.min(), gray.max()], dtype=np.float32)

        # Edges and shape
        edges = cv2.Canny(gray, 80, 160)
        edge_density = np.array([edges.mean() / 255.0], dtype=np.float32)
        moments = cv2.HuMoments(cv2.moments(gray)).flatten()
        moments = np.sign(moments) * np.log1p(np.abs(moments))

        # Compact DCT signature similar to pHash but as dense features
        small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        dct = cv2.dct(small)
        dct_low = dct[:8, :8].flatten()[1:]
        dct_low = dct_low / (np.linalg.norm(dct_low) + 1e-8)

        feature_parts = [
            self._l2_norm(h_hist),
            self._l2_norm(s_hist),
            self._l2_norm(v_hist),
            self._l2_norm(l_hist),
            self._l2_norm(a_hist),
            self._l2_norm(b_hist),
            rgb_stats.astype(np.float32) / 255.0,
            gray_stats.astype(np.float32) / 255.0,
            edge_density,
            moments.astype(np.float32),
            dct_low.astype(np.float32),
        ]
        vector = np.concatenate(feature_parts).astype(np.float32)
        vector = self._l2_norm(vector)
        return FeatureOutput(vector=vector, preview_rgb=resized)

    @staticmethod
    def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b)) + 1e-8
        return float(np.dot(vec_a, vec_b) / denom)

    @staticmethod
    def _l2_norm(vector: np.ndarray) -> np.ndarray:
        vector = vector.astype(np.float32)
        denom = np.linalg.norm(vector) + 1e-8
        return vector / denom
