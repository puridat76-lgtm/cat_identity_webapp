from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from services.decision import DecisionConfig, decide_label
from services.features import FeatureExtractor
from services.gallery import GalleryIndex
from services.quality import QualityChecker


@dataclass
class PipelineConfig:
    gallery_dir: Path
    not_cat_dir: Path
    unknown_cat_dir: Path
    index_dir: Path


class CatIdentityPipeline:
    def __init__(self, cfg: PipelineConfig, gallery_index: GalleryIndex):
        self.cfg = cfg
        self.gallery_index = gallery_index
        self.extractor = FeatureExtractor()
        self.quality_checker = QualityChecker()
        self.decision_cfg = DecisionConfig()

    def predict(self, image: Image.Image) -> dict:
        quality = self.quality_checker.check(image)
        feature_output = self.extractor.extract(image)

        known_matches = self.gallery_index.search(feature_output.vector, split='gallery', top_k=5)
        unknown_matches = self.gallery_index.search(feature_output.vector, split='unknown_cat', top_k=3)
        not_cat_matches = self.gallery_index.search(feature_output.vector, split='not_cat', top_k=3)

        best_known_name = known_matches[0]['label'] if known_matches else None
        best_known_score = float(known_matches[0]['score']) if known_matches else 0.0
        second_known_score = self._second_distinct_score(known_matches, best_known_name)
        best_unknown_score = float(unknown_matches[0]['score']) if unknown_matches else None
        best_not_cat_score = float(not_cat_matches[0]['score']) if not_cat_matches else None

        final_label = decide_label(
            has_gallery=bool(known_matches),
            quality_passed=quality.passed,
            best_known_score=best_known_score,
            second_known_score=second_known_score,
            best_known_name=best_known_name,
            best_not_cat_score=best_not_cat_score,
            best_unknown_score=best_unknown_score,
            cfg=self.decision_cfg,
        )

        return {
            'final_label': final_label,
            'best_known_name': best_known_name,
            'best_known_score': round(best_known_score, 4),
            'second_known_score': round(second_known_score, 4),
            'best_unknown_score': round(best_unknown_score, 4) if best_unknown_score is not None else None,
            'best_not_cat_score': round(best_not_cat_score, 4) if best_not_cat_score is not None else None,
            'quality_pass': quality.passed,
            'quality_reasons': quality.reasons,
            'blur_score': round(quality.blur_score, 2),
            'brightness': round(quality.brightness, 2),
            'top_matches': [
                {'label': row['label'], 'score': round(float(row['score']), 4)}
                for row in known_matches
            ],
        }

    @staticmethod
    def _second_distinct_score(matches: list[dict], first_label: str | None) -> float:
        if not matches:
            return 0.0
        for row in matches[1:]:
            if row['label'] != first_label:
                return float(row['score'])
        return 0.0
