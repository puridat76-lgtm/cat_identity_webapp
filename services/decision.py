from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DecisionConfig:
    accept_threshold: float = 0.93
    margin_threshold: float = 0.015
    not_cat_threshold: float = 0.96
    unknown_ref_threshold: float = 0.955


def decide_label(
    *,
    has_gallery: bool,
    quality_passed: bool,
    best_known_score: float,
    second_known_score: float,
    best_known_name: str | None,
    best_not_cat_score: float | None,
    best_unknown_score: float | None,
    cfg: DecisionConfig,
) -> str:
    if not has_gallery:
        raise ValueError('Gallery is empty. Add dataset images first, then rebuild the index.')

    if not quality_passed:
        return 'low_quality'

    if best_not_cat_score is not None and best_not_cat_score >= cfg.not_cat_threshold and best_not_cat_score > best_known_score:
        return 'not_cat'

    if best_unknown_score is not None and best_unknown_score >= cfg.unknown_ref_threshold and best_unknown_score >= best_known_score:
        return 'unknown'

    if best_known_name is None:
        return 'unknown'

    if best_known_score < cfg.accept_threshold:
        return 'unknown'

    if (best_known_score - second_known_score) < cfg.margin_threshold:
        return 'unknown'

    return best_known_name
