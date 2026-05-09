from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score


@dataclass(frozen=True)
class PixelMetrics:
    auroc: float
    best_f1: float
    best_iou: float
    threshold: float


def compute_pixel_metrics(scores: list[np.ndarray], masks: list[np.ndarray]) -> PixelMetrics:
    if len(scores) != len(masks):
        raise ValueError(f"scores/masks length mismatch: {len(scores)} != {len(masks)}")
    y_score = np.concatenate([s.reshape(-1).astype(np.float32) for s in scores])
    y_true = np.concatenate([m.reshape(-1).astype(np.uint8) for m in masks])

    positives = int(y_true.sum())
    negatives = int(len(y_true) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("Pixel AUROC requires both positive and negative pixels.")

    auroc = float(roc_auc_score(y_true, y_score))
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best_index = int(np.nanargmax(f1))
    threshold = float(thresholds[min(best_index, len(thresholds) - 1)]) if len(thresholds) else 0.0
    best_f1 = float(f1[best_index])

    pred = y_score >= threshold
    intersection = float(np.logical_and(pred, y_true == 1).sum())
    union = float(np.logical_or(pred, y_true == 1).sum())
    best_iou = intersection / max(union, 1.0)
    return PixelMetrics(auroc=auroc, best_f1=best_f1, best_iou=best_iou, threshold=threshold)


def placeholder_pro_auc() -> float | None:
    """PRO can be added here without changing callers."""
    return None
