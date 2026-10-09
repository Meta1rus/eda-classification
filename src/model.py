"""Финальная модель проекта (выбрана в lab5_models) и выбор порога.

Один и тот же код используется в lab5, итоговом проекте и дашборде (lab6),
поэтому прогнозы во всех частях работы получаются одинаковыми.
"""
from __future__ import annotations

import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import precision_recall_curve

from .features import RANDOM_STATE

# Гиперпараметры подобраны на 5-кратной кросс-валидации в lab5 (критерий — PR-AUC)
FINAL_PARAMS = {
    "iterations": 500,
    "learning_rate": 0.05,
    "depth": 6,
    "scale_pos_weight": 1.0,  # без перевзвешивания классов
}


def make_final_model() -> CatBoostClassifier:
    return CatBoostClassifier(**FINAL_PARAMS, verbose=0, thread_count=1, random_seed=RANDOM_STATE,
                              allow_writing_files=False)


def choose_threshold(y_true, proba) -> tuple[float, float]:
    """Порог, максимизирующий F1 (по out-of-fold вероятностям на train)."""
    pr, rc, th = precision_recall_curve(y_true, proba)
    f1 = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-12, None)
    i = int(np.nanargmax(f1))
    return float(th[i]), float(f1[i])
