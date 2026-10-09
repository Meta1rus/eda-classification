"""Подготовка признаков (этап Data preparation по CRISP-DM).

Решения обоснованы в lab4_features/lab4_features.ipynb:
  * номинальные признаки (Month, VisitorType и целочисленные КОДЫ OperatingSystems,
    Browser, Region, TrafficType) -> one-hot; категории реже 1 % объединяются в «other»;
  * Weekend -> 0/1;
  * числовые счётчики, длительности и PageValues сильно скошены -> log1p
    (только для моделей, чувствительных к масштабу: kNN, логистическая регрессия, SVM);
  * стандартизация для тех же моделей; деревьям масштабирование не нужно.
Все преобразования обучаются только на train внутри Pipeline — без утечки из test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .data import BINARY, COUNT_DURATION, NOMINAL, NUMERIC, TARGET

RANDOM_STATE = 42
TEST_SIZE = 0.2
RARE_MIN_FREQUENCY = 0.01  # доля, ниже которой категория считается редкой

SKEWED = COUNT_DURATION + ["PageValues"]
NOT_SKEWED = [c for c in NUMERIC if c not in SKEWED]  # BounceRates, ExitRates, SpecialDay


def split(df: pd.DataFrame, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    """Стратифицированное разбиение: доля покупок в train и test одинакова."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def _onehot() -> OneHotEncoder:
    return OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=RARE_MIN_FREQUENCY,
        sparse_output=False,
        dtype=np.int8,
    )


def make_preprocessor(kind: str = "linear", log: bool = True) -> ColumnTransformer:
    """kind='linear' — для kNN / логистической регрессии / SVM; kind='tree' — для деревьев и ансамблей."""
    if kind == "tree":
        num = ("num", "passthrough", NUMERIC)
    elif kind == "linear":
        steps = []
        if log:
            steps.append(("log1p", ColumnTransformer(
                [("skewed", FunctionTransformer(np.log1p, feature_names_out="one-to-one"), SKEWED)],
                remainder="passthrough", verbose_feature_names_out=False)))
        steps.append(("scale", StandardScaler()))
        num = ("num", Pipeline(steps), NUMERIC)
    else:
        raise ValueError(kind)
    ct = ColumnTransformer(
        [num, ("cat", _onehot(), NOMINAL), ("bin", "passthrough", BINARY)],
        verbose_feature_names_out=False,
    )
    return ct.set_output(transform="pandas")


def pretty_names(columns) -> list[str]:
    """'Browser_infrequent_sklearn' -> 'Browser_other'."""
    return [str(c).replace("infrequent_sklearn", "other") for c in columns]


def encode_for_trees(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """One-hot кодирование для древесных моделей и дашборда (исходные числовые значения сохраняются)."""
    pre = make_preprocessor("tree").fit(X_train)
    Xtr = pre.transform(X_train)
    Xte = pre.transform(X_test)
    Xtr.columns = pretty_names(Xtr.columns)
    Xte.columns = pretty_names(Xte.columns)
    return Xtr.astype(float), Xte.astype(float), pre
