"""Сборка дашборда: обучение финальной модели, расчёт объяснений, сохранение в YAML + joblib.

Запускается один раз при сборке Docker-образа (RUN в Dockerfile) или вручную:
    python lab6_dashboard/build_dashboard.py
Результат: lab6_dashboard/artifacts/dashboard.yaml и explainer.joblib.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

# explainerdashboard 0.5.8 импортирует pkg_resources, которого нет в setuptools >= 81.
# В requirements зафиксирован setuptools<81; эта заглушка — на случай, если окружение собрано иначе.
try:
    import pkg_resources  # noqa: F401
except ImportError:
    import types
    from packaging.version import parse as _parse_version
    _pr = types.ModuleType("pkg_resources")
    _pr.parse_version = _parse_version
    sys.modules["pkg_resources"] = _pr

from explainerdashboard import ClassifierExplainer, ExplainerDashboard  # noqa: E402
from explainerdashboard.custom import (ImportancesComposite, ClassifierModelStatsComposite,  # noqa: E402
                                       IndividualPredictionsComposite, WhatIfComposite, ShapDependenceComposite)
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.data import load, NOMINAL, DESCRIPTIONS, CLASS_LABELS  # noqa: E402
from src.features import split, encode_for_trees  # noqa: E402
from src.model import make_final_model  # noqa: E402
from eda_tabs import ProjectTab, EdaTab  # noqa: E402

ARTIFACTS = HERE / "artifacts"


def main() -> None:
    t0 = time.time()
    df = load()
    X_train, X_test, y_train, y_test = split(df)
    Xtr, Xte, _ = encode_for_trees(X_train, X_test)
    model = make_final_model().fit(Xtr, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(Xte)[:, 1])
    print(f"Финальная модель обучена: ROC-AUC на test = {auc:.4f} ({time.time() - t0:.0f} с)")

    expected = json.loads((ROOT / "results" / "final_metrics.json").read_text(encoding="utf-8"))
    if abs(expected["test_roc_auc"] - auc) > 1e-3:
        print(f"ВНИМАНИЕ: ROC-AUC отличается от lab5 ({expected['test_roc_auc']:.4f})")
    thr = float(expected["threshold"])

    Xte.index = [f"сессия {i}" for i in Xte.index]
    y_te = y_test.copy(); y_te.index = Xte.index
    descriptions = {c: DESCRIPTIONS[c] for c in DESCRIPTIONS if c in NOMINAL or c in Xte.columns}
    explainer = ClassifierExplainer(
        model, Xte, y_te,
        shap="tree",                      # точный и быстрый TreeSHAP для CatBoost
        cats=NOMINAL,                     # one-hot столбцы Month_*, Browser_* … собираются обратно в признак
        labels=CLASS_LABELS,
        descriptions=descriptions,
        target="Revenue",
        index_name="Сессия",
        precision="float32",
    )
    explainer.calculate_properties(include_interactions=False)
    print(f"Объяснения рассчитаны ({time.time() - t0:.0f} с)")

    tabs = [
        ProjectTab,
        EdaTab,
        ImportancesComposite(explainer, title="Важность признаков"),
        ClassifierModelStatsComposite(explainer, title="Качество модели", cutoff=round(thr, 2)),
        IndividualPredictionsComposite(explainer, title="Отдельные прогнозы"),
        WhatIfComposite(explainer, title="Что если"),
        ShapDependenceComposite(explainer, title="Зависимости SHAP"),
    ]
    db = ExplainerDashboard(
        explainer, tabs=tabs,
        title="Покупки в интернет-магазине: EDA и модель",
        description="Разведочный анализ датасета Online Shoppers Purchasing Intention и интерпретация модели CatBoost",
        header_hide_selector=True, hide_poweredby=True,
    )
    ARTIFACTS.mkdir(exist_ok=True)
    db.to_yaml(ARTIFACTS / "dashboard.yaml", explainerfile="explainer.joblib", dump_explainer=True)
    print(f"Сохранено: {ARTIFACTS / 'dashboard.yaml'}, {ARTIFACTS / 'explainer.joblib'} ({time.time() - t0:.0f} с)")


if __name__ == "__main__":
    main()
