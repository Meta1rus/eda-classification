"""Загрузка датасета Online Shoppers Purchasing Intention и справочник признаков.

Один и тот же файл data/online_shoppers_intention.csv используется во всех
лабораторных (lab2–lab5), в итоговом проекте и в дашборде (lab6).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "online_shoppers_intention.csv"
# Контрольная сумма файла, с которым выполнялись все тетрадки (см. scripts/download_data.py)
DATA_SHA256 = "b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9"

TARGET = "Revenue"

# Группы признаков по смыслу и по типу шкалы
COUNT_DURATION = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
]
GA_METRICS = ["BounceRates", "ExitRates", "PageValues"]
NUMERIC = COUNT_DURATION + GA_METRICS + ["SpecialDay"]
# Целочисленные коды без порядка (номинальные!) и текстовые категории
NOMINAL = ["Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType"]
BINARY = ["Weekend"]

MONTH_ORDER = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_RU = {
    "Feb": "фев", "Mar": "мар", "May": "май", "June": "июн", "Jul": "июл",
    "Aug": "авг", "Sep": "сен", "Oct": "окт", "Nov": "ноя", "Dec": "дек",
}

DESCRIPTIONS = {
    "Administrative": "Число просмотренных страниц типа «администрирование» (аккаунт, вход и т.п.)",
    "Administrative_Duration": "Суммарное время на страницах «администрирование», с",
    "Informational": "Число просмотренных информационных страниц (о магазине, доставка, контакты)",
    "Informational_Duration": "Суммарное время на информационных страницах, с",
    "ProductRelated": "Число просмотренных страниц, связанных с товарами",
    "ProductRelated_Duration": "Суммарное время на товарных страницах, с",
    "BounceRates": "Средний показатель отказов (Google Analytics) по страницам сессии",
    "ExitRates": "Средний показатель выходов (Google Analytics) по страницам сессии",
    "PageValues": "Средняя «ценность страницы» (Google Analytics) по страницам сессии",
    "SpecialDay": "Близость даты сессии к «особому дню» (14 февраля, 8 марта и т.п.), 0–1",
    "Month": "Месяц сессии",
    "OperatingSystems": "Код операционной системы (номинальный)",
    "Browser": "Код браузера (номинальный)",
    "Region": "Код региона (номинальный)",
    "TrafficType": "Код источника трафика (номинальный)",
    "VisitorType": "Тип посетителя: новый / вернувшийся / другое",
    "Weekend": "Сессия пришлась на выходной день",
    "Revenue": "Целевая переменная: сессия закончилась покупкой",
}

CLASS_LABELS = ["Нет покупки", "Покупка"]
CLASS_COLORS = {"Нет покупки": "#2a78d6", "Покупка": "#eb6834"}  # синий / оранжевый


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(path: Path | str = DATA_PATH, check: bool = True) -> pd.DataFrame:
    """Исходный датасет без изменений (12330 x 18)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Нет файла {path}. Скачайте его: python scripts/download_data.py"
        )
    if check and sha256(path) != DATA_SHA256:
        raise ValueError("Контрольная сумма датасета не совпадает с ожидаемой")
    df = pd.read_csv(path)
    return df


def load(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Датасет с аккуратными типами: категории как category, флаги как int."""
    df = load_raw(path)
    df["Weekend"] = df["Weekend"].astype(int)
    df[TARGET] = df[TARGET].astype(int)
    df["Month"] = pd.Categorical(df["Month"], categories=MONTH_ORDER, ordered=False)
    for c in ["OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType"]:
        df[c] = df[c].astype("category")
    return df
