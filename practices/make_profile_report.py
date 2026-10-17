"""Отчёт ydata-profiling (бывший pandas-profiling) по diamonds_good.csv.

Зачем отдельный скрипт. ydata-profiling 4.18 ставится под Python 3.13, но требует
pandas < 3, matplotlib ≤ 3.10 и scipy < 1.17, а весь репозиторий работает на pandas 3.0.
Поэтому отчёт строится в ОТДЕЛЬНОМ виртуальном окружении, а в тетрадке
practice_eda_basic.ipynb лежит готовый html и ручной обзор данных.

Запуск из корня репозитория:
    python -m venv venv_profiling
    source venv_profiling/bin/activate            # Windows: venv_profiling\\Scripts\\activate
    pip install -r practices/requirements-profiling.txt
    python practices/make_profile_report.py
Результат: practices/Diamonds_report.html
"""
from pathlib import Path

import pandas as pd
from ydata_profiling import ProfileReport

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "diamonds_good.csv"
OUT = HERE / "Diamonds_report.html"

if __name__ == "__main__":
    # как в тетрадке преподавателя: отчёт по «сырому» файлу, до удаления дубликатов и заполнения пропусков
    df_copy = pd.read_csv(DATA)
    profile_report = ProfileReport(df_copy, title="Pandas Profiling Report")
    profile_report.to_file(OUT)
    print(f"Отчёт сохранён: {OUT} ({OUT.stat().st_size / 1e6:.1f} МБ)")
