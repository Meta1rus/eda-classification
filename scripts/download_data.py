"""Скачать датасеты в data/, если их там нет, и сверить контрольные суммы.

* online_shoppers_intention.csv — Online Shoppers Purchasing Intention, UCI Machine Learning Repository, id 468
  https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
  Используется в lab2–lab6, в проекте и в дашборде. Файл берётся с зеркала на GitHub и проверен
  по описанию UCI: 12 330 строк × 18 столбцов, 1 908 сессий с Revenue = True.
* diamonds_good.csv, diamonds_clean.csv — данные практик по алмазам из материалов курса
  https://github.com/aiedu-courses/stepik_eda_and_dev_tools/tree/main/datasets

Нужна только стандартная библиотека Python, поэтому скрипт можно запустить и перед сборкой Docker-образа.

    python scripts/download_data.py
"""
import csv
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
COURSE = "https://raw.githubusercontent.com/aiedu-courses/stepik_eda_and_dev_tools/main/datasets"
DATASETS = [
    ("online_shoppers_intention.csv",
     "https://raw.githubusercontent.com/fenago/datasets/main/online_shoppers_intention.csv",
     "b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9"),
    ("diamonds_good.csv", f"{COURSE}/diamonds_good.csv",
     "58af71e60850c6c72e1090f17651c7e2c32c0dfa30a1d78fe04b8d2f2b3b4ec9"),
    ("diamonds_clean.csv", f"{COURSE}/diamonds_clean.csv",
     "e087603ede468c02fa966ee0560b72e8a4b4a07f6a7a0a70edacdc9819bdccc2"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_shoppers(path: Path) -> None:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    revenue = header.index("Revenue")
    assert (len(body), len(header)) == (12330, 18)
    assert sum(r[revenue] == "TRUE" for r in body) == 1908


def main() -> None:
    DATA.mkdir(exist_ok=True)
    for name, url, expected in DATASETS:
        target = DATA / name
        if target.exists() and sha256(target) == expected:
            print(f"Файл уже на месте и совпадает с эталоном: {target}")
            continue
        print(f"Скачиваю {url} ...")
        urllib.request.urlretrieve(url, target)
        if sha256(target) != expected:
            sys.exit(f"Контрольная сумма {name} не совпала — файл в источнике изменился, проверьте его вручную")
        print("Готово:", target)
    check_shoppers(DATA / "online_shoppers_intention.csv")


if __name__ == "__main__":
    main()
