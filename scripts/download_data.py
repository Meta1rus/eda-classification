"""Скачать датасет Online Shoppers Purchasing Intention, если его нет в data/.

Оригинал: UCI Machine Learning Repository, id 468
https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
Файл в репозитории (1 МБ) взят с зеркала на GitHub и проверен по описанию UCI:
12 330 строк × 18 столбцов, 1 908 сессий с Revenue = True.

    python scripts/download_data.py
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "data" / "online_shoppers_intention.csv"
MIRROR = "https://raw.githubusercontent.com/fenago/datasets/main/online_shoppers_intention.csv"
SHA256 = "b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if TARGET.exists() and sha256(TARGET) == SHA256:
        print(f"Файл уже на месте и совпадает с эталоном: {TARGET}")
        return
    TARGET.parent.mkdir(exist_ok=True)
    print(f"Скачиваю {MIRROR} ...")
    urllib.request.urlretrieve(MIRROR, TARGET)
    if sha256(TARGET) != SHA256:
        sys.exit("Контрольная сумма не совпала — файл на зеркале изменился, проверьте его вручную")
    import pandas as pd
    df = pd.read_csv(TARGET)
    assert df.shape == (12330, 18) and int(df["Revenue"].sum()) == 1908
    print("Готово:", TARGET, df.shape)


if __name__ == "__main__":
    main()
