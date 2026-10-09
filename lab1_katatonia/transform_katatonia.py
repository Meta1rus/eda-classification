"""
Лабораторная работа 1. Пример преобразования признаков листа catatonia_base.

Скрипт реализует правила, описанные в README.md этой папки:
  * убирает служебные строки (пустые и строки-подсказки с занятия);
  * чистит текст (пробелы по краям, «ё» -> «е», двойные пробелы);
  * разбирает ответы с множественным выбором НЕ наивным split(','),
    а по словарю известных вариантов ответа (в самих вариантах бывают запятые);
  * строит multi-hot / бинарные / порядковые признаки и индикаторы пропуска.

Запуск из корня репозитория:
    python lab1_katatonia/transform_katatonia.py
Результат: lab1_katatonia/output/katatonia_encoded.csv и сводка в консоли.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE / "data" / "katatonia_lab1.xlsx"
OUT_DIR = HERE / "output"

N_PATIENTS = 814  # строки 2..815 листа; ниже пустые строки и подсказки с занятия


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------
def norm(text) -> str | None:
    """Нормализация строки: None для пустого, обрезка пробелов, ё->е, схлопывание пробелов."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return None
    s = str(text).replace("ё", "е").replace("Ё", "Е")
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def parse_multi(value, options: list[str]) -> tuple[set[str], list[str]]:
    """Разбор ответа с множественным выбором.

    Сначала ищем известные варианты (самые длинные первыми, т.к. в некоторых
    вариантах есть запятые: «Ипохондрические, соматоформные и ...»),
    затем остаток режем по запятой/точке с запятой -> свободный текст.
    """
    s = norm(value)
    if s is None:
        return set(), []
    found: set[str] = set()
    rest = s
    for opt in sorted(options, key=len, reverse=True):
        pat = re.escape(norm(opt))
        if re.search(pat, rest, flags=re.IGNORECASE):
            found.add(opt)
            rest = re.sub(pat, " ", rest, flags=re.IGNORECASE)
    free = [t.strip(" .") for t in re.split(r"[,;]", rest)]
    free = [t for t in free if t]
    return found, free


# ---------------------------------------------------------------------------
# 1. Наследственность (7 признаков-«строк» одной табличной формы)
# ---------------------------------------------------------------------------
FAMILY_COLS = {
    "Аффективные_расстройства": "аффект",
    "ПсихПоздВозр": "псих_позд",
    "НеврСтресс": "невр_стресс",
    "УО": "уо",
    "Эпилепсия": "эпилепсия",
    "ОргПорЦНС": "орг_цнс",
    "Химическая зависимость]": "хим_завис",
}
# Группы родственников — та же схема, что обсуждалась на занятии
# (подсказка в строках 831–837 листа: «onehot: мать, отец, дети, родст_матери, родств_отца, сиблинги»).
RELATIVE_GROUP = {
    "Мать": "мать",
    "Отец": "отец",
    "Сиблинги": "сиблинги",
    "Дети": "дети",
    "Бабка по линии матери": "родств_матери",
    "Дед по линии матери": "родств_матери",
    "Дядя/Тетя по линии матери": "родств_матери",
    "Бабка по линии отца": "родств_отца",
    "Дед по линии отца": "родств_отца",
    "Дядя/Тетя по линии отца": "родств_отца",
}
GROUPS = ["мать", "отец", "сиблинги", "дети", "родств_матери", "родств_отца"]
FIRST_DEGREE = {"мать", "отец", "сиблинги", "дети"}  # родственники 1-й степени (50 % общих генов)


def encode_family(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    group_any = {g: np.zeros(len(df), dtype=int) for g in GROUPS}
    closeness = np.zeros(len(df), dtype=int)
    n_disorders = np.zeros(len(df), dtype=int)
    for col, short in FAMILY_COLS.items():
        per_group = {g: np.zeros(len(df), dtype=int) for g in GROUPS}
        for i, v in enumerate(df[col].tolist()):
            found, free = parse_multi(v, list(RELATIVE_GROUP))
            if free:
                raise ValueError(f"{col}: неизвестный родственник {free}")
            for rel in found:
                g = RELATIVE_GROUP[rel]
                per_group[g][i] = 1
                group_any[g][i] = 1
                closeness[i] = max(closeness[i], 2 if g in FIRST_DEGREE else 1)
        has = np.max(np.vstack(list(per_group.values())), axis=0)
        out[f"насл_{short}"] = has
        n_disorders += has
        # подробный multi-hot по группам делаем только там, где хватает наблюдений
        if short == "хим_завис":
            for g in GROUPS:
                out[f"насл_{short}_{g}"] = per_group[g]
    for g in GROUPS:
        out[f"насл_любое_{g}"] = group_any[g]
    out["насл_число_расстройств"] = n_disorders
    # 0 — отягощённости нет/не указана, 1 — только 2-я степень родства, 2 — есть 1-я степень
    out["насл_близость_родства"] = closeness
    return out


# ---------------------------------------------------------------------------
# 2. Беременность и роды у матери
# ---------------------------------------------------------------------------
PREG_OPTS = {
    "Без патологии": None,
    "Не оценивались (в т.ч. не известно)": None,
    "Патология беременности": "бер_патология",
    "Токсикоз первой половины беременности": "бер_токсикоз_1",
    "Токсикоз второй половины беременности": "бер_токсикоз_2",
    "Психотравмирующие события": "бер_психотравма",
    "Инфекционные заболевания": "бер_инфекции",
    "Невынашивание беременности в анамнезе": "бер_невынашивание",
    "Другие тяжелые заболевания": "бер_другие_заболевания",
}


def encode_pregnancy(s: pd.Series) -> pd.DataFrame:
    flags = [c for c in PREG_OPTS.values() if c]
    out = pd.DataFrame(0, index=s.index, columns=flags + ["бер_не_оценивалось"])
    for i, v in s.items():
        found, free = parse_multi(v, list(PREG_OPTS))
        if free:
            raise ValueError(f"БерРодыМать: неожиданный текст {free}")
        if not found:  # по легенде: пусто = «Без патологии»
            continue
        patho = [PREG_OPTS[o] for o in found if PREG_OPTS[o]]
        if patho:  # конкретная патология важнее отметки «Без патологии»
            out.loc[i, patho] = 1
        elif "Не оценивались (в т.ч. не известно)" in found:
            # в т.ч. противоречие «Не оценивались + Без патологии» -> считаем неизвестным
            out.loc[i, "бер_не_оценивалось"] = 1
    out["бер_любая_патология"] = out[flags].max(axis=1)
    # редкие пункты объединяем, чтобы не плодить почти пустые столбцы
    rare = ["бер_психотравма", "бер_инфекции", "бер_невынашивание", "бер_другие_заболевания"]
    out["бер_прочее"] = out[rare].max(axis=1)
    return out.drop(columns=rare)


# ---------------------------------------------------------------------------
# 3. Патология родов + «Другие патологии родов» (свободный текст)
# ---------------------------------------------------------------------------
BIRTH_OPTS = {
    "Нет": None,
    "Преждевременные роды": "роды_преждевременные",
    "Гипоксия плода в родах": "роды_гипоксия",
    "Кесарево сечение": "роды_кесарево",
    "Затяжные роды": "роды_затяжные",
    "Наложение щипцов": "роды_прочее",
    "Стремительные роды": "роды_прочее",
    "Эклампсия (нефропатия+АД)": "роды_прочее",
}
# ключевые слова для свободного текста (ПатРодов «Другое» и ПатРодовДруг)
BIRTH_TEXT_RULES = [
    (r"асфикси|гипокси|апгар|закричал не сразу|остановка сердца", "роды_гипоксия"),
    (r"34 недел|недонош|7 месяце", "роды_преждевременные"),
    (r"кесарев", "роды_кесарево"),
    (r".", "роды_прочее"),  # всё остальное содержательное — «прочая патология»
]
NO_INFO = {"нет", "нет данных"}


def text_to_birth_flag(text: str) -> str | None:
    t = text.lower().strip(" .")
    if t in NO_INFO:
        return None
    for pat, flag in BIRTH_TEXT_RULES:
        if re.search(pat, t):
            return flag
    return None


def encode_birth(main: pd.Series, other: pd.Series) -> pd.DataFrame:
    flags = ["роды_преждевременные", "роды_гипоксия", "роды_кесарево", "роды_затяжные", "роды_прочее"]
    out = pd.DataFrame(0, index=main.index, columns=flags + ["роды_нет_данных"])
    for i in main.index:
        found, free = parse_multi(main[i], list(BIRTH_OPTS))
        for o in found:
            if BIRTH_OPTS[o]:
                out.loc[i, BIRTH_OPTS[o]] = 1
        o_txt = norm(other[i])
        texts = free + ([o_txt] if o_txt else [])
        for t in texts:
            f = text_to_birth_flag(t)
            if f:
                out.loc[i, f] = 1
        if not found and not texts:
            out.loc[i, "роды_нет_данных"] = 1
    out["роды_любая_патология"] = out[flags].max(axis=1)
    return out


# ---------------------------------------------------------------------------
# 4. Раннее развитие (до 1 года) + уточнение Ут2
# ---------------------------------------------------------------------------
EARLY_MAP = {
    "В соответствии с возрастом": 0,
    "Наличие отклонений, уточнить, в т.ч. обращаемость за медицинской помощью": 1,
    "Не оценивалось": np.nan,
}
UT2_RULES = {
    "ранразв_задержка": r"задер|отстав|поздно",
    "ранразв_неврология": r"невролог|гипертонус|гидроцефал|вчд|судорог|парез|энцефалопат|дисфункц|приступ|гемиплег",
    "ранразв_соматика": r"бронхит|пневмони|гепатит|аллерг|поллиноз|пиелонефрит|нефрэктом|анеми|желтух|грипп|удушья|маленького роста|набирал",
}


def encode_early(main: pd.Series, ut2: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=main.index)
    m = main.map(norm).map(lambda v: EARLY_MAP.get(v, np.nan) if v else np.nan)
    out["ранразв_отклонения"] = m  # 0/1, пропуск = не оценивалось или пусто
    out["ранразв_не_оценивалось"] = m.isna().astype(int)
    txt = ut2.map(norm).fillna("").str.lower()
    for name, pat in UT2_RULES.items():
        out[name] = txt.str.contains(pat, regex=True).astype(int)
    return out


# ---------------------------------------------------------------------------
# 5. Невротические расстройства детского возраста
# ---------------------------------------------------------------------------
CHILD_OPTS = {
    "Простые фобии": "детство_фобии",
    "Симбиотическая привязанность к значимому взрослому": "детство_симбиоз",
    "Логоневроз": "детство_логоневроз",
    "Энурез": "детство_энурез",
    "Эмоциональная лабильность (неустойчивость)": "детство_эмоц_лабильность",
    "Парейдолические переживания": "детство_прочее",
    "Энкопрез": "детство_прочее",
    "Снохождение": "детство_прочее",
    "Сноговорение": "детство_прочее",
    "Ипохондрические, соматоформные и соматизированные расстройства": "детство_прочее",
}


def encode_childhood(s: pd.Series) -> pd.DataFrame:
    cols = sorted(set(CHILD_OPTS.values()))
    out = pd.DataFrame(0, index=s.index, columns=cols)
    n = np.zeros(len(s), dtype=int)
    for k, (i, v) in enumerate(s.items()):
        found, free = parse_multi(v, list(CHILD_OPTS))
        if free:
            raise ValueError(f"НеротРасстрДетство: неожиданный текст {free}")
        for o in found:
            out.loc[i, CHILD_OPTS[o]] = 1
        n[k] = len(found)
    out["детство_число_расстройств"] = n
    return out


# ---------------------------------------------------------------------------
# 6. Дошкольный возраст: НевротРасстрШкола + НевротРасстрШколаДоп
#    (один ответ, разрезанный по первой запятой — см. README)
# ---------------------------------------------------------------------------
PRESCHOOL_OPTS = {
    "В соответствии с возрастом": None,
    "Наличие отклонений, уточнить, в т.ч. обращаемость за медицинской помощью": None,
    "Наличие отклонений": None,
    "уточнить, в т.ч. обращаемость за медицинской помощью": None,
    "Не оценивалось": None,
    "Отставание в психомоторном развитии": "дошк_отставание",
    "Значительное опережение в психомоторном развитии": "дошк_опережение",
    "Гиперактивность": "дошк_гиперактивность",
    "Дефицит внимания": "дошк_дефицит_внимания",
    "Диссоциация психического и физического развития": "дошк_диссоциация",
    "Трудности установления контактов и общения в детском коллективе": "дошк_трудности_общения",
    "Особые увлечения": "дошк_прочее",
}


def encode_preschool(main: pd.Series, extra: pd.Series) -> pd.DataFrame:
    flags = sorted({v for v in PRESCHOOL_OPTS.values() if v})
    out = pd.DataFrame(0, index=main.index, columns=flags)
    out["дошк_отклонения"] = np.nan
    for i in main.index:
        parts = [p for p in (main[i], extra[i]) if norm(p)]
        joined = ", ".join(str(p) for p in parts)  # склеиваем разрезанный ответ обратно
        found, free = parse_multi(joined, list(PRESCHOOL_OPTS))
        for o in found:
            if PRESCHOOL_OPTS[o]:
                out.loc[i, PRESCHOOL_OPTS[o]] = 1
        if free:  # клаустрофобия, тревожность, соматика и т.п.
            out.loc[i, "дошк_прочее"] = 1
        specific = out.loc[i, flags].sum() > 0
        if specific or any(o.startswith("Наличие отклонений") for o in found):
            out.loc[i, "дошк_отклонения"] = 1
        elif "В соответствии с возрастом" in found:
            out.loc[i, "дошк_отклонения"] = 0
        # «Не оценивалось» и пусто -> остаётся NaN
    out["дошк_не_оценивалось"] = out["дошк_отклонения"].isna().astype(int)
    # порядковый признак с очевидным порядком: отставание < норма < опережение
    tempo = np.where(out["дошк_отставание"] == 1, -1, np.where(out["дошк_опережение"] == 1, 1, 0)).astype(float)
    tempo[out["дошк_не_оценивалось"].to_numpy() == 1] = np.nan
    out["дошк_темп_развития"] = tempo
    return out


# ---------------------------------------------------------------------------
# 7. Симптомы е-3 и з-2: где отмечен симптом
# ---------------------------------------------------------------------------
SYMPT_OPTS = ["Статус до лечения = Анамнез", "Анамнез", "Статус до лечения", "Статус после лечения", "Статус"]


def encode_symptom(s: pd.Series, short: str) -> pd.DataFrame:
    out = pd.DataFrame(0, index=s.index, columns=[f"{short}_анамнез", f"{short}_статус", f"{short}_после_лечения"])
    for i, v in s.items():
        found, free = parse_multi(v, SYMPT_OPTS)
        if free:
            raise ValueError(f"{short}: неожиданный текст {free}")
        if "Статус до лечения = Анамнез" in found:
            found |= {"Анамнез", "Статус до лечения"}
        if "Анамнез" in found:
            out.loc[i, f"{short}_анамнез"] = 1
        if found & {"Статус", "Статус до лечения"}:
            out.loc[i, f"{short}_статус"] = 1
        if "Статус после лечения" in found:
            out.loc[i, f"{short}_после_лечения"] = 1
    out[f"{short}_есть"] = out.max(axis=1)
    return out


# ---------------------------------------------------------------------------
def main() -> None:
    raw = pd.read_excel(SRC, sheet_name="catatonia_base")
    print(f"Лист catatonia_base: {raw.shape[0]} строк x {raw.shape[1]} столбцов")
    tail = raw.iloc[N_PATIENTS:]
    print(f"Служебных строк ниже данных: {len(tail)}, из них непустых (подсказки): {int(tail.notna().any(axis=1).sum())}")
    df = raw.iloc[:N_PATIENTS].copy()
    assert df.notna().any(axis=1).all(), "среди пациентов не должно быть пустых строк"

    parts = [
        encode_family(df),
        encode_pregnancy(df["БерРодыМать"]),
        encode_birth(df["ПатРодов"], df["ПатРодовДруг"]),
        encode_early(df["РанРазы"], df["Ут2"]),
        encode_childhood(df["НеротРасстрДетство"]),
        encode_preschool(df["НевротРасстрШкола"], df["НевротРасстрШколаДоп"]),
        encode_symptom(df["Симптомы_е3"], "симпт_е3_разорв_речи"),
        encode_symptom(df["Симптомы_з2"], "симпт_з2_бедность_речи"),
    ]
    # Свободный текст в «дошкольном» поле иногда описывает невротические расстройства детства
    # (логоневроз, фобии, сноговорение) — переносим их в соответствующие флаги.
    child = parts[4]
    extra = (df["НевротРасстрШкола"].fillna("").astype(str) + ", "
             + df["НевротРасстрШколаДоп"].fillna("").astype(str)).str.lower()
    for pat, col in [("логоневроз", "детство_логоневроз"), ("фоби", "детство_фобии"),
                     ("сноговорен|снохожден", "детство_прочее")]:
        m = extra.str.contains(pat, regex=True) & (child[col] == 0)
        child.loc[m, col] = 1
        child.loc[m, "детство_число_расстройств"] += 1
        if m.any():
            print(f"Из дошкольного поля перенесено в {col}: {int(m.sum())}")

    res = pd.concat(parts, axis=1)
    # столбцы-константы (например, хим. зависимость у детей пациента — ни одного случая) не несут информации
    const = [c for c in res.columns if res[c].nunique(dropna=False) <= 1]
    if const:
        print("Удалены столбцы-константы:", ", ".join(const))
        res = res.drop(columns=const)
    OUT_DIR.mkdir(exist_ok=True)
    res.to_csv(OUT_DIR / "katatonia_encoded.csv", index=False, encoding="utf-8-sig")

    print(f"Пациентов: {len(res)}; признаков после преобразования: {res.shape[1]} (было 17)")
    summary = pd.DataFrame({
        "единиц/среднее": res.mean(numeric_only=True).round(3),
        "сумма": res.sum(numeric_only=True),
        "пропусков": res.isna().sum(),
    })
    pd.set_option("display.width", 160)
    print(summary.to_string())
    summary.to_csv(OUT_DIR / "summary.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
