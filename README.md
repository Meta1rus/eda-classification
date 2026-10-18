# Разведочный анализ и алгоритмы классификации — лабораторные и итоговый проект

**Подольский Филипп, магистратура ПМИ, ФМИ ПГНИУ**
Дисциплина «Разведочный анализ и алгоритмы классификации» (Барулина Марина Александровна), зачёт.

Сквозная задача — **прогноз покупки по сессии интернет-магазина** на датасете
[Online Shoppers Purchasing Intention](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) (UCI, id 468): 12 330 сессий, 17 признаков, целевая переменная `Revenue` (покупка — 15,5 % сессий).
**Один и тот же датасет используется во всех работах lab2–lab6, в проекте и в дашборде.** Практики курса из Colab выполнены на своём датасете алмазов, как в тетрадках преподавателя, — см. раздел «Практики». Работа построена по CRISP-DM: понимание задачи → понимание данных → подготовка → моделирование → оценка → развёртывание.

## Лабораторные → папки

| Лаба | Папка | Этап CRISP-DM | Что внутри |
|---|---|---|---|
| lab1 | [`lab1_katatonia/`](lab1_katatonia/) | Data preparation (отдельный датасет из задания) | [`README.md`](lab1_katatonia/README.md) — как преобразовать каждый из 17 признаков «Кататонии» и почему; `transform_katatonia.py` — пример реализации; `output/` — результат (814 × 63) |
| lab2 | [`lab2_eda_overview/`](lab2_eda_overview/) | Business + Data understanding | общее исследование (типы, пропуски, дубликаты, допустимые значения), целевая переменная, однофакторный анализ числовых и категориальных признаков и в разрезе классов |
| lab3 | [`lab3_correlations/`](lab3_correlations/) | Data understanding | корреляции Пирсона / Спирмена / Кендалла, V Крамера, η, φK; мультиколлинеарность; влияние пар факторов на целевую переменную (тепловые карты, диаграммы рассеяния, V пары) |
| lab4 | [`lab4_features/`](lab4_features/) | Data preparation | обоснование кодирования (one-hot для номинальных, почему не label encoding), редкие категории, `log1p`, масштабирование, выбросы, дубликаты, VIF, разбиение; проверка влияния подготовки на модели |
| lab5 | [`lab5_models/`](lab5_models/) | Modeling + Evaluation | kNN, логистическая регрессия (веса классов, SMOTE), решающее дерево, случайный лес, SVM, CatBoost; кросс-валидация, тест, порог, калибровка; словесная оценка качества; коэффициенты, перестановочная важность, SHAP, разбор отдельных прогнозов; проверка зависимости от `PageValues` |
| lab6 | [`lab6_dashboard/`](lab6_dashboard/) | Deployment | интерактивный дашборд (ExplainerDashboard + собственные вкладки EDA), Dockerfile, запуск одной командой — **см. ниже** |
| проект | [`project/`](project/) | всё вместе | `project.ipynb` — связный итоговый отчёт; `presentation/` — презентация для защиты (10 слайдов, .pptx и .pdf) |

Общий код — в [`src/`](src/): `data.py` (загрузка и справочник признаков), `features.py` (конвейер подготовки), `model.py` (финальная модель и выбор порога), `viz.py` (единый стиль графиков). Результаты lab5, которые использует дашборд, — в [`results/`](results/).

## Практики курса (датасет алмазов) → [`practices/`](practices/)

Две тетрадки практик, которые преподаватель выдавала в Colab, выполнены целиком. Это **отдельные учебные работы на датасете diamonds** (как в исходных тетрадках), к сквозному проекту Online Shoppers они не относятся. Текст и код преподавателя сохранены, заглушки «# ваш код здесь» заполнены, после каждого задания идёт markdown с ответом и пояснением, к каждому заданию есть графики.

| Тетрадка | Данные | Темы |
|---|---|---|
| [`practice_eda_basic.ipynb`](practices/practice_eda_basic.ipynb) | `data/diamonds_good.csv` (файл преподавателя, 53 940 строк) | дубликаты и пропуски, опечатка `Goood` → `Good`, однофакторный анализ (carat, cut, table/depth, clarity, color), группировки; profiling: [`Diamonds_report.html`](practices/Diamonds_report.html) + ручной обзор |
| [`practice_eda_deep.ipynb`](practices/practice_eda_deep.ipynb) | `data/diamonds_clean.csv` (по ссылке из тетрадки, 53 805 строк) | корреляции Пирсона / Спирмена / Кендалла, V Крамера, ANOVA (+ η²), Polars vs Pandas (eager / lazy), линейная регрессия и MAE |

**Ответы — «Разведочный анализ данных: начало» (basic)**

| Задание | Ответ | | Домашнее задание | Ответ |
|---|---|---|---|---|
| 1. Медиана `y` | 5.71 | | 1. Максимальная из медиан `x`, `y`, `z` | 5.71 |
| 2. Макс. вес огранки Fair | 5.01 | | 2. Строк с нулевым размером | 19 |
| 3. Доля с оптимальными table и depth | 0.62 | | 3. Строк после удаления | 53 786 |
| 4. Чья медианная цена больше | всех алмазов (2401 > 2215) | | 4. Сильнее всего коррелирует с ценой | `x` (r = 0.887) |
| 5. Градаций `clarity` | 8 | | 5. Вид зависимости `price` от `x` | экспоненциальная (нелинейный ускоряющийся рост) |
| 6. Алмазов VVS1 + VVS2 | 8704 | | 6. Доля с `x` и `y` выше средних | 47 % |
| 7. Самая дорогая `clarity` | SI2 | | 7. Доля крупных с `z` < медианы | 0.0002 |
| 8. Огранка с макс. средней ценой в паре (cut, clarity) | Premium (с SI2) | | 8. Группы цвета в данных | Colorless (D–F), Near Colorless (G–J) |
| | | | 9. Самый дорогой цвет | J |
| | | | 10. CaratMedian не возрастает при убывании PriceMedian | верно |
| | | | 11. Самая дорогая тройка | (Very Good,D,IF) |

**Ответы — «Разведочный анализ данных: продолжение» (deep)**

| Задание | Ответ |
|---|---|
| 1. С чем сильнее всего коррелирует `depth` (Кендалл) | `table` (τ = −0.177 по модулю) |
| 2. Фильтр `color = "J"`, `price > 1000`: pandas vs polars | 119 493 строки в обоих; polars быстрее в несколько раз (в сохранённом запуске ≈ 8.7×) |
| 3. Медиана цены и средний вес по цвету: eager vs lazy | таблицы совпадают; polars быстрее pandas в 2–3 раза (в сохранённом запуске 2.0× eager, 3.0× lazy) |
| 4. MAE регрессии `price ~ table` | 2989.02 |
| 5. MAE в % от средней цены | ≈ 76 % (75.99 %) |
| 6. MAE при `price < 10 000` / `≥ 10 000` | 2279.63 / 9608.52, больше на дорогих |

**Что пришлось поправить в коде из Colab** (подробно — в начале каждой тетрадки): убрано `google.colab`/`drive.mount`, пути относительные (`../data/…`); `!pip install` заменён импортом с понятной ошибкой и пакетом в `requirements.txt`; pandas 3: `df['depth'].fillna(..., inplace=True)` больше не меняет таблицу (Copy-on-Write) → явное присваивание, `df.corr()` → `numeric_only=True`, `dtype == 'object'` → проверки `is_string_dtype`/`is_numeric_dtype` (текст теперь имеет тип `str`); пакет `association-metrics` с pandas 3 не работает → V Крамера на `scipy` по той же формуле; polars 2.0: `groupby` → `group_by`, `collect(streaming=True)` → `collect(engine="streaming")`; `%%time` перенесён в первую строку ячеек (иначе `UsageError`). `ydata-profiling` 4.18 под Python 3.13 ставится, но требует pandas < 3, поэтому отчёт `Diamonds_report.html` собран скриптом `practices/make_profile_report.py` в отдельном окружении, а в тетрадке сделан ручной обзор. Файл `data/diamonds_big.csv` (~140 МБ) создаётся тетрадкой deep и в репозиторий не входит (`.gitignore`).

## Главные результаты

* **Данные:** пропусков нет; 125 полных дубликатов (одностраничные сессии без покупки); счётчики и длительности сильно скошены; `OperatingSystems`, `Browser`, `Region`, `TrafficType` — номинальные коды; в данных нет января и апреля.
* **EDA (только наблюдения):** сильнее всего классы различаются по `PageValues` (ненулевое значение у 80,6 % сессий с покупкой и у 11,4 % без неё), затем по `ExitRates`, `ProductRelated(_Duration)`, `Month`, `TrafficType`; в таблице `PageValues` × `ExitRates` доля покупок меняется от 0–6 % до 82 %.
* **Финальная модель — CatBoost** (глубина 6, 500 деревьев, без весов классов): на тестовой выборке **ROC-AUC = 0,929**, **PR-AUC = 0,744** (у случайной модели 0,5 и 0,155). При пороге 0,27 (подобран по F1 на out-of-fold прогнозах train) модель находит 286 из 382 покупок (recall 0,75) при precision 0,61.
* **Интерпретация:** главный вклад в прогноз даёт `PageValues`, затем `Month`, `ExitRates`/`BounceRates`, `Administrative`, `TrafficType`. Без `PageValues` качество падает до ROC-AUC 0,765 / PR-AUC 0,355 — модель во многом опирается на метрику Google Analytics, связанную с историей транзакций (подробно — lab5, раздел 8).

## Как запустить

### Дашборд (lab6) — одной командой

Нужен Docker с Compose. Из корня репозитория:

```bash
docker compose up --build
```

Открыть **http://localhost:9050**. Первая сборка занимает 5–15 минут в зависимости от интернета и компьютера (скачиваются пакеты ~1 ГБ, внутри образа обучается модель и считаются SHAP-значения); повторный запуск — секунды. Остановить: `Ctrl+C` и `docker compose down`. Подробности, запуск без Compose и без Docker — в [`lab6_dashboard/README.md`](lab6_dashboard/README.md).

### Тетрадки

Python 3.13 (проверено на 3.13.16).

```bash
git clone https://github.com/Meta1rus/eda-classification.git && cd eda-classification
python -m venv venv && source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_data.py                        # проверит data/online_shoppers_intention.csv (или скачает с зеркала)
jupyter nbconvert --to notebook --execute --inplace lab2_eda_overview/lab2_eda_overview.ipynb   # и т. д.
python lab1_katatonia/transform_katatonia.py           # пример для lab1
cd practices && jupyter nbconvert --to notebook --execute --inplace practice_eda_basic.ipynb practice_eda_deep.ipynb && cd ..
```

Отчёт profiling для практики basic пересобирается в **отдельном** окружении, потому что `ydata-profiling` требует pandas < 3:

```bash
python -m venv venv_profiling && source venv_profiling/bin/activate
pip install -r practices/requirements-profiling.txt
python practices/make_profile_report.py                # → practices/Diamonds_report.html
```

Все тетрадки уже выполнены и сохранены с результатами — их можно просто открыть на GitHub. Время выполнения на ноутбуке: lab2–lab4 и проект — 1–3 минуты, lab5 (подбор гиперпараметров восьми моделей) — около 10 минут, практики по алмазам — меньше минуты каждая (deep создаёт временный файл `data/diamonds_big.csv` ~140 МБ).

### Как проверить, что всё работает

1. `python scripts/download_data.py` → «Файл уже на месте и совпадает с эталоном».
2. `python lab1_katatonia/transform_katatonia.py` → «Пациентов: 814; признаков после преобразования: 63».
3. `docker compose up --build`, затем `curl -s -o /dev/null -w "%{http_code}\n" http://localhost:9050/` → `200`; в браузере открываются 7 вкладок дашборда.

## Данные

`data/diamonds_good.csv` и `data/diamonds_clean.csv` (по 2.7 МБ) — данные практик: первый — файл, который раздавала преподаватель, второй скачан по ссылке из тетрадки (`raw.githubusercontent.com/aiedu-courses/stepik_eda_and_dev_tools/.../diamonds_clean.csv`); если его нет, тетрадка deep читает его по той же ссылке.

`data/online_shoppers_intention.csv` (1 МБ) лежит в репозитории. Файл взят с зеркала на GitHub; соответствие оригиналу проверено: 12 330 × 18, `Revenue = True` у 1 908 сессий, контрольная сумма SHA-256 зафиксирована в `src/data.py` и `scripts/download_data.py`.

## Проверка, выполненная при подготовке

* Все тетрадки выполнены целиком командой `jupyter nbconvert --to notebook --execute --inplace` на Python 3.13.16 и сохранены с результатами (ошибок и невыполненных ячеек нет). Все числа в выводах, README и презентации взяты из этих запусков.
* `python lab1_katatonia/transform_katatonia.py` — отрабатывает, 814 пациентов × 63 признака.
* Практики по алмазам выполнены тем же `nbconvert --execute --inplace` (pandas 3.0.5, polars 2.0.0): 66 и 62 кодовые ячейки выполнены по порядку, ошибок и заглушек нет.
* **Docker:** образ собран и запущен командой `docker compose up --build` по файлам `docker-compose.yml` и `lab6_dashboard/Dockerfile` из репозитория; контейнер получил статус `healthy`, `GET /` → `HTTP 200`; при имитации загрузки страницы вызваны все 82 callback'а дашборда — 70 ответили `200`, 12 — `204` (штатный PreventUpdate), ошибок в логах контейнера нет; интерактивные графики вкладки «Разведочный анализ» возвращаются для любого выбранного признака, пары признаков и типа корреляции.
* Во время проверки найдена и исправлена несовместимость: `explainerdashboard 0.5.8` импортирует `pkg_resources`, которого нет в свежих `setuptools` (≥ 81), — в requirements зафиксирован `setuptools==80.10.2`, а в скриптах дашборда есть запасная заглушка.

## Структура репозитория

```
├── README.md                 ← этот файл
├── Как сдать.md              ← чек-лист сдачи
├── requirements.txt
├── docker-compose.yml        ← запуск дашборда одной командой
├── data/                     ← online_shoppers_intention.csv; diamonds_good.csv и diamonds_clean.csv (практики)
├── scripts/download_data.py
├── src/                      ← общий код (данные, признаки, модель, стиль графиков)
├── results/                  ← метрики и таблицы сравнения моделей из lab5
├── lab1_katatonia/           ← README.md (описание), transform_katatonia.py, data/, output/
├── lab2_eda_overview/lab2_eda_overview.ipynb
├── lab3_correlations/lab3_correlations.ipynb
├── lab4_features/lab4_features.ipynb
├── lab5_models/lab5_models.ipynb
├── lab6_dashboard/           ← Dockerfile, build_dashboard.py, run_dashboard.py, eda_tabs.py, README.md
├── project/                  ← project.ipynb, figures/, presentation/
└── practices/                ← практики по алмазам: practice_eda_basic.ipynb, practice_eda_deep.ipynb,
                                Diamonds_report.html, make_profile_report.py, requirements-profiling.txt
```
