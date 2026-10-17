# Лабораторная работа 6. Интерактивный дашборд в Docker

**Подольский Филипп, магистратура ПМИ, ФМИ ПГНИУ**
Дисциплина «Разведочный анализ и алгоритмы классификации»

**Этап CRISP-DM:** Deployment. Датасет тот же, что во всех работах, — Online Shoppers Purchasing Intention (`data/online_shoppers_intention.csv`), модель — финальная модель из lab5 (`src/model.py`).

## Как запустить (одна команда)

Нужен установленный Docker с плагином Compose (Docker Desktop или `docker-compose-plugin`). Из **корня репозитория**:

```bash
docker compose up --build
```

Первая сборка занимает 5–15 минут (скачиваются пакеты ~1 ГБ, внутри образа обучается модель и считаются SHAP-значения — шаг `build_dashboard.py` идёт 2–5 минут). Когда в логе появится строка `Дашборд: http://localhost:9050`, откройте в браузере **http://localhost:9050**.

Остановить: `Ctrl+C`, затем `docker compose down`.
Запустить в фоне: `docker compose up --build -d`, логи — `docker compose logs -f`.

Если порт 9050 занят, поменяйте левую часть в `docker-compose.yml`: `"8080:9050"` → дашборд будет на http://localhost:8080.

### Без Compose (как в практике по Docker)

```bash
docker build -f lab6_dashboard/Dockerfile -t shoppers-dashboard .
docker run --rm -p 9050:9050 shoppers-dashboard
```

### Без Docker

```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r lab6_dashboard/requirements.txt
python lab6_dashboard/build_dashboard.py               # обучение модели + dashboard.yaml и explainer.joblib в artifacts/
python lab6_dashboard/run_dashboard.py                 # http://localhost:9050
```

### Как проверить, что всё работает

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:9050/    # должно быть 200
docker compose ps                                                  # STATUS: Up … (healthy)
```

## Что внутри

| Файл | Назначение |
|---|---|
| `Dockerfile` | образ на `python:3.13-slim`: ставит зависимости, копирует код и данные, при сборке запускает `build_dashboard.py`, при старте — `run_dashboard.py` |
| `../docker-compose.yml` | запуск одной командой, проброс порта 9050, healthcheck |
| `requirements.txt` | только пакеты дашборда (версии зафиксированы) |
| `build_dashboard.py` | обучает финальную модель CatBoost на train, строит `ClassifierExplainer` на test, собирает дашборд и сериализует его: `artifacts/dashboard.yaml` + `artifacts/explainer.joblib` (метод `to_yaml`, как в практике) |
| `run_dashboard.py` | поднимает дашборд из конфигурации: `ExplainerDashboard.from_config("dashboard.yaml")`, сервер waitress на `0.0.0.0:9050` |
| `eda_tabs.py` | две собственные вкладки (`ExplainerComponent`): «О проекте» и «Разведочный анализ» |

Схема та же, что в `intro.ipynb` с практики: Dockerfile → `docker build` → образ → `docker run` → контейнер; модель и объяснения готовятся на этапе сборки образа, поэтому контейнер стартует за секунды.

## Вкладки дашборда

1. **О проекте** — цель, данные, ключевые метрики финальной модели на тесте, сравнение всех моделей из lab5, словесная оценка качества и ограничения.
2. **Разведочный анализ** (интерактивно):
   * выбор признака → его распределение в двух классах и доля покупок по значениям;
   * выбор двух признаков → тепловая карта доли покупок по их сочетаниям (влияние пары факторов на целевую переменную);
   * матрицы корреляций: Спирмен, Кендалл, Пирсон, φK.
3. **Важность признаков** — перестановочная важность и средние |SHAP| (one-hot столбцы собраны обратно в исходные признаки).
4. **Качество модели** — матрица ошибок, ROC- и PR-кривые, precision по квантилям; порог по умолчанию — подобранный в lab5.
5. **Отдельные прогнозы** — выбор сессии из тестовой выборки и разложение её прогноза на вклады признаков (SHAP).
6. **Что если** — меняете значения признаков сессии и сразу видите, как меняется вероятность покупки.
7. **Зависимости SHAP** — как вклад признака зависит от его значения.

## Проверка, выполненная при подготовке

* Образ собран и запущен командой `docker compose up --build` по этим `Dockerfile` и `docker-compose.yml`: контейнер `healthy`, `GET /` → `HTTP 200`, все 82 callback'а дашборда отвечают без ошибок (70 — `200`, 12 — `204`, штатный PreventUpdate); графики вкладки «Разведочный анализ» строятся для любого выбора.
* Найдено и исправлено при проверке: `explainerdashboard 0.5.8` импортирует `pkg_resources`, которого нет в `setuptools` ≥ 81, поэтому в `requirements.txt` зафиксирован `setuptools==80.10.2`.
