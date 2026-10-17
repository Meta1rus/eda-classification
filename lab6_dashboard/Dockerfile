# Дашборд по проекту «Online Shoppers Purchasing Intention»
# Сборка из КОРНЯ репозитория:  docker compose up --build
# (или: docker build -f lab6_dashboard/Dockerfile -t shoppers-dashboard . && docker run -p 9050:9050 shoppers-dashboard)
FROM python:3.13-slim

LABEL description="EDA + CatBoost + ExplainerDashboard для датасета Online Shoppers Purchasing Intention" \
      author="Подольский Филипп, магистратура ПМИ, ФМИ ПГНИУ"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=9050

WORKDIR /app

# 1) зависимости отдельным слоем — при правке кода не переустанавливаются
COPY lab6_dashboard/requirements.txt lab6_dashboard/requirements.txt
RUN pip install -r lab6_dashboard/requirements.txt

# 2) код, данные и результаты lab5 (метрики, порог, таблица сравнения моделей)
COPY src/ src/
COPY data/online_shoppers_intention.csv data/online_shoppers_intention.csv
COPY results/ results/
COPY lab6_dashboard/*.py lab6_dashboard/

# 3) обучение финальной модели и сериализация дашборда (dashboard.yaml + explainer.joblib)
RUN python lab6_dashboard/build_dashboard.py

EXPOSE 9050
CMD ["python", "lab6_dashboard/run_dashboard.py"]
