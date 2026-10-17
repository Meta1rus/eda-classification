"""Собственные вкладки дашборда: «О проекте» и «Разведочный анализ».

Остальные вкладки (важность признаков, качество модели, отдельные прогнозы,
«что если», зависимости SHAP) — стандартные компоненты ExplainerDashboard.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
from explainerdashboard.custom import ExplainerComponent

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import (load, NUMERIC, NOMINAL, TARGET, DESCRIPTIONS, MONTH_ORDER, MONTH_RU,  # noqa: E402
                      CLASS_COLORS)
from src.features import SKEWED  # noqa: E402

RESULTS = ROOT / "results"
BLUE, ORANGE = CLASS_COLORS["Нет покупки"], CLASS_COLORS["Покупка"]
SEQ_SCALE = [[0, "#f4f8fd"], [0.2, "#cde2fb"], [0.4, "#86b6ef"], [0.6, "#3987e5"], [0.8, "#256abf"], [1, "#0d366b"]]
DIV_SCALE = [[0, "#104281"], [0.25, "#3987e5"], [0.5, "#f0efec"], [0.75, "#e34948"], [1, "#9e2223"]]
FEATURES = NUMERIC + NOMINAL + ["Weekend"]
LAYOUT = dict(template="plotly_white", margin=dict(l=60, r=20, t=50, b=50), font=dict(size=12))


# ---------------------------------------------------------------------------
# Чистые функции построения графиков (их можно проверить без Dash)
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    df = load()
    df["Month"] = df["Month"].astype(str)
    for c in ["OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType"]:
        df[c] = df[c].astype(str)
    return df


def bin_feature(df: pd.DataFrame, col: str) -> pd.Series:
    """Числовой признак: ноль — отдельная группа, ненулевые значения — квартили.
    Номинальный: категории реже 1 % объединяются в «прочие»."""
    s = df[col]
    if col in NUMERIC and col != "SpecialDay":
        out = pd.Series("0", index=s.index, dtype=object)
        pos = s[s > 0]
        edges = np.unique(np.quantile(pos, np.linspace(0, 1, 5)))
        labels = [f"{edges[i]:.3g}–{edges[i + 1]:.3g}" for i in range(len(edges) - 1)]
        out.loc[pos.index] = pd.cut(pos, edges, labels=labels, include_lowest=True).astype(str)
        return pd.Series(pd.Categorical(out, categories=["0"] + labels, ordered=True), index=s.index)
    if col == "Month":
        return pd.Series(pd.Categorical(s.map(MONTH_RU), categories=[MONTH_RU[m] for m in MONTH_ORDER], ordered=True),
                         index=s.index)
    s = s.astype(str)
    freq = s.value_counts(normalize=True)
    s = s.where(s.map(freq) >= 0.01, "прочие")
    cats = sorted([c for c in s.unique() if c != "прочие"], key=lambda v: (len(v), v)) + (["прочие"] if (s == "прочие").any() else [])
    return pd.Series(pd.Categorical(s, categories=cats, ordered=True), index=s.index)


def fig_distribution(df: pd.DataFrame, col: str, log: bool = True) -> go.Figure:
    fig = go.Figure()
    names = {0: "Нет покупки", 1: "Покупка"}
    if col in NUMERIC and col != "SpecialDay":
        use_log = log and col in SKEWED
        for cls in [0, 1]:
            x = df.loc[df[TARGET] == cls, col]
            fig.add_trace(go.Histogram(x=np.log1p(x) if use_log else x, name=names[cls], histnorm="percent",
                                       nbinsx=50, marker_color=[BLUE, ORANGE][cls], opacity=0.6))
        fig.update_layout(barmode="overlay", xaxis_title=f"log1p({col})" if use_log else col,
                          yaxis_title="% сессий класса")
    else:
        b = bin_feature(df, col)
        ct = pd.crosstab(b, df[TARGET], normalize="columns") * 100
        for cls in [0, 1]:
            fig.add_trace(go.Bar(x=[str(i) for i in ct.index], y=ct[cls], name=names[cls], marker_color=[BLUE, ORANGE][cls]))
        fig.update_layout(barmode="group", xaxis_title=col, yaxis_title="% сессий класса")
    fig.update_layout(title=f"Распределение {col} в двух классах", legend=dict(orientation="h", y=1.1), **LAYOUT)
    return fig


def fig_rate(df: pd.DataFrame, col: str) -> go.Figure:
    b = bin_feature(df, col)
    g = df.groupby(b, observed=True)[TARGET].agg(["mean", "size"])
    fig = go.Figure(go.Bar(x=[str(i) for i in g.index], y=g["mean"] * 100, marker_color=ORANGE,
                           customdata=g["size"], hovertemplate="%{x}: %{y:.1f}% покупок<br>сессий: %{customdata}<extra></extra>"))
    fig.add_hline(y=df[TARGET].mean() * 100, line_dash="dash", line_color="#52514e",
                  annotation_text="в среднем 15,5 %", annotation_position="top left")
    fig.update_layout(title=f"Доля покупок по значениям {col}", xaxis_title=col, yaxis_title="% сессий с покупкой", **LAYOUT)
    return fig


def fig_pair(df: pd.DataFrame, a: str, b: str, min_n: int = 30) -> go.Figure:
    A, B = bin_feature(df, a), bin_feature(df, b)
    rate = pd.crosstab(A, B, values=df[TARGET], aggfunc="mean") * 100
    cnt = pd.crosstab(A, B)
    z = rate.where(cnt >= min_n)
    text = [[("" if np.isnan(v) else f"{v:.0f}%") for v in row] for row in z.values]
    fig = go.Figure(go.Heatmap(z=z.values, x=[str(c) for c in z.columns], y=[str(i) for i in z.index],
                               colorscale=SEQ_SCALE, zmin=0, zmax=max(60, np.nanmax(z.values) if np.isfinite(np.nanmax(z.values)) else 60),
                               text=text, texttemplate="%{text}", customdata=cnt.values,
                               hovertemplate=f"{a}: %{{y}}<br>{b}: %{{x}}<br>доля покупок: %{{z:.1f}}%<br>сессий: %{{customdata}}<extra></extra>",
                               colorbar=dict(title="% покупок")))
    fig.update_layout(title=f"Доля покупок по сочетаниям {a} × {b} (пусто — меньше {min_n} сессий)",
                      xaxis_title=b, yaxis_title=a, **LAYOUT)
    return fig


def correlation_matrices(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    num = df[NUMERIC + [TARGET]].astype(float)
    out = {"Спирмен": num.corr("spearman"), "Кендалл": num.corr("kendall"), "Пирсон": num.corr("pearson")}
    try:
        import phik  # noqa: F401
        cols = NUMERIC + NOMINAL + ["Weekend", TARGET]
        out["φK (все признаки)"] = df[cols].phik_matrix(interval_cols=[c for c in NUMERIC if c != "SpecialDay"], verbose=False)
    except Exception:  # phik необязателен для работы дашборда
        pass
    return out


def fig_corr(m: pd.DataFrame, method: str) -> go.Figure:
    signed = not method.startswith("φK")
    fig = go.Figure(go.Heatmap(z=m.values, x=list(m.columns), y=list(m.index),
                               colorscale=DIV_SCALE if signed else SEQ_SCALE,
                               zmin=-1 if signed else 0, zmax=1, text=np.round(m.values, 2), texttemplate="%{text}",
                               textfont=dict(size=9), hovertemplate="%{y} × %{x}: %{z:.3f}<extra></extra>"))
    fig.update_layout(title=f"Матрица: {method}", height=640, yaxis=dict(autorange="reversed"), **LAYOUT)
    return fig


# ---------------------------------------------------------------------------
# Вкладка «О проекте»
# ---------------------------------------------------------------------------
def _kpi(title, value, note=""):
    return dbc.Card(dbc.CardBody([
        html.Div(title, style={"fontSize": "0.85rem", "color": "#52514e"}),
        html.Div(value, style={"fontSize": "1.7rem", "fontWeight": 600}),
        html.Div(note, style={"fontSize": "0.8rem", "color": "#6b6a66"}),
    ]), className="h-100")


class ProjectTab(ExplainerComponent):
    def __init__(self, explainer, title="О проекте", name=None):
        super().__init__(explainer, title, name)
        self.metrics = json.loads((RESULTS / "final_metrics.json").read_text(encoding="utf-8"))
        self.test_table = pd.read_csv(RESULTS / "model_comparison_test.csv")

    def _models_fig(self):
        t = self.test_table[~self.test_table["модель"].str.startswith("Dummy")].sort_values("PR-AUC")
        fig = go.Figure()
        fig.add_trace(go.Bar(y=t["модель"], x=t["ROC-AUC"], orientation="h", name="ROC-AUC", marker_color=BLUE))
        fig.add_trace(go.Bar(y=t["модель"], x=t["PR-AUC"], orientation="h", name="PR-AUC", marker_color=ORANGE))
        fig.add_vline(x=0.155, line_dash="dot", line_color="#52514e", annotation_text="PR-AUC случайной модели")
        fig.update_layout(barmode="group", title="Сравнение моделей на тестовой выборке (lab5)", xaxis_range=[0, 1],
                          height=430, legend=dict(orientation="h", y=1.08), **LAYOUT)
        return fig

    def layout(self):
        m = self.metrics
        cm = m["confusion_matrix_at_threshold"]
        tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
        table = self.test_table.copy()
        for c in table.columns[1:]:
            table[c] = table[c].round(3)
        return dbc.Container([
            dbc.Row(dbc.Col([
                html.H3("Прогноз покупки по сессии интернет-магазина"),
                html.P("Подольский Филипп, магистратура ПМИ, ФМИ ПГНИУ · «Разведочный анализ и алгоритмы классификации»",
                       style={"color": "#52514e"}),
                html.P(["Данные — UCI Online Shoppers Purchasing Intention (12 330 сессий, 17 признаков). "
                        "Задача — бинарная классификация: закончится ли сессия покупкой (Revenue). "
                        "Работа выполнена по CRISP-DM: разведочный анализ (lab2–lab3) → подготовка признаков (lab4) → "
                        "модели и интерпретация (lab5) → этот дашборд (lab6)."]),
            ])),
            dbc.Row([
                dbc.Col(_kpi("Сессий", "12 330", "из них с покупкой 1 908"), md=2),
                dbc.Col(_kpi("Доля покупок", "15,5 %", "классы несбалансированы"), md=2),
                dbc.Col(_kpi("Финальная модель", m["model"], m["params_short"]), md=2),
                dbc.Col(_kpi("ROC-AUC (test)", f"{m['test_roc_auc']:.3f}", "0,5 — случайная модель"), md=2),
                dbc.Col(_kpi("PR-AUC (test)", f"{m['test_pr_auc']:.3f}", "0,155 — случайная модель"), md=2),
                dbc.Col(_kpi(f"F1 при пороге {m['threshold']:.2f}", f"{m['test_f1_at_threshold']:.3f}",
                             f"recall {m['test_recall_at_threshold']:.2f}, precision {m['test_precision_at_threshold']:.2f}"), md=2),
            ], className="g-2 mb-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=self._models_fig()), md=7),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("Хорошее ли качество?"),
                    html.P(f"ROC-AUC = {m['test_roc_auc']:.3f}: в {m['test_roc_auc'] * 100:.0f} случаях из 100 модель ставит "
                           "случайную сессию с покупкой выше случайной сессии без покупки. PR-AUC = "
                           f"{m['test_pr_auc']:.3f} при уровне случайной модели 0,155."),
                    html.P(f"На тестовой выборке при пороге {m['threshold']:.2f} модель нашла {tp} из {tp + fn} покупок "
                           f"(recall {tp / (tp + fn):.0%}); из {tp + fp} сессий, отмеченных как «покупка», верно {tp} "
                           f"(precision {tp / (tp + fp):.0%}); ложных срабатываний {fp} из {tn + fp} сессий без покупки."),
                    html.P("Для задачи «кому показать персональное предложение» это хорошее качество: модель отбирает "
                           "небольшую группу сессий, где покупка встречается в несколько раз чаще среднего. "
                           "Для задачи «точно предсказать каждую покупку» — недостаточное: заметная часть покупок "
                           "пропускается."),
                    html.H5("Ограничения"),
                    html.Ul([
                        html.Li("Главный признак модели — PageValues (метрика Google Analytics, рассчитанная по доходу "
                                "от транзакций). Без него качество заметно ниже — подробно в lab5."),
                        html.Li("В данных нет января и апреля; прогноз для этих месяцев не проверен."),
                        html.Li("Вероятности модели нужно читать как ранжирование сессий; порог подобран по F1 на train."),
                    ]),
                ])), md=5),
            ], className="mb-3"),
            dbc.Row(dbc.Col([
                html.H5("Таблица сравнения моделей (тестовая выборка)"),
                dash_table.DataTable(data=table.to_dict("records"), columns=[{"name": c, "id": c} for c in table.columns],
                                     style_cell={"fontSize": 12, "padding": "4px"}, style_header={"fontWeight": "bold"}),
            ])),
        ], fluid=True)


# ---------------------------------------------------------------------------
# Вкладка «Разведочный анализ»
# ---------------------------------------------------------------------------
class EdaTab(ExplainerComponent):
    def __init__(self, explainer, title="Разведочный анализ", name=None):
        super().__init__(explainer, title, name)
        self.df = load_data()
        self.corr = correlation_matrices(self.df)

    def layout(self):
        opts = [{"label": f"{c} — {DESCRIPTIONS[c]}", "value": c} for c in FEATURES]
        return dbc.Container([
            dbc.Row(dbc.Col([
                html.H4("Разведочный анализ датасета"),
                dbc.Alert("Как и в lab2–lab3, здесь показано только то, что видно в данных: распределения и доли покупок. "
                          "Причинные выводы на этом этапе не делаются.", color="light"),
            ])),
            html.H5("1. Один признак"),
            dbc.Row([
                dbc.Col(dcc.Dropdown(id="eda-feature-" + self.name, options=opts, value="PageValues", clearable=False), md=8),
                dbc.Col(dbc.Checklist(id="eda-log-" + self.name, options=[{"label": " логарифмическая шкала (log1p)", "value": "log"}],
                                      value=["log"], switch=True), md=4),
            ], className="mb-2"),
            dbc.Row([
                dbc.Col(dcc.Graph(id="eda-dist-" + self.name), md=6),
                dbc.Col(dcc.Graph(id="eda-rate-" + self.name), md=6),
            ]),
            html.Hr(),
            html.H5("2. Пара признаков и целевая переменная"),
            dbc.Row([
                dbc.Col(dcc.Dropdown(id="eda-a-" + self.name, options=[{"label": c, "value": c} for c in FEATURES],
                                     value="PageValues", clearable=False), md=6),
                dbc.Col(dcc.Dropdown(id="eda-b-" + self.name, options=[{"label": c, "value": c} for c in FEATURES],
                                     value="ExitRates", clearable=False), md=6),
            ], className="mb-2"),
            dcc.Graph(id="eda-pair-" + self.name),
            html.Hr(),
            html.H5("3. Корреляции"),
            dbc.Row(dbc.Col(dcc.Dropdown(id="eda-corr-" + self.name, options=[{"label": k, "value": k} for k in self.corr],
                                         value="Спирмен", clearable=False), md=4), className="mb-2"),
            dcc.Graph(id="eda-corrfig-" + self.name),
        ], fluid=True)

    def component_callbacks(self, app):
        @app.callback(Output("eda-dist-" + self.name, "figure"), Output("eda-rate-" + self.name, "figure"),
                      Input("eda-feature-" + self.name, "value"), Input("eda-log-" + self.name, "value"))
        def _one(col, log):
            return fig_distribution(self.df, col, log=bool(log)), fig_rate(self.df, col)

        @app.callback(Output("eda-pair-" + self.name, "figure"),
                      Input("eda-a-" + self.name, "value"), Input("eda-b-" + self.name, "value"))
        def _pair(a, b):
            return fig_pair(self.df, a, b)

        @app.callback(Output("eda-corrfig-" + self.name, "figure"), Input("eda-corr-" + self.name, "value"))
        def _corr(method):
            return fig_corr(self.corr[method], method)
