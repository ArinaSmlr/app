"""Веб-приложение: прогноз направления цены акции на 5 торговых дней вперёд.

Запуск:  streamlit run app.py
"""
import datetime as dt
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import yaml

from src.features import TICKERS
from src.predict import DEFAULT_THRESHOLD, FEATURE_NAMES_RU, explain_row, load_model, predict

ROOT = Path(__file__).resolve().parent
HISTORY_DAYS = 600  # календарных дней истории: хватает на годовые признаки (252 торговых дня)

st.set_page_config(page_title="Прогноз направления акций", page_icon="📈", layout="wide")


# ---------------------------------------------------------------- данные и модель
@st.cache_resource
def get_model():
    return load_model()


@st.cache_data(ttl=3600, show_spinner=False)
def download(tickers, start, end):
    import yfinance as yf

    frames, failed = [], []
    for t in tickers:
        try:
            df = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
        except Exception:
            df = pd.DataFrame()
        if df.empty:
            failed.append(t)
            continue
        df = df[["Close", "Volume"]].copy()
        df.columns = ["Close", "Volume"]
        df["Ticker"] = t
        df.index.name = "Date"
        frames.append(df.reset_index())
    prices = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return prices, failed


def read_csv(file):
    df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]
    rename = {c: c.capitalize() for c in df.columns if c.lower() in {"date", "ticker", "close", "volume"}}
    df = df.rename(columns=rename)
    missing = {"Date", "Ticker", "Close", "Volume"} - set(df.columns)
    if missing:
        raise ValueError(f"В файле нет столбцов: {', '.join(sorted(missing))}")
    df["Date"] = pd.to_datetime(df["Date"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    return df[["Date", "Ticker", "Close", "Volume"]].dropna()


def load_config():
    try:
        return yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    except Exception:
        return {}


def load_summary():
    path = ROOT / "reports" / "summary.json"
    return json.load(open(path, encoding="utf-8")) if path.exists() else None


# ---------------------------------------------------------------- боковая панель
st.sidebar.header("Входные данные")
source = st.sidebar.radio("Источник котировок", ["Yahoo Finance", "Свой CSV-файл"])

prices, chosen = None, []
if source == "Yahoo Finance":
    chosen = st.sidebar.multiselect("Акции из обучающей выборки", TICKERS, default=TICKERS)
    extra = st.sidebar.text_input("Другие тикеры через запятую", placeholder="например: TSLA, META")
    extra = [t.strip().upper() for t in extra.split(",") if t.strip()]
    chosen = list(dict.fromkeys(chosen + extra))
    as_of = st.sidebar.date_input("Дата прогноза (последний день котировок)", dt.date.today(),
                                  max_value=dt.date.today())
else:
    st.sidebar.caption("Столбцы: Date, Ticker, Close, Volume — дневные данные, "
                       "минимум ~1 год истории по каждой акции.")
    sample = pd.DataFrame({"Date": ["2026-10-13", "2026-10-14"], "Ticker": ["KCEL", "KCEL"],
                           "Close": [2450.0, 2462.5], "Volume": [15230, 18020]})
    st.sidebar.download_button("Скачать пример файла", sample.to_csv(index=False),
                               "example.csv", "text/csv")
    file = st.sidebar.file_uploader("CSV с котировками", type="csv")
    add_market = st.sidebar.checkbox("Рынок = 10 акций обучения (загрузить из Yahoo)", value=True,
                                     help="Рыночные признаки модель считает по 10 крупным акциям США. "
                                          "Если выключить, «рынком» будут акции из вашего файла.")

threshold = st.sidebar.slider("Порог сигнала «покупать»", 0.50, 0.65, float(DEFAULT_THRESHOLD), 0.0005,
                              format="%.4f",
                              help="Порог 0.5815 выбран на validation по максимальному Sharpe.")
run = st.sidebar.button("Рассчитать прогноз", type="primary", width="stretch")

# ---------------------------------------------------------------- шапка
st.title("📈 Прогноз направления цены акции на 5 торговых дней")
st.markdown(
    "Модель **LightGBM** оценивает вероятность того, что цена акции через 5 торговых дней будет "
    "выше текущей, и объясняет, какие факторы повлияли на прогноз. "
    "Признаки считаются только по прошлым ценам и объёмам."
)

tab_forecast, tab_about = st.tabs(["Прогноз", "О модели и её качестве"])

# ---------------------------------------------------------------- расчёт
if run:
    try:
        with st.spinner("Загружаю котировки и считаю прогноз…"):
            if source == "Yahoo Finance":
                if not chosen:
                    st.error("Выберите хотя бы одну акцию."); st.stop()
                end = as_of + dt.timedelta(days=1)
                start = as_of - dt.timedelta(days=HISTORY_DAYS)
                need = list(dict.fromkeys(TICKERS + chosen))
                prices, failed = download(tuple(need), start, end)
                if failed:
                    st.warning(f"Нет данных по тикерам: {', '.join(failed)}")
                chosen = [t for t in chosen if t not in failed]
            else:
                if file is None:
                    st.error("Загрузите CSV-файл."); st.stop()
                user = read_csv(file)
                chosen = sorted(user["Ticker"].unique())
                prices = user
                if add_market:
                    base = [t for t in TICKERS if t not in chosen]
                    mkt, failed = download(tuple(base), user["Date"].min().date(),
                                           user["Date"].max().date() + dt.timedelta(days=1))
                    if failed:
                        st.warning("Не удалось загрузить рынок из Yahoo — «рынком» будут акции из файла.")
                    else:
                        prices = pd.concat([user, mkt], ignore_index=True)
            res = predict(prices, model=get_model(), threshold=threshold, explain=True)
            res = res[res["Ticker"].isin(chosen)]
        if res.empty:
            st.error("Недостаточно истории для расчёта признаков (нужно ~260 торговых дней на акцию)."); st.stop()
        st.session_state["res"] = res
        st.session_state["threshold"] = threshold
    except ValueError as e:
        st.error(str(e)); st.stop()

# ---------------------------------------------------------------- вкладка «Прогноз»
with tab_forecast:
    res = st.session_state.get("res")
    if res is None:
        st.info("Выберите акции или загрузите файл слева и нажмите **«Рассчитать прогноз»**.")
    else:
        thr = st.session_state["threshold"]
        latest = res.sort_values("Date").groupby("Ticker").tail(1).sort_values("p", ascending=False)

        c1, c2, c3 = st.columns(3)
        c1.metric("Акций в прогнозе", len(latest))
        c2.metric("Сигналов «покупать»", int(latest["signal"].sum()))
        c3.metric("Дата последних котировок", latest["Date"].max().strftime("%d.%m.%Y"))

        st.subheader("Сигналы на ближайшие 5 торговых дней")
        table = pd.DataFrame({
            "Тикер": latest["Ticker"],
            "Дата котировки": latest["Date"].dt.strftime("%d.%m.%Y"),
            "Цена закрытия": latest["Close"].round(2),
            "Вероятность роста": latest["p"],
            "Сигнал": np.where(latest["signal"] == 1, "🟢 Покупать", "⚪ Не покупать"),
        })
        st.dataframe(table, hide_index=True, width="stretch",
                     column_config={"Вероятность роста": st.column_config.ProgressColumn(
                         format="%.3f", min_value=0.4, max_value=0.7)})
        st.download_button("Скачать прогноз (CSV)", table.to_csv(index=False), "forecast.csv", "text/csv")

        st.subheader("Почему модель так считает")
        ticker = st.selectbox("Акция", latest["Ticker"].tolist())
        row = latest[latest["Ticker"] == ticker].iloc[0]
        expl = explain_row(row, top=8)
        col_a, col_b = st.columns([3, 2])
        with col_a:
            chart = alt.Chart(expl).mark_bar().encode(
                x=alt.X("Вклад:Q", title="Вклад в прогноз (лог-шансы)"),
                y=alt.Y("Описание:N", sort=alt.EncodingSortField("Вклад", op="sum", order="descending"),
                        title=None, axis=alt.Axis(labelLimit=320)),
                color=alt.condition(alt.datum["Вклад"] > 0, alt.value("#2e7d32"), alt.value("#c62828")),
                tooltip=["Описание", alt.Tooltip("Значение:Q", format=".4f"),
                         alt.Tooltip("Вклад:Q", format=".4f")],
            ).properties(height=320)
            st.altair_chart(chart, width="stretch")
        with col_b:
            verdict = "выше порога → сигнал «покупать»" if row["p"] > thr else "ниже порога → сигнал «не покупать»"
            st.markdown(f"**{ticker}**, котировка на {row['Date']:%d.%m.%Y}  \n"
                        f"Вероятность роста: **{row['p']:.3f}** ({verdict}, порог {thr:.4f})")
            top = expl.iloc[0]
            st.markdown(f"Главный фактор: **{top['Описание']}** = {top['Значение']:.4f} — {top['Влияние'].lower()}.")
            st.dataframe(expl[["Описание", "Значение", "Влияние"]], hide_index=True,
                         width="stretch",
                         column_config={"Значение": st.column_config.NumberColumn(format="%.4f")})

        st.subheader("История цены и прогнозов")
        hist = res[res["Ticker"] == ticker].tail(120)
        base = alt.Chart(hist).encode(x=alt.X("Date:T", title=None))
        price = base.mark_line(color="#455a64").encode(y=alt.Y("Close:Q", title="Цена", scale=alt.Scale(zero=False)))
        prob = base.mark_line(color="#1565c0", opacity=0.7).encode(
            y=alt.Y("p:Q", title="Вероятность роста", scale=alt.Scale(domain=[0.45, 0.65])))
        rule = alt.Chart(pd.DataFrame({"y": [thr]})).mark_rule(strokeDash=[4, 4], color="#1565c0").encode(y="y:Q")
        st.altair_chart(alt.layer(price, prob + rule).resolve_scale(y="independent"),
                        width="stretch")
        st.caption("Серая линия — цена закрытия, синяя — вероятность роста, пунктир — порог сигнала.")

# ---------------------------------------------------------------- вкладка «О модели»
with tab_about:
    cfg = load_config()
    summary = load_summary()
    st.markdown("""
**Что предсказывает модель.** Вероятность того, что цена закрытия акции через 5 торговых дней
будет выше цены закрытия в день прогноза.

**На чём обучена.** Дневные котировки 10 крупных акций США (AAPL, MSFT, AMZN, GOOGL, NVDA, JPM, KO,
WMT, XOM, JNJ) за 2016 — март 2023 года; 21 признак: доходности, волатильность, скользящие средние,
RSI, MACD, объёмы и рыночные признаки. Параметры подобраны на validation (март 2023 — декабрь 2024),
финальная проверка — на out-of-time test (январь 2025 — октябрь 2026).
""")
    st.markdown("**Качество модели**")
    st.table(pd.DataFrame({
        "Выборка": ["Validation", "Test (out-of-time)"],
        "ROC-AUC": [0.536, 0.502],
        "Gini": [0.071, 0.003],
    }).set_index("Выборка"))
    st.markdown("**Бэктест на test** (порог 0.5815, ребалансировка раз в 5 дней, комиссия 0.1%)")
    st.table(pd.DataFrame({
        "Стратегия": ["По сигналам модели", "Купи и держи"],
        "Доходность": ["+5.3%", "+57.5%"],
        "Sharpe": [0.37, 1.84],
        "Макс. просадка": ["−8.4%", "−18.0%"],
    }).set_index("Стратегия"))
    st.warning(
        "**Ограничения.** На out-of-time данных прогнозная сила модели близка к случайной, "
        "стратегия по её сигналам уступила «купи и держи». Главный признак — волатильность рынка — "
        "нестабилен (PSI = 0.87). Для акций вне обучающей выборки и других рынков качество не проверялось. "
        "Приложение — учебный проект и не является инвестиционной рекомендацией."
    )
    with st.expander("Описание признаков"):
        st.table(pd.DataFrame({"Признак": list(FEATURE_NAMES_RU), "Описание": list(FEATURE_NAMES_RU.values())}))
