"""Прогноз для новых данных.

Пример:
    from src.features import load_prices, TICKERS
    from src.predict import predict
    prices = load_prices(TICKERS + ["TSLA"], start="2024-06-01", end="2026-10-21")
    signals = predict(prices, threshold=0.5815)
    print(signals[signals["Date"] == signals["Date"].max()])
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features import FEATURES, TICKERS, build_features

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_THRESHOLD = 0.5815

FEATURE_NAMES_RU = {
    "ret_1": "Доходность за 1 день",
    "ret_5": "Доходность за 5 дней",
    "ret_20": "Доходность за 20 дней",
    "momentum_60": "Доходность за 60 дней (тренд)",
    "volatility_5": "Волатильность акции за 5 дней",
    "volatility_20": "Волатильность акции за 20 дней",
    "ma_ratio_20": "Отклонение от средней за 20 дней",
    "ma_ratio_60": "Отклонение от средней за 60 дней",
    "volume_change": "Изменение объёма за день",
    "relative_volume_20": "Объём относительно среднего за 20 дней",
    "rsi_14": "RSI (14 дней)",
    "macd": "MACD (EMA12 / EMA26)",
    "zscore_20": "Z-score цены (полосы Боллинджера)",
    "vol_ratio": "Волатильность 5 дн. / 20 дн.",
    "drawdown_252": "Просадка от годового максимума",
    "mom_12_1": "Моментум 12-1 (год без последнего месяца)",
    "volume_trend": "Тренд объёма (5 дн. / 60 дн.)",
    "mkt_ret_5": "Доходность рынка за 5 дней",
    "mkt_vol_20": "Волатильность рынка за 20 дней",
    "rel_ret_20": "Доходность акции минус рынок за 20 дней",
    "rank_ret_20": "Место акции среди 10 по доходности за 20 дней",
}


def load_model(model_path=ROOT / "models" / "model.pkl"):
    return joblib.load(model_path)


def predict(df, model=None, threshold=DEFAULT_THRESHOLD, market_tickers=None, explain=False):
    """df — история цен: столбцы Date, Ticker, Close, Volume (минимум ~260 торговых дней
    на акцию, чтобы рассчитались годовые признаки).

    market_tickers — акции, по которым считаются рыночные признаки. По умолчанию это
    10 акций обучения, если все они есть в df, иначе — все акции df.

    Возвращает Date, Ticker, p (вероятность роста цены через 5 торговых дней),
    signal (1 = «покупать», если p > threshold) и признаки.
    При explain=True добавляются столбцы contrib_<признак> — вклад признака в прогноз
    (в лог-шансах, аналог SHAP) и contrib_base — средний прогноз модели.
    """
    model = model or load_model()
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    if market_tickers is None and set(TICKERS) <= set(df["Ticker"]):
        market_tickers = TICKERS
    data = build_features(df, market_tickers=market_tickers).dropna(subset=FEATURES)
    data = data.reset_index(drop=True)

    out = data[["Date", "Ticker", "Close"] + FEATURES].copy()
    out["p"] = model.predict_proba(data[FEATURES])[:, 1]
    out["signal"] = (out["p"] > threshold).astype(int)
    if explain:
        contrib = model.booster_.predict(data[FEATURES], pred_contrib=True)
        for i, f in enumerate(FEATURES):
            out[f"contrib_{f}"] = contrib[:, i]
        out["contrib_base"] = contrib[:, -1]
    return out


def explain_row(row, top=8):
    """Таблица главных факторов прогноза для одной строки результата predict(explain=True)."""
    items = [(f, row[f], row[f"contrib_{f}"]) for f in FEATURES]
    items.sort(key=lambda t: abs(t[2]), reverse=True)
    table = pd.DataFrame(items[:top], columns=["Признак", "Значение", "Вклад"])
    table["Описание"] = table["Признак"].map(FEATURE_NAMES_RU)
    table["Влияние"] = np.where(table["Вклад"] > 0, "↑ повышает шанс роста", "↓ понижает шанс роста")
    return table[["Описание", "Значение", "Вклад", "Влияние", "Признак"]]
