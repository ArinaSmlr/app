"""Загрузка котировок, целевая переменная, признаки и разбиение по времени.

Все признаки считаются отдельно по каждой акции и сдвинуты на 1 день (shift(1)),
то есть используют только информацию, известную до дня прогноза.
"""
import numpy as np
import pandas as pd

TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "JPM", "KO", "WMT", "XOM", "JNJ"]
HORIZON = 5  # горизонт прогноза, торговых дней

FEATURES_STAGE1 = [
    "ret_1", "ret_5", "ret_20", "momentum_60", "volatility_5", "volatility_20",
    "ma_ratio_20", "ma_ratio_60", "volume_change", "relative_volume_20",
]
FEATURES_STAGE2 = [
    "rsi_14", "macd", "zscore_20", "vol_ratio", "drawdown_252", "mom_12_1",
    "volume_trend", "mkt_ret_5", "mkt_vol_20", "rel_ret_20", "rank_ret_20",
]
FEATURES = FEATURES_STAGE1 + FEATURES_STAGE2


def load_prices(tickers=TICKERS, start="2015-01-01", end="2026-10-15"):
    """Дневные скорректированные цены закрытия и объёмы из Yahoo Finance."""
    import yfinance as yf

    frames = []
    for ticker in tickers:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            raise RuntimeError(f"Не удалось загрузить данные: {ticker}")
        df = df[["Close", "Volume"]].copy()
        df.columns = ["Close", "Volume"]
        df["Ticker"] = ticker
        df.index.name = "Date"
        frames.append(df.reset_index())
    prices = pd.concat(frames, ignore_index=True)
    return prices.sort_values(["Ticker", "Date"]).reset_index(drop=True)


def _rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / down)


def add_target(prices):
    """target = 1, если цена через 5 торговых дней выше текущей (shift(-5))."""
    df = prices.sort_values(["Ticker", "Date"]).reset_index(drop=True).copy()
    g = df.groupby("Ticker")["Close"]
    df["future_return_5d"] = g.shift(-HORIZON) / df["Close"] - 1
    df["target"] = np.where(df["future_return_5d"].notna(),
                            (df["future_return_5d"] > 0).astype(int), np.nan)
    return df


def build_features(prices, market_tickers=None):
    """Возвращает таблицу Date, Ticker, Close, Volume + 21 признак (без целевой переменной).

    market_tickers — акции, по которым считается «рынок» (None = все акции таблицы).
    """
    df = prices[["Date", "Ticker", "Close", "Volume"]].sort_values(["Ticker", "Date"])
    df = df.reset_index(drop=True).copy()
    gc = df.groupby("Ticker")["Close"]
    gv = df.groupby("Ticker")["Volume"]

    # --- Этап 1 ---
    df["ret_1"] = gc.transform(lambda s: s.pct_change().shift(1))
    df["ret_5"] = gc.transform(lambda s: s.pct_change(5).shift(1))
    df["ret_20"] = gc.transform(lambda s: s.pct_change(20).shift(1))
    df["momentum_60"] = gc.transform(lambda s: s.pct_change(60).shift(1))
    df["volatility_5"] = gc.transform(lambda s: s.pct_change().rolling(5).std().shift(1))
    df["volatility_20"] = gc.transform(lambda s: s.pct_change().rolling(20).std().shift(1))
    df["ma_ratio_20"] = gc.transform(lambda s: (s / s.rolling(20).mean() - 1).shift(1))
    df["ma_ratio_60"] = gc.transform(lambda s: (s / s.rolling(60).mean() - 1).shift(1))
    df["volume_change"] = gv.transform(lambda v: v.pct_change().shift(1))
    df["relative_volume_20"] = gv.transform(lambda v: (v / v.rolling(20).mean()).shift(1))

    # --- Этап 2: технические индикаторы ---
    df["rsi_14"] = gc.transform(lambda s: _rsi(s).shift(1))
    df["macd"] = gc.transform(lambda s: (s.ewm(span=12, adjust=False).mean()
                                         / s.ewm(span=26, adjust=False).mean() - 1).shift(1))
    df["zscore_20"] = gc.transform(lambda s: ((s - s.rolling(20).mean())
                                              / s.rolling(20).std()).shift(1))
    df["vol_ratio"] = gc.transform(lambda s: (s.pct_change().rolling(5).std()
                                              / s.pct_change().rolling(20).std()).shift(1))
    df["drawdown_252"] = gc.transform(lambda s: (s / s.rolling(252).max() - 1).shift(1))
    df["mom_12_1"] = gc.transform(lambda s: (s.shift(21) / s.shift(252) - 1).shift(1))
    df["volume_trend"] = gv.transform(lambda v: (v.rolling(5).mean()
                                                 / v.rolling(60).mean()).shift(1))

    # --- Этап 2: рыночные признаки (рынок = среднее по акциям market_tickers) ---
    # По умолчанию рынок = все акции таблицы (как при обучении). Для прогноза по
    # «чужой» акции рынок считается по 10 акциям обучения, а акция сравнивается с ними.
    r = gc.pct_change()
    ret_20 = gc.transform(lambda s: s.pct_change(20).shift(1))
    in_mkt = (df["Ticker"].isin(market_tickers) if market_tickers is not None
              else pd.Series(True, index=df.index))
    mkt = r[in_mkt].groupby(df.loc[in_mkt, "Date"]).mean().sort_index()
    mkt_idx = (1 + mkt.fillna(0)).cumprod()
    mkt_ret_5 = mkt_idx.pct_change(5).shift(1)
    mkt_ret_20 = mkt_idx.pct_change(20).shift(1)
    mkt_vol_20 = mkt.rolling(20).std().shift(1)
    df["mkt_ret_5"] = df["Date"].map(mkt_ret_5)
    df["mkt_vol_20"] = df["Date"].map(mkt_vol_20)
    df["rel_ret_20"] = ret_20 - df["Date"].map(mkt_ret_20)

    # Место акции среди акций рынка по доходности за 20 дней (доля тех, у кого не больше).
    # Для акций рынка совпадает с rank(pct=True) внутри даты.
    peers = {d: np.sort(v.dropna().to_numpy())
             for d, v in ret_20[in_mkt].groupby(df.loc[in_mkt, "Date"])}
    def _rank(d, x):
        arr = peers.get(d)
        if arr is None or len(arr) == 0 or pd.isna(x):
            return np.nan
        return np.searchsorted(arr, x, side="right") / len(arr)
    df["rank_ret_20"] = [_rank(d, x) for d, x in zip(df["Date"], ret_20)]

    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    return df


def time_split(data, train_share=0.70, valid_share=0.15, gap=HORIZON):
    """Разбиение по уникальным датам 70/15/15 с зазором в `gap` дней между выборками.

    Как в ноутбуке 01_eda: доли считаются по строкам с известными target и признаками
    этапа 1; затем удаляются строки, где не набралось окно для признаков этапа 2.
    """
    base = data.dropna(subset=["target"] + FEATURES_STAGE1).copy()
    base["target"] = base["target"].astype(int)
    dates = np.sort(base["Date"].unique())
    n = len(dates)
    train_end, valid_end = int(n * train_share), int(n * (train_share + valid_share))
    parts = {
        "train": dates[:train_end - gap],
        "valid": dates[train_end:valid_end - gap],
        "test": dates[valid_end:],
    }
    out = {}
    for name, d in parts.items():
        part = base[base["Date"].isin(d)].dropna(subset=FEATURES)
        out[name] = part.reset_index(drop=True)
    return out["train"], out["valid"], out["test"]


def make_dataset(prices):
    """Цены -> признаки + target -> train / valid / test."""
    data = build_features(prices).merge(
        add_target(prices)[["Date", "Ticker", "future_return_5d", "target"]],
        on=["Date", "Ticker"], how="left")
    return time_split(data)
