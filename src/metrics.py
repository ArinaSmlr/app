"""Метрики качества, бэктест стратегии и стабильность (PSI)."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def gini(y, p):
    return 2 * roc_auc_score(y, p) - 1


def classification_metrics(y, p):
    auc = roc_auc_score(y, p)
    return {"ROC-AUC": round(float(auc), 4), "Gini": round(float(2 * auc - 1), 4)}


def ks_stat(y, p):
    order = np.argsort(p)
    y = np.asarray(y)[order]
    cum_bad = np.cumsum(y) / y.sum()
    cum_good = np.cumsum(1 - y) / (1 - y).sum()
    return float(np.max(np.abs(cum_bad - cum_good)))


def next_day_returns(prices):
    """Доходность каждой акции с закрытия дня t до закрытия дня t+1."""
    px = prices.sort_values(["Ticker", "Date"]).copy()
    px["r_next"] = px.groupby("Ticker")["Close"].shift(-1) / px["Close"] - 1
    return px[["Date", "Ticker", "r_next"]]


def backtest(pred, rets, thr, step=5, fee=0.001):
    """Стратегия «в позиции, если p > thr».

    Раз в `step` торговых дней (горизонт прогноза) каждой акции с p > thr выделяется
    1/N капитала, остальное — в деньгах (0% доходности). Комиссия `fee` от оборота.
    Бенчмарк — равновзвешенный портфель тех же акций («купи и держи»).
    Возвращает (дневная доходность стратегии, дневная доходность бенчмарка, позиции).
    """
    df = pred[["Date", "Ticker", "p"]].merge(rets, on=["Date", "Ticker"])
    sig = df.pivot(index="Date", columns="Ticker", values="p")
    ret = df.pivot(index="Date", columns="Ticker", values="r_next").fillna(0)

    pos = (sig > thr).astype(float)
    rebalance_days = sig.index[::step]
    pos.loc[~pos.index.isin(rebalance_days)] = np.nan
    pos = pos.ffill()
    w = pos / pos.shape[1]

    turnover = w.diff().abs().sum(axis=1)
    turnover.iloc[0] = w.iloc[0].sum()
    strat = (w * ret).sum(axis=1) - fee * turnover

    bh = ret.mean(axis=1)
    bh.iloc[0] -= fee
    return strat, bh, pos


def performance(r, periods=252):
    """Доходность, годовая доходность, Sharpe (без безрисковой ставки), макс. просадка."""
    eq = (1 + r).cumprod()
    return {
        "Доходность за период": float(eq.iloc[-1] - 1),
        "Годовая доходность": float(eq.iloc[-1] ** (periods / len(r)) - 1),
        "Sharpe": float(r.mean() / r.std() * np.sqrt(periods)),
        "Макс. просадка": float((eq / eq.cummax() - 1).min()),
    }


def choose_threshold(valid_pred, rets, quantiles=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)):
    """Порог выбирается на validation по максимальному Sharpe (test не используется)."""
    thresholds = np.unique(np.round(np.quantile(valid_pred["p"], quantiles), 4))
    rows = []
    for thr in thresholds:
        s, _, pos = backtest(valid_pred, rets, thr)
        rows.append({"Порог": float(thr), "Доля времени в позиции": float(pos.mean().mean()),
                     **performance(s)})
    table = pd.DataFrame(rows)
    best = float(table.loc[table["Sharpe"].idxmax(), "Порог"])
    return best, table


def psi(expected, actual, bins=10):
    """Population Stability Index: > 0.25 — нестабильно, 0.10–0.25 — умеренный сдвиг."""
    expected, actual = np.asarray(expected), np.asarray(actual)
    cuts = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    cuts[0], cuts[-1] = -np.inf, np.inf
    e = np.histogram(expected, cuts)[0] / len(expected) + 1e-6
    a = np.histogram(actual, cuts)[0] / len(actual) + 1e-6
    return float(np.sum((a - e) * np.log(a / e)))


def psi_report(train, test, columns):
    rows = [{"Переменная": c, "PSI": psi(train[c], test[c])} for c in columns]
    table = pd.DataFrame(rows)
    table["Вывод"] = np.select([table["PSI"] > 0.25, table["PSI"] > 0.10],
                               ["нестабилен (> 0.25)", "умеренный сдвиг"], default="стабилен")
    return table
