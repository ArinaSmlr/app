"""Полный пайплайн одной командой:  python -m src.train

1. Загрузка котировок (кэш в data/prices.csv).
2. Признаки, целевая переменная, разбиение 70/15/15 по времени с зазором 5 дней.
3. Логистическая регрессия (подбор C на validation) и LightGBM.
   По умолчанию LightGBM обучается с параметрами из config.yaml;
   с флагом --search заново подбираются гиперпараметры на validation и config.yaml обновляется.
4. Однократная оценка на test: ROC-AUC/Gini, бэктест против «купи и держи», PSI, SHAP.
5. Сохранение models/model.pkl, models/logreg.pkl и отчётов в reports/.
"""
import argparse
import json
import warnings
from itertools import product
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import features as ft
from src import metrics as mt

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*eval_set.*")

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"


def get_prices(cfg, refresh=False):
    path = ROOT / "data" / "prices.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, parse_dates=["Date"])
    prices = ft.load_prices(cfg["data"]["tickers"], cfg["data"]["start"], cfg["data"]["end"])
    path.parent.mkdir(exist_ok=True)
    prices.to_csv(path, index=False)
    return prices


def fit_logreg(train, valid, C_grid, seed):
    best = None
    for C in C_grid:
        model = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(C=C, max_iter=1000, class_weight="balanced", random_state=seed),
        ).fit(train[ft.FEATURES], train["target"])
        m = mt.classification_metrics(valid["target"], model.predict_proba(valid[ft.FEATURES])[:, 1])
        if best is None or m["Gini"] > best[2]["Gini"]:
            best = (C, model, m)
    return best


def search_lgbm(train, valid, grid, base_params):
    rows, best = [], None
    for num_leaves, min_child, reg_lambda in product(
            grid["num_leaves"], grid["min_child_samples"], grid["reg_lambda"]):
        params = {**base_params, "n_estimators": 3000, "num_leaves": num_leaves,
                  "min_child_samples": min_child, "reg_lambda": reg_lambda}
        model = lgb.LGBMClassifier(**params, metric="auc", verbose=-1)
        model.fit(train[ft.FEATURES], train["target"],
                  eval_set=[(valid[ft.FEATURES], valid["target"])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        auc = model.best_score_["valid_0"]["auc"]
        rows.append({"num_leaves": num_leaves, "min_child_samples": min_child,
                     "reg_lambda": reg_lambda, "best_iteration": model.best_iteration_,
                     "ROC-AUC": round(auc, 4), "Gini": round(2 * auc - 1, 4)})
        if best is None or auc > best[0]:
            best = (auc, {**params, "n_estimators": int(model.best_iteration_)})
    return best[1], pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)


def shap_summary(model, X, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap
    except ImportError:
        print("shap не установлен — SHAP-график пропущен")
        return
    exp = shap.TreeExplainer(model)(X)
    values = exp.values[:, :, 1] if exp.values.ndim == 3 else exp.values
    shap.summary_plot(values, X, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main(search=False, refresh=False):
    cfg = yaml.safe_load(open(CONFIG_PATH, encoding="utf-8"))
    seed = cfg["random_state"]
    (ROOT / "models").mkdir(exist_ok=True)
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)

    prices = get_prices(cfg, refresh)
    train, valid, test = ft.make_dataset(prices)
    for name, d in [("train", train), ("valid", valid), ("test", test)]:
        print(f"{name:5s}: {d['Date'].min().date()} — {d['Date'].max().date()}, "
              f"{len(d)} строк, доля роста {d['target'].mean():.3f}")

    # --- Базовая модель ---
    C, logreg, lr_valid = fit_logreg(train, valid, cfg["logreg"]["C_grid"], seed)
    print(f"LogReg: C={C}, validation {lr_valid}")

    # --- LightGBM ---
    if search:
        params, table = search_lgbm(train, valid, cfg["lgbm_search_grid"], cfg["lgbm_base_params"])
        table.to_csv(reports / "lgbm_search.csv", index=False)
        cfg["model"]["params"] = params
    params = cfg["model"]["params"]
    model = lgb.LGBMClassifier(**params, verbose=-1).fit(train[ft.FEATURES], train["target"])

    for d in (train, valid, test):
        d["p"] = model.predict_proba(d[ft.FEATURES])[:, 1]
    lgb_valid = mt.classification_metrics(valid["target"], valid["p"])
    print(f"LightGBM: validation {lgb_valid}")

    cfg["logreg"]["best_C"] = C
    cfg["model"]["validation_metrics"] = lgb_valid
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    joblib.dump(model, ROOT / "models" / "model.pkl")
    joblib.dump(logreg, ROOT / "models" / "logreg.pkl")

    # --- Однократная оценка на test ---
    lr_test = mt.classification_metrics(test["target"], logreg.predict_proba(test[ft.FEATURES])[:, 1])
    lgb_test = mt.classification_metrics(test["target"], test["p"])
    comparison = pd.DataFrame({
        ("Логистическая регрессия", "validation"): lr_valid,
        ("LightGBM", "validation"): lgb_valid,
        ("Логистическая регрессия", "test"): lr_test,
        ("LightGBM", "test"): lgb_test,
    }).T
    comparison.to_csv(reports / "model_comparison.csv")
    print(comparison)

    rets = mt.next_day_returns(prices)
    thr, thr_table = mt.choose_threshold(valid, rets)
    thr_table.to_csv(reports / "threshold_selection_valid.csv", index=False)
    strat, bh, pos = mt.backtest(test, rets, thr)
    backtest = pd.DataFrame({"Стратегия (LightGBM)": mt.performance(strat),
                             "Купи и держи": mt.performance(bh)}).T
    backtest["Доля времени в позиции"] = [float(pos.mean().mean()), 1.0]
    backtest.to_csv(reports / "backtest_test.csv")
    print(f"\nПорог (выбран на validation): {thr}\n{backtest.round(4)}")

    importance = pd.Series(model.booster_.feature_importance(importance_type="gain"),
                           index=ft.FEATURES).sort_values(ascending=False)
    psi_table = mt.psi_report(train, test, ["p"] + list(importance.index[:5]))
    psi_table.to_csv(reports / "psi.csv", index=False)
    print(f"\n{psi_table.round(4)}")

    shap_summary(model, test[ft.FEATURES], reports / "fig_shap_summary.png")

    with open(reports / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"threshold": thr, "test": lgb_test, "validation": lgb_valid,
                   "backtest": backtest.to_dict()}, f, ensure_ascii=False, indent=2)
    print("\nГотово: models/model.pkl, reports/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", action="store_true", help="заново подобрать параметры LightGBM")
    parser.add_argument("--refresh", action="store_true", help="заново скачать котировки")
    args = parser.parse_args()
    main(search=args.search, refresh=args.refresh)
