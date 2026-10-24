# Прогноз направления доходности акций (вариант 3)

Практическое задание «ML-моделирование в финансах». Проверяется, можно ли по прошлым ценам
и объёмам предсказать знак доходности акции на 5 торговых дней вперёд и заработать на этом
больше, чем стратегия «купи и держи».

## Данные
- Источник: Yahoo Finance через `yfinance` (скорректированные цены, `auto_adjust=True`).
- 10 акций: AAPL, MSFT, AMZN, GOOGL, NVDA, JPM, KO, WMT, XOM, JNJ.
- Период: 2015-01-01 — 2026-10-14, дневные данные.
- Целевая переменная: `target = 1`, если `Close(t+5) > Close(t)` (через `shift(-5)`).
- 21 признак (доходности, волатильность, скользящие средние, RSI, MACD, объёмы,
  рыночные признаки), все сдвинуты на 1 день — только прошлая информация.
- Разбиение по датам: train 70% / validation 15% / test 15%, зазор 5 торговых дней между выборками.

## Как запустить
```bash
git clone (https://github.com/ArinaSmlr/app)
cd fintech_theme3
pip install -r requirements.txt
python -m src.train
```
Команда скачивает котировки (кэш в `data/prices.csv`), строит признаки, обучает
логистическую регрессию и LightGBM с параметрами из `config.yaml`, один раз оценивает
модель на test и сохраняет результаты:
- `models/model.pkl` — финальная модель LightGBM, `models/logreg.pkl` — базовая модель;
- `reports/` — сравнение моделей, бэктест, PSI, SHAP summary plot.

Дополнительные флаги:
- `python -m src.train --search` — заново подобрать гиперпараметры LightGBM на validation
  (перебор 18 комбинаций с ранней остановкой) и обновить `config.yaml`;
- `python -m src.train --refresh` — заново скачать котировки.

## Веб-приложение
Готовый интерфейс для пользователей: выбираете акции (или загружаете свой CSV с котировками),
нажимаете «Рассчитать прогноз» и получаете вероятность роста на 5 торговых дней, сигнал
«покупать / не покупать», объяснение главных факторов прогноза и историю сигналов.

Запуск на своём компьютере:
```bash
pip install -r requirements.txt
streamlit run app.py
```
Ссылка на приложение: https://8vosusjctabysxqfswfdbm.streamlit.app/

## Прогноз для новых данных (из кода)
```python
from src.features import load_prices
from src.predict import predict

prices = load_prices(start="2024-06-01", end="2026-10-21")
signals = predict(prices, threshold=0.5815, explain=True)
print(signals[signals["Date"] == signals["Date"].max()])
```
`p` — вероятность роста цены через 5 торговых дней, `signal = 1` — «покупать».

## Структура
```
app.py                   веб-приложение (Streamlit)
config.yaml              параметры данных, подбора и финальной модели
requirements.txt         версии пакетов
src/features.py          загрузка цен, target, признаки, разбиение по времени
src/train.py             полный пайплайн (обучение + оценка)
src/metrics.py           Gini, KS, бэктест, Sharpe, просадка, PSI
src/predict.py           predict(df) для новых данных
notebooks/01_eda.ipynb          этап 1: данные и разбиение
notebooks/02_baseline.ipynb     этап 2: признаки и логистическая регрессия
notebooks/03_boosting.ipynb     этап 3: LightGBM и подбор параметров
notebooks/04_evaluation.ipynb   этап 4: test, бэктест, SHAP, PSI
report.pdf               отчёт
```

## Основные результаты
| Модель | ROC-AUC validation | ROC-AUC test | Gini test |
| --- | --- | --- | --- |
| Логистическая регрессия | 0.506 | 0.485 | −0.030 |
| LightGBM | 0.536 | 0.502 | 0.003 |

Бэктест на test (январь 2025 — октябрь 2026), порог p > 0.5815, комиссия 0.1%:
стратегия +5.3% (Sharpe 0.37, просадка −8.4%) против +57.5% (Sharpe 1.84, просадка −18.0%)
у «купи и держи». Прогнозная сила на out-of-time данных не подтвердилась; главная причина —
смена рыночного режима (PSI признака `mkt_vol_20` = 0.87). Подробности — в отчёте.
