# preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_and_prepare_data(filepath="data/yahoo_data.xlsx"):
    """Загрузка и первичная обработка данных"""
    df = pd.read_excel(filepath)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df["Log_Volume"] = np.log1p(df["Volume"])
    return df


def create_features(df, target_col="Close"):
    """Создание дополнительных признаков для временного ряда"""
    df = df.copy()

    for lag in [1, 2, 3, 5, 7, 14]:
        df[f"Close_lag_{lag}"] = df[target_col].shift(lag)
        df[f"Volume_lag_{lag}"] = df["Log_Volume"].shift(lag)

    # Скользящие статистики
    df["MA7"] = df[target_col].rolling(window=7).mean()
    df["MA30"] = df[target_col].rolling(window=30).mean()
    df["Volatility_7d"] = df[target_col].rolling(window=7).std()
    df["Volatility_30d"] = df[target_col].rolling(window=30).std()

    # Процентные изменения
    df["Daily_Return"] = df[target_col].pct_change() * 100
    df["Return_MA7"] = df["Daily_Return"].rolling(window=7).mean()

    # RSI (относительная сила индекса)
    delta = df[target_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # День недели и месяц
    df["Weekday"] = df.index.dayofweek
    df["Month"] = df.index.month
    df["Day"] = df.index.day

    # Период (до/во время/после кризиса)
    crisis_start = pd.Timestamp("2020-02-01")
    crisis_end = pd.Timestamp("2020-06-30")
    conditions = [
        df.index < crisis_start,
        (df.index >= crisis_start) & (df.index <= crisis_end),
        df.index > crisis_end,
    ]
    choices = [0, 1, 2]
    df["Period"] = np.select(conditions, choices, default=2)

    # Разница между High и Low (внутридневной диапазон)
    df["High_Low_Range"] = df["High"] - df["Low"]
    df["Open_Close_Range"] = abs(df["Close"] - df["Open"])

    # Целевая переменная (цена закрытия на следующий день)
    df["Target"] = df[target_col].shift(-1)

    # Удаляем строки с NaN
    df.dropna(inplace=True)

    return df


def prepare_data_for_training(
    df, target_col="Target", test_size=0.2, feature_cols=None
):
    """Подготовка данных для обучения моделей"""

    # Признаки для модели
    if feature_cols is None:
        featured_df = create_features(df, target_col=target_col)
        feature_cols = [
            "Open",
            "High",
            "Low",
            "Log_Volume",
            "Close_lag_1",
            "Close_lag_2",
            "Close_lag_3",
            "Volume_lag_1",
            "Volume_lag_2",
            "MA7",
            "MA30",
            "Volatility_7d",
            "Volatility_30d",
            "Daily_Return",
            "Return_MA7",
            "RSI",
            "High_Low_Range",
            "Open_Close_Range",
            "Weekday",
            "Month",
            "Period",
        ]

    X = df[feature_cols]
    y = df[target_col]

    # Разделение на train/test с сохранением временного порядка
    split_idx = int(len(X) * (1 - test_size))
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    # Масштабирование признаков
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols
