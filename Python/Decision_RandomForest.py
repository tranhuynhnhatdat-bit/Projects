import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import ParameterSampler
import pandas_ta as ta

df_gold = pd.read_csv(
    r"C:\Users\Mr.Dat\Desktop\Projects\Data\2026.6.25XAUUSD_ftmo-D1-Forex_245.csv",
    index_col=0,
    parse_dates=True,
)

df_gold["Daily_Return"] = df_gold["Close"].pct_change()
df_gold = df_gold.dropna()
df_gold["Weekday"] = df_gold.index.day_name()


df_silver = pd.read_csv(
    r"C:\Users\Mr.Dat\Desktop\Projects\Data\2026.6.25XAGUSD_ftmo-D1-Forex_245.csv",
    index_col=0,
    parse_dates=True,
)

df_silver["Daily_Return"] = df_silver["Close"].pct_change()
df_silver["Weekday"] = df_silver.index.day_name()
df_silver["Vol_20"] = df_silver["Daily_Return"].rolling(20).std()
df_silver["Gold_Return"] = df_gold["Daily_Return"]
df_silver["Return_1"] = df_silver["Close"].pct_change(1)
df_silver["Return_5"] = df_silver["Close"].pct_change(5)

df_silver["RSI_14"] = ta.rsi(df_silver["Close"], length=14)

df_silver["ATR_Pct"] = (
    ta.atr(df_silver["High"], df_silver["Low"], df_silver["Close"], length=14)
    / df_silver["Close"]
)

df_silver["Dist_EMA20"] = (
    df_silver["Close"] - ta.ema(df_silver["Close"], length=20)
) / ta.ema(df_silver["Close"], length=20)
df_silver["Daily_Return"] = df_silver["Daily_Return"].shift(-1)
df_silver = df_silver.dropna()
copy_df = df_silver.copy().drop(columns={"Open", "High", "Low", "Close"})
inputs = copy_df.drop(columns="Daily_Return")
targets = df_silver["Daily_Return"]
# Splitting Data(Training Set and Testing Set)
x_train, x_test, y_train, y_test = train_test_split(
    inputs, targets, test_size=0.4, shuffle=False
)
train_inputs = x_train.copy()
train_targets = y_train.copy()
test_inputs = x_test.copy()
test_targets = y_test.copy()
numeric_cols = train_inputs.select_dtypes(np.number).columns.tolist()
categorical_cols = train_inputs.select_dtypes("object").columns.tolist()
# Scale numerical features to range(0,1)
scaler = MinMaxScaler().fit(train_inputs[numeric_cols])
train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])
# One-hot encode categorical features
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(
    train_inputs[categorical_cols]
)
encoder_cols = list(encoder.get_feature_names_out(categorical_cols))
train_inputs[encoder_cols] = encoder.transform(train_inputs[categorical_cols])
test_inputs[encoder_cols] = encoder.transform(test_inputs[categorical_cols])

train_inputs = train_inputs[numeric_cols + encoder_cols]
test_inputs = test_inputs[numeric_cols + encoder_cols]

# Training Model

param_grid = {
    "n_estimators": range(100, 1100, 100),
    "max_depth": range(1, 11),
    "min_samples_leaf": range(5, 30, 5),
}
sampler = ParameterSampler(param_grid, n_iter=5, random_state=42)
results = []
for params in sampler:
    model = RandomForestRegressor(**params, random_state=42).fit(
        train_inputs, train_targets
    )
    preds = model.predict(test_inputs)
    signal = np.where(preds > 0, 1, -1)
    strategy_returns = signal * test_targets
    sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
    results.append({**params, "Sharpe": sharpe})
results_df = pd.DataFrame(results)

best = results_df.sort_values("Sharpe", ascending=False)

print(best.head())
