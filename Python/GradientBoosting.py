import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import ParameterSampler
from xgboost import XGBRegressor
from xgboost import plot_tree
from matplotlib.pylab import rcParams
from sklearn.model_selection import TimeSeriesSplit
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
numeric_cols = inputs.select_dtypes(np.number).columns.tolist()
categorical_cols = inputs.select_dtypes("object").columns.tolist()
# Encoding Categorical Features
encoder = OneHotEncoder(sparse_output=False).fit(inputs[categorical_cols])
encoder_cols = list(encoder.get_feature_names_out(categorical_cols))
inputs[encoder_cols] = encoder.transform(inputs[categorical_cols])
inputs = inputs[numeric_cols + encoder_cols]
model = XGBRegressor(random_state=42, n_estimators=20, n_jobs=-1, max_depth=4)
model.fit(inputs, targets)
preds = model.predict(inputs)

importance_df = pd.DataFrame(
    {"feature": inputs.columns, "importance": model.feature_importances_}
).sort_values("importance", ascending=False)
sns.barplot(data=importance_df, x="importance", y="feature")


def train_and_evaluate(
    train_inputs, train_targets, test_inputs, test_targets, **params
):
    model = XGBRegressor(**params).fit(train_inputs, train_targets)
    train_predictions = model.predict(train_inputs)
    test_predictions = model.predict(test_inputs)
    train_r2score = r2_score(train_targets, train_predictions)
    test_r2score = r2_score(test_targets, test_predictions)
    return train_r2score, test_r2score


models = []
tscv = TimeSeriesSplit(n_splits=10)
for train_idx, test_idx in tscv.split(inputs):
    train_inputs, train_targets = inputs.iloc[train_idx], targets.iloc[train_idx]
    test_inputs, test_targets = inputs.iloc[test_idx], targets.iloc[test_idx]
    train_r2score, test_r2score = train_and_evaluate(
        train_inputs,
        train_targets,
        test_inputs,
        test_targets,
        n_estimators=10,
        max_depth=3,
        learning_rate=0.1,
    )
    models.append(model)
    print(f"Train R2 Score: {train_r2score} || Test R2 Score: {test_r2score}")
