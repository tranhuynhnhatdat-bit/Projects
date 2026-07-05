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
from sklearn.cluster import KMeans
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
model = KMeans(n_clusters=3, random_state=42).fit(inputs[numeric_cols])
preds = model.predict(inputs[numeric_cols])
