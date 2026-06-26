import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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
df_silver = df_silver.dropna()
df_silver["Weekday"] = df_silver.index.day_name()
df_silver = pd.get_dummies(df_silver, columns=["Weekday"], drop_first=True)
df_silver["Close_Gold"] = df_gold[["Close"]]
df_silver["Vol_20"] = df_silver["Daily_Return"].rolling(20).std()
df_silver["Gold_Return"] = df_gold["Daily_Return"]
dummy_cols = [
    "Weekday_Monday",
    "Weekday_Tuesday",
    "Weekday_Thursday",
    "Weekday_Wednesday",
]

df_silver[dummy_cols] = df_silver[dummy_cols].astype(int)
df_silver = df_silver.dropna()
inputs = df_silver[["Vol_20", "Gold_Return", "Close_Gold"]]
targets = df_silver["Close"]


model = LinearRegression().fit(inputs, targets)

# Generate predictions
predictions = model.predict(inputs)

# Compute loss to evalute the model
loss = root_mean_squared_error(targets, predictions)
print("Full Sample Loss:", loss)


continuous_cols = ["Vol_20", "Close_Gold", "Gold_Return"]

X = inputs.copy()

scaler = StandardScaler()
X[continuous_cols] = scaler.fit_transform(X[continuous_cols])

model = LinearRegression()
model.fit(X, targets)

coef_df = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_})

print(coef_df.sort_values("Coefficient", key=abs, ascending=False))

inputs_train, inputs_test, targets_train, targets_test = train_test_split(
    inputs, targets, test_size=0.2, shuffle=False
)
X_train = inputs_train.copy()
X_test = inputs_test.copy()

X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])

X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])
model = LinearRegression().fit(X_train, targets_train)
# Generate predictions
predictions_train = model.predict(X_train)

# Compute loss to evalute the model
loss = root_mean_squared_error(targets_train, predictions_train)
print("Training Loss:", loss)

# Generate predictions
predictions_test = model.predict(X_test)

# Compute loss to evalute the model
loss = root_mean_squared_error(targets_test, predictions_test)
print("Test Loss:", loss)
coef_compare = pd.DataFrame({"Feature": X_train.columns, "Coefficient": model.coef_})
print(coef_compare.sort_values("Coefficient", key=abs, ascending=False))

results = pd.DataFrame({"Actual": targets_test, "Predicted": predictions_test})

print(results.head())
