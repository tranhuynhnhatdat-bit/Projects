import vectorbt as vbt
from GettingData import getting_M1_data
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import talib
from numba import njit

df, df_1h = getting_M1_data(
    r"C:\Users\Mr.Dat\Desktop\Projects\Data\2026.6.30XAUUSD_ftmo-M1-Forex_245.csv",
    frequency="h",
)


def apply_func(
    close: pd.Series, start_hour, start_minute, end_hour, end_minute, weekday: str
):
    print(close)

    return


indicator = vbt.IndicatorFactory(
    class_name="Seasonality",
    short_name="season",
    input_names=["close"],
    param_names=["start_hour", "end_hour", "start_minute", "end_minute", "weekday"],
    output_names=["value"],
).from_apply_func(apply_func, keep_pd=True)

result = indicator.run(
    df["Close"], start_hour=10, start_minute=5, end_hour=20, end_minute=5, weekday=0
)


close = df[["Close"]]

close["Weekday"] = close.index.weekday
required_times: tuple = ((10, 5), (23, 5))
valid_days = set()
for date, group in close.groupby(close.index.date):
    times = set(zip(close.index.hour, close.index.minute))
    if all(t in times for t in required_times):
        valid_days.add(date)

date_Series = pd.Series(close.index.date, index=close.index)
close["Valid_Day"] = date_Series.map(lambda date: date in valid_days)
