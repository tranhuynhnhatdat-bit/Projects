import vectorbt as vbt
from GettingData import getting_M1_data
from Metrics import Trade_Metrics, Asset_Metrics
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from numba import njit
import talib

df, df_1h = getting_M1_data(
    r"C:\Users\Mr.Dat\Desktop\Projects\Data\2026.6.30XAUUSD_ftmo-M1-Forex_245.csv",
    frequency="h",
)


@njit
def _produce_signal_nb(
    hour,
    minute,
    weekday,
    valid_day,
    start_hour,
    start_minute,
    end_hour,
    end_minute,
    target_weekday,
):
    signal = np.zeros(len(hour), dtype=np.int64)
    for i in range(len(hour)):
        if (
            valid_day[i]
            and hour[i] == start_hour
            and minute[i] == start_minute
            and weekday[i] == target_weekday
        ):
            signal[i] = 1
        if (
            valid_day[i]
            and hour[i] == end_hour
            and minute[i] == end_minute
            and weekday[i] == target_weekday
        ):
            signal[i] = -1
    return signal


def produce_signal(close_df, start_hour, start_minute, end_hour, end_minute, weekday):
    hour = close_df.index.hour.values
    minute = close_df.index.minute.values
    weekday_arr = close_df.index.weekday.values
    valid_day = close_df["Valid_Day"].values
    return _produce_signal_nb(
        hour,
        minute,
        weekday_arr,
        valid_day,
        start_hour,
        start_minute,
        end_hour,
        end_minute,
        weekday,
    )


def apply_func(
    close: pd.Series, start_hour, start_minute, end_hour, end_minute, weekday: int
):
    close_df = pd.DataFrame(close, columns=["Close"])
    required_times = (
        (start_hour, start_minute),
        (end_hour, end_minute),
    )
    dates = close.index.normalize()
    # Create boolean masks for each required (hour, minute) pair
    masks = [
        (close.index.hour == h) & (close.index.minute == m) for h, m in required_times
    ]

    # For each mask, extract the set of dates where that time exists
    date_sets = [set(dates[m]) for m in masks]

    # Valid days = intersection of all sets (days that have ALL required times)
    valid_days = set.intersection(*date_sets) if date_sets else set()
    close_df["Valid_Day"] = dates.isin(valid_days)

    return produce_signal(
        close_df, start_hour, start_minute, end_hour, end_minute, weekday
    )


indicator = vbt.IndicatorFactory(
    class_name="Seasonality",
    short_name="season",
    input_names=["close"],
    param_names=["start_hour", "start_minute", "end_hour", "end_minute", "weekday"],
    output_names=["value"],
).from_apply_func(apply_func, keep_pd=True)

result = indicator.run(
    df["Close"], start_hour=2, start_minute=5, end_hour=23, end_minute=5, weekday=4
)

entries = result.value == 1
exits = result.value == -1

atr = talib.ATR(df["High"], df["Low"], df["Close"], timeperiod=14)
sl_stop = (2 * atr) / df["Close"]


pf = vbt.Portfolio.from_signals(
    df["Close"], entries, exits, size=1, init_cash=10000, sl_stop=sl_stop
)

metrics = Trade_Metrics(df, pf, exit_or_entry_time=True)
asset_metrics = Asset_Metrics(df, "Close")
