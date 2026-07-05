"""
Data cleaning & feature engineering for M1 OHLC data.

The signal-generation and feature-engineering steps used by `clean_data_M1`
are fully customizable: pass your own callables via `signal_func` /
`feature_func` (plus optional `signal_kwargs` / `feature_kwargs`) instead of
editing this file every time you want to try a different strategy or
feature set.

Example
-------
    from Data.analyzeM1 import clean_data_M1, SignalConfig, default_signal_func

    # Use the built-in logic but with different buy/sell times:
    my_config = SignalConfig(
        required_times=((2, 0), (10, 0)),
        buy_time=(2, 0),
        sell_time=(10, 0),
    )
    df_m1, df_1h, df_D = clean_data_M1(
        "data.csv",
        signal_kwargs={"config": my_config},
    )

    # Or plug in a completely custom signal generator:
    def my_signal_func(df_m1):
        df_m1 = df_m1.copy()
        df_m1["Signal"] = 0
        # ... your own entry/exit rules ...
        return df_m1

    df_m1, df_1h, df_D = clean_data_M1("data.csv", signal_func=my_signal_func)
"""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta


# ──────────────────────────────────────────────
# Customizable signal generation
# ──────────────────────────────────────────────
@dataclass
class SignalConfig:
    """Configuration for `default_signal_func`."""

    required_times: Sequence[Tuple[int, int]] = ((1, 0), (9, 0))
    buy_time: Tuple[int, int] = (1, 0)
    sell_time: Tuple[int, int] = (9, 0)


def default_signal_func(
    df_m1: pd.DataFrame, config: Optional[SignalConfig] = None
) -> pd.DataFrame:
    """
    Default entry/exit logic:
      - A day is only "valid" for trading if all bars listed in
        `config.required_times` exist for that day.
      - Buy at `config.buy_time` on valid days, close the position at
        `config.sell_time`.

    Swap this out for your own logic by passing a different `signal_func`
    to `clean_data_M1`. A custom signal function must:
      * accept a DataFrame with a DatetimeIndex and OHLC columns
        (plus "Hour"/"Minute"/"Weekday" already added)
      * return that DataFrame with a "Signal" column added
        (1 = buy, -1 = sell/close, 0 = no action)
    """
    config = config or SignalConfig()
    df_m1 = df_m1.copy()

    valid_days = set()
    for date, group in df_m1.groupby(df_m1.index.date):
        times = set(zip(group.index.hour, group.index.minute))
        if all(t in times for t in config.required_times):
            valid_days.add(date)

    date_series = pd.Series(df_m1.index.date, index=df_m1.index)
    df_m1["Valid_Day"] = date_series.map(lambda d: d in valid_days)

    df_m1["Signal"] = 0

    buy_hour, buy_minute = config.buy_time
    sell_hour, sell_minute = config.sell_time

    buy = (
        df_m1["Valid_Day"]
        & (df_m1.index.hour == buy_hour)
        & (df_m1.index.minute == buy_minute)
    )
    sell = (df_m1.index.hour == sell_hour) & (df_m1.index.minute == sell_minute)

    df_m1.loc[buy, "Signal"] = 1
    df_m1.loc[sell, "Signal"] = -1

    return df_m1


# ──────────────────────────────────────────────
# Customizable feature engineering (hourly)
# ──────────────────────────────────────────────
def default_feature_func(df_h1: pd.DataFrame) -> pd.DataFrame:
    """
    Default hourly feature set (ATR%, EMA distance, RSI, volatility, returns).

    Swap this out for your own logic by passing a different `feature_func`
    to `clean_data_M1`. A custom feature function must:
      * accept a resampled 1h OHLC DataFrame
      * return it with your own additional feature columns
    """
    df_h1 = df_h1.copy()

    df_h1["Log_Return"] = np.log(df_h1["Close"] / df_h1["Close"].shift(1))
    df_h1["Return1"] = df_h1["Close"].pct_change()
    df_h1["Return5"] = df_h1["Close"].pct_change(5)
    df_h1["ATR_Pct"] = (
        ta.atr(df_h1["High"], df_h1["Low"], df_h1["Close"], length=14) / df_h1["Close"]
    )
    ema20 = ta.ema(df_h1["Close"], length=20)
    df_h1["Dist_EMA20"] = (df_h1["Close"] - ema20) / ema20
    df_h1["RSI14"] = ta.rsi(df_h1["Close"], length=14)
    df_h1["Volatility20"] = df_h1["Close"].pct_change().rolling(20).std()

    return df_h1


# ──────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────
def clean_data_M1(
    file,
    signal_func: Callable[..., pd.DataFrame] = default_signal_func,
    feature_func: Callable[..., pd.DataFrame] = default_feature_func,
    signal_kwargs: Optional[dict] = None,
    feature_kwargs: Optional[dict] = None,
):
    """
    Load a raw M1 OHLC csv, generate trade signals, and build hourly/daily
    features.

    Parameters
    ----------
    file : str | Path
        Path to the M1 OHLC CSV file (columns: Datetime, Open, High, Low, Close).
    signal_func : callable, optional
        `signal_func(df_m1, **signal_kwargs) -> df_m1` with a "Signal" column.
        Defaults to `default_signal_func`.
    feature_func : callable, optional
        `feature_func(df_h1, **feature_kwargs) -> df_h1` with extra feature
        columns. Defaults to `default_feature_func`.
    signal_kwargs, feature_kwargs : dict, optional
        Extra keyword arguments forwarded to `signal_func` / `feature_func`.

    Returns
    -------
    df_m1, df_h1, df_D : pd.DataFrame
    """
    signal_kwargs = signal_kwargs or {}
    feature_kwargs = feature_kwargs or {}
    base_df = pd.read_csv(file)
    columns = ["Datetime"]
    if columns not in base_df.columns.tolist():
        df_m1 = pd.read_csv(
            file,
            names=["Datetime", "Open", "High", "Low", "Close"],
            index_col=0,
            parse_dates=True,
            header=None,
        )
    else:
        df_m1 = pd.read_csv(file, index_col=0, parse_dates=True)

    if df_m1.empty:
        raise ValueError(f"No data loaded from '{file}'. Check the file path/contents.")

    missing_cols = {"Open", "High", "Low", "Close"} - set(df_m1.columns)
    if missing_cols:
        raise ValueError(f"CSV is missing required columns: {missing_cols}")

    df_m1["Weekday"] = df_m1.index.day_name()
    df_m1["Hour"] = df_m1.index.hour
    df_m1["Minute"] = df_m1.index.minute

    df_m1 = signal_func(df_m1, **signal_kwargs)
    if "Signal" not in df_m1.columns:
        raise ValueError("signal_func must return a DataFrame with a 'Signal' column.")

    df_h1 = df_m1.resample("1h").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    )
    df_h1 = feature_func(df_h1, **feature_kwargs)

    df_D = df_m1.resample("D").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    )
    df_D["Return"] = df_D["Close"].pct_change()

    return df_m1, df_h1, df_D


def cleaning_backtest(stats, df_m1, df_1h, drop_columns=None):
    """
    Merge backtest trade outcomes onto the M1 data and build the ML-ready
    `labeled` feature table.

    Parameters
    ----------
    stats : backtesting.backtesting._Stats
        Result object returned by `Backtester.backtester.backtest`.
    df_m1, df_1h : pd.DataFrame
        Outputs of `clean_data_M1`.
    drop_columns : list[str], optional
        Columns to drop from the final `labeled` table. Defaults to the
        raw OHLC/time/signal helper columns.
    """
    if stats is None:
        raise ValueError(
            "cleaning_backtest received no `stats` (backtest did not run)."
        )

    trades = stats._trades.copy()
    if trades.empty:
        raise ValueError(
            "Backtest produced no trades. Check your signal logic / data range."
        )

    trades["Profitable"] = (trades["PnL"] > 0).astype(int)

    df_m1 = df_m1.copy()
    df_m1["Profitable"] = np.nan

    # If multiple trades share the same EntryTime, keep the last outcome
    # (avoids a length-mismatch error from `.loc[...] = values`).
    profitable_by_time = trades.groupby("EntryTime")["Profitable"].last()
    df_m1.loc[profitable_by_time.index, "Profitable"] = profitable_by_time.values

    # Merge 1h features (shifted to avoid look-ahead bias)
    df_1h_features = df_1h.drop(
        columns=["Open", "High", "Low", "Close"], errors="ignore"
    ).shift(1)
    df_m1 = pd.merge_asof(
        df_m1.sort_index(),
        df_1h_features.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
    )

    # Defensive: drop any rows where 1h features are NaN (early rows in the
    # series where rolling windows like ATR/RSI/volatility aren't ready yet)
    feature_cols = [c for c in df_1h_features.columns if c in df_m1.columns]
    if feature_cols:
        df_m1 = df_m1.dropna(subset=feature_cols)

    # Keep only rows with trade labels
    labeled = df_m1[df_m1["Profitable"].isin([0, 1])].copy()

    default_drop_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Hour",
        "Minute",
        "Signal",
        "Valid_Day",
        "Weekday",
    ]
    labeled = labeled.drop(
        columns=drop_columns if drop_columns is not None else default_drop_columns,
        errors="ignore",
    )

    if labeled.empty:
        raise ValueError(
            "No labeled samples after cleaning. Check that trade entry times "
            "align with the M1 index."
        )

    return trades, df_m1, df_1h, labeled
