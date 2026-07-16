import talib
import pandas as pd
import numpy as np
from numba import njit
from Data.DataManager import DataManager

data = DataManager("XAUUSD")
df = data.return_dataframe()
higher_df = data.extract_new_data("h")


class IndicatorCal:
    def __init__(self, higher_df, indicator_period):
        self._higher_df = higher_df
        self._indicator_period = indicator_period

    def ATR(self):
        return talib.ATR(
            self._higher_df["High"],
            self._higher_df["Low"],
            self._higher_df["Close"],
            self._indicator_period,
        )

    def ADX(self):
        return talib.ADX(
            self._higher_df["High"],
            self._higher_df["Low"],
            self._higher_df["Close"],
            self._indicator_period,
        )

    def ADXR(self):
        return talib.ADX(
            self._higher_df["High"],
            self._higher_df["Low"],
            self._higher_df["Close"],
            self._indicator_period,
        )

    def AROON(self):
        return talib.AROON(
            self._higher_df["High"],
            self._hihger_df["Low"],
            self._indicator_period,
        )


indicator = IndicatorCal(higher_df)
