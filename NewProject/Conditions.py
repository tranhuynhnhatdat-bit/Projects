import pandas as pd
import numpy as np
from Data.DataManager import DataManager
from IndicatorsCal import IndicatorCal


class Conditions:
    def __init__(self, higher_df=None):
        self._higher_df = higher_df

    def greater_than(self, a, b):
        return a > b

    def less_than(self, a, b):
        return a < b

    def greater_or_equal(self, a, b):
        return a >= b

    def less_or_equal(self, a, b):
        return a <= b

    def equal(self, a, b):
        return a == b

    # ── Cross detection ──────────────────────────────────────────────

    def cross_above(self, a, b):

        if isinstance(a, pd.Series) and isinstance(b, pd.Series):
            prev_condition = a.shift(1) <= b.shift(1)
        elif isinstance(a, pd.Series):
            prev_condition = a.shift(1) <= b
        elif isinstance(b, pd.Series):
            prev_condition = a <= b.shift(1)
        else:
            raise TypeError("At least one of a or b must be a pandas Series.")

        return prev_condition & (a > b)

    def cross_below(self, a, b):

        if isinstance(a, pd.Series) and isinstance(b, pd.Series):
            prev_condition = a.shift(1) >= b.shift(1)
        elif isinstance(a, pd.Series):
            prev_condition = a.shift(1) >= b
        elif isinstance(b, pd.Series):
            prev_condition = a >= b.shift(1)
        else:
            raise TypeError("At least one of a or b must be a pandas Series.")

        return prev_condition & (a < b)


data = DataManager("XAUUSD")
df = data.return_dataframe()
df_1h = data.extract_new_data("h")
condition = Conditions(df_1h)
indicator = IndicatorCal(df_1h)
std = indicator.StdDev(20)
ema20 = indicator.EMA(20)
