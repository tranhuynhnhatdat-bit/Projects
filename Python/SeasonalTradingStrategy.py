import pandas_ta as ta
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from backtesting import Backtest, Strategy

df_1h = pd.read_csv(
    r"C:\Users\Mr.Dat\Desktop\Projects\Data\2026.6.25XAUUSD_ftmo-H1-Forex_245.csv",
    index_col=0,
    parse_dates=True,
)
df_1h.columns = df_1h.columns.str.strip()
df_1h["Weekday"] = df_1h.index.weekday
df_1h["Weekday"] = df_1h["Weekday"].replace({5: 0})
df_1h["hour"] = df_1h.index.hour


def passthrough(x):
    return x


class SeasonalStrategy(Strategy):
    start_hour = 1
    end_hour = 9

    def init(self):
        self.hour = self.I(passthrough, self.data.hour)
        self.weekday = self.I(passthrough, self.data.Weekday)

    def next(self):

        # ENTRY at hour 1 (only once per day condition optional)
        if (
            self.hour[-1] == self.start_hour
            and not self.position
            and self.weekday[-1] == 0
        ):
            self.buy()

        # EXIT at hour 9
        if self.hour[-1] == self.end_hour and self.position and self.weekday[-1] == 0:
            self.position.close()


bt = Backtest(df_1h, SeasonalStrategy, cash=10000)
stats = bt.run()
print(stats)
bt.plot()
