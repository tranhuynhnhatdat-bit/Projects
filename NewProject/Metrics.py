import pandas as pd
import numpy as np
from backtesting import Strategy, Backtest


class Asset_Metrics:
    def __init__(self, df: pd.DataFrame, column: str) -> None:
        self._df = df
        self._column = column
        if self._column not in list(self._df.columns):
            raise ValueError("The column is not in the dataframe")
        self._daily_return = None
        self._years = None

    def __getitem__(self, key: int) -> pd.Series:
        return self._df.iloc[key]

    @property
    def daily_return(self) -> pd.Series:
        if self._daily_return is None:
            self._daily_return = self._df[self._column].pct_change().dropna()
        return self._daily_return

    @property
    def years(self) -> int:
        if self._years is None:
            self_years = (self._df.index[-1] - self._df.index[0]).days / 365.25
        return self_years

    def sharpe_ratio(self, print_result: bool = False) -> float:
        sharpe = float(
            (self.daily_return.mean() / self.daily_return.std()) * np.sqrt(252)
        )
        if print_result:
            print(f"Sharpe Ratio is: {sharpe:.2f}")
        return sharpe

    def sortino_ratio(self, print_result: bool = False) -> float:
        downside_return = self.daily_return.copy()
        downside_return[downside_return > 0] = 0
        downside_std = downside_return.std()
        if downside_std == 0.0:
            return 0.0
        sortino = float((self.daily_return.mean() / downside_std) * np.sqrt(252))
        if print_result:
            print(f"Sortino Ratio is: {sortino:.2f}")
        return sortino

    def maximum_dd(self, print_result: bool = False) -> float:
        running_max = self._df[self._column].cummax()
        drawdown = self._df[self._column] / running_max - 1
        max_drawdown = drawdown.min()
        if print_result:
            print(f"Maximum Drawdown is: {float(max_drawdown * 100):.2f}")
        return float(max_drawdown)

    def profit_factor(self, print_result: bool = False) -> float:
        gross_profit = self.daily_return[self.daily_return > 0].sum()
        gross_loss = self.daily_return[self.daily_return < 0].sum()

        if gross_loss == 0:
            return 0.0
        profitfactor = float(gross_profit / abs(gross_loss))
        if print_result:
            print(f"Profit factor is: {profitfactor:.2f}")
        return profitfactor

    def win_rate(self, print_result: bool = False) -> float:
        winrate = float(
            self.daily_return[self.daily_return > 0].count() / self.daily_return.count()
        )
        if print_result:
            print(f"Win rate is: {winrate * 100:.2f}%")
        return winrate

    def total_return(self, print_result: bool = False) -> float:
        # Get the first and last price from the specified column
        prices = self._df[self._column]

        beginning_price = prices.iloc[0]
        ending_price = prices.iloc[-1]

        if beginning_price == 0:
            return 0.0
        total = float((ending_price - beginning_price) / beginning_price)
        if print_result:
            print(f"Total Return is: {total * 100:.2f}%")
        return total

    def CAGR(self, print_result: bool = False) -> float:
        prices = self._df[self._column]

        beginning_price = prices.iloc[0]
        ending_price = prices.iloc[-1]

        cagr = float((ending_price / beginning_price) ** (1 / self.years) - 1)
        if print_result:
            print(f"CAGR is: {cagr * 100:.2f}%")
        return cagr

    def max_stagnation_days(self, print_result: bool = False) -> int:
        prices = self._df[self._column]

        running_max = prices.cummax()

        is_stagnant = prices < running_max

        new_highs = (~is_stagnant).cumsum()

        stagnant_durations = prices.groupby(new_highs).apply(
            lambda x: (x.index[-1] - x.index[0]).days
        )
        if print_result:
            print(f"Stagnation in days: {int(stagnant_durations.max())}")
        return int(stagnant_durations.max())

    def annualized_volatility(self, print_result: bool = False) -> float:
        daily_std = self.daily_return.std()
        annual_vol = float(daily_std * np.sqrt(252))
        if print_result:
            print(f"Annualized Volatility is: {annual_vol * 100:.2f}%")
        return annual_vol

    def expectancy(self, print_result: bool = False) -> float:
        wins = self.daily_return[self.daily_return > 0]
        losses = self.daily_return[self.daily_return < 0]

        win_prob = len(wins) / len(self.daily_return)
        loss_prob = 1 - win_prob

        average_win = wins.mean()
        average_loss = abs(losses.mean())

        ev = float((win_prob * average_win) - (loss_prob * average_loss))
        if print_result:
            print(f"Expectancy is: {ev * 100}%")
        return ev

    def return_drawdown_ratio(self, print_result: bool = False) -> float:
        return_dd = self.CAGR() / self.maximum_dd()
        if print_result:
            print(f"Return/Drawdown Ratio: {return_dd:.2}")
        return return_dd


class Trade_Metrics():
    def __init__(self,asset_df: pd.DataFrame, stats: pd.DataFrame, exit_or_entry_time:bool = True):
        self._asset_df = asset_df
        self._trade_df = stats._trades[["PnL", "ReturnPct", "EntryTime", "ExitTime"]]
        self._exit_or_entry_time: bool = exit_or_entry_time
        self._equity: pd.DataFrame = pd.DataFrame(self._asset_df.index)
    def __getitem__(self, key: int):
        return self._trade_df[key]
    @property
    def pnl(self) -> pd.Series:
        return self._trade_df["PnL"]
    @property
    def return_pct(self) -> pd.Series:
        return self._trade_df['ReturnPct']
    @property
    def entrytime(self) -> pd.Series:
        return self._trade_df['EntryTime']
    @property
    def exittime(self) -> pd.Series:
        return self._trade_df['ExitTime']
    @property
    def number_of_trades(self) -> int:
        return len(self._trade_df)
    @property
    def exit_equity(self) -> pd.DataFrame:
        exit_pnl = self._trade_df['ExitTime'][['PnL']].reindex(self._asset_df.index).fillna(0).cumsum()
        self._equity['Exit_Equity'] = exit_pnl['PnL'] + 10000
        return self._equity['Exit_Equity']
    @property
    def entry_equity(self) -> pd.DataFrame:
        entry_pnl = self._trade_df['EntryTime'][['PnL']].reindex(self._asset_df.index).fillna(0).cumsum()
        self._equity['Entry_Equity'] = entry_pnl['PnL'] + 10000
        return self._equity['Entry_Equity']
    def sharpe_ratio(self, print_result= False) -> float:
        
    
    
        
    

df = pd.read_csv(
    r"C:\Users\Mr.Dat\Desktop\Projects\Data\2026.6.25XAUUSD_ftmo-H1-Forex_245.csv",
    index_col=0,
    parse_dates=True,
)
required_times = [0, 8]
valid_days = set()
for date, group in df.groupby(df.index.date):
    times = set(group.index.hour)
    if all(t in times for t in required_times):
        valid_days.add(date)  # Using a set is much faster

date_series = pd.Series(df.index.date, index=df.index)
df["Valid_Day"] = date_series.map(lambda date: date in valid_days)
df["Signal"] = 0

buy: pd.Series = (df["Valid_Day"]) & (df.index.hour == 1)
sell: pd.Series = df.index.hour == 9
df.loc[buy, "Signal"] = 1
df.loc[sell, "Signal"] = -1


# backtesting
class SeasonalityStrategy(Strategy):
    def init(self):
        self.signal = self.I(lambda x: x, self.data.Signal)

    def next(self):
        if self.signal[-1] == 1:
            self.buy(size=0.01)
        if self.signal[-1] == -1:
            self.position.close()


bt = Backtest(df, SeasonalityStrategy, cash=10000, margin=0.01)
stats = bt.run()
trades_df = stats._trades[["PnL", "ReturnPct", "EntryTime", "ExitTime"]]
pnl = trades_df.set_index("ExitTime")[["PnL"]]
pnl = pnl.reindex(df.index).fillna(0).cumsum()
equity_df = pd.DataFrame(index=df.index)
equity_df['Exit_Equity'] = 10000 + pnl.PnL
equity_df['Entry_Equity']



metrics = Asset_Metrics(df, "Close")
