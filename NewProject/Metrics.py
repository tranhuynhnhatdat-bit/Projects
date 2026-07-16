import pandas as pd
import numpy as np


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
    def years(self) -> float:
        if self._years is None:
            self._years = (self._df.index[-1] - self._df.index[0]).days / 365.25
        return self._years

    def sharpe_ratio(self, print_result: bool = False) -> float:
        sharpe = float(
            (self.daily_return.mean() / self.daily_return.std()) * np.sqrt(252)
        )
        if print_result:
            print(f"Sharpe Ratio is: {sharpe:.2f}")
        return sharpe

    def sortino_ratio(self, print_result: bool = False) -> float:
        downside_return = self.daily_return.where(self.daily_return <= 0, 0)
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
        is_stagnant = (prices < running_max).values
        boundaries = np.diff(np.concatenate(([0], is_stagnant.astype(int), [0])))
        run_starts = np.where(boundaries == 1)[0]
        run_ends = np.where(boundaries == -1)[0]
        if len(run_starts) > 0 and len(run_ends) > 0:
            # Clip to valid index range to handle edge case where the
            # stagnation run extends to the last data point (run_ends can
            # contain index N which is out of bounds for a size-N array).
            n = len(prices.index)
            run_starts = np.clip(run_starts, 0, n - 1)
            run_ends = np.clip(run_ends, 0, n - 1)
            durations = prices.index[run_ends] - prices.index[run_starts]
            max_dd = int(durations.max().days)
        else:
            max_dd = 0
        if print_result:
            print(f"Stagnation in days: {max_dd}")
        return max_dd

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

    def __str__(self) -> str:
        """Print all asset metrics in a formatted table."""
        lines = [
            "=" * 67,
            "  Asset Metrics",
            "=" * 67,
            f"  Column:                        {self._column}",
            f"  Sharpe Ratio:                  {self.sharpe_ratio():>8.2f}",
            f"  Sortino Ratio:                 {self.sortino_ratio():>8.2f}",
            f"  Maximum Drawdown:              {self.maximum_dd() * 100:>7.2f}%",
            f"  Total Return:                  {self.total_return() * 100:>7.2f}%",
            f"  CAGR:                          {self.CAGR() * 100:>7.2f}%",
            f"  Annualized Volatility:         {self.annualized_volatility() * 100:>7.2f}%",
            f"  Profit Factor:                 {self.profit_factor():>8.2f}",
            f"  Win Rate:                      {self.win_rate() * 100:>7.2f}%",
            f"  Expectancy:                    {self.expectancy() * 100:>7.2f}%",
            f"  Max Stagnation Days:           {self.max_stagnation_days():>8}",
            f"  Return/Drawdown Ratio:         {self.return_drawdown_ratio():>8.2f}",
            "=" * 67,
        ]
        self.plot()
        return "\n".join(lines)

    def plot(self) -> None:
        """Plot the daily OHLC close price chart (resampled from any timeframe)."""
        import matplotlib.pyplot as plt

        ohlc_cols = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
        daily_ohlc = self._df.resample("D").agg(ohlc_cols).dropna()

        plt.figure(figsize=(12, 6))
        plt.plot(daily_ohlc.index, daily_ohlc["Close"], label=f"Close ({self._column})")
        plt.title("Asset Price (Daily)")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.show()


class Trade_Metrics:
    def __init__(
        self,
        asset_df: pd.DataFrame,
        pf,
        exit_or_entry_time: bool = True,
        initial_capital: float = 10000,
    ):
        self._asset_df = asset_df
        records = pf.trades.records
        self._trade_df = pd.DataFrame(
            {
                "PnL": records["pnl"],
                "ReturnPct": records["return"],
                "EntryTime": asset_df.index[records["entry_idx"]],
                "ExitTime": asset_df.index[records["exit_idx"]],
            }
        )
        self._exit_or_entry_time = exit_or_entry_time
        self._initial_capital = initial_capital
        self._equity_df = pd.DataFrame(index=self._asset_df.index)
        self._daily_return = None
        self._years = None
        self._trade_return_pct = None

    def __getitem__(self, key: int):
        return self._trade_df.iloc[key]

    @property
    def pnl(self) -> pd.Series:
        return self._trade_df["PnL"]

    @property
    def return_pct(self) -> pd.Series:
        return self._trade_df["ReturnPct"]

    @property
    def entrytime(self) -> pd.Series:
        return self._trade_df["EntryTime"]

    @property
    def exittime(self) -> pd.Series:
        return self._trade_df["ExitTime"]

    @property
    def number_of_trades(self) -> int:
        return len(self._trade_df)

    @property
    def exit_equity(self) -> pd.Series:
        """Equity curve using exit timestamps to assign PnL."""
        if "Exit_Equity" in self._equity_df.columns:
            return self._equity_df["Exit_Equity"]
        exit_pnl = (
            self._trade_df.set_index("ExitTime")[["PnL"]]
            .reindex(self._asset_df.index)
            .fillna(0)
            .cumsum()
        )
        self._equity_df["Exit_Equity"] = exit_pnl["PnL"] + self._initial_capital
        return self._equity_df["Exit_Equity"]

    @property
    def entry_equity(self) -> pd.Series:
        """Equity curve using entry timestamps to assign PnL."""
        if "Entry_Equity" in self._equity_df.columns:
            return self._equity_df["Entry_Equity"]
        entry_pnl = (
            self._trade_df.set_index("EntryTime")[["PnL"]]
            .reindex(self._asset_df.index)
            .fillna(0)
            .cumsum()
        )
        self._equity_df["Entry_Equity"] = entry_pnl["PnL"] + self._initial_capital
        return self._equity_df["Entry_Equity"]

    @property
    def equity(self) -> pd.Series:
        """Return the equity curve based on the exit_or_entry_time flag."""
        if self._exit_or_entry_time:
            return self.exit_equity
        return self.entry_equity

    @property
    def daily_return(self) -> pd.Series:
        """Daily returns of the equity curve."""
        if self._daily_return is None:
            self._daily_return = self.equity.pct_change().dropna()
        return self._daily_return

    @property
    def years(self) -> float:
        """Number of years spanned by the data."""
        if self._years is None:
            self._years = (
                self._asset_df.index[-1] - self._asset_df.index[0]
            ).days / 365.25
        return self._years

    @property
    def trade_return_pct(self) -> pd.Series:
        """Individual trade returns (non-zero), used for trade-level metrics."""
        if self._trade_return_pct is None:
            self._trade_return_pct = self._trade_df["ReturnPct"][
                self._trade_df["ReturnPct"] != 0
            ]
        return self._trade_return_pct

    @property
    def average_win(self) -> float:
        """Average winning trade in dollars."""
        wins = self.pnl[self.pnl > 0]
        return float(wins.mean()) if len(wins) > 0 else 0.0

    @property
    def average_loss(self) -> float:
        """Average losing trade in dollars (negative value)."""
        losses = self.pnl[self.pnl < 0]
        return float(losses.mean()) if len(losses) > 0 else 0.0

    @property
    def largest_win(self) -> float:
        """Largest winning trade in dollars."""
        return float(self.pnl.max())

    @property
    def largest_loss(self) -> float:
        """Largest losing trade in dollars (negative value)."""
        return float(self.pnl.min())

    @property
    def average_holding_time(self) -> str:
        """Average holding time per trade. In hours if < 72h, else 'X days, Y hours'."""
        holding_times = self._trade_df["ExitTime"] - self._trade_df["EntryTime"]
        avg_td = holding_times.mean()
        total_hours = avg_td.total_seconds() / 3600
        if total_hours < 72:
            return f"{total_hours:.1f} hours"
        days = int(total_hours // 24)
        hours = int(total_hours % 24)
        return f"{days} days, {hours} hours"

    # ---------- Equity-based metrics (same logic as Asset_Metrics) ----------

    def sharpe_ratio(self, print_result: bool = False) -> float:
        sharpe = float(
            (self.daily_return.mean() / self.daily_return.std()) * np.sqrt(252)
        )
        if print_result:
            print(f"Sharpe Ratio is: {sharpe:.2f}")
        return sharpe

    def sortino_ratio(self, print_result: bool = False) -> float:
        downside_return = self.daily_return.where(self.daily_return <= 0, 0)
        downside_std = downside_return.std()
        if downside_std == 0.0:
            return 0.0
        sortino = float((self.daily_return.mean() / downside_std) * np.sqrt(252))
        if print_result:
            print(f"Sortino Ratio is: {sortino:.2f}")
        return sortino

    def maximum_dd(self, print_result: bool = False) -> float:
        running_max = self.equity.cummax()
        drawdown = self.equity / running_max - 1
        max_drawdown = drawdown.min()
        if print_result:
            print(f"Maximum Drawdown is: {float(max_drawdown * 100):.2f}%")
        return float(max_drawdown)

    def total_return(self, print_result: bool = False) -> float:
        beginning_equity = self.equity.iloc[0]
        ending_equity = self.equity.iloc[-1]

        if beginning_equity == 0:
            return 0.0
        total = float((ending_equity - beginning_equity) / beginning_equity)
        if print_result:
            print(f"Total Return is: {total * 100:.2f}%")
        return total

    def CAGR(self, print_result: bool = False) -> float:
        beginning_equity = self.equity.iloc[0]
        ending_equity = self.equity.iloc[-1]

        cagr = float((ending_equity / beginning_equity) ** (1 / self.years) - 1)
        if print_result:
            print(f"CAGR is: {cagr * 100:.2f}%")
        return cagr

    def max_stagnation_days(self, print_result: bool = False) -> int:
        equity = self.equity
        running_max = equity.cummax()
        is_stagnant = (equity < running_max).values
        boundaries = np.diff(np.concatenate(([0], is_stagnant.astype(int), [0])))
        run_starts = np.where(boundaries == 1)[0]
        run_ends = np.where(boundaries == -1)[0]
        if len(run_starts) > 0 and len(run_ends) > 0:
            # Clip to valid index range to handle edge case where the
            # stagnation run extends to the last data point (run_ends can
            # contain index N which is out of bounds for a size-N array).
            n = len(equity.index)
            run_starts = np.clip(run_starts, 0, n - 1)
            run_ends = np.clip(run_ends, 0, n - 1)
            durations = equity.index[run_ends] - equity.index[run_starts]
            max_dd = int(durations.max().days)
        else:
            max_dd = 0
        if print_result:
            print(f"Stagnation in days: {max_dd}")
        return max_dd

    def annualized_volatility(self, print_result: bool = False) -> float:
        daily_std = self.daily_return.std()
        annual_vol = float(daily_std * np.sqrt(252))
        if print_result:
            print(f"Annualized Volatility is: {annual_vol * 100:.2f}%")
        return annual_vol

    def return_drawdown_ratio(self, print_result: bool = False) -> float:
        return_dd = self.CAGR() / self.maximum_dd()
        if print_result:
            print(f"Return/Drawdown Ratio: {return_dd:.2}")
        return return_dd

    # ---------- Trade-level metrics (from individual trade ReturnPct) ----------

    def profit_factor(self, print_result: bool = False) -> float:
        returns = self.trade_return_pct
        gross_profit = returns[returns > 0].sum()
        gross_loss = returns[returns < 0].sum()

        if gross_loss == 0:
            return 0.0
        profitfactor = float(gross_profit / abs(gross_loss))
        if print_result:
            print(f"Profit factor is: {profitfactor:.2f}")
        return profitfactor

    def win_rate(self, print_result: bool = False) -> float:
        returns = self.trade_return_pct
        winrate = float(returns[returns > 0].count() / returns.count())
        if print_result:
            print(f"Win rate is: {winrate * 100:.2f}%")
        return winrate

    def expectancy(self, print_result: bool = False) -> float:
        returns = self.trade_return_pct
        wins = returns[returns > 0]
        losses = returns[returns < 0]

        win_prob = len(wins) / len(returns)
        loss_prob = 1 - win_prob

        average_win = wins.mean()
        average_loss = abs(losses.mean())

        ev = float((win_prob * average_win) - (loss_prob * average_loss))
        if print_result:
            print(f"Expectancy is: {ev * 100}%")
        return ev

    def __str__(self) -> str:
        """Print all metrics in a formatted table."""
        lines = [
            "=" * 67,
            "  Trade Metrics",
            "=" * 67,
            f"  Number of Trades:              {self.number_of_trades}",
            f"  Sharpe Ratio:                  {self.sharpe_ratio():>8.2f}",
            f"  Sortino Ratio:                 {self.sortino_ratio():>8.2f}",
            f"  Maximum Drawdown:              {self.maximum_dd() * 100:>7.2f}%",
            f"  Total Return:                  {self.total_return() * 100:>7.2f}%",
            f"  CAGR:                          {self.CAGR() * 100:>7.2f}%",
            f"  Annualized Volatility:         {self.annualized_volatility() * 100:>7.2f}%",
            f"  Profit Factor:                 {self.profit_factor():>8.2f}",
            f"  Win Rate:                      {self.win_rate() * 100:>7.2f}%",
            f"  Expectancy:                    {self.expectancy() * 100:>7.2f}%",
            f"  Max Stagnation Days:           {self.max_stagnation_days():>8}",
            f"  Return/Drawdown Ratio:         {self.return_drawdown_ratio():>8.2f}",
            f"  Average Win:                   ${self.average_win:>7.2f}",
            f"  Average Loss:                 -${abs(self.average_loss):>7.2f}",
            f"  Largest Win:                   ${self.largest_win:>7.2f}",
            f"  Largest Loss:                 -${abs(self.largest_loss):>7.2f}",
            f"  Average Holding Time:          {self.average_holding_time}",
            "=" * 67,
        ]
        self.plot()
        return "\n".join(lines)

    def plot(self) -> None:
        """Plot two charts side-by-side: asset close price and exit equity curve."""
        import matplotlib.pyplot as plt

        ohlc_cols = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
        daily_ohlc = self._asset_df.resample("D").agg(ohlc_cols).dropna()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        ax1.plot(daily_ohlc.index, daily_ohlc["Close"])
        ax1.set_title("Asset Close Price (Daily)")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price")
        ax1.grid(True)

        exit_equity = self.exit_equity
        ax2.plot(exit_equity.index, exit_equity.values)
        ax2.set_title("Equity Curve (Exit-based)")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Equity ($)")
        ax2.grid(True)

        plt.tight_layout()
        plt.show()


# ============================================================
# Test / Demo Code
# ============================================================

if __name__ == "__main__":
    from backtesting import Strategy, Backtest

    df = pd.read_csv(
        r"C:\Users\Mr.Dat\Desktop\Projects\Data\2026.6.25XAUUSD_ftmo-H1-Forex_245.csv",
        index_col=0,
        parse_dates=True,
    )
    required_times = [0, 1, 8, 9]
    valid_days = set()
    for date, group in df.groupby(df.index.date):
        times = set(group.index.hour)
        if all(t in times for t in required_times):
            valid_days.add(date)

    date_series = pd.Series(df.index.date, index=df.index)
    df["Valid_Day"] = date_series.map(lambda date: date in valid_days)
    df["Signal"] = 0

    buy = (df["Valid_Day"]) & (df.index.hour == 0)
    sell = df.index.hour == 8
    df.loc[buy, "Signal"] = 1
    df.loc[sell, "Signal"] = -1

    # Backtesting
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

    # Test Asset_Metrics with __str__
    metrics = Asset_Metrics(df, "Close")
    print(metrics)

    print()

    # Test Trade_Metrics with __str__ (vectorbt portfolio)
    # pf = vbt.Portfolio.from_orders(...)  # create your vectorbt portfolio
    # trade_metrics = Trade_Metrics(
    #     df, pf, exit_or_entry_time=True, initial_capital=10000
    # )
    # print(trade_metrics)
