from backtesting import Backtest, Strategy


def backtest(df, run_backtest=False):

    if not run_backtest:
        return

    def passthrough(x):
        return x

    class SignalStrategy(Strategy):
        def init(self):
            self.signal = self.I(passthrough, self.data.Signal)

        def next(self):

            if self.signal[-1] == 1 and not self.position:
                self.buy(size=0.01)

            elif self.signal[-1] == -1 and self.position:
                self.position.close()

    bt = Backtest(df, SignalStrategy, cash=10000, margin=0.01)

    stats = bt.run()

    print("--- Base Model ---")
    print(f"Sharpe: {stats['Sharpe Ratio']:.2f}")
    print(f"Win Rate: {stats['Win Rate [%]']:.2f}")
    print(f"Max Drawdown: {stats['Max. Drawdown [%]']:.2f}")

    return stats
