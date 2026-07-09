import pandas as pd
from GettingData import getting_data_mt5
from scipy.stats import ttest_1samp


class Seasonality:
    def __init__(self, df_1h: pd.DataFrame):
        self._df_1h = df_1h
        self._df_D = (
            self._df_1h.resample("D")
            .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
            .dropna()
        )

    @property
    def return_1h(self) -> pd.DataFrame:
        df_1h = self._df_1h.copy()
        df_1h["Return"] = df_1h["Close"].pct_change()
        df_1h["Weekday"] = df_1h.index.day_name()
        df_1h = df_1h.dropna()
        return df_1h[["Return", "Weekday"]]

    @property
    def return_D(self) -> pd.DataFrame:
        df_D = self._df_D.copy()
        df_D["Return"] = df_D["Close"].pct_change()
        df_D["Weekday"] = df_D.index.day_name()
        df_D = df_D.dropna()
        return df_D[["Return", "Weekday"]]

    def stats_hour(self):
        results = []
        for name, group in self.return_1h.groupby(self.return_1h.index.hour):
            Return = group["Return"]
            t_stats, p_value = ttest_1samp(Return, popmean=0)
            results.append(
                {
                    "Group": name,
                    "Mean Return %": Return.mean() * 100,
                    "Median Return %": Return.median() * 100,
                    "T-stats": t_stats,
                    "P-value": p_value,
                    "Win Rate %": (Return >= 0).mean() * 100,
                }
            )
            stats = pd.DataFrame(results)
            significant = stats[(abs(stats["T-stats"]) > 2) & (stats["P-value"] < 0.05)]
        return significant


if __name__ == "__main__":
    df_1h = pd.read_csv(
        r"C:\Users\Mr.Dat\Desktop\Projects\Data\2026.6.25XAUUSD_ftmo-H1-Forex_245.csv",
        index_col=0,
        parse_dates=True,
    )
    season = Seasonality(df_1h)
