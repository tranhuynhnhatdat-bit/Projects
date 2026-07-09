import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

    def seasonality_stats(self, group_cols):
        df = self.return_1h.copy()

        # Dynamically add columns derived from the datetime index
        if isinstance(group_cols, str):
            group_cols = [group_cols]

        # Ensure Hour column exists if Session needs it
        if "Session" in group_cols and "Hour" not in df.columns:
            df["Hour"] = df.index.hour

        for col in group_cols:
            if col == "Hour" and col not in df.columns:
                df["Hour"] = df.index.hour
            elif col == "Month" and col not in df.columns:
                df["Month"] = df.index.month
            elif col == "Day" and col not in df.columns:
                df["Day"] = df.index.day
            elif col == "Session" and col not in df.columns:

                def assign_session(hour):
                    if 1 <= hour <= 9:
                        return "Asian"
                    elif 10 <= hour <= 13:
                        return "London"
                    elif 14 <= hour <= 18:
                        return "Overlap"
                    elif 19 <= hour <= 23:
                        return "New York"
                    return None

                df["Session"] = df["Hour"].apply(assign_session)
                df = df.dropna(subset=["Session"])

        results = []
        for name, group in df.groupby(group_cols):
            returns = group["Return"]
            t_stat, p_value = ttest_1samp(returns, popmean=0)
            # Unpack tuple if single-column group
            group_name = name[0] if isinstance(name, tuple) and len(name) == 1 else name
            results.append(
                {
                    "Group": group_name,
                    "Count": len(returns),
                    "Mean Return %": returns.mean() * 100,
                    "Median Return %": returns.median() * 100,
                    "T-stats": t_stat,
                    "P-value": p_value,
                    "Win Rate %": (returns >= 0).mean() * 100,
                }
            )

        stats = pd.DataFrame(results)
        significant = stats[(abs(stats["T-stats"]) > 2) & (stats["P-value"] < 0.05)]
        return significant, stats

    def stats_hour(self, plot: bool = False):
        significant, stats = self.seasonality_stats("Hour")

        if plot:
            fig, ax = plt.subplots(1, 2, figsize=(12, 8))

            ax[0].plot(stats["Group"], stats["Mean Return %"], marker="o")
            ax[0].axhline(0, linestyle="--", color="gray")
            ax[0].set_title("Mean Return by Hour")
            ax[0].set_xlabel("Hour")
            ax[0].set_ylabel("Mean Return %")
            ax[0].set_xticks(range(24))
            ax[0].grid(True)

            ax[1].plot(stats["Group"], stats["Median Return %"], marker="o")
            ax[1].axhline(0, linestyle="--", color="gray")
            ax[1].set_title("Median Return by Hour")
            ax[1].set_xlabel("Hour")
            ax[1].set_ylabel("Median Return %")
            ax[1].set_xticks(range(24))
            ax[1].grid(True)

            plt.tight_layout()
            plt.show()

        return significant

    def stats_weekday(self, visualization: bool = False):
        significant, stats = self.seasonality_stats("Weekday")

        return significant

    def stats_weekday_hour(self, visualization: bool = False):
        significant, stats = self.seasonality_stats(["Weekday", "Hour"])

        return significant

    def stats_month(self, visualization: bool = False):
        significant, stats = self.seasonality_stats("Month")

        return significant

    def stats_month_day(self, visualization: bool = False):
        significant, stats = self.seasonality_stats(["Month", "Day"])

        return significant

    def stats_session(self, visualization: bool = False):
        significant, stats = self.seasonality_stats("Session")

        return significant

    def stats_session_weekday(self, visualization: bool = False):
        significant, stats = self.seasonality_stats(["Session", "Weekday"])

        return significant


if __name__ == "__main__":
    df_1h = pd.read_csv(
        r"C:\Users\Mr.Dat\Desktop\Projects\Data\2026.6.25XAUUSD_ftmo-H1-Forex_245.csv",
        index_col=0,
        parse_dates=True,
    )
    season = Seasonality(df_1h)
