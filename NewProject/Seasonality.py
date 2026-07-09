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

        # Exclude weekend data
        if "Weekday" in df.columns:
            df = df[~df["Weekday"].isin(["Saturday", "Sunday"])]

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
        significant = significant.sort_values("T-stats", key=abs, ascending=False)
        return significant, stats

    def stats_hour(self, plot: bool = False):
        significant, stats = self.seasonality_stats("Hour")
        print(significant.head(10))

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

    def stats_weekday(self, plot: bool = False):
        significant, stats = self.seasonality_stats("Weekday")
        print(significant.head(10))

        if plot:
            # Order weekdays properly
            weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

            stats["Group"] = pd.Categorical(
                stats["Group"], categories=weekday_order, ordered=True
            )
            stats = stats.sort_values("Group")

            fig, ax = plt.subplots(2, 2, figsize=(20, 12))

            ax[0, 0].plot(stats["Group"], stats["Mean Return %"], marker="o")
            ax[0, 0].axhline(0, linestyle="--", color="gray")
            ax[0, 0].set_title("Mean Return by Weekday (Line)")
            ax[0, 0].set_ylabel("Mean Return %")
            ax[0, 0].grid(True)
            ax[0, 0].tick_params(axis="x", rotation=45)

            ax[0, 1].plot(stats["Group"], stats["Median Return %"], marker="o")
            ax[0, 1].axhline(0, linestyle="--", color="gray")
            ax[0, 1].set_title("Median Return by Weekday (Line)")
            ax[0, 1].set_ylabel("Median Return %")
            ax[0, 1].grid(True)
            ax[0, 1].tick_params(axis="x", rotation=45)

            ax[1, 0].bar(stats["Group"], stats["Mean Return %"])
            ax[1, 0].axhline(0, linestyle="--", color="gray")
            ax[1, 0].set_title("Mean Return by Weekday (Bar)")
            ax[1, 0].set_ylabel("Mean Return %")
            ax[1, 0].grid(True)
            ax[1, 0].tick_params(axis="x", rotation=45)

            ax[1, 1].bar(stats["Group"], stats["Median Return %"])
            ax[1, 1].axhline(0, linestyle="--", color="gray")
            ax[1, 1].set_title("Median Return by Weekday (Bar)")
            ax[1, 1].set_ylabel("Median Return %")
            ax[1, 1].grid(True)
            ax[1, 1].tick_params(axis="x", rotation=45)

            plt.tight_layout()
            plt.show()

        return significant

    def stats_weekday_hour(self, plot: bool = False):
        significant, stats = self.seasonality_stats(["Weekday", "Hour"])
        print(significant.head(10))

        if plot:
            # Split tuple Group into separate columns
            stats[["Weekday", "Hour"]] = pd.DataFrame(
                stats["Group"].tolist(), index=stats.index
            )

            # Pivot for heatmap
            weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            pivot = stats.pivot_table(
                values="Mean Return %",
                index="Hour",
                columns="Weekday",
                aggfunc="first",
            )
            pivot = pivot[weekday_order]

            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(
                pivot,
                cmap="RdYlGn",
                center=0,
                annot=True,
                fmt=".3f",
                ax=ax,
            )
            ax.set_title("Mean Return % by Hour and Weekday")
            ax.set_ylabel("Hour")
            ax.set_xlabel("Weekday")
            plt.tight_layout()
            plt.show()

        return significant

    def stats_month(self, plot: bool = False):
        significant, stats = self.seasonality_stats("Month")
        print(significant.head(10))

        if plot:
            fig, ax = plt.subplots(1, 2, figsize=(12, 8))

            ax[0].plot(stats["Group"], stats["Mean Return %"], marker="o")
            ax[0].axhline(0, linestyle="--", color="gray")
            ax[0].set_title("Mean Return by Month")
            ax[0].set_xlabel("Month")
            ax[0].set_ylabel("Mean Return %")
            ax[0].set_xticks(range(1, 13))
            ax[0].grid(True)

            ax[1].plot(stats["Group"], stats["Median Return %"], marker="o")
            ax[1].axhline(0, linestyle="--", color="gray")
            ax[1].set_title("Median Return by Month")
            ax[1].set_xlabel("Month")
            ax[1].set_ylabel("Median Return %")
            ax[1].set_xticks(range(1, 13))
            ax[1].grid(True)

            plt.tight_layout()
            plt.show()

        return significant

    def stats_month_day(self, plot: bool = False):
        significant, stats = self.seasonality_stats(["Month", "Day"])
        print(significant.head(10))

        if plot:
            # Split tuple Group into separate columns
            stats[["Month", "Day"]] = pd.DataFrame(
                stats["Group"].tolist(), index=stats.index
            )

            # Pivot for heatmap
            pivot = stats.pivot_table(
                values="Mean Return %",
                index="Day",
                columns="Month",
                aggfunc="first",
            )

            fig, ax = plt.subplots(figsize=(14, 8))
            sns.heatmap(
                pivot,
                cmap="RdYlGn",
                center=0,
                annot=True,
                fmt=".3f",
                ax=ax,
            )
            ax.set_title("Mean Return % by Month and Day")
            ax.set_xlabel("Month")
            ax.set_ylabel("Day")
            plt.tight_layout()
            plt.show()

        return significant

    def stats_monthday(self, plot: bool = False):
        significant, stats = self.seasonality_stats("Day")
        print(significant.head(10))

        if plot:
            fig, ax = plt.subplots(1, 2, figsize=(20, 8))

            ax[0].plot(stats["Group"], stats["Mean Return %"], marker="o")
            ax[0].axhline(0, linestyle="--", color="gray")
            ax[0].set_title("Mean Return by Day of Month")
            ax[0].set_xlabel("Day")
            ax[0].set_ylabel("Mean Return %")
            ax[0].set_xticks(range(1, 32))
            ax[0].tick_params(axis="x", rotation=90)
            ax[0].grid(True)

            ax[1].plot(stats["Group"], stats["Median Return %"], marker="o")
            ax[1].axhline(0, linestyle="--", color="gray")
            ax[1].set_title("Median Return by Day of Month")
            ax[1].set_xlabel("Day")
            ax[1].set_ylabel("Median Return %")
            ax[1].set_xticks(range(1, 32))
            ax[1].tick_params(axis="x", rotation=90)
            ax[1].grid(True)

            plt.tight_layout()
            plt.show()

        return significant

    def stats_session(self, plot: bool = False):
        significant, stats = self.seasonality_stats("Session")
        print(significant.head(10))

        if plot:
            # Order sessions properly
            session_order = ["Asian", "London", "Overlap", "New York"]
            stats["Group"] = pd.Categorical(
                stats["Group"], categories=session_order, ordered=True
            )
            stats = stats.sort_values("Group")

            fig, ax = plt.subplots(2, 2, figsize=(12, 6))

            ax[0, 0].plot(stats["Group"], stats["Mean Return %"], marker="o")
            ax[0, 0].axhline(0, linestyle="--", color="gray")
            ax[0, 0].set_title("Mean Return by Session (Line)")
            ax[0, 0].set_ylabel("Mean Return %")
            ax[0, 0].grid(True)
            ax[0, 0].tick_params(axis="x", rotation=45)

            ax[0, 1].plot(stats["Group"], stats["Median Return %"], marker="o")
            ax[0, 1].axhline(0, linestyle="--", color="gray")
            ax[0, 1].set_title("Median Return by Session (Line)")
            ax[0, 1].set_ylabel("Median Return %")
            ax[0, 1].grid(True)
            ax[0, 1].tick_params(axis="x", rotation=45)

            ax[1, 0].bar(stats["Group"], stats["Mean Return %"])
            ax[1, 0].axhline(0, linestyle="--", color="gray")
            ax[1, 0].set_title("Mean Return by Session (Bar)")
            ax[1, 0].set_ylabel("Mean Return %")
            ax[1, 0].grid(True)
            ax[1, 0].tick_params(axis="x", rotation=45)

            ax[1, 1].bar(stats["Group"], stats["Median Return %"])
            ax[1, 1].axhline(0, linestyle="--", color="gray")
            ax[1, 1].set_title("Median Return by Session (Bar)")
            ax[1, 1].set_ylabel("Median Return %")
            ax[1, 1].grid(True)
            ax[1, 1].tick_params(axis="x", rotation=45)

            plt.tight_layout()
            plt.show()

        return significant

    def stats_session_hour(self, plot: bool = False):
        significant, stats = self.seasonality_stats(["Session", "Hour"])
        print(significant.head(10))

        if plot:
            # Split tuple Group into separate columns
            stats[["Session", "Hour"]] = pd.DataFrame(
                stats["Group"].tolist(), index=stats.index
            )

            # Order sessions properly
            session_order = ["Asian", "London", "Overlap", "New York"]

            fig, ax = plt.subplots(2, 2, figsize=(12, 6))

            for i, session in enumerate(session_order):
                row, col = i // 2, i % 2
                session_data = stats[stats["Session"] == session]
                ax[row, col].plot(
                    session_data["Hour"], session_data["Mean Return %"], marker="o"
                )
                ax[row, col].axhline(0, linestyle="--", color="gray")
                ax[row, col].set_title(f"{session} Session - Mean Return by Hour")
                ax[row, col].set_xlabel("Hour")
                ax[row, col].set_ylabel("Mean Return %")
                ax[row, col].grid(True)

            plt.tight_layout()
            plt.show()

        return significant

    def stats_session_weekday(self, plot: bool = False):
        significant, stats = self.seasonality_stats(["Session", "Weekday"])
        print(significant.head(10))

        if plot:
            # Split tuple Group into separate columns
            stats[["Session", "Weekday"]] = pd.DataFrame(
                stats["Group"].tolist(), index=stats.index
            )

            # Pivot for heatmap
            session_order = ["Asian", "London", "Overlap", "New York"]
            weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            pivot = stats.pivot_table(
                values="Mean Return %",
                index="Weekday",
                columns="Session",
                aggfunc="first",
            )
            pivot = pivot.reindex(weekday_order)
            pivot = pivot[session_order]

            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(
                pivot,
                cmap="RdYlGn",
                center=0,
                annot=True,
                fmt=".3f",
                ax=ax,
            )
            ax.set_title("Mean Return % by Session and Weekday")
            ax.set_xlabel("Session")
            ax.set_ylabel("Weekday")
            plt.tight_layout()
            plt.show()

        return significant


if __name__ == "__main__":
    df_1h = pd.read_csv(
        r"C:\Users\Mr.Dat\Desktop\Projects\Data\2026.6.25XAUUSD_ftmo-H1-Forex_245.csv",
        index_col=0,
        parse_dates=True,
    )
    season = Seasonality(df_1h)
