import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp


def analyze_intraday_seasonality(file_path):

    # =====================
    # LOAD & PREPROCESS
    # =====================
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)

    df.columns = df.columns.str.strip()

    df["Return"] = df["Close"].pct_change()
    df = df.dropna()

    df["Hour"] = df.index.hour
    df["Weekday"] = df.index.day_name()

    # Reassign early Saturday hours to Monday
    mask = (df["Weekday"] == "Saturday") & (df["Hour"].between(0, 6))

    df.loc[mask, "Weekday"] = "Monday"

    # Session classification
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

    # =====================
    # STATISTICAL TESTS
    # =====================

    def seasonality_stats(df, group_cols):

        results = []

        for name, group in df.groupby(group_cols):
            returns = group["Return"].dropna()

            t_stat, p_value = ttest_1samp(returns, popmean=0)

            results.append(
                {
                    "Group": name,
                    "Mean Return (%)": returns.mean() * 100,
                    "Count": len(returns),
                    "Win Rate (%)": (returns > 0).mean() * 100,
                    "t-stat": t_stat,
                    "p-value": p_value,
                }
            )

        stats = pd.DataFrame(results)

        significant = stats[
            (stats["p-value"] < 0.05)
            & (abs(stats["t-stat"]) > 2)
            & (stats["Count"] > 1000)
        ]

        return significant

    print("\nHourly Effects")
    print(seasonality_stats(df, "Hour"))

    print("\nWeekday Effects")
    print(seasonality_stats(df, "Weekday"))

    print("\nSession Effects")
    print(seasonality_stats(df, "Session"))

    print("\nSession + Weekday Effects")
    print(seasonality_stats(df, ["Session", "Weekday"]))

    # =====================
    # CREATE TABLES
    # =====================

    hourly_mean = df.groupby("Hour")["Return"].mean() * 100

    hour_weekday = (
        df.pivot_table(values="Return", index="Hour", columns="Weekday", aggfunc="mean")
        * 100
    )

    hour_weekday = hour_weekday[
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    ]

    session_weekday = (
        df.pivot_table(
            values="Return", index="Session", columns="Weekday", aggfunc="mean"
        )
        * 100
    )

    session_weekday = session_weekday.loc[
        ["Asian", "London", "Overlap", "New York"],
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
    ]

    session_mean = df.groupby("Session")["Return"].mean() * 100

    session_mean = session_mean.reindex(["Asian", "London", "Overlap", "New York"])

    # =====================
    # PLOTS
    # =====================

    fig, ax = plt.subplots(2, 2, figsize=(20, 12))

    # Hourly mean
    ax[0, 0].plot(hourly_mean.index, hourly_mean.values, marker="o")

    ax[0, 0].axhline(0, linestyle="--")
    ax[0, 0].grid(True)
    ax[0, 0].set_title("Mean Return by Hour")
    ax[0, 0].set_xlabel("Hour")
    ax[0, 0].set_ylabel("Mean Return (%)")
    ax[0, 0].set_xticks(range(24))

    # Weekday x Hour heatmap
    sns.heatmap(
        hour_weekday, cmap="RdYlGn", center=0, annot=True, fmt=".3f", ax=ax[0, 1]
    )

    ax[0, 1].set_title("Mean Return by Hour and Weekday")

    # Session x Weekday heatmap
    sns.heatmap(
        session_weekday, cmap="RdYlGn", center=0, annot=True, fmt=".3f", ax=ax[1, 0]
    )

    ax[1, 0].set_title("Mean Return by Session and Weekday")

    # Session mean
    ax[1, 1].bar(session_mean.index, session_mean.values)

    ax[1, 1].axhline(0, linestyle="--")
    ax[1, 1].grid(True)

    ax[1, 1].set_title("Mean Return by Session")

    ax[1, 1].set_ylabel("Mean Return (%)")

    plt.tight_layout()
    plt.show()

    return df


analyze_intraday_seasonality(
    r"C:\Users\Mr.Dat\Desktop\Projects\Data\2026.6.23GBPUSD_ftmo-H1-Forex_247.csv"
)
