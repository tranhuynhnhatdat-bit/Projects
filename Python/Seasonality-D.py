import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp


def analyze_weekday_effect(file_path):
    """
    Analyze weekday effects in daily market data.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.
    """

    # Load data
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Add weekday and returns
    df["Weekday"] = df.index.day_name()
    df["Return"] = df["Close"].pct_change()

    # Remove NaN and weekends
    df = df.dropna()
    df = df[~df["Weekday"].isin(["Saturday", "Sunday"])]
    # Extract day of month
    df["Day"] = df.index.day

    # Calculate median return for each day of month
    monthly_seasonality = df.groupby("Day")["Return"].mean() * 100
    # Extract month
    df["Month"] = df.index.month

    # Calculate mean return for each month
    monthly_returns = df.groupby("Month")["Return"].mean() * 100

    # Rename month numbers to names
    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    monthly_returns.index = month_names
    # Create summary table
    summary = df.groupby("Weekday")["Return"].agg(
        Wins=lambda x: (x > 0).sum(),
        Losses=lambda x: (x < 0).sum(),
        Mean=lambda x: x.mean() * 100,
    )

    # Order weekdays
    summary = summary.reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])

    # Calculate Win/Loss Ratio
    summary["Win/Loss Ratio"] = summary["Wins"] / summary["Losses"]

    # Print statistics
    print("\nWeekday Statistics")
    print("-" * 50)
    for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
        returns = df[df["Weekday"] == day]["Return"]

        t_stat, p_value = ttest_1samp(returns, popmean=0)

        print(
            f"{day}: "
            f"Mean = {returns.mean() * 100:.4f}% | "
            f"Win/Loss Ratio = {summary.loc[day]['Win/Loss Ratio']:.4f} |"
            f"t-stat = {t_stat:.2f} | "
            f"p-value = {p_value:.4f}"
        )
    # Create monthly summary table
    monthly_summary = df.groupby("Month")["Return"].agg(
        Wins=lambda x: (x > 0).sum(),
        Losses=lambda x: (x < 0).sum(),
        Mean=lambda x: x.mean() * 100,
    )

    # Add Win/Loss Ratio
    monthly_summary["Win/Loss Ratio"] = (
        monthly_summary["Wins"] / monthly_summary["Losses"]
    )

    # Replace month numbers with names
    monthly_summary.index = month_names
    print("\nMonthly Statistics")
    print("-" * 60)

    for month in month_names:
        returns = df[df["Month"] == month_names.index(month) + 1]["Return"]

        t_stat, p_value = ttest_1samp(returns, popmean=0)

        print(
            f"{month:<5}"
            f" Mean = {returns.mean() * 100:.4f}% | "
            f"Win/Loss Ratio = {monthly_summary.loc[month, 'Win/Loss Ratio']:.2f} | "
            f"t-stat = {t_stat:.2f} | "
            f"p-value = {p_value:.4f}"
        )
    # Plot charts
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))

    # Chart 1: Wins vs Losses
    summary[["Wins", "Losses"]].plot(kind="bar", ax=ax[0, 0])

    ax[0, 0].set_title("Wins and Losses by Weekday")
    ax[0, 0].set_xlabel("Weekday")
    ax[0, 0].set_ylabel("Count")
    ax[0, 0].tick_params(axis="x", rotation=0)

    # Chart 2: Median Returns by Weekday
    ax[0, 1].bar(summary.index, summary["Mean"])

    ax[0, 1].set_title("Mean Return by Weekday")
    ax[0, 1].set_xlabel("Weekday")
    ax[0, 1].set_ylabel("Mean Return (%)")
    ax[0, 1].tick_params(axis="x", rotation=0)

    # Chart 3: Monthly Seasonality
    ax[1, 0].plot(monthly_seasonality.index, monthly_seasonality.values, marker="o")

    ax[1, 0].axhline(y=0, linestyle="--")

    ax[1, 0].set_title("Mean Return by Day of Month")
    ax[1, 0].set_xlabel("Day of Month")
    ax[1, 0].set_ylabel("Mean Return (%)")

    # Chart 4: Empty for future use
    # Chart 4: Mean Return by Month
    ax[1, 1].bar(monthly_returns.index, monthly_returns.values)

    ax[1, 1].axhline(y=0, linestyle="--")

    ax[1, 1].set_title("Mean Return by Month")
    ax[1, 1].set_xlabel("Month")
    ax[1, 1].set_ylabel("Mean Return (%)")
    ax[1, 1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()

    return summary


summary = analyze_weekday_effect(
    r"C:\Users\Mr.Dat\Desktop\Projects\Data\2026.6.20XAUUSD_ftmo-D1-Forex_247.csv"
)
